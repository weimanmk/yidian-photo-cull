from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np

from .config import CACHE_DB
from .imaging import file_sequence, photo_id
from .internal_models import (
    BodyObservation,
    DepthObservation,
    FaceObservation,
    PhotoObservation,
    PoseObservation,
    VisualDescriptor,
)


def _array_blob(array: np.ndarray | None) -> bytes | None:
    if array is None:
        return None
    return np.asarray(array, dtype=np.float16).reshape(-1).tobytes()


def _blob_array(blob: bytes | None) -> np.ndarray | None:
    if blob is None:
        return None
    return np.frombuffer(blob, dtype=np.float16).copy()


class FeatureCache:
    """SQLite 特征缓存；只缓存模型输入事实，不缓存人物聚类和最终分组。"""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or CACHE_DB
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._error = ""
        self._initialize()

    @property
    def error(self) -> str:
        return self._error

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS photo_features (
                    cache_key TEXT PRIMARY KEY,
                    photo_id TEXT NOT NULL,
                    pipeline_signature TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    capture_time TEXT,
                    file_sequence INTEGER NOT NULL,
                    phash TEXT NOT NULL,
                    dhash TEXT NOT NULL,
                    layout BLOB NOT NULL,
                    color BLOB NOT NULL,
                    edge BLOB NOT NULL,
                    semantic BLOB,
                    metrics_json TEXT NOT NULL,
                    score REAL NOT NULL,
                    issues_json TEXT NOT NULL,
                    cached_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_photo_features_source ON photo_features(source_path, pipeline_signature);
                CREATE TABLE IF NOT EXISTS face_features (
                    cache_key TEXT NOT NULL,
                    face_index INTEGER NOT NULL,
                    face_id TEXT NOT NULL,
                    bbox_json TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    area_ratio REAL NOT NULL,
                    embedding BLOB,
                    eye_state TEXT NOT NULL,
                    open_probability REAL,
                    sharpness REAL NOT NULL,
                    profile INTEGER NOT NULL,
                    smile_score REAL NOT NULL,
                    high_res_sharpness REAL NOT NULL,
                    eye_sharpness REAL NOT NULL,
                    yaw REAL NOT NULL,
                    pitch REAL NOT NULL,
                    roll REAL NOT NULL,
                    occlusion_risk REAL NOT NULL,
                    expression TEXT NOT NULL DEFAULT 'unknown',
                    expression_confidence REAL,
                    expression_score REAL NOT NULL DEFAULT 0,
                    fiqa_score REAL,
                    PRIMARY KEY(cache_key, face_index),
                    FOREIGN KEY(cache_key) REFERENCES photo_features(cache_key) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS body_features (
                    cache_key TEXT NOT NULL,
                    body_index INTEGER NOT NULL,
                    bbox_json TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    area_ratio REAL NOT NULL,
                    embedding BLOB,
                    detector TEXT NOT NULL,
                    PRIMARY KEY(cache_key, body_index),
                    FOREIGN KEY(cache_key) REFERENCES photo_features(cache_key) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS pose_features (
                    cache_key TEXT NOT NULL,
                    pose_index INTEGER NOT NULL,
                    bbox_json TEXT NOT NULL,
                    detection_confidence REAL NOT NULL,
                    presence_confidence REAL NOT NULL,
                    area_ratio REAL NOT NULL,
                    landmarks_2d BLOB NOT NULL,
                    descriptor BLOB,
                    visibility REAL NOT NULL,
                    foreground_score REAL,
                    model TEXT NOT NULL,
                    PRIMARY KEY(cache_key, pose_index),
                    FOREIGN KEY(cache_key) REFERENCES photo_features(cache_key) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS depth_features (
                    cache_key TEXT PRIMARY KEY,
                    descriptor BLOB,
                    subject_depth REAL,
                    background_depth REAL,
                    foreground_separation REAL NOT NULL,
                    subject_focus_score REAL,
                    background_blur_score REAL,
                    occlusion_risk REAL NOT NULL,
                    subject_confidence REAL NOT NULL,
                    model TEXT NOT NULL,
                    FOREIGN KEY(cache_key) REFERENCES photo_features(cache_key) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS model_versions (
                    name TEXT PRIMARY KEY,
                    signature TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    recorded_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS user_actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id TEXT NOT NULL,
                    photo_id TEXT NOT NULL,
                    group_id TEXT,
                    previous_category TEXT,
                    selected_category TEXT NOT NULL,
                    stars INTEGER,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS vlm_decisions (
                    decision_key TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    decision_json TEXT NOT NULL,
                    raw_response TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            existing_face_columns = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(face_features)").fetchall()
            }
            for column, declaration in (
                ("expression", "TEXT NOT NULL DEFAULT 'unknown'"),
                ("expression_confidence", "REAL"),
                ("expression_score", "REAL NOT NULL DEFAULT 0"),
                ("fiqa_score", "REAL"),
            ):
                if column not in existing_face_columns:
                    connection.execute(f"ALTER TABLE face_features ADD COLUMN {column} {declaration}")
            existing_pose_columns = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(pose_features)").fetchall()
            }
            if "foreground_score" not in existing_pose_columns:
                connection.execute("ALTER TABLE pose_features ADD COLUMN foreground_score REAL")

    @staticmethod
    def _cache_key(path: Path, pipeline_signature: str) -> tuple[str, str]:
        identifier = photo_id(path)
        import hashlib

        signature_hash = hashlib.sha256(pipeline_signature.encode("utf-8")).hexdigest()[:16]
        return f"{identifier}:{signature_hash}", identifier

    def load(self, path: Path, source_root: Path, pipeline_signature: str) -> PhotoObservation | None:
        cache_key, identifier = self._cache_key(path, pipeline_signature)
        try:
            stat = path.stat()
        except OSError:
            return None
        try:
            with self._lock, self._connect() as connection:
                row = connection.execute("SELECT * FROM photo_features WHERE cache_key = ?", (cache_key,)).fetchone()
                if row is None:
                    return None
                if int(row["file_size"]) != stat.st_size or int(row["mtime_ns"]) != stat.st_mtime_ns:
                    return None
                face_rows = connection.execute(
                    "SELECT * FROM face_features WHERE cache_key = ? ORDER BY face_index", (cache_key,)
                ).fetchall()
                body_rows = connection.execute(
                    "SELECT * FROM body_features WHERE cache_key = ? ORDER BY body_index", (cache_key,)
                ).fetchall()
                pose_rows = connection.execute(
                    "SELECT * FROM pose_features WHERE cache_key = ? ORDER BY pose_index", (cache_key,)
                ).fetchall()
                depth_row = connection.execute(
                    "SELECT * FROM depth_features WHERE cache_key = ?", (cache_key,)
                ).fetchone()
        except sqlite3.Error as exc:
            self._error = str(exc)
            return None

        faces: list[FaceObservation] = []
        for face in face_rows:
            embedding = _blob_array(face["embedding"])
            if embedding is not None:
                embedding = embedding.astype(np.float32)
                norm = float(np.linalg.norm(embedding))
                embedding = embedding / norm if norm > 1e-8 else None
            faces.append(
                FaceObservation(
                    face_id=face["face_id"],
                    bbox=tuple(json.loads(face["bbox_json"])),
                    confidence=float(face["confidence"]),
                    area_ratio=float(face["area_ratio"]),
                    embedding=embedding,
                    eye_state=face["eye_state"],
                    open_probability=None if face["open_probability"] is None else float(face["open_probability"]),
                    sharpness=float(face["sharpness"]),
                    profile=bool(face["profile"]),
                    smile_score=float(face["smile_score"]),
                    high_res_sharpness=float(face["high_res_sharpness"]),
                    eye_sharpness=float(face["eye_sharpness"]),
                    yaw=float(face["yaw"]),
                    pitch=float(face["pitch"]),
                    roll=float(face["roll"]),
                    occlusion_risk=float(face["occlusion_risk"]),
                    expression=str(face["expression"]),
                    expression_confidence=(
                        None if face["expression_confidence"] is None else float(face["expression_confidence"])
                    ),
                    expression_score=float(face["expression_score"]),
                    fiqa_score=None if face["fiqa_score"] is None else float(face["fiqa_score"]),
                )
            )
        bodies: list[BodyObservation] = []
        for body in body_rows:
            embedding = _blob_array(body["embedding"])
            if embedding is not None:
                embedding = embedding.astype(np.float32)
                norm = float(np.linalg.norm(embedding))
                embedding = embedding / norm if norm > 1e-8 else None
            bodies.append(
                BodyObservation(
                    bbox=tuple(json.loads(body["bbox_json"])),
                    confidence=float(body["confidence"]),
                    area_ratio=float(body["area_ratio"]),
                    embedding=embedding,
                    detector=str(body["detector"]),
                )
            )
        poses: list[PoseObservation] = []
        for pose in pose_rows:
            descriptor = _blob_array(pose["descriptor"])
            if descriptor is not None:
                descriptor = descriptor.astype(np.float32)
                norm = float(np.linalg.norm(descriptor))
                descriptor = descriptor / norm if norm > 1e-8 else None
            landmarks = _blob_array(pose["landmarks_2d"])
            if landmarks is None or landmarks.size % 3:
                continue
            poses.append(
                PoseObservation(
                    bbox=tuple(json.loads(pose["bbox_json"])),
                    detection_confidence=float(pose["detection_confidence"]),
                    presence_confidence=float(pose["presence_confidence"]),
                    area_ratio=float(pose["area_ratio"]),
                    landmarks_2d=landmarks.astype(np.float32).reshape(-1, 3),
                    descriptor=descriptor,
                    visibility=float(pose["visibility"]),
                    model=str(pose["model"]),
                    foreground_score=(
                        None if pose["foreground_score"] is None else float(pose["foreground_score"])
                    ),
                )
            )
        depth_observation: DepthObservation | None = None
        if depth_row is not None:
            depth_descriptor = _blob_array(depth_row["descriptor"])
            if depth_descriptor is not None:
                depth_descriptor = depth_descriptor.astype(np.float32)
                norm = float(np.linalg.norm(depth_descriptor))
                depth_descriptor = depth_descriptor / norm if norm > 1e-8 else None
            depth_observation = DepthObservation(
                descriptor=depth_descriptor,
                subject_depth=(
                    None if depth_row["subject_depth"] is None else float(depth_row["subject_depth"])
                ),
                background_depth=(
                    None if depth_row["background_depth"] is None else float(depth_row["background_depth"])
                ),
                foreground_separation=float(depth_row["foreground_separation"]),
                subject_focus_score=(
                    None
                    if depth_row["subject_focus_score"] is None
                    else float(depth_row["subject_focus_score"])
                ),
                background_blur_score=(
                    None
                    if depth_row["background_blur_score"] is None
                    else float(depth_row["background_blur_score"])
                ),
                occlusion_risk=float(depth_row["occlusion_risk"]),
                subject_confidence=float(depth_row["subject_confidence"]),
                model=str(depth_row["model"]),
            )
        captured = datetime.fromisoformat(row["capture_time"]) if row["capture_time"] else None
        return PhotoObservation(
            id=identifier,
            path=path,
            source_root=source_root,
            filename=path.name,
            relative_path=str(path.relative_to(source_root)),
            width=int(row["width"]),
            height=int(row["height"]),
            capture_time=captured,
            # 文件名序号解析可独立升级，不必因此让昂贵的视觉特征缓存失效。
            file_sequence=file_sequence(path),
            descriptor=VisualDescriptor(
                phash=int(row["phash"], 16),
                dhash=int(row["dhash"], 16),
                layout=_blob_array(row["layout"]),
                color=_blob_array(row["color"]),
                edge=_blob_array(row["edge"]),
                semantic=_blob_array(row["semantic"]),
            ),
            faces=faces,
            bodies=bodies,
            poses=poses,
            depth=depth_observation,
            metrics={key: float(value) for key, value in json.loads(row["metrics_json"]).items()},
            score=float(row["score"]),
            issues=list(json.loads(row["issues_json"])),
        )

    def dominant_pipeline_signature(self, source_root: Path) -> str | None:
        """返回指定素材目录中缓存覆盖最多的特征管线版本。"""
        root_text = str(source_root.expanduser().resolve()).rstrip("\\/")
        prefix = f"{root_text}{os.sep}%"
        try:
            with self._lock, self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT pipeline_signature, COUNT(*) AS cached_count, MAX(cached_at) AS latest
                    FROM photo_features
                    WHERE source_path LIKE ?
                    GROUP BY pipeline_signature
                    ORDER BY cached_count DESC, latest DESC
                    LIMIT 1
                    """,
                    (prefix,),
                ).fetchone()
            return str(row["pipeline_signature"]) if row is not None else None
        except sqlite3.Error as exc:
            self._error = str(exc)
            return None

    def save(self, photo: PhotoObservation, pipeline_signature: str) -> None:
        cache_key, _ = self._cache_key(photo.path, pipeline_signature)
        stat = photo.path.stat()
        now = datetime.now().astimezone().isoformat(timespec="seconds")
        values = (
            cache_key,
            photo.id,
            pipeline_signature,
            str(photo.path),
            stat.st_size,
            stat.st_mtime_ns,
            photo.width,
            photo.height,
            photo.capture_time.isoformat() if photo.capture_time else None,
            photo.file_sequence,
            f"{photo.descriptor.phash:016x}",
            f"{photo.descriptor.dhash:016x}",
            sqlite3.Binary(_array_blob(photo.descriptor.layout)),
            sqlite3.Binary(_array_blob(photo.descriptor.color)),
            sqlite3.Binary(_array_blob(photo.descriptor.edge)),
            None if photo.descriptor.semantic is None else sqlite3.Binary(_array_blob(photo.descriptor.semantic)),
            json.dumps(photo.metrics, ensure_ascii=False, separators=(",", ":")),
            photo.score,
            json.dumps(photo.issues, ensure_ascii=False, separators=(",", ":")),
            now,
        )
        try:
            with self._lock, self._connect() as connection:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO photo_features VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    values,
                )
                connection.execute("DELETE FROM face_features WHERE cache_key = ?", (cache_key,))
                connection.executemany(
                    """
                    INSERT INTO face_features (
                        cache_key, face_index, face_id, bbox_json, confidence, area_ratio, embedding,
                        eye_state, open_probability, sharpness, profile, smile_score, high_res_sharpness,
                        eye_sharpness, yaw, pitch, roll, occlusion_risk, expression,
                        expression_confidence, expression_score, fiqa_score
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    [
                        (
                            cache_key,
                            index,
                            face.face_id,
                            json.dumps(face.bbox, separators=(",", ":")),
                            face.confidence,
                            face.area_ratio,
                            None if face.embedding is None else sqlite3.Binary(_array_blob(face.embedding)),
                            face.eye_state,
                            face.open_probability,
                            face.sharpness,
                            int(face.profile),
                            face.smile_score,
                            face.high_res_sharpness,
                            face.eye_sharpness,
                            face.yaw,
                            face.pitch,
                            face.roll,
                            face.occlusion_risk,
                            face.expression,
                            face.expression_confidence,
                            face.expression_score,
                            face.fiqa_score,
                        )
                        for index, face in enumerate(photo.faces)
                    ],
                )
                connection.execute("DELETE FROM body_features WHERE cache_key = ?", (cache_key,))
                connection.executemany(
                    """
                    INSERT INTO body_features (
                        cache_key, body_index, bbox_json, confidence, area_ratio, embedding, detector
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            cache_key,
                            index,
                            json.dumps(body.bbox, separators=(",", ":")),
                            body.confidence,
                            body.area_ratio,
                            None if body.embedding is None else sqlite3.Binary(_array_blob(body.embedding)),
                            body.detector,
                        )
                        for index, body in enumerate(photo.bodies)
                    ],
                )
                connection.execute("DELETE FROM pose_features WHERE cache_key = ?", (cache_key,))
                connection.executemany(
                    """
                    INSERT INTO pose_features (
                        cache_key, pose_index, bbox_json, detection_confidence, presence_confidence,
                        area_ratio, landmarks_2d, descriptor, visibility, foreground_score, model
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            cache_key,
                            index,
                            json.dumps(pose.bbox, separators=(",", ":")),
                            pose.detection_confidence,
                            pose.presence_confidence,
                            pose.area_ratio,
                            sqlite3.Binary(_array_blob(pose.landmarks_2d)),
                            None if pose.descriptor is None else sqlite3.Binary(_array_blob(pose.descriptor)),
                            pose.visibility,
                            pose.foreground_score,
                            pose.model,
                        )
                        for index, pose in enumerate(photo.poses)
                    ],
                )
                connection.execute("DELETE FROM depth_features WHERE cache_key = ?", (cache_key,))
                if photo.depth is not None:
                    connection.execute(
                        """
                        INSERT INTO depth_features (
                            cache_key, descriptor, subject_depth, background_depth,
                            foreground_separation, subject_focus_score, background_blur_score,
                            occlusion_risk, subject_confidence, model
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            cache_key,
                            (
                                None
                                if photo.depth.descriptor is None
                                else sqlite3.Binary(_array_blob(photo.depth.descriptor))
                            ),
                            photo.depth.subject_depth,
                            photo.depth.background_depth,
                            photo.depth.foreground_separation,
                            photo.depth.subject_focus_score,
                            photo.depth.background_blur_score,
                            photo.depth.occlusion_risk,
                            photo.depth.subject_confidence,
                            photo.depth.model,
                        ),
                    )
        except (sqlite3.Error, OSError) as exc:
            self._error = str(exc)

    def record_model(self, name: str, signature: str, metadata: dict[str, Any]) -> None:
        try:
            with self._lock, self._connect() as connection:
                connection.execute(
                    "INSERT OR REPLACE INTO model_versions VALUES (?, ?, ?, ?)",
                    (
                        name,
                        signature,
                        json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
                        datetime.now().astimezone().isoformat(timespec="seconds"),
                    ),
                )
        except sqlite3.Error as exc:
            self._error = str(exc)

    def load_vlm_decision(self, decision_key: str) -> tuple[dict[str, Any], str] | None:
        try:
            with self._lock, self._connect() as connection:
                row = connection.execute(
                    "SELECT decision_json, raw_response FROM vlm_decisions WHERE decision_key = ?",
                    (decision_key,),
                ).fetchone()
            if row is None:
                return None
            return json.loads(row["decision_json"]), str(row["raw_response"])
        except (sqlite3.Error, ValueError, TypeError) as exc:
            self._error = str(exc)
            return None

    def save_vlm_decision(
        self,
        decision_key: str,
        model_id: str,
        prompt_version: str,
        decision: dict[str, Any],
        raw_response: str,
    ) -> None:
        try:
            with self._lock, self._connect() as connection:
                connection.execute(
                    "INSERT OR REPLACE INTO vlm_decisions VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        decision_key,
                        model_id,
                        prompt_version,
                        json.dumps(decision, ensure_ascii=False, separators=(",", ":")),
                        raw_response,
                        datetime.now().astimezone().isoformat(timespec="seconds"),
                    ),
                )
        except sqlite3.Error as exc:
            self._error = str(exc)

    def record_user_action(
        self,
        project_id: str,
        photo_id_value: str,
        group_id: str | None,
        previous_category: str | None,
        selected_category: str,
        stars: int | None,
    ) -> None:
        try:
            with self._lock, self._connect() as connection:
                connection.execute(
                    "INSERT INTO user_actions(project_id, photo_id, group_id, previous_category, selected_category, stars, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        project_id,
                        photo_id_value,
                        group_id,
                        previous_category,
                        selected_category,
                        stars,
                        datetime.now().astimezone().isoformat(timespec="seconds"),
                    ),
                )
        except sqlite3.Error as exc:
            self._error = str(exc)
