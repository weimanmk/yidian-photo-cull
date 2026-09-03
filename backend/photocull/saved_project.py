from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .internal_models import (
    BodyObservation,
    DepthObservation,
    FaceObservation,
    PhotoGroupInternal,
    PhotoObservation,
    PoseObservation,
    VisualDescriptor,
)


def hydrate_face(payload: dict[str, Any]) -> FaceObservation:
    return FaceObservation(
        face_id=str(payload["face_id"]),
        bbox=tuple(float(value) for value in payload["bbox"]),
        confidence=float(payload.get("confidence", 0.0)),
        area_ratio=float(payload.get("area_ratio", 0.0)),
        embedding=None,
        eye_state=str(payload.get("eye_state", "Unknown")),
        open_probability=payload.get("open_probability"),
        sharpness=float(payload.get("sharpness", 0.0)),
        profile=bool(payload.get("profile", False)),
        smile_score=float(payload.get("smile_score", 0.0)),
        high_res_sharpness=float(payload.get("high_res_sharpness", 0.0)),
        eye_sharpness=float(payload.get("eye_sharpness", 0.0)),
        yaw=float(payload.get("yaw", 0.0)),
        pitch=float(payload.get("pitch", 0.0)),
        roll=float(payload.get("roll", 0.0)),
        occlusion_risk=float(payload.get("occlusion_risk", 0.0)),
        person_id=payload.get("person_id"),
        expression=str(payload.get("expression", "unknown")),
        expression_confidence=payload.get("expression_confidence"),
        expression_score=float(payload.get("expression_score", 0.0)),
        fiqa_score=None if payload.get("fiqa_score") is None else float(payload["fiqa_score"]),
    )


def hydrate_body(payload: dict[str, Any]) -> BodyObservation:
    return BodyObservation(
        bbox=tuple(float(value) for value in payload["bbox"]),
        confidence=float(payload.get("confidence", 0.0)),
        area_ratio=float(payload.get("area_ratio", 0.0)),
        embedding=None,
        detector=str(payload.get("detector", "saved-project")),
    )


def hydrate_pose(payload: dict[str, Any]) -> PoseObservation:
    landmarks = np.asarray(payload.get("landmarks", []), dtype=np.float32)
    if landmarks.size == 0:
        landmarks = np.empty((0, 3), dtype=np.float32)
    else:
        landmarks = landmarks.reshape(-1, 3)
    return PoseObservation(
        bbox=tuple(float(value) for value in payload["bbox"]),
        detection_confidence=float(payload.get("detection_confidence", 0.0)),
        presence_confidence=float(payload.get("presence_confidence", 0.0)),
        area_ratio=float(payload.get("area_ratio", 0.0)),
        landmarks_2d=landmarks,
        descriptor=None,
        visibility=float(payload.get("visibility", 0.0)),
        model=str(payload.get("model", "saved-project")),
        foreground_score=(
            None if payload.get("foreground_score") is None else float(payload["foreground_score"])
        ),
    )


def hydrate_depth(payload: dict[str, Any] | None) -> DepthObservation | None:
    if not payload:
        return None
    return DepthObservation(
        descriptor=None,
        subject_depth=None if payload.get("subject_depth") is None else float(payload["subject_depth"]),
        background_depth=(
            None if payload.get("background_depth") is None else float(payload["background_depth"])
        ),
        foreground_separation=float(payload.get("foreground_separation", 0.0)),
        subject_focus_score=(
            None if payload.get("subject_focus_score") is None else float(payload["subject_focus_score"])
        ),
        background_blur_score=(
            None if payload.get("background_blur_score") is None else float(payload["background_blur_score"])
        ),
        occlusion_risk=float(payload.get("occlusion_risk", 0.0)),
        subject_confidence=float(payload.get("subject_confidence", 0.0)),
        model=str(payload.get("model", "saved-project")),
    )


def hydrate_photo(payload: dict[str, Any], source_path: str, sequence: int) -> PhotoObservation:
    path = Path(source_path)
    capture_time = None
    if payload.get("capture_time"):
        try:
            capture_time = datetime.fromisoformat(str(payload["capture_time"]))
        except ValueError:
            capture_time = None
    return PhotoObservation(
        id=str(payload["id"]),
        path=path,
        source_root=path.parent,
        filename=str(payload["filename"]),
        relative_path=str(payload.get("relative_path", payload["filename"])),
        width=int(payload.get("width", 0)),
        height=int(payload.get("height", 0)),
        capture_time=capture_time,
        file_sequence=sequence,
        descriptor=VisualDescriptor(
            phash=0,
            layout=np.zeros(1, dtype=np.float32),
            color=np.zeros(1, dtype=np.float32),
            edge=np.zeros(1, dtype=np.float32),
        ),
        faces=[hydrate_face(face) for face in payload.get("faces", [])],
        bodies=[hydrate_body(body) for body in payload.get("bodies", [])],
        poses=[hydrate_pose(pose) for pose in payload.get("poses", [])],
        depth=hydrate_depth(payload.get("depth")),
        person_ids=[str(value) for value in payload.get("person_ids", [])],
        metrics={key: float(value) for key, value in payload.get("metrics", {}).items()},
        score=float(payload.get("score", 0.0)),
        issues=[str(value) for value in payload.get("issues", [])],
        group_id=str(payload.get("group_id", "")),
        is_best_pick=bool(payload.get("is_best_pick", False)),
        rank_in_group=int(payload.get("rank_in_group", 0)),
        category=str(payload.get("category", "selected")),
        stars=int(payload.get("stars", 0)),
        rating_tier=str(payload.get("rating_tier", "legacy")),
        rating_origin=str(payload.get("rating_origin", "legacy")),
        rating_reason=str(payload.get("rating_reason", "legacy_score")),
        rating_locked=bool(payload.get("rating_locked", False)),
        needs_review=bool(payload.get("needs_review", False)),
        coverage_keys=[str(value) for value in payload.get("coverage_keys", [])],
        strict_duplicate_cluster_id=str(payload.get("strict_duplicate_cluster_id", "")),
        beat_id=str(payload.get("beat_id", "")),
        selection_reasons=[str(value) for value in payload.get("selection_reasons", [])],
        stage_id=str(payload.get("stage_id", "")),
        stage_label=str(payload.get("stage_label", "")),
        coverage_protected=bool(payload.get("coverage_protected", False)),
        coverage_person_ids=[str(value) for value in payload.get("coverage_person_ids", [])],
        coverage_original_category=payload.get("coverage_original_category"),
    )


def build_saved_groups(results: dict[str, Any], files: dict[str, str]) -> list[PhotoGroupInternal]:
    payload_by_id = {str(photo["id"]): photo for photo in results.get("photos", [])}
    photo_by_id: dict[str, PhotoObservation] = {}
    for sequence, (identifier, payload) in enumerate(payload_by_id.items()):
        source_path = files.get(identifier)
        if source_path:
            photo_by_id[identifier] = hydrate_photo(payload, source_path, sequence)
    groups: list[PhotoGroupInternal] = []
    for payload in results.get("groups", []):
        photos = [photo_by_id[identifier] for identifier in payload.get("photo_ids", []) if identifier in photo_by_id]
        if photos:
            groups.append(
                PhotoGroupInternal(
                    id=str(payload["id"]),
                    photos=photos,
                    confidence=float(payload.get("confidence", 0.0)),
                    reason=str(payload.get("scene_reason", "")),
                )
            )
    return groups


def reference_stems(directory: Path) -> set[str]:
    if not directory.is_dir():
        raise ValueError(f"人工参考目录不存在: {directory}")
    return {path.stem.casefold() for path in directory.iterdir() if path.is_file()}
