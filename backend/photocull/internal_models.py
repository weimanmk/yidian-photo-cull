from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
import math
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class VisualDescriptor:
    phash: int
    layout: np.ndarray
    color: np.ndarray
    edge: np.ndarray
    dhash: int = 0
    semantic: np.ndarray | None = None


@dataclass(slots=True)
class FaceObservation:
    face_id: str
    bbox: tuple[float, float, float, float]
    confidence: float
    area_ratio: float
    embedding: np.ndarray | None
    eye_state: str = "Unknown"
    open_probability: float | None = None
    sharpness: float = 0.0
    profile: bool = False
    smile_score: float = 0.0
    high_res_sharpness: float = 0.0
    eye_sharpness: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    occlusion_risk: float = 0.0
    person_id: str | None = None
    expression: str = "unknown"
    expression_confidence: float | None = None
    expression_score: float = 0.0
    fiqa_score: float | None = None

    def public_dict(self) -> dict[str, Any]:
        return {
            "face_id": self.face_id,
            "person_id": self.person_id,
            "confidence": round(self.confidence, 4),
            "bbox": [round(value, 6) for value in self.bbox],
            "area_ratio": round(self.area_ratio, 6),
            "eye_state": self.eye_state,
            "open_probability": None if self.open_probability is None else round(self.open_probability, 4),
            "sharpness": round(self.sharpness, 2),
            "profile": self.profile,
            "smile_score": round(self.smile_score, 2),
            "high_res_sharpness": round(self.high_res_sharpness, 2),
            "eye_sharpness": round(self.eye_sharpness, 2),
            "yaw": round(self.yaw, 2),
            "pitch": round(self.pitch, 2),
            "roll": round(self.roll, 2),
            "occlusion_risk": round(self.occlusion_risk, 4),
            "expression": self.expression,
            "expression_confidence": (
                None if self.expression_confidence is None else round(self.expression_confidence, 4)
            ),
            "expression_score": round(self.expression_score, 2),
            "fiqa_score": None if self.fiqa_score is None else round(self.fiqa_score, 2),
        }


def is_significant_face(face: FaceObservation) -> bool:
    """返回足以参与人物覆盖与软分组的人脸。"""
    return bool(
        face.person_id
        and (
            (face.area_ratio >= 0.0006 and face.confidence >= 0.82)
            or (face.area_ratio >= 0.0010 and face.confidence >= 0.65)
            or (face.area_ratio >= 0.0025 and face.confidence >= 0.55)
        )
    )


def is_reliable_face(face: FaceObservation) -> bool:
    """返回可作为人物硬约束或单张覆盖依据的高可信人脸。"""
    return bool(
        face.person_id
        and (
            (face.area_ratio >= 0.0007 and face.confidence >= 0.90)
            or (face.area_ratio >= 0.0018 and face.confidence >= 0.78)
            or (face.area_ratio >= 0.0035 and face.confidence >= 0.65)
        )
    )


@dataclass(slots=True)
class BodyObservation:
    bbox: tuple[float, float, float, float]
    confidence: float
    area_ratio: float
    embedding: np.ndarray | None
    detector: str

    def public_dict(self) -> dict[str, Any]:
        return {
            "bbox": [round(value, 6) for value in self.bbox],
            "confidence": round(self.confidence, 4),
            "area_ratio": round(self.area_ratio, 6),
            "detector": self.detector,
            "reid_available": self.embedding is not None,
        }


@dataclass(slots=True)
class PoseObservation:
    bbox: tuple[float, float, float, float]
    detection_confidence: float
    presence_confidence: float
    area_ratio: float
    landmarks_2d: np.ndarray
    descriptor: np.ndarray | None
    visibility: float
    model: str
    foreground_score: float | None = None

    def public_dict(self) -> dict[str, Any]:
        landmarks = np.asarray(self.landmarks_2d, dtype=np.float32).reshape(-1, 3)
        return {
            "bbox": [round(value, 6) for value in self.bbox],
            "detection_confidence": round(self.detection_confidence, 4),
            "presence_confidence": round(self.presence_confidence, 4),
            "area_ratio": round(self.area_ratio, 6),
            "visibility": round(self.visibility, 4),
            "foreground_score": None if self.foreground_score is None else round(self.foreground_score, 4),
            "landmarks": [[round(float(value), 5) for value in row] for row in landmarks],
            "model": self.model,
            "world_3d_available": self.descriptor is not None,
        }


@dataclass(slots=True)
class DepthObservation:
    descriptor: np.ndarray | None
    subject_depth: float | None
    background_depth: float | None
    foreground_separation: float
    subject_focus_score: float | None
    background_blur_score: float | None
    occlusion_risk: float
    subject_confidence: float
    model: str

    def public_dict(self) -> dict[str, Any]:
        return {
            "subject_depth": None if self.subject_depth is None else round(self.subject_depth, 4),
            "background_depth": None if self.background_depth is None else round(self.background_depth, 4),
            "foreground_separation": round(self.foreground_separation, 4),
            "subject_focus_score": (
                None if self.subject_focus_score is None else round(self.subject_focus_score, 2)
            ),
            "background_blur_score": (
                None if self.background_blur_score is None else round(self.background_blur_score, 2)
            ),
            "occlusion_risk": round(self.occlusion_risk, 4),
            "subject_confidence": round(self.subject_confidence, 4),
            "model": self.model,
        }


@dataclass(slots=True)
class PhotoObservation:
    id: str
    path: Path
    source_root: Path
    filename: str
    relative_path: str
    width: int
    height: int
    capture_time: datetime | None
    file_sequence: int
    descriptor: VisualDescriptor
    faces: list[FaceObservation] = field(default_factory=list)
    bodies: list[BodyObservation] = field(default_factory=list)
    poses: list[PoseObservation] = field(default_factory=list)
    depth: DepthObservation | None = None
    person_ids: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    score: float = 0.0
    issues: list[str] = field(default_factory=list)
    group_id: str = ""
    is_best_pick: bool = False
    rank_in_group: int = 0
    category: str = "selected"
    stars: int = 0
    rating_tier: str = "waste"
    rating_origin: str = "ai"
    rating_reason: str = "technical_reject"
    rating_locked: bool = False
    needs_review: bool = False
    coverage_keys: list[str] = field(default_factory=list)
    strict_duplicate_cluster_id: str = ""
    beat_id: str = ""
    selection_reasons: list[str] = field(default_factory=list)
    vlm_rank: int | None = None
    vlm_confidence: float | None = None
    vlm_reasons: list[str] = field(default_factory=list)
    stage_id: str = ""
    stage_label: str = ""
    coverage_protected: bool = False
    coverage_person_ids: list[str] = field(default_factory=list)
    coverage_original_category: str | None = None

    @property
    def significant_person_ids(self) -> set[str]:
        """返回足以参与分组的人脸身份。

        全身活动照中的主角人脸通常只占画面千分之一到千分之三；旧阈值会把这些
        人脸全部忽略，导致同一人物在变焦后被拆组。这里同时约束检测置信度和面积，
        小而低置信度的观众席误检仍不会参与分组。
        """
        return {face.person_id for face in self.faces if is_significant_face(face) and face.person_id}

    @property
    def reliable_person_ids(self) -> set[str]:
        """返回可作为“不同人物不得合组”硬约束的高可靠身份。"""
        return {face.person_id for face in self.faces if is_reliable_face(face) and face.person_id}

    @property
    def largest_face_ratio(self) -> float:
        return max((face.area_ratio for face in self.faces), default=0.0)

    def public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "filename": self.filename,
            "relative_path": self.relative_path,
            "width": self.width,
            "height": self.height,
            "capture_time": self.capture_time.isoformat() if self.capture_time else None,
            "thumbnail_url": f"/api/thumbnails/{self.id}",
            "image_url": f"/api/images/{self.id}",
            "score": round(self.score, 2),
            "stars": self.stars,
            "rating_tier": self.rating_tier,
            "rating_origin": self.rating_origin,
            "rating_reason": self.rating_reason,
            "rating_locked": self.rating_locked,
            "needs_review": self.needs_review,
            "coverage_keys": list(self.coverage_keys),
            "strict_duplicate_cluster_id": self.strict_duplicate_cluster_id,
            "beat_id": self.beat_id,
            "category": self.category,
            "issues": list(self.issues),
            "metrics": {key: round(float(value), 3) for key, value in self.metrics.items()},
            "faces": [face.public_dict() for face in self.faces],
            "bodies": [body.public_dict() for body in self.bodies],
            "poses": [pose.public_dict() for pose in self.poses],
            "depth": None if self.depth is None else self.depth.public_dict(),
            "person_ids": list(self.person_ids),
            "group_id": self.group_id,
            "is_best_pick": self.is_best_pick,
            "rank_in_group": self.rank_in_group,
            "selection_reasons": list(self.selection_reasons),
            "vlm_rank": self.vlm_rank,
            "vlm_confidence": None if self.vlm_confidence is None else round(self.vlm_confidence, 4),
            "vlm_reasons": list(self.vlm_reasons),
            "stage_id": self.stage_id,
            "stage_label": self.stage_label,
            "coverage_protected": self.coverage_protected,
            "coverage_person_ids": list(self.coverage_person_ids),
            "coverage_original_category": self.coverage_original_category,
        }


@dataclass(slots=True)
class SimilarityEvidence:
    total: float
    scene: float
    semantic: float | None
    phash: float
    dhash: float
    layout: float
    color: float
    edge: float
    composition: float
    people: float
    body: float | None
    pose: float | None
    depth: float | None
    temporal: float
    compatible_people: bool
    strong_duplicate: bool


@dataclass(slots=True)
class PhotoGroupInternal:
    id: str
    photos: list[PhotoObservation]
    confidence: float
    reason: str
    vlm_decision: dict[str, Any] | None = None

    def public_dict(self, keep_per_group: int) -> dict[str, Any]:
        # 人物覆盖保底可以有意识地突破“每组保留 N 张”，因此不能再截断已标记的保留照片。
        _ = keep_per_group
        best = [photo.id for photo in self.photos if photo.is_best_pick]
        people_counts = Counter(person for photo in self.photos for person in photo.significant_person_ids)
        minimum = max(1, math.ceil(len(self.photos) * 0.20))
        people = sorted(person for person, count in people_counts.items() if count >= minimum)
        first = self.photos[0] if self.photos else None
        return {
            "id": self.id,
            "photo_ids": [photo.id for photo in self.photos],
            "best_photo_ids": best,
            "person_ids": people,
            "size": len(self.photos),
            "confidence": round(self.confidence, 4),
            "scene_reason": self.reason,
            "vlm_decision": self.vlm_decision,
            "stage_id": first.stage_id if first else "",
            "stage_label": first.stage_label if first else "",
            "coverage_protected": any(photo.coverage_protected for photo in self.photos),
        }
