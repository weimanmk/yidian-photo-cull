from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .internal_models import FaceObservation, PhotoObservation


@dataclass(slots=True)
class IdentityCluster:
    temporary_id: int
    centroid: np.ndarray
    count: int = 1
    photo_ids: set[str] = field(default_factory=set)
    faces: list[FaceObservation] = field(default_factory=list)

    def add(self, face: FaceObservation, photo_id: str, *, update_centroid: bool = True) -> None:
        if face.embedding is None:
            return
        if update_centroid:
            embedding = face.embedding / np.linalg.norm(face.embedding)
            merged = self.centroid * self.count + embedding
            norm = float(np.linalg.norm(merged))
            self.centroid = merged / norm if norm > 1e-8 else self.centroid
        self.count += 1
        self.photo_ids.add(photo_id)
        self.faces.append(face)


class IdentityClusterer:
    """先用可靠脸建立身份，再保守分配弱脸；同图不能共享身份。"""

    def __init__(self, threshold: float = 0.42, ambiguity_margin: float = 0.025) -> None:
        self.threshold = threshold
        self.ambiguity_margin = ambiguity_margin

    def assign(self, photos: list[PhotoObservation]) -> list[IdentityCluster]:
        clusters: list[IdentityCluster] = []
        records = []
        for photo in photos:
            photo.person_ids = []
            for face in photo.faces:
                face.person_id = None
                if face.embedding is None:
                    continue
                embedding = np.asarray(face.embedding, dtype=np.float32)
                if embedding.ndim != 1 or not embedding.size or not np.isfinite(embedding).all():
                    continue
                norm = float(np.linalg.norm(embedding))
                if norm <= 1e-8 or not np.isfinite(norm):
                    continue
                quality = self._quality(face)
                records.append((photo, face, embedding / norm, quality))

        # Stable tie breaks use source IDs rather than input enumeration. Faces
        # with reliable pose/size/sharpness establish centroids before weak ones.
        records.sort(key=lambda item: (-item[3][1], -len(item[0].faces), item[0].id, item[1].face_id))
        for strong_pass in (True, False):
            for photo, face, embedding, (strong, _) in records:
                if strong != strong_pass:
                    continue
                similarities = [
                    (float(np.dot(embedding, cluster.centroid)), cluster)
                    for cluster in clusters
                    if photo.id not in cluster.photo_ids and cluster.centroid.shape == embedding.shape
                ]
                similarities.sort(key=lambda item: item[0], reverse=True)
                best_score = similarities[0][0] if similarities else -1.0
                second_score = similarities[1][0] if len(similarities) > 1 else -1.0
                threshold = self.threshold if strong else max(self.threshold + 0.10, 0.55)
                margin = self.ambiguity_margin if strong else max(self.ambiguity_margin, 0.05)
                unambiguous = best_score - second_score >= margin
                if similarities and best_score >= threshold and unambiguous:
                    cluster = similarities[0][1]
                    cluster.add(face, photo.id, update_centroid=strong)
                elif strong and best_score < self.threshold:
                    cluster = IdentityCluster(
                        temporary_id=len(clusters),
                        centroid=embedding.copy(),
                        photo_ids={photo.id},
                        faces=[face],
                    )
                    clusters.append(cluster)
                # Ambiguity is unknown, never evidence for a new identity.

        ranked = sorted(clusters, key=lambda cluster: (-len(cluster.photo_ids), -cluster.count, cluster.temporary_id))
        for index, cluster in enumerate(ranked, start=1):
            person_id = f"人物 {index:02d}"
            for face in cluster.faces:
                face.person_id = person_id

        for photo in photos:
            photo.person_ids = sorted({face.person_id for face in photo.faces if face.person_id})
        return ranked

    @staticmethod
    def _quality(face: FaceObservation) -> tuple[bool, float]:
        # Blurred or missing sharpness cannot establish an identity. Optional
        # learned quality models are additional evidence, not prerequisites.
        sharpness = max(face.high_res_sharpness, face.sharpness)
        values = (face.confidence, face.area_ratio, sharpness, face.yaw, face.pitch, face.roll, face.occlusion_risk)
        if not np.isfinite(values).all() or (face.fiqa_score is not None and not np.isfinite(face.fiqa_score)):
            return False, -1.0
        strong = (
            face.confidence >= 0.82 and face.area_ratio >= 0.001
            and sharpness >= 45.0
            and abs(face.yaw) <= 35.0 and abs(face.pitch) <= 30.0
            and abs(face.roll) <= 35.0 and not face.profile
            and face.occlusion_risk < 0.45
            and (face.fiqa_score is None or face.fiqa_score >= 35.0)
        )
        score = (
            face.confidence + min(face.area_ratio, 0.10)
            + min(sharpness, 200.0) / 1000.0
            - abs(face.yaw) / 180.0 - face.occlusion_risk * 0.25
        )
        return strong, score
