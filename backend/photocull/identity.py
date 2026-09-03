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

    def add(self, face: FaceObservation, photo_id: str) -> None:
        if face.embedding is None:
            return
        merged = self.centroid * self.count + face.embedding
        norm = float(np.linalg.norm(merged))
        self.centroid = merged / norm if norm > 1e-8 else self.centroid
        self.count += 1
        self.photo_ids.add(photo_id)
        self.faces.append(face)


class IdentityClusterer:
    """ArcFace 增量聚类，并禁止同一照片中的两张脸落入同一身份。"""

    def __init__(self, threshold: float = 0.42, ambiguity_margin: float = 0.025) -> None:
        self.threshold = threshold
        self.ambiguity_margin = ambiguity_margin

    def assign(self, photos: list[PhotoObservation]) -> list[IdentityCluster]:
        clusters: list[IdentityCluster] = []
        for photo in photos:
            used_in_photo: set[int] = set()
            faces = sorted(
                (face for face in photo.faces if face.embedding is not None),
                key=lambda face: (face.area_ratio, face.confidence),
                reverse=True,
            )
            for face in faces:
                similarities = [
                    (float(np.dot(face.embedding, cluster.centroid)), cluster)
                    for cluster in clusters
                    if cluster.temporary_id not in used_in_photo
                ]
                similarities.sort(key=lambda item: item[0], reverse=True)
                best_score = similarities[0][0] if similarities else -1.0
                second_score = similarities[1][0] if len(similarities) > 1 else -1.0
                unambiguous = best_score - second_score >= self.ambiguity_margin or best_score >= self.threshold + 0.12
                if similarities and best_score >= self.threshold and unambiguous:
                    cluster = similarities[0][1]
                    cluster.add(face, photo.id)
                else:
                    cluster = IdentityCluster(
                        temporary_id=len(clusters),
                        centroid=face.embedding.copy(),
                        photo_ids={photo.id},
                        faces=[face],
                    )
                    clusters.append(cluster)
                used_in_photo.add(cluster.temporary_id)
                face.person_id = f"tmp-{cluster.temporary_id}"

        ranked = sorted(clusters, key=lambda cluster: (len(cluster.photo_ids), cluster.count), reverse=True)
        mapping: dict[int, str] = {}
        for index, cluster in enumerate(ranked, start=1):
            person_id = f"人物 {index:02d}"
            mapping[cluster.temporary_id] = person_id
            for face in cluster.faces:
                face.person_id = person_id

        for photo in photos:
            photo.person_ids = sorted({face.person_id for face in photo.faces if face.person_id})
        return ranked
