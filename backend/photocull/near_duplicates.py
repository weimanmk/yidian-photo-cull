from __future__ import annotations

from dataclasses import dataclass

from .grouping import compare_photos
from .internal_models import PhotoGroupInternal, PhotoObservation, SimilarityEvidence


@dataclass(frozen=True, slots=True)
class DuplicateLayers:
    strict_cluster_by_photo: dict[str, str]
    beat_by_photo: dict[str, str]


def _sort_key(photo: PhotoObservation) -> tuple[float, int, str, str]:
    timestamp = photo.capture_time.timestamp() if photo.capture_time else float("inf")
    sequence = photo.file_sequence if photo.file_sequence >= 0 else 10**12
    return timestamp, sequence, photo.filename.casefold(), photo.id


class _EvidenceCache:
    def __init__(self) -> None:
        self._values: dict[tuple[str, str], SimilarityEvidence] = {}

    def get(self, left: PhotoObservation, right: PhotoObservation) -> SimilarityEvidence:
        key = tuple(sorted((left.id, right.id)))
        cached = self._values.get(key)
        if cached is None:
            cached = compare_photos(left, right)
            self._values[key] = cached
        return cached


class _UnionFind:
    def __init__(self, photo_ids: list[str]) -> None:
        self._parent = {photo_id: photo_id for photo_id in photo_ids}

    def find(self, photo_id: str) -> str:
        parent = self._parent[photo_id]
        if parent != photo_id:
            self._parent[photo_id] = self.find(parent)
        return self._parent[photo_id]

    def union(self, left_id: str, right_id: str) -> None:
        left_root = self.find(left_id)
        right_root = self.find(right_id)
        if left_root != right_root:
            self._parent[right_root] = left_root


def _strict_clusters(
    members: list[PhotoObservation],
    evidence: _EvidenceCache,
) -> list[list[PhotoObservation]]:
    union_find = _UnionFind([photo.id for photo in members])
    for left_index, left in enumerate(members):
        for right in members[left_index + 1 :]:
            pair = evidence.get(left, right)
            if pair.strong_duplicate and pair.compatible_people:
                union_find.union(left.id, right.id)

    by_root: dict[str, list[PhotoObservation]] = {}
    for photo in members:
        by_root.setdefault(union_find.find(photo.id), []).append(photo)
    return sorted(by_root.values(), key=lambda cluster: min(_sort_key(photo) for photo in cluster))


def _same_beat(
    left: PhotoObservation,
    right: PhotoObservation,
    evidence: _EvidenceCache,
) -> bool:
    pair = evidence.get(left, right)
    if not pair.compatible_people or pair.scene < 0.90:
        return False
    if pair.pose is not None and pair.pose < 0.90:
        return False
    if left.file_sequence < 0 or right.file_sequence < 0:
        return False
    if abs(left.file_sequence - right.file_sequence) > 3:
        return False
    if left.capture_time and right.capture_time:
        seconds = abs((left.capture_time - right.capture_time).total_seconds())
        if seconds > 4.0:
            return False
    return True


def _beats(
    members: list[PhotoObservation],
    evidence: _EvidenceCache,
) -> list[list[PhotoObservation]]:
    ordered = sorted(members, key=_sort_key)
    if not ordered:
        return []

    beats: list[list[PhotoObservation]] = [[ordered[0]]]
    for photo in ordered[1:]:
        current = beats[-1]
        if _same_beat(current[-1], photo, evidence) and _same_beat(current[0], photo, evidence):
            current.append(photo)
        else:
            beats.append([photo])
    return beats


def build_duplicate_layers(groups: list[PhotoGroupInternal]) -> DuplicateLayers:
    """为每张照片生成严格重复簇与较宽松的同瞬间标识。"""
    strict_cluster_by_photo: dict[str, str] = {}
    beat_by_photo: dict[str, str] = {}

    for group in groups:
        members = sorted(group.photos, key=_sort_key)
        evidence = _EvidenceCache()
        for cluster_index, cluster in enumerate(_strict_clusters(members, evidence), start=1):
            cluster_id = f"{group.id}:strict-{cluster_index:04d}"
            for photo in cluster:
                strict_cluster_by_photo[photo.id] = cluster_id

        for beat_index, beat in enumerate(_beats(members, evidence), start=1):
            beat_id = f"{group.id}:beat-{beat_index:04d}"
            for photo in beat:
                beat_by_photo[photo.id] = beat_id

    return DuplicateLayers(
        strict_cluster_by_photo=strict_cluster_by_photo,
        beat_by_photo=beat_by_photo,
    )
