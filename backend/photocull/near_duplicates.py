from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

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


# Fixed conservative limits, shared with audit provenance.
DUPLICATE_CONFIG = {
    "version": "global-complete-link-v1",
    "seconds": 4.0,
    "sequence_gap_without_time": 3,
    "minimum_pose_similarity": 0.90,
    "minimum_hash_similarity": 0.90,
    "distant_candidates": "identical_visual_descriptor",
}


def _fingerprint(photo: PhotoObservation) -> bytes:
    """Descriptor equality is a retrieval signal, not proof of file equality."""
    descriptor = photo.descriptor
    digest = hashlib.sha256(f"{descriptor.phash}:{descriptor.dhash}".encode())
    for vector in (descriptor.layout, descriptor.color, descriptor.edge, descriptor.semantic):
        if vector is None:
            digest.update(b"none")
        else:
            value = np.asarray(vector, dtype=np.float32)
            digest.update(str(value.shape).encode())
            digest.update(value.tobytes())
    return digest.digest()


def _changed_face_moment(left: PhotoObservation, right: PhotoObservation) -> bool:
    left_faces = {face.person_id: face for face in left.faces if face.person_id and face.confidence >= 0.8}
    right_faces = {face.person_id: face for face in right.faces if face.person_id and face.confidence >= 0.8}
    for person in left_faces.keys() & right_faces.keys():
        a, b = left_faces[person], right_faces[person]
        eyes = {a.eye_state.casefold(), b.eye_state.casefold()}
        if eyes == {"open", "closed"}:
            return True
        if (a.expression not in {"unknown", ""} and b.expression not in {"unknown", ""}
                and (a.expression_confidence or 0) >= 0.8
                and (b.expression_confidence or 0) >= 0.8 and a.expression != b.expression):
            return True
        if max(abs(a.yaw - b.yaw), abs(a.pitch - b.pitch)) > 15:
            return True
    return False


def _strict_pair(left: PhotoObservation, right: PhotoObservation,
                 evidence: _EvidenceCache, fingerprints: dict[str, bytes]) -> bool:
    if left.width <= 0 or left.height <= 0 or right.width <= 0 or right.height <= 0:
        return False
    pair = evidence.get(left, right)
    if not pair.strong_duplicate or not pair.compatible_people:
        return False
    if min(pair.phash, pair.dhash) < DUPLICATE_CONFIG["minimum_hash_similarity"]:
        return False
    if pair.pose is not None and pair.pose < DUPLICATE_CONFIG["minimum_pose_similarity"]:
        return False
    if _changed_face_moment(left, right):
        return False
    if fingerprints[left.id] == fingerprints[right.id]:
        return True
    if left.capture_time and right.capture_time:
        return abs((left.capture_time - right.capture_time).total_seconds()) <= DUPLICATE_CONFIG["seconds"]
    return (left.file_sequence >= 0 and right.file_sequence >= 0
            and abs(left.file_sequence - right.file_sequence) <= DUPLICATE_CONFIG["sequence_gap_without_time"])


def _strict_clusters(members: list[PhotoObservation], evidence: _EvidenceCache) -> list[list[PhotoObservation]]:
    """Global deterministic complete-link clusters; no transitive bridges.

    Time/sequence buckets retrieve only nearby moments. Exact descriptor buckets
    additionally retrieve copies anywhere in the project. Every cluster admission
    must pass against every member, including people and visible moment guards.
    """
    clusters: list[list[PhotoObservation]] = []
    fingerprints = {photo.id: _fingerprint(photo) for photo in members}
    by_fingerprint: dict[bytes, set[int]] = {}
    by_time: dict[int, set[int]] = {}
    by_sequence: dict[int, set[int]] = {}
    for photo in sorted(members, key=_sort_key):
        fingerprint = fingerprints[photo.id]
        candidates = set(by_fingerprint.get(fingerprint, ()))
        time_bucket = int(photo.capture_time.timestamp() // 4) if photo.capture_time else None
        sequence_bucket = photo.file_sequence // 3 if photo.file_sequence >= 0 else None
        if time_bucket is not None:
            for bucket in range(time_bucket - 1, time_bucket + 2):
                candidates.update(by_time.get(bucket, ()))
        if sequence_bucket is not None:
            for bucket in range(sequence_bucket - 1, sequence_bucket + 2):
                candidates.update(by_sequence.get(bucket, ()))
        target = next((index for index in sorted(candidates)
                       if all(_strict_pair(other, photo, evidence, fingerprints) for other in clusters[index])), None)
        if target is None:
            target = len(clusters)
            clusters.append([])
        clusters[target].append(photo)
        by_fingerprint.setdefault(fingerprint, set()).add(target)
        if time_bucket is not None:
            by_time.setdefault(time_bucket, set()).add(target)
        if sequence_bucket is not None:
            by_sequence.setdefault(sequence_bucket, set()).add(target)
    return clusters


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

    evidence = _EvidenceCache()
    owners = {photo.id: group.id for group in groups for photo in group.photos}
    members = [photo for group in groups for photo in group.photos]
    owner_counts: dict[str, int] = {}
    for cluster in _strict_clusters(members, evidence):
        owner = owners[cluster[0].id]
        owner_counts[owner] = owner_counts.get(owner, 0) + 1
        cluster_id = f"{owner}:strict-{owner_counts[owner]:04d}"
        for photo in cluster:
            strict_cluster_by_photo[photo.id] = cluster_id

    for group in groups:
        members = sorted(group.photos, key=_sort_key)
        for beat_index, beat in enumerate(_beats(members, evidence), start=1):
            beat_id = f"{group.id}:beat-{beat_index:04d}"
            for photo in beat:
                beat_by_photo[photo.id] = beat_id

    return DuplicateLayers(
        strict_cluster_by_photo=strict_cluster_by_photo,
        beat_by_photo=beat_by_photo,
    )
