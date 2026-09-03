from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math

import numpy as np

from .imaging import cosine_similarity, dhash_similarity, phash_similarity, temporal_similarity
from .internal_models import PhotoGroupInternal, PhotoObservation, SimilarityEvidence


PRESET_THRESHOLDS = {
    "cautious": 0.81,
    "balanced": 0.73,
    "aggressive": 0.66,
}
SEQUENCE_LIMITS = {"cautious": 7, "balanced": 16, "aggressive": 28}
CONSOLIDATION_SEQUENCE_LIMITS = {"cautious": 10, "balanced": 28, "aggressive": 44}
CONSOLIDATION_SECONDS = {"cautious": 10.0, "balanced": 28.0, "aggressive": 48.0}
GROUP_SPAN_SECONDS = {"cautious": 7.0, "balanced": 16.0, "aggressive": 30.0}
GROUP_SPAN_SEQUENCES = {"cautious": 7, "balanced": 16, "aggressive": 28}
POSE_STAGE_FLOORS = {"cautious": 0.48, "balanced": 0.34, "aggressive": 0.24}


def _unit_interval_cosine(value: float | None, neutral: float = 0.5) -> float:
    return neutral if value is None else float(np.clip((value + 1.0) / 2.0, 0.0, 1.0))


def _body_similarity(left: PhotoObservation, right: PhotoObservation) -> float | None:
    """计算一对一人体外观匹配；衣着相似只能软加分，不能形成身份硬约束。"""
    left_embeddings = [body.embedding for body in left.bodies if body.embedding is not None]
    right_embeddings = [body.embedding for body in right.bodies if body.embedding is not None]
    if not left_embeddings or not right_embeddings:
        return None
    candidates = sorted(
        (
            (float(np.dot(left_embedding, right_embedding)), left_index, right_index)
            for left_index, left_embedding in enumerate(left_embeddings)
            for right_index, right_embedding in enumerate(right_embeddings)
        ),
        reverse=True,
    )
    used_left: set[int] = set()
    used_right: set[int] = set()
    matches: list[float] = []
    for similarity, left_index, right_index in candidates:
        if left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        matches.append(similarity)
        if len(matches) >= min(len(left_embeddings), len(right_embeddings)):
            break
    if not matches:
        return None
    coverage = min(len(left_embeddings), len(right_embeddings)) / max(len(left_embeddings), len(right_embeddings))
    raw_similarity = float(np.mean(matches)) * (0.88 + 0.12 * coverage)
    return float(np.clip((raw_similarity - 0.35) / 0.45, 0.0, 1.0))


def _pose_similarity(left: PhotoObservation, right: PhotoObservation) -> float | None:
    """匹配主要人物的 3D 姿态；这是动作阶段证据，不作为人物身份。"""
    left_poses = [
        pose
        for pose in left.poses
        if pose.descriptor is not None and pose.visibility >= 0.20 and pose.presence_confidence >= 0.60
    ]
    right_poses = [
        pose
        for pose in right.poses
        if pose.descriptor is not None and pose.visibility >= 0.20 and pose.presence_confidence >= 0.60
    ]
    for poses in (left_poses, right_poses):
        scored = [pose.foreground_score for pose in poses if pose.foreground_score is not None]
        if not scored:
            continue
        maximum = max(scored)
        poses[:] = [
            pose
            for pose in poses
            if pose.foreground_score is not None
            and pose.foreground_score >= max(0.46, maximum - 0.17)
        ]
    if not left_poses or not right_poses:
        return None
    candidates: list[tuple[float, float, int, int]] = []
    for left_index, left_pose in enumerate(left_poses):
        left_center = (
            (left_pose.bbox[0] + left_pose.bbox[2]) * 0.5,
            (left_pose.bbox[1] + left_pose.bbox[3]) * 0.5,
        )
        for right_index, right_pose in enumerate(right_poses):
            right_center = (
                (right_pose.bbox[0] + right_pose.bbox[2]) * 0.5,
                (right_pose.bbox[1] + right_pose.bbox[3]) * 0.5,
            )
            action = float(np.clip(np.dot(left_pose.descriptor, right_pose.descriptor), 0.0, 1.0))
            distance = math.hypot(left_center[0] - right_center[0], left_center[1] - right_center[1])
            spatial = float(np.clip(1.0 - distance / 0.75, 0.0, 1.0))
            candidates.append((0.88 * action + 0.12 * spatial, action, left_index, right_index))
    candidates.sort(reverse=True)
    used_left: set[int] = set()
    used_right: set[int] = set()
    matches: list[float] = []
    for _combined, action, left_index, right_index in candidates:
        if left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        matches.append(action)
        if len(matches) >= min(len(left_poses), len(right_poses)):
            break
    if not matches:
        return None
    coverage = min(len(left_poses), len(right_poses)) / max(len(left_poses), len(right_poses))
    return float(np.clip(np.mean(matches) * (0.90 + 0.10 * coverage), 0.0, 1.0))


def _depth_similarity(left: PhotoObservation, right: PhotoObservation) -> float | None:
    if (
        left.depth is None
        or right.depth is None
        or left.depth.descriptor is None
        or right.depth.descriptor is None
    ):
        return None
    return _unit_interval_cosine(cosine_similarity(left.depth.descriptor, right.depth.descriptor))


def _pose_stage_compatible(evidence: SimilarityEvidence, preset: str, slack: float = 0.0) -> bool:
    if evidence.strong_duplicate or evidence.pose is None:
        return True
    return evidence.pose >= max(0.0, POSE_STAGE_FLOORS[preset] - slack)


def _pose_supported_scene_match(evidence: SimilarityEvidence, preset: str) -> bool:
    if evidence.pose is None or evidence.pose < {"cautious": 0.95, "balanced": 0.92, "aggressive": 0.89}[preset]:
        return False
    if evidence.depth is not None and evidence.depth < {"cautious": 0.78, "balanced": 0.72, "aggressive": 0.66}[preset]:
        return False
    threshold = PRESET_THRESHOLDS[preset]
    return (
        evidence.compatible_people
        and evidence.temporal >= {"cautious": 0.70, "balanced": 0.48, "aggressive": 0.28}[preset]
        and evidence.scene >= threshold - 0.15
        and evidence.total >= threshold - 0.17
    )


def compare_photos(left: PhotoObservation, right: PhotoObservation) -> SimilarityEvidence:
    phash = phash_similarity(left.descriptor.phash, right.descriptor.phash)
    dhash = dhash_similarity(left.descriptor.dhash, right.descriptor.dhash)
    layout = _unit_interval_cosine(cosine_similarity(left.descriptor.layout, right.descriptor.layout))
    color = max(0.0, cosine_similarity(left.descriptor.color, right.descriptor.color) or 0.0)
    edge = max(0.0, cosine_similarity(left.descriptor.edge, right.descriptor.edge) or 0.0)
    semantic_raw = cosine_similarity(left.descriptor.semantic, right.descriptor.semantic)
    semantic = float(np.clip(semantic_raw, 0.0, 1.0)) if semantic_raw is not None else None
    hash_similarity = 0.66 * phash + 0.34 * dhash
    composition = 0.56 * layout + 0.25 * color + 0.19 * edge
    body = _body_similarity(left, right)
    pose = _pose_similarity(left, right)
    depth = _depth_similarity(left, right)

    left_people = left.significant_person_ids
    right_people = right.significant_person_ids
    left_reliable_people = left.reliable_person_ids
    right_reliable_people = right.reliable_person_ids
    compatible_people = True
    hard_person_conflict = (
        bool(left_reliable_people)
        and bool(right_reliable_people)
        and not (left_reliable_people & right_reliable_people)
    )
    if hard_person_conflict:
        people = 0.0
        compatible_people = False
    elif left_people and right_people:
        intersection = left_people & right_people
        union = left_people | right_people
        people = len(intersection) / len(union)
    elif left_people or right_people:
        people = 0.52 + 0.34 * body if body is not None else 0.52
    elif body is not None:
        people = 0.58 + 0.30 * body
    else:
        people = 0.58

    temporal = temporal_similarity(left.capture_time, right.capture_time, left.file_sequence, right.file_sequence)
    strong_duplicate = (hash_similarity >= 0.925 and layout >= 0.89) or (
        semantic is not None and semantic >= 0.965 and composition >= 0.80 and hash_similarity >= 0.79
    )
    if semantic is None:
        total = 0.31 * temporal + 0.23 * people + 0.27 * hash_similarity + 0.19 * composition
        scene = 0.57 * hash_similarity + 0.43 * composition
    else:
        total = 0.40 * semantic + 0.20 * temporal + 0.15 * people + 0.15 * hash_similarity + 0.10 * composition
        scene = 0.52 * semantic + 0.27 * hash_similarity + 0.21 * composition
    if strong_duplicate:
        total = max(total, 0.92)
    if not compatible_people:
        total = min(total, 0.40)
    if depth is not None and not strong_duplicate:
        total = 0.95 * total + 0.05 * depth
        scene = 0.92 * scene + 0.08 * depth
    return SimilarityEvidence(
        total=total,
        scene=scene,
        semantic=semantic,
        phash=phash,
        dhash=dhash,
        layout=layout,
        color=color,
        edge=edge,
        composition=composition,
        people=people,
        body=body,
        pose=pose,
        depth=depth,
        temporal=temporal,
        compatible_people=compatible_people,
        strong_duplicate=strong_duplicate,
    )


@dataclass(slots=True)
class _GroupCandidate:
    group: PhotoGroupInternal
    evidence: SimilarityEvidence


def _sort_key(photo: PhotoObservation) -> tuple[float, int, str]:
    timestamp = photo.capture_time.timestamp() if photo.capture_time else float("inf")
    sequence = photo.file_sequence if photo.file_sequence >= 0 else 10**12
    return timestamp, sequence, photo.filename.casefold()


def _people_compatible_with_group(photo: PhotoObservation, group: PhotoGroupInternal) -> bool:
    photo_people = photo.reliable_person_ids
    stable_group_people = _stable_group_reliable_people(group)
    if photo_people and stable_group_people and not (photo_people & stable_group_people):
        return False
    return True


def _stable_group_people(group: PhotoGroupInternal) -> set[str]:
    counts = Counter(person for member in group.photos for person in member.significant_person_ids)
    minimum = max(1, math.ceil(len(group.photos) * 0.35))
    return {person for person, count in counts.items() if count >= minimum}


def _stable_group_reliable_people(group: PhotoGroupInternal) -> set[str]:
    counts = Counter(person for member in group.photos for person in member.reliable_person_ids)
    minimum = max(1, math.ceil(len(group.photos) * 0.35))
    return {person for person, count in counts.items() if count >= minimum}


def _sequence_close(left: PhotoObservation, right: PhotoObservation, preset: str) -> bool:
    if left.capture_time and right.capture_time:
        seconds = abs((left.capture_time - right.capture_time).total_seconds())
        time_limit = {"cautious": 7.0, "balanced": 16.0, "aggressive": 30.0}[preset]
        if seconds <= time_limit:
            return True
        if left.file_sequence >= 0 and right.file_sequence >= 0:
            return seconds <= 45.0 and abs(left.file_sequence - right.file_sequence) <= SEQUENCE_LIMITS[preset]
        return False
    if left.file_sequence >= 0 and right.file_sequence >= 0:
        return abs(left.file_sequence - right.file_sequence) <= SEQUENCE_LIMITS[preset]
    return True


def _group_span_is_coherent(photos: list[PhotoObservation], preset: str) -> bool:
    if len(photos) <= 1:
        return True
    ordered = sorted(photos, key=_sort_key)
    times = [photo.capture_time for photo in ordered if photo.capture_time]
    sequences = [photo.file_sequence for photo in ordered if photo.file_sequence >= 0]
    seconds_span = (max(times) - min(times)).total_seconds() if len(times) >= 2 else 0.0
    sequence_span = max(sequences) - min(sequences) if len(sequences) >= 2 else 0
    has_complete_capture_times = len(times) == len(photos)
    within_normal_span = (
        seconds_span <= GROUP_SPAN_SECONDS[preset]
        if has_complete_capture_times
        else sequence_span <= GROUP_SPAN_SEQUENCES[preset]
    )
    if within_normal_span:
        return True
    exceeds_extended_span = (
        seconds_span > GROUP_SPAN_SECONDS[preset] * 2.0
        if has_complete_capture_times
        else sequence_span > GROUP_SPAN_SEQUENCES[preset] * 2
    )
    if exceeds_extended_span:
        return False
    endpoint = compare_photos(ordered[0], ordered[-1])
    return endpoint.strong_duplicate and endpoint.compatible_people


@dataclass(slots=True)
class _MergeCandidate:
    left_index: int
    right_index: int
    evidence: SimilarityEvidence
    shared_people: set[str]


def _consolidation_pair_close(left: PhotoObservation, right: PhotoObservation, preset: str) -> bool:
    if left.capture_time and right.capture_time:
        return abs((left.capture_time - right.capture_time).total_seconds()) <= CONSOLIDATION_SECONDS[preset]
    if left.file_sequence >= 0 and right.file_sequence >= 0:
        return abs(left.file_sequence - right.file_sequence) <= CONSOLIDATION_SEQUENCE_LIMITS[preset]
    return False


def _right_group_beyond_window(
    left: PhotoGroupInternal,
    right: PhotoGroupInternal,
    preset: str,
    ordered_by_time: bool,
    ordered_by_sequence: bool,
) -> bool:
    if ordered_by_time:
        left_latest = max(photo.capture_time for photo in left.photos if photo.capture_time)
        right_earliest = min(photo.capture_time for photo in right.photos if photo.capture_time)
        return (right_earliest - left_latest).total_seconds() > CONSOLIDATION_SECONDS[preset]
    if ordered_by_sequence:
        left_latest = max(photo.file_sequence for photo in left.photos)
        right_earliest = min(photo.file_sequence for photo in right.photos)
        return right_earliest - left_latest > CONSOLIDATION_SEQUENCE_LIMITS[preset]
    return False


def _merge_is_safe(
    left: PhotoGroupInternal,
    right: PhotoGroupInternal,
    evidence: SimilarityEvidence,
    shared_people: set[str],
    preset: str,
) -> bool:
    left_people = _stable_group_reliable_people(left)
    right_people = _stable_group_reliable_people(right)
    if left_people and right_people and not shared_people:
        return False
    if evidence.strong_duplicate:
        return True
    if not _pose_stage_compatible(evidence, preset):
        return False
    if _pose_supported_scene_match(evidence, preset):
        return True

    threshold = PRESET_THRESHOLDS[preset]
    if shared_people:
        temporal_floor = {"cautious": 0.68, "balanced": 0.46, "aggressive": 0.26}[preset]
        return (
            evidence.compatible_people
            and evidence.temporal >= temporal_floor
            and evidence.scene >= threshold - 0.10
            and evidence.total >= threshold - 0.12
        )

    temporal_floor = {"cautious": 0.76, "balanced": 0.56, "aggressive": 0.34}[preset]
    scene_floor = threshold - {"cautious": 0.02, "balanced": 0.05, "aggressive": 0.07}[preset]
    total_floor = threshold - {"cautious": 0.04, "balanced": 0.07, "aggressive": 0.09}[preset]
    semantic_floor = {"cautious": 0.91, "balanced": 0.82, "aggressive": 0.74}[preset]
    learned_match = evidence.semantic is not None and evidence.semantic >= semantic_floor
    structural_match = evidence.composition >= 0.76 or max(evidence.phash, evidence.dhash) >= 0.84
    return (
        evidence.compatible_people
        and evidence.temporal >= temporal_floor
        and evidence.scene >= scene_floor
        and evidence.total >= total_floor
        and (learned_match or structural_match)
    )


def consolidate_split_groups(groups: list[PhotoGroupInternal], preset: str = "balanced") -> list[PhotoGroupInternal]:
    """二次合并被变焦、姿态或曝光变化切开的相邻连拍组。"""
    preset = preset if preset in PRESET_THRESHOLDS else "balanced"
    working = sorted(groups, key=lambda group: min(_sort_key(photo) for photo in group.photos))
    all_photos = [photo for group in working for photo in group.photos]
    ordered_by_time = all(photo.capture_time is not None for photo in all_photos)
    ordered_by_sequence = (
        not any(photo.capture_time is not None for photo in all_photos)
        and all(photo.file_sequence >= 0 for photo in all_photos)
    )

    while True:
        candidates: list[_MergeCandidate] = []
        for left_index, left in enumerate(working):
            left_people = _stable_group_people(left)
            for right_index in range(left_index + 1, len(working)):
                right = working[right_index]
                if _right_group_beyond_window(
                    left,
                    right,
                    preset,
                    ordered_by_time,
                    ordered_by_sequence,
                ):
                    break
                right_people = _stable_group_people(right)
                shared_people = left_people & right_people
                best_evidence: SimilarityEvidence | None = None
                for left_photo in left.photos:
                    for right_photo in right.photos:
                        if not _consolidation_pair_close(left_photo, right_photo, preset):
                            continue
                        evidence = compare_photos(left_photo, right_photo)
                        if not _merge_is_safe(left, right, evidence, shared_people, preset):
                            continue
                        if not _group_span_is_coherent([*left.photos, *right.photos], preset):
                            continue
                        if best_evidence is None or (
                            evidence.strong_duplicate,
                            evidence.total,
                            evidence.scene,
                        ) > (
                            best_evidence.strong_duplicate,
                            best_evidence.total,
                            best_evidence.scene,
                        ):
                            best_evidence = evidence
                if best_evidence:
                    candidates.append(_MergeCandidate(left_index, right_index, best_evidence, shared_people))

        if not candidates:
            break
        best = max(
            candidates,
            key=lambda candidate: (
                candidate.evidence.strong_duplicate,
                bool(candidate.shared_people),
                candidate.evidence.total,
                candidate.evidence.scene,
            ),
        )
        left = working[best.left_index]
        right = working[best.right_index]
        left.photos = sorted([*left.photos, *right.photos], key=_sort_key)
        left.confidence = float(np.mean((left.confidence, right.confidence, best.evidence.total)))
        left.reason = (
            "二次校验：同人物 + AI 场景 + 连拍时序"
            if best.shared_people
            else "二次校验：AI 场景 + 连拍时序"
        )
        del working[best.right_index]

    working.sort(key=lambda group: min(_sort_key(photo) for photo in group.photos))
    for index, group in enumerate(working, start=1):
        group.id = f"group-{index:05d}"
        for photo in group.photos:
            photo.group_id = group.id
    return working


def group_similar_photos(photos: list[PhotoObservation], preset: str = "balanced") -> list[PhotoGroupInternal]:
    preset = preset if preset in PRESET_THRESHOLDS else "balanced"
    threshold = PRESET_THRESHOLDS[preset]
    groups: list[PhotoGroupInternal] = []
    confidence_samples: dict[str, list[float]] = {}

    for photo in sorted(photos, key=_sort_key):
        candidates: list[_GroupCandidate] = []
        for group in groups[-24:]:
            if not _people_compatible_with_group(photo, group):
                continue
            representative = max(group.photos, key=lambda member: member.score)
            latest = group.photos[-1]
            if not _sequence_close(photo, latest, preset):
                continue
            representative_evidence = compare_photos(photo, representative)
            latest_evidence = compare_photos(photo, latest)
            evidence = representative_evidence if representative_evidence.total >= latest_evidence.total else latest_evidence
            reference_floor = min(representative_evidence.scene, latest_evidence.scene)
            accepted = (
                evidence.strong_duplicate
                or (
                    (
                        evidence.total >= threshold
                        and evidence.compatible_people
                        and reference_floor >= threshold - 0.13
                    )
                    or _pose_supported_scene_match(evidence, preset)
                )
            )
            pose_coherent = _pose_stage_compatible(latest_evidence, preset) and _pose_stage_compatible(
                representative_evidence,
                preset,
                slack=0.08,
            )
            accepted = accepted and pose_coherent and _group_span_is_coherent([*group.photos, photo], preset)
            if accepted:
                candidates.append(_GroupCandidate(group=group, evidence=evidence))

        if not candidates:
            group_id = f"group-{len(groups) + 1:05d}"
            new_group = PhotoGroupInternal(id=group_id, photos=[photo], confidence=1.0, reason="独立画面")
            photo.group_id = group_id
            groups.append(new_group)
            confidence_samples[group_id] = []
            continue

        best = max(candidates, key=lambda candidate: candidate.evidence.total)
        photo.group_id = best.group.id
        best.group.photos.append(photo)
        confidence_samples[best.group.id].append(best.evidence.total)

    for group in groups:
        samples = confidence_samples[group.id]
        group.confidence = float(np.mean(samples)) if samples else 1.0
        if len(group.photos) == 1:
            group.reason = "独立画面"
            continue
        people = {person for photo in group.photos for person in photo.significant_person_ids}
        semantic_available = any(photo.descriptor.semantic is not None for photo in group.photos)
        if people and semantic_available:
            group.reason = "AI 场景特征 + 同人物确认"
        elif people:
            group.reason = "画面相似 + 同人物确认"
        elif semantic_available:
            group.reason = "AI 场景特征 + 连拍时序"
        else:
            group.reason = "感知特征 + 连拍时序"
    return consolidate_split_groups(groups, preset)
