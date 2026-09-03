from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any

from .internal_models import (
    FaceObservation,
    PhotoGroupInternal,
    PhotoObservation,
    is_reliable_face,
    is_significant_face,
)


SEVERE_ISSUES = {
    "主要人物闭眼",
    "主体清晰度不足",
    "主要人物可能被遮挡",
    "主要人物姿态异常",
    "曝光偏差明显",
    "高感噪声明显",
}


@dataclass(slots=True)
class CoverageStage:
    id: str
    label: str
    photos: list[PhotoObservation]

    def public_dict(self, eligible_people: set[str]) -> dict[str, Any]:
        people = sorted(
            {
                person
                for photo in self.photos
                for person in photo.significant_person_ids
                if person in eligible_people
            }
        )
        times = sorted(photo.capture_time for photo in self.photos if photo.capture_time)
        return {
            "id": self.id,
            "label": self.label,
            "photo_count": len(self.photos),
            "person_count": len(people),
            "start_time": times[0].isoformat() if times else None,
            "end_time": times[-1].isoformat() if times else None,
        }


@dataclass(frozen=True, slots=True)
class CoverageSelection:
    selected_photo_ids: frozenset[str]
    keys_by_photo: dict[str, tuple[str, ...]]
    required_keys: tuple[str, ...]
    already_covered_keys: tuple[str, ...]
    unresolved_keys: tuple[str, ...]
    stages: tuple[CoverageStage, ...]
    stage_source: str
    eligible_people: tuple[str, ...]


def _sort_key(photo: PhotoObservation) -> tuple[float, int, str]:
    timestamp = photo.capture_time.timestamp() if photo.capture_time else float("inf")
    sequence = photo.file_sequence if photo.file_sequence >= 0 else 10**12
    return timestamp, sequence, photo.filename.casefold()


def _top_folder(photo: PhotoObservation) -> str | None:
    path = PurePosixPath(photo.relative_path.replace("\\", "/"))
    return path.parts[0] if len(path.parts) > 1 else None


def _format_time_range(index: int, photos: list[PhotoObservation]) -> str:
    times = sorted(photo.capture_time for photo in photos if photo.capture_time)
    if not times:
        return f"自动环节 {index:02d}"
    start = times[0].strftime("%H:%M")
    end = times[-1].strftime("%H:%M")
    return f"自动环节 {index:02d} · {start}–{end}"


def _folder_stages(photos: list[PhotoObservation]) -> list[CoverageStage] | None:
    nested = [photo for photo in photos if _top_folder(photo)]
    folders = {_top_folder(photo) for photo in nested}
    if len(folders) < 2 or len(nested) < max(2, round(len(photos) * 0.80)):
        return None

    buckets: dict[str, list[PhotoObservation]] = defaultdict(list)
    for photo in photos:
        buckets[_top_folder(photo) or "根目录"].append(photo)
    ordered = sorted(buckets.items(), key=lambda item: min(_sort_key(photo) for photo in item[1]))
    stages: list[CoverageStage] = []
    for index, (folder, members) in enumerate(ordered, start=1):
        stage = CoverageStage(id=f"stage-{index:03d}", label=f"环节 {index:02d} · {folder}", photos=members)
        stages.append(stage)
    return stages


def _time_stages(photos: list[PhotoObservation], window_minutes: int) -> list[CoverageStage]:
    ordered = sorted(photos, key=_sort_key)
    if not ordered:
        return []
    window_seconds = max(5, min(60, int(window_minutes))) * 60.0
    gap_seconds = max(180.0, window_seconds * 0.50)
    buckets: list[list[PhotoObservation]] = [[]]
    stage_start: datetime | None = None
    previous_time: datetime | None = None
    previous_group = ""

    for photo in ordered:
        timestamp = photo.capture_time
        same_burst = bool(previous_group and photo.group_id == previous_group)
        new_stage = False
        if buckets[-1] and timestamp and not same_burst:
            if stage_start and (timestamp - stage_start).total_seconds() >= window_seconds:
                new_stage = True
            elif previous_time and (timestamp - previous_time).total_seconds() >= gap_seconds:
                new_stage = True
        if new_stage:
            buckets.append([])
            stage_start = timestamp
        elif not buckets[-1] and timestamp:
            stage_start = timestamp
        buckets[-1].append(photo)
        if timestamp:
            previous_time = timestamp
            stage_start = stage_start or timestamp
        previous_group = photo.group_id

    return [
        CoverageStage(id=f"stage-{index:03d}", label=_format_time_range(index, members), photos=members)
        for index, members in enumerate(buckets, start=1)
        if members
    ]


def assign_coverage_stages(
    groups: list[PhotoGroupInternal],
    window_minutes: int,
) -> tuple[list[CoverageStage], str]:
    photos = [photo for group in groups for photo in group.photos]
    stages = _folder_stages(photos)
    source = "folder" if stages is not None else "time"
    if stages is None:
        stages = _time_stages(photos, window_minutes)
    for stage in stages:
        for photo in stage.photos:
            photo.stage_id = stage.id
            photo.stage_label = stage.label
    return stages, source


def _person_faces(photo: PhotoObservation, person_id: str) -> list[FaceObservation]:
    return [
        face
        for face in photo.faces
        if face.person_id == person_id and is_significant_face(face)
    ]


def _eligible_people(photos: list[PhotoObservation]) -> set[str]:
    support: dict[str, set[str]] = defaultdict(set)
    prominent_singletons: set[str] = set()
    for photo in photos:
        for person in photo.significant_person_ids:
            support[person].add(photo.id)
            faces = _person_faces(photo, person)
            if any(is_reliable_face(face) and face.area_ratio >= 0.008 for face in faces):
                prominent_singletons.add(person)
    # 两张以上的身份聚类可排除大部分一次性背景误检；
    # 只出现一次时，仅接受画面占比足够大的高可信人脸。
    return {
        person
        for person, photo_ids in support.items()
        if len(photo_ids) >= 2 or person in prominent_singletons
    }


def _target_quality_key(
    photo: PhotoObservation,
    person_id: str,
    newly_covered: int,
) -> tuple[float, ...]:
    faces = _person_faces(photo, person_id)
    closed = sum(face.eye_state == "Closed" for face in faces)
    partial = sum(face.eye_state == "Partial" for face in faces)
    open_probability = max(
        (face.open_probability for face in faces if face.open_probability is not None),
        default=0.50,
    )
    face_quality = max(
        (
            face.fiqa_score
            if face.fiqa_score is not None
            else max(face.high_res_sharpness, face.eye_sharpness, face.sharpness)
            for face in faces
        ),
        default=0.0,
    )
    severe = sum(issue in SEVERE_ISSUES for issue in photo.issues)
    return (
        -float(closed),
        -float(severe),
        -float(partial),
        open_probability,
        face_quality,
        float(newly_covered),
        photo.metrics.get("subject_sharpness_score", photo.metrics.get("sharpness_score", 0.0)),
        photo.metrics.get("exposure_score", 0.0),
        photo.metrics.get("group_ranking_score", photo.score),
        photo.score,
    )


def _unique_reasons(reasons: list[str]) -> list[str]:
    return list(dict.fromkeys(reason for reason in reasons if reason))[:5]


def _coverage_key(stage_id: str, person_id: str) -> str:
    return f"{stage_id}:{person_id}"


def select_person_stage_coverage(
    groups: list[PhotoGroupInternal],
    *,
    primary_photo_ids: set[str],
    window_minutes: int = 15,
) -> CoverageSelection:
    """返回补齐人物×环节所需的最小贪心集合，不修改分类或星级。"""
    stages, stage_source = assign_coverage_stages(groups, window_minutes)
    photos = [
        photo
        for stage in stages
        for photo in stage.photos
        if photo.width > 0 and photo.height > 0
    ]
    eligible_people = _eligible_people(photos)
    candidates: dict[str, list[PhotoObservation]] = defaultdict(list)
    photo_cells: dict[str, set[str]] = defaultdict(set)
    person_by_key: dict[str, str] = {}
    for stage in stages:
        for photo in stage.photos:
            if photo.width <= 0 or photo.height <= 0:
                continue
            for person in photo.significant_person_ids & eligible_people:
                key = _coverage_key(stage.id, person)
                candidates[key].append(photo)
                photo_cells[photo.id].add(key)
                person_by_key[key] = person

    required = set(candidates)
    covered = {
        key
        for key, options in candidates.items()
        if any(photo.id in primary_photo_ids for photo in options)
    }
    missing = required - covered
    selected_photo_ids: set[str] = set()
    keys_by_photo: dict[str, tuple[str, ...]] = {}

    while missing:
        options = sorted(
            {
                photo.id: photo
                for key in missing
                for photo in candidates[key]
                if photo.id not in primary_photo_ids and photo.id not in selected_photo_ids
            }.values(),
            key=_sort_key,
        )
        if not options:
            break

        def candidate_key(photo: PhotoObservation) -> tuple[float, ...]:
            contribution = photo_cells[photo.id] & missing
            quality_keys = [
                _target_quality_key(photo, person_by_key[key], len(contribution))
                for key in contribution
            ]
            weakest_person = min(quality_keys) if quality_keys else (-1.0,)
            return (float(len(contribution)), *weakest_person)

        chosen = max(options, key=candidate_key)
        newly_covered = photo_cells[chosen.id] & missing
        if not newly_covered:
            break
        selected_photo_ids.add(chosen.id)
        keys_by_photo[chosen.id] = tuple(sorted(newly_covered))
        covered.update(newly_covered)
        missing.difference_update(newly_covered)

    return CoverageSelection(
        selected_photo_ids=frozenset(selected_photo_ids),
        keys_by_photo=keys_by_photo,
        required_keys=tuple(sorted(required)),
        already_covered_keys=tuple(sorted(required & {
            key for key, options in candidates.items() if any(photo.id in primary_photo_ids for photo in options)
        })),
        unresolved_keys=tuple(sorted(missing)),
        stages=tuple(stages),
        stage_source=stage_source,
        eligible_people=tuple(sorted(eligible_people)),
    )


def apply_person_stage_coverage(
    groups: list[PhotoGroupInternal],
    *,
    enabled: bool,
    window_minutes: int = 15,
) -> dict[str, Any]:
    """旧扫描流程的兼容包装；新语义策略直接使用纯选择器。"""
    if not enabled:
        return {
            "enabled": False,
            "stage_source": "disabled",
            "window_minutes": int(window_minutes),
            "stages": [],
            "eligible_people": 0,
            "required_cells": 0,
            "already_covered_cells": 0,
            "protected_photos": 0,
            "protected_cells": 0,
            "unresolved_cells": 0,
        }

    primary_ids = {
        photo.id
        for group in groups
        for photo in group.photos
        if photo.category == "selected"
    }
    selection = select_person_stage_coverage(
        groups,
        primary_photo_ids=primary_ids,
        window_minutes=window_minutes,
    )
    by_id = {photo.id: photo for group in groups for photo in group.photos}
    for photo_id in selection.selected_photo_ids:
        photo = by_id[photo_id]
        keys = selection.keys_by_photo[photo_id]
        people = sorted({key.split(":", 1)[1] for key in keys})
        if not photo.coverage_protected:
            photo.coverage_original_category = photo.category
        photo.coverage_protected = True
        photo.coverage_person_ids = people
        photo.category = "selected"
        photo.is_best_pick = True
        photo.stars = max(1, photo.stars)
        reason = f"覆盖保底：{photo.stage_label or photo.stage_id} 为 {', '.join(people)} 至少保留一张"
        warning = (
            f"保留不代表画质合格，仍需复核：{' / '.join(photo.issues[:2])}"
            if photo.issues
            else ""
        )
        photo.selection_reasons = _unique_reasons([reason, warning, *photo.selection_reasons])

    eligible_people = set(selection.eligible_people)
    return {
        "enabled": True,
        "stage_source": selection.stage_source,
        "window_minutes": int(window_minutes),
        "stages": [stage.public_dict(eligible_people) for stage in selection.stages],
        "eligible_people": len(selection.eligible_people),
        "required_cells": len(selection.required_keys),
        "already_covered_cells": len(selection.already_covered_keys),
        "protected_photos": len(selection.selected_photo_ids),
        "protected_cells": sum(len(keys) for keys in selection.keys_by_photo.values()),
        "unresolved_cells": len(selection.unresolved_keys),
    }
