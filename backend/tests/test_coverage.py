from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from photocull.coverage import select_person_stage_coverage
from photocull.internal_models import FaceObservation, PhotoGroupInternal, PhotoObservation, VisualDescriptor


def make_photo(
    identifier: str,
    minute: int,
    people: tuple[str, ...],
    *,
    category: str,
    issues: list[str] | None = None,
    relative_path: str | None = None,
    area_ratio: float = 0.04,
    eye_state: str = "Open",
    score: float = 75.0,
) -> PhotoObservation:
    faces = [
        FaceObservation(
            face_id=f"{identifier}-{index}",
            bbox=(0.1 + index * 0.2, 0.1, 0.25 + index * 0.2, 0.45),
            confidence=0.97,
            area_ratio=area_ratio,
            embedding=None,
            person_id=person,
            eye_state=eye_state,
            open_probability=0.08 if eye_state == "Closed" else 0.94,
            sharpness=72.0,
            high_res_sharpness=74.0,
            fiqa_score=68.0,
        )
        for index, person in enumerate(people)
    ]
    return PhotoObservation(
        id=identifier,
        path=Path(f"{identifier}.jpg"),
        source_root=Path("."),
        filename=f"{identifier}.jpg",
        relative_path=relative_path or f"{identifier}.jpg",
        width=1600,
        height=1000,
        capture_time=datetime(2026, 5, 1, 9, 0) + timedelta(minutes=minute),
        file_sequence=minute,
        descriptor=VisualDescriptor(
            phash=1,
            layout=np.ones(4, dtype=np.float32),
            color=np.ones(4, dtype=np.float32),
            edge=np.ones(4, dtype=np.float32),
        ),
        faces=faces,
        person_ids=list(people),
        metrics={
            "sharpness_score": 72.0,
            "subject_sharpness_score": 70.0,
            "exposure_score": 80.0,
            "group_ranking_score": score,
        },
        score=score,
        issues=list(issues or []),
        category=category,
        is_best_pick=category == "selected",
        stars=2 if category == "selected" else 0,
    )


def make_group(index: int, *photos: PhotoObservation) -> PhotoGroupInternal:
    group_id = f"group-{index:05d}"
    for photo in photos:
        photo.group_id = group_id
    return PhotoGroupInternal(id=group_id, photos=list(photos), confidence=0.9, reason="test")


def test_coverage_selects_least_bad_photo_without_mutating_diagnostics() -> None:
    selected = make_photo("selected-a", 0, ("人物 01",), category="selected")
    closed = make_photo(
        "closed-b",
        1,
        ("人物 02",),
        category="closed_eyes",
        issues=["主要人物闭眼"],
        eye_state="Closed",
        score=82.0,
    )
    blurred = make_photo(
        "blurred-b",
        2,
        ("人物 02",),
        category="duplicate",
        issues=["主体清晰度不足"],
        score=68.0,
    )
    groups = [make_group(1, selected), make_group(2, closed, blurred)]

    selection = select_person_stage_coverage(
        groups,
        primary_photo_ids={selected.id},
        window_minutes=15,
    )

    assert len(selection.required_keys) == 2
    assert len(selection.already_covered_keys) == 1
    assert selection.selected_photo_ids == frozenset({blurred.id})
    assert len(selection.keys_by_photo[blurred.id]) == 1
    assert blurred.coverage_protected is False
    assert blurred.category == "duplicate"
    assert blurred.stars == 0
    assert blurred.issues == ["主体清晰度不足"]
    assert closed.category == "closed_eyes"


def test_same_person_is_protected_once_in_each_time_stage() -> None:
    first = make_photo("stage-one", 0, ("人物 01",), category="selected")
    second = make_photo(
        "stage-two",
        20,
        ("人物 01",),
        category="blurred",
        issues=["主体清晰度不足"],
    )
    groups = [make_group(1, first), make_group(2, second)]

    selection = select_person_stage_coverage(
        groups,
        primary_photo_ids={first.id},
        window_minutes=15,
    )

    assert len(selection.stages) == 2
    assert len(selection.required_keys) == 2
    assert selection.selected_photo_ids == frozenset({second.id})
    assert first.stage_id != second.stage_id


def test_folder_stages_and_multi_person_photo_minimize_extra_picks() -> None:
    opening = make_photo(
        "opening",
        0,
        ("人物 01", "人物 02", "人物 03"),
        category="selected",
        relative_path="开幕/opening.jpg",
    )
    award_a = make_photo(
        "award-a",
        30,
        ("人物 01",),
        category="selected",
        relative_path="颁奖/award-a.jpg",
    )
    award_bc = make_photo(
        "award-bc",
        31,
        ("人物 02", "人物 03"),
        category="duplicate",
        relative_path="颁奖/award-bc.jpg",
    )
    groups = [make_group(1, opening), make_group(2, award_a), make_group(3, award_bc)]

    selection = select_person_stage_coverage(
        groups,
        primary_photo_ids={opening.id, award_a.id},
        window_minutes=60,
    )

    assert selection.stage_source == "folder"
    assert len(selection.stages) == 2
    assert selection.selected_photo_ids == frozenset({award_bc.id})
    assert len(selection.keys_by_photo[award_bc.id]) == 2


def test_small_one_shot_background_face_does_not_expand_selection() -> None:
    background = make_photo(
        "background",
        0,
        ("人物 99",),
        category="rejected",
        area_ratio=0.002,
    )
    groups = [make_group(1, background)]

    selection = select_person_stage_coverage(
        groups,
        primary_photo_ids=set(),
        window_minutes=15,
    )

    assert selection.eligible_people == ()
    assert selection.selected_photo_ids == frozenset()
    assert background.category == "rejected"
