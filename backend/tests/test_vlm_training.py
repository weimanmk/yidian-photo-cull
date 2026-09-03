from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from photocull.internal_models import PhotoGroupInternal, PhotoObservation, VisualDescriptor
from photocull.vlm_training import (
    build_sft_record,
    choose_training_candidates,
    deterministic_split,
    find_single_manual_photo,
)


def make_photo(tmp_path: Path, name: str, score: float, rank: int) -> PhotoObservation:
    path = tmp_path / name
    descriptor = VisualDescriptor(
        phash=0,
        layout=np.zeros(1, dtype=np.float32),
        color=np.zeros(1, dtype=np.float32),
        edge=np.zeros(1, dtype=np.float32),
    )
    return PhotoObservation(
        id=path.stem,
        path=path,
        source_root=tmp_path,
        filename=path.name,
        relative_path=path.name,
        width=900,
        height=600,
        capture_time=None,
        file_sequence=rank,
        descriptor=descriptor,
        metrics={
            "group_ranking_score": score,
            "technical_score": score,
            "face_quality_score": 80.0,
            "eye_score": 90.0,
            "motion_blur_score": 85.0,
            "exposure_score": 82.0,
            "composition_score": 78.0,
        },
        score=score,
        rank_in_group=rank,
    )


def test_only_a_group_with_one_manual_pick_becomes_training_label(tmp_path: Path) -> None:
    first = make_photo(tmp_path, "A.jpg", 90.0, 1)
    second = make_photo(tmp_path, "B.jpg", 80.0, 2)
    group = PhotoGroupInternal("group-1", [first, second], 0.9, "连拍")

    assert find_single_manual_photo(group, {"b"}) is second
    assert find_single_manual_photo(group, set()) is None
    assert find_single_manual_photo(group, {"a", "b"}) is None


def test_candidate_selection_keeps_manual_pick_and_hardest_negatives(tmp_path: Path) -> None:
    photos = [
        make_photo(tmp_path, "A.jpg", 95.0, 1),
        make_photo(tmp_path, "B.jpg", 92.0, 2),
        make_photo(tmp_path, "C.jpg", 88.0, 3),
        make_photo(tmp_path, "D.jpg", 70.0, 4),
        make_photo(tmp_path, "E.jpg", 40.0, 5),
    ]
    group = PhotoGroupInternal("group-1", photos, 0.9, "连拍")

    selected = choose_training_candidates(group, photos[3], 4, "活动A", permutation=0)

    assert {photo.id for photo in selected} == {"A", "B", "C", "D"}
    assert selected == choose_training_candidates(group, photos[3], 4, "活动A", permutation=0)
    assert selected != choose_training_candidates(group, photos[3], 4, "活动A", permutation=1)


def test_split_is_group_stable_and_locked_test_wins() -> None:
    locked = {"group-locked"}

    assert deterministic_split("活动A", "group-locked", locked, 0.15) == "test"
    assert deterministic_split("活动A", "group-20", locked, 0.15) == deterministic_split(
        "活动A", "group-20", locked, 0.15
    )
    assert deterministic_split("活动A", "group-20", locked, 0.15) in {"train", "validation"}


def test_sft_record_uses_manual_pick_as_top1_without_false_rejects(tmp_path: Path) -> None:
    first = make_photo(tmp_path, "A.jpg", 90.0, 1)
    manual = make_photo(tmp_path, "B.jpg", 80.0, 2)
    group = PhotoGroupInternal("group-1", [first, manual], 0.9, "连拍")
    image_path = tmp_path / "group-1.jpg"

    record = build_sft_record(group, [first, manual], manual, image_path)
    answer = json.loads(record["messages"][-1]["content"])

    assert record["images"] == [str(image_path.resolve())]
    assert record["messages"][1]["content"].startswith("<image>")
    assert answer["best_photo_id"] == "B"
    assert answer["ranking"][0]["photo_id"] == "B"
    assert answer["rejected_photo_ids"] == []
    assert answer["confidence"] < 1.0
