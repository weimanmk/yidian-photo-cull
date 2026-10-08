from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from photocull.internal_models import (
    FaceObservation,
    PhotoGroupInternal,
    PhotoObservation,
    VisualDescriptor,
)
from photocull.rating_model import RatingModelError
from photocull.rating_policy import assign_semantic_ratings


def descriptor(seed: int) -> VisualDescriptor:
    layout = np.zeros(16, dtype=np.float32)
    color = np.zeros(16, dtype=np.float32)
    edge = np.zeros(16, dtype=np.float32)
    semantic = np.zeros(16, dtype=np.float32)
    layout[seed % 16] = 1.0
    color[(seed + 3) % 16] = 1.0
    edge[(seed + 6) % 16] = 1.0
    semantic[(seed + 9) % 16] = 1.0
    return VisualDescriptor(
        phash=1 << (seed % 63),
        dhash=1 << ((seed + 11) % 63),
        layout=layout,
        color=color,
        edge=edge,
        semantic=semantic,
    )


def make_photo(
    identifier: str,
    sequence: int,
    *,
    seed: int,
    score: float = 75.0,
    person: str | None = None,
    issues: list[str] | None = None,
    decodable: bool = True,
) -> PhotoObservation:
    faces = []
    if person:
        faces.append(
            FaceObservation(
                face_id=f"{identifier}-face",
                bbox=(0.2, 0.1, 0.7, 0.9),
                confidence=0.98,
                area_ratio=0.12,
                embedding=None,
                person_id=person,
                eye_state="Open",
                open_probability=0.96,
                sharpness=80.0,
                high_res_sharpness=82.0,
                fiqa_score=78.0,
            )
        )
    return PhotoObservation(
        id=identifier,
        path=Path(f"{identifier}.jpg"),
        source_root=Path("."),
        filename=f"IMG_{sequence:04d}.jpg",
        relative_path=f"IMG_{sequence:04d}.jpg",
        width=1600 if decodable else 0,
        height=1000 if decodable else 0,
        capture_time=datetime(2026, 8, 30, 9, 0) + timedelta(seconds=sequence),
        file_sequence=sequence,
        descriptor=descriptor(seed),
        faces=faces,
        person_ids=[person] if person else [],
        metrics={
            "sharpness_score": score,
            "subject_sharpness_score": score,
            "exposure_score": 78.0,
            "contrast_score": 76.0,
            "composition_score": 72.0,
            "technical_score": score,
            "generic_group_score": score,
            "group_ranking_score": score,
            "learned_test_score": score / 100.0,
        },
        score=score,
        issues=list(issues or []),
    )


def make_group(identifier: str, *photos: PhotoObservation) -> PhotoGroupInternal:
    for rank, photo in enumerate(photos, start=1):
        photo.group_id = identifier
        photo.rank_in_group = rank
    return PhotoGroupInternal(identifier, list(photos), 0.95, "test")


class StubModel:
    last_profile = "custom"

    def predict(self, photos, *, feature_provider=None):
        return {photo.id: photo.metrics["learned_test_score"] for photo in photos}


def test_primary_uses_fixed_budget_without_two_strict_duplicates() -> None:
    first = make_photo("duplicate-a", 1, seed=1, score=72.0)
    duplicate = make_photo("duplicate-b", 2, seed=1, score=99.0)
    duplicate.descriptor = VisualDescriptor(
        phash=first.descriptor.phash,
        dhash=first.descriptor.dhash,
        layout=first.descriptor.layout.copy(),
        color=first.descriptor.color.copy(),
        edge=first.descriptor.edge.copy(),
        semantic=first.descriptor.semantic.copy(),
    )
    others = [make_photo(f"unique-{index}", index + 2, seed=index + 2, score=70.0 + index) for index in range(1, 7)]
    group = make_group("group-00001", first, duplicate, *others)

    report = assign_semantic_ratings([group], window_minutes=15, model=StubModel())
    primary = [photo for photo in group.photos if photo.stars == 3]

    assert report.target_reduction == 0.35
    assert len(primary) == round(len(group.photos) * 0.65)
    assert len({photo.strict_duplicate_cluster_id for photo in primary}) == len(primary)
    assert not ({first.id, duplicate.id} <= {photo.id for photo in primary})
    assert report.primary_duplicate_leaks == 0
    assert report.rating_model_profile == "custom"
    assert report.rating_model_fallback_reason == ""


def test_coverage_adds_two_star_without_downgrading_primary() -> None:
    primary = make_photo("primary", 1, seed=1, score=88.0, person="人物 01")
    gap_fill = make_photo("gap-fill", 2, seed=2, score=68.0, person="人物 02")
    group = make_group("group-00001", primary, gap_fill)

    report = assign_semantic_ratings([group], window_minutes=15, model=StubModel())

    assert primary.stars == 3
    assert gap_fill.stars == 2
    assert gap_fill.rating_reason == "person_stage_gap"
    assert gap_fill.coverage_keys
    assert report.unresolved_coverage_keys == 0


def test_delivery_reserve_keeps_a_second_person_stage_photo() -> None:
    group = make_group(
        "group-00001",
        *(
            make_photo(
                f"photo-{index}",
                index,
                seed=index,
                score=82.0 - index,
                person="人物 01",
            )
            for index in range(1, 9)
        ),
    )

    report = assign_semantic_ratings([group], window_minutes=15, model=StubModel())
    reserve = [photo for photo in group.photos if photo.rating_reason == "person_stage_reserve"]

    assert report.delivery_target_reduction == 0.25
    assert report.delivery_actual_reduction == 0.25
    assert report.coverage_reserve_count == 1
    assert report.delivery_budget_shortfall == 0
    assert len(reserve) == 1
    assert reserve[0].stars == 2
    assert reserve[0].coverage_keys == [f"{reserve[0].stage_id}:人物 01"]


def test_delivery_reserve_does_not_pad_without_reliable_people() -> None:
    group = make_group(
        "group-00001",
        *(make_photo(f"photo-{index}", index, seed=index) for index in range(1, 9)),
    )

    report = assign_semantic_ratings([group], window_minutes=15, model=StubModel())

    assert report.coverage_reserve_count == 0
    assert report.delivery_budget_shortfall == 1
    assert sum(photo.stars >= 2 for photo in group.photos) == 5
    assert all(photo.rating_reason != "person_stage_reserve" for photo in group.photos)


def test_delivery_reserve_never_reintroduces_a_strict_duplicate() -> None:
    photos = [
        make_photo(
            f"photo-{index}",
            index,
            seed=index,
            score=82.0 - index,
            person="人物 01",
        )
        for index in range(1, 9)
    ]
    photos[-1].descriptor = VisualDescriptor(
        phash=photos[0].descriptor.phash,
        dhash=photos[0].descriptor.dhash,
        layout=photos[0].descriptor.layout.copy(),
        color=photos[0].descriptor.color.copy(),
        edge=photos[0].descriptor.edge.copy(),
        semantic=photos[0].descriptor.semantic.copy(),
    )
    group = make_group("group-00001", *photos)

    assign_semantic_ratings([group], window_minutes=15, model=StubModel())
    delivered = [photo for photo in group.photos if photo.stars >= 2]

    assert len({photo.strict_duplicate_cluster_id for photo in delivered}) == len(delivered)


def test_delivery_reserve_prefers_clean_candidate_over_soft_issue() -> None:
    photos = [
        make_photo(
            f"photo-{index}",
            index,
            seed=index,
            score=82.0 - index,
            person="人物 01",
        )
        for index in range(1, 9)
    ]
    photos[5].issues = ["曝光偏差明显"]
    group = make_group("group-00001", *photos)

    assign_semantic_ratings([group], window_minutes=15, model=StubModel())
    reserve = [photo for photo in group.photos if photo.rating_reason == "person_stage_reserve"]

    assert [photo.id for photo in reserve] == ["photo-7"]


def test_value_layer_keeps_at_most_one_non_duplicate_backup_per_group() -> None:
    group = make_group(
        "group-00001",
        *(make_photo(f"photo-{index}", index, seed=index, score=60.0 + index) for index in range(1, 6)),
    )

    assign_semantic_ratings([group], window_minutes=15, model=StubModel())

    assert sum(photo.stars == 1 for photo in group.photos) <= 1


def test_unreadable_photo_is_always_zero() -> None:
    unreadable = make_photo(
        "unreadable",
        1,
        seed=1,
        score=0.0,
        issues=["文件读取失败：损坏"],
        decodable=False,
    )
    group = make_group("group-00001", unreadable)

    assign_semantic_ratings([group], window_minutes=15, model=StubModel())

    assert unreadable.stars == 0
    assert unreadable.rating_reason == "technical_reject"


def test_model_failure_uses_stable_fallback_instead_of_aborting() -> None:
    group = make_group(
        "group-00001",
        make_photo("first", 1, seed=1, score=80.0),
        make_photo("second", 2, seed=2, score=70.0),
    )

    class FailingModel:
        def predict(self, photos, *, feature_provider=None):
            raise RatingModelError("fixture failure")

    report = assign_semantic_ratings([group], window_minutes=15, model=FailingModel())

    assert report.primary_count == 1
    assert sum(photo.stars == 3 for photo in group.photos) == 1
    assert report.rating_model_profile == "stable_fallback"
    assert "fixture failure" in report.rating_model_fallback_reason


@pytest.mark.parametrize("issue", ["主体清晰度不足", "主要人物闭眼", "曝光偏差明显"])
def test_decodable_severe_photo_never_becomes_primary(issue) -> None:
    photo = make_photo("bad", 1, seed=1, score=90, issues=[issue])
    report = assign_semantic_ratings([make_group("g", photo)], window_minutes=15)
    assert photo.stars == 0
    assert photo.rating_reason == "technical_reject"
    assert photo.needs_review
    assert report.primary_count == 0


def test_all_low_quality_group_has_no_primary_even_on_model_fallback() -> None:
    class FailingModel:
        def predict(self, photos):
            raise RatingModelError("unavailable")

    photos = [make_photo(str(i), i, seed=i, score=5) for i in range(1, 5)]
    report = assign_semantic_ratings([make_group("g", *photos)], window_minutes=15, model=FailingModel())
    assert all(photo.stars == 0 for photo in photos)
    assert all(photo.rating_reason == "technical_reject" for photo in photos)
    assert report.primary_count == 0
    assert report.primary_budget_shortfall > 0


def test_primary_budget_does_not_reintroduce_technical_rejects() -> None:
    photos = [make_photo(str(i), i, seed=i, score=90) for i in range(1, 9)]
    for photo in photos[2:]:
        photo.issues = ["主体清晰度不足"]
    report = assign_semantic_ratings([make_group("g", *photos)], window_minutes=15, model=StubModel())
    assert sum(photo.stars == 3 for photo in photos) == 2
    assert all(photo.stars == 0 for photo in photos[2:])
    assert report.primary_budget_shortfall == 3


def test_only_available_bad_face_photo_is_flagged_coverage_not_primary() -> None:
    photo = make_photo("only", 1, seed=1, score=5, person="人物 01", issues=["主体清晰度不足"])
    report = assign_semantic_ratings([make_group("g", photo)], window_minutes=15, model=StubModel())
    assert photo.stars == 2
    assert photo.rating_reason == "person_stage_gap"
    assert photo.needs_review
    assert report.unresolved_coverage_keys == 0


def test_uncertain_warning_is_reviewable_instead_of_hard_rejection() -> None:
    photo = make_photo("profile", 1, seed=1, score=80, issues=["主要人物姿态异常"])
    assign_semantic_ratings([make_group("g", photo)], window_minutes=15, model=StubModel())
    assert photo.stars == 3
    assert photo.needs_review


def test_ordinary_reserve_cannot_pad_with_bad_photos() -> None:
    photos = [make_photo(str(i), i, seed=i, score=80, person="人物 01") for i in range(1, 9)]
    for photo in photos[5:]:
        photo.issues = ["主体清晰度不足"]
    report = assign_semantic_ratings([make_group("g", *photos)], window_minutes=15, model=StubModel())
    assert all(photo.stars == 0 for photo in photos[5:])
    assert report.coverage_reserve_count == 0


def lock(photo, stars):
    from photocull.rating_types import RatingOrigin, RatingReason, TIER_BY_STAR, apply_rating
    apply_rating(photo, tier=TIER_BY_STAR[stars], origin=RatingOrigin.MANUAL,
                 reason=RatingReason.MANUAL_OVERRIDE, locked=True)


def test_locked_zero_does_not_consume_primary_seed_or_coverage_slot() -> None:
    rejected = make_photo("manual-reject", 1, seed=1, score=99, person="人物 01")
    alternate = make_photo("alternate", 2, seed=2, score=70, person="人物 01")
    lock(rejected, 0)
    report = assign_semantic_ratings([make_group("g", rejected, alternate)], window_minutes=15, model=StubModel())
    assert rejected.stars == 0 and rejected.rating_locked
    assert alternate.stars == 3
    assert report.unresolved_coverage_keys == 0


def test_locked_zero_only_coverage_candidate_reports_gap() -> None:
    photo = make_photo("manual-reject", 1, seed=1, score=99, person="人物 01")
    lock(photo, 0)
    report = assign_semantic_ratings([make_group("g", photo)], window_minutes=15, model=StubModel())
    assert photo.stars == 0 and photo.rating_locked
    assert report.required_coverage_keys == 1
    assert report.unresolved_coverage_keys == 1


def test_locked_primary_wins_global_duplicate_conflict(monkeypatch) -> None:
    from photocull.near_duplicates import DuplicateLayers
    first = make_photo("first", 1, seed=1, score=90)
    manual = make_photo("manual", 2, seed=1, score=5, issues=["主体清晰度不足"])
    lock(manual, 3)
    monkeypatch.setattr("photocull.rating_policy.build_duplicate_layers", lambda _groups: DuplicateLayers(
        strict_cluster_by_photo={"first": "same", "manual": "same"},
        beat_by_photo={"first": "a", "manual": "b"},
    ))
    report = assign_semantic_ratings([make_group("a", first), make_group("b", manual)],
                                    window_minutes=15, model=StubModel())
    assert manual.stars == 3 and manual.rating_locked
    assert first.stars == 0
    assert report.primary_duplicate_leaks == 0


def test_global_duplicate_seeds_choose_better_candidate(monkeypatch) -> None:
    from photocull.near_duplicates import DuplicateLayers
    weak = make_photo("weak", 1, seed=1, score=60)
    best = make_photo("best", 2, seed=1, score=95)
    monkeypatch.setattr("photocull.rating_policy.build_duplicate_layers", lambda _groups: DuplicateLayers(
        strict_cluster_by_photo={"weak": "same", "best": "same"},
        beat_by_photo={"weak": "a", "best": "b"},
    ))
    report = assign_semantic_ratings([make_group("a", weak), make_group("b", best)],
                                    window_minutes=15, model=StubModel())
    assert best.stars == 3 and weak.stars == 0
    assert report.primary_duplicate_leaks == 0


def test_persisted_coverage_evidence_includes_primary_and_rejected_photos() -> None:
    photos = [make_photo(str(i), i, seed=i, score=80, person="人物 01") for i in range(1, 9)]
    lock(photos[-1], 0)
    assign_semantic_ratings([make_group("g", *photos)], window_minutes=15, model=StubModel())
    assert photos[0].stars == 3
    assert photos[-1].stars == 0
    assert all(photo.coverage_keys == [f"{photo.stage_id}:人物 01"] for photo in photos)
    assert photos[0].coverage_person_ids == []
    assert photos[-1].coverage_person_ids == []
