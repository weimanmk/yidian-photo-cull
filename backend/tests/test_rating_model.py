from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from photocull.internal_models import PhotoObservation, VisualDescriptor
from photocull.rating_model import (
    BuiltInRatingFeatures,
    FrozenRatingModel,
    IqaRatingFeatures,
    RatingModelError,
)


def make_photos() -> list[PhotoObservation]:
    photos: list[PhotoObservation] = []
    for index, score in enumerate((62.0, 81.0), start=1):
        photos.append(
            PhotoObservation(
                id=f"photo-{index}",
                path=Path(f"photo-{index}.jpg"),
                source_root=Path("."),
                filename=f"photo-{index}.jpg",
                relative_path=f"photo-{index}.jpg",
                width=1600,
                height=1000,
                capture_time=datetime(2026, 8, 30, 9, 0) + timedelta(seconds=index),
                file_sequence=index,
                descriptor=VisualDescriptor(
                    phash=index,
                    layout=np.ones(4, dtype=np.float32),
                    color=np.ones(4, dtype=np.float32),
                    edge=np.ones(4, dtype=np.float32),
                ),
                metrics={
                    "sharpness_score": score,
                    "motion_blur_score": score - 2.0,
                    "exposure_score": 76.0,
                    "contrast_score": 72.0,
                    "noise_score": 80.0,
                    "composition_score": 68.0 + index,
                    "technical_score": score,
                    "generic_group_score": score,
                    "group_relative_score": 40.0 + index * 20.0,
                    "group_quality_percentile": float(index - 1),
                    "group_size": 2.0,
                    "subject_sharpness_score": score,
                },
                score=score,
                group_id="group-00001",
                rank_in_group=index,
            )
        )
    return photos


def test_frozen_model_asset_has_training_hashes_and_two_profiles() -> None:
    model = FrozenRatingModel.load_default()

    assert model.version == "rating-pointwise-v1"
    assert set(model.profiles) == {"base", "iqa"}
    assert set(model.training_hashes) == {"活动A", "活动B", "活动C"}
    assert model.selection_parameters == {
        "target_reduction": 0.35,
        "learned_alpha": 0.75,
        "group_demote": 0.55,
        "duplicate_demote": 0.10,
        "duplicate_scene_floor": 0.90,
        "duplicate_pose_floor": 0.90,
        "duplicate_max_sequence_span": 3,
        "duplicate_max_time_span_seconds": 4.0,
    }


def test_missing_iqa_uses_base_profile_without_changing_result_shape() -> None:
    photos = make_photos()
    model = FrozenRatingModel.load_default()

    scores = model.predict(photos, feature_provider=IqaRatingFeatures())

    assert set(scores) == {photo.id for photo in photos}
    assert all(0.0 <= value <= 1.0 for value in scores.values())
    assert model.last_profile == "base"


def test_complete_finite_iqa_metrics_select_iqa_profile() -> None:
    photos = make_photos()
    for index, photo in enumerate(photos):
        photo.metrics["musiq_score"] = 65.0 + index * 10.0
        photo.metrics["qualiclip_score"] = 0.55 + index * 0.12
    model = FrozenRatingModel.load_default()

    scores = model.predict(photos, feature_provider=IqaRatingFeatures())

    assert len(scores) == 2
    assert model.last_profile == "iqa"


def test_external_iqa_mapping_selects_iqa_profile_without_mutating_photos() -> None:
    photos = make_photos()
    provider = IqaRatingFeatures(
        scores_by_photo={
            "photo-1": {"musiq": 68.0, "qualiclip+": 0.61},
            "photo-2": {"musiq": 76.0, "qualiclip+": 0.72},
        }
    )
    model = FrozenRatingModel.load_default()

    scores = model.predict(photos, feature_provider=provider)

    assert set(scores) == {"photo-1", "photo-2"}
    assert model.last_profile == "iqa"
    assert all("musiq_score" not in photo.metrics for photo in photos)


def test_non_finite_iqa_metric_falls_back_to_base() -> None:
    photos = make_photos()
    for photo in photos:
        photo.metrics["musiq_score"] = 70.0
        photo.metrics["qualiclip_score"] = 0.62
    photos[1].metrics["musiq_score"] = float("nan")
    model = FrozenRatingModel.load_default()

    model.predict(photos, feature_provider=IqaRatingFeatures())

    assert model.last_profile == "base"


def test_provider_dimension_mismatch_raises_typed_error() -> None:
    photos = make_photos()
    model = FrozenRatingModel.load_default()

    class WrongWidthProvider:
        def matrix(self, members, profiles):
            return np.zeros((len(members), 1), dtype=np.float64), "base"

    with pytest.raises(RatingModelError, match="特征维度"):
        model.predict(photos, feature_provider=WrongWidthProvider())


def test_base_prediction_is_deterministic() -> None:
    photos = make_photos()
    model = FrozenRatingModel.load_default()

    first = model.predict(photos, feature_provider=BuiltInRatingFeatures())
    second = model.predict(photos, feature_provider=BuiltInRatingFeatures())

    assert first == second
