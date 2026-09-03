from pathlib import Path

import numpy as np

from photocull.internal_models import PhotoGroupInternal, PhotoObservation, VisualDescriptor
from photocull.preference import PreferenceModel, fit_pairwise_preference_model, fit_preference_model
from photocull.scoring import rank_groups


def make_photo(index: int, preferred: bool, preference_score: float | None = None) -> PhotoObservation:
    good = 86.0 + index % 5 if preferred else 42.0 + index % 7
    metrics = {
        "sharpness": good * 2.0,
        "sharpness_score": good,
        "motion_blur_score": good,
        "exposure_score": 92.0 if preferred else 55.0,
        "contrast_score": 84.0 if preferred else 49.0,
        "noise_score": 90.0 if preferred else 52.0,
        "composition_score": 88.0 if preferred else 48.0,
        "face_quality_score": 0.0,
        "min_face_score": 0.0,
        "face_sharpness_score": good,
        "eye_sharpness_score": 0.0,
        "eye_score": 72.0,
        "expression_score": 60.0,
        "technical_score": good,
        "bad_face_count": 0.0,
    }
    if preference_score is not None:
        metrics["preference_score"] = preference_score
        metrics["preference_threshold"] = 50.0
        metrics["preference_strength"] = 1.0
        metrics["preference_selection_enabled"] = 1.0
    zeros = np.zeros(8, dtype=np.float32)
    return PhotoObservation(
        id=f"photo-{index}",
        path=Path(f"IMG_{index:04d}.jpg"),
        source_root=Path("."),
        filename=f"IMG_{index:04d}.jpg",
        relative_path=f"IMG_{index:04d}.jpg",
        width=1200,
        height=800,
        capture_time=None,
        file_sequence=index,
        descriptor=VisualDescriptor(phash=index, layout=zeros, color=zeros, edge=zeros),
        metrics=metrics,
        score=good,
    )


def test_preference_model_learns_and_round_trips(tmp_path: Path) -> None:
    photos = [make_photo(index, preferred=index % 3 == 0) for index in range(90)]
    labels = np.asarray([index % 3 == 0 for index in range(90)], dtype=np.int8)

    model = fit_preference_model(photos, labels)
    probabilities = model.predict_probabilities(photos)

    assert float(np.mean(probabilities[labels == 1])) > float(np.mean(probabilities[labels == 0])) + 0.6
    destination = model.save(tmp_path / "preference.json")
    restored = PreferenceModel.load(destination)
    assert restored is not None
    assert np.allclose(restored.predict_probabilities(photos), probabilities)


def test_preference_model_defaults_to_its_training_source(tmp_path: Path) -> None:
    photos = [make_photo(index, preferred=index % 3 == 0) for index in range(90)]
    labels = np.asarray([index % 3 == 0 for index in range(90)], dtype=np.int8)
    model = fit_preference_model(photos, labels)
    trained_source = tmp_path / "event-a"
    model.metadata["source_dir"] = str(trained_source)

    assert model.applies_to_source(trained_source)
    assert not model.applies_to_source(tmp_path / "event-b")

    model.metadata["scope"] = "global"
    assert model.applies_to_source(tmp_path / "event-b")


def test_pairwise_preference_learns_group_winners() -> None:
    photos: list[PhotoObservation] = []
    labels: list[int] = []
    for group_index in range(16):
        for offset, preferred in enumerate((True, False, False)):
            photo = make_photo(group_index * 3 + offset, preferred=preferred)
            photo.group_id = f"group-{group_index}"
            photos.append(photo)
            labels.append(int(preferred))

    model = fit_pairwise_preference_model(photos, np.asarray(labels, dtype=np.int8))
    probabilities = model.predict_probabilities(photos)

    assert model.metadata["objective"] == "pairwise"
    assert all(
        probabilities[index] > max(probabilities[index + 1], probabilities[index + 2])
        for index in range(0, len(photos), 3)
    )


def test_preference_breaks_soft_quality_tie_without_changing_grouping() -> None:
    preferred = make_photo(1, preferred=True, preference_score=91.0)
    alternative = make_photo(2, preferred=True, preference_score=18.0)
    preferred.score = alternative.score = 82.0
    for key in set(preferred.metrics) - {"preference_score", "preference_threshold"}:
        alternative.metrics[key] = preferred.metrics[key]
    group = PhotoGroupInternal("group-1", [alternative, preferred], 0.9, "测试相似组")

    rank_groups([group], keep_per_group=1)

    assert group.photos[0] is preferred
    assert preferred.category == "selected"
    assert alternative.category == "duplicate"


def test_low_preference_singleton_is_not_exported_as_selected() -> None:
    photo = make_photo(1, preferred=True, preference_score=22.0)
    group = PhotoGroupInternal("group-1", [photo], 1.0, "独立照片")

    rank_groups([group], keep_per_group=1)

    assert photo.is_best_pick is True
    assert photo.category == "rejected"
    assert any("偏好阈值" in reason for reason in photo.selection_reasons)
