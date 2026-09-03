from __future__ import annotations

import math

import pytest

from photocull.eye_evidence import (
    DEFAULT_EYE_EVIDENCE_PROFILE_NAME,
    EYE_EVIDENCE_PROFILES,
    active_eye_evidence_profile,
    evaluate_eye_evidence,
)
from photocull.internal_models import FaceObservation


def face(**changes: object) -> FaceObservation:
    values: dict[str, object] = {
        "face_id": "face-1",
        "bbox": (0.35, 0.25, 0.65, 0.75),
        "confidence": 0.95,
        "area_ratio": 0.010,
        "embedding": None,
        "eye_state": "Closed",
        "open_probability": 0.08,
        "sharpness": 190.0,
        "eye_sharpness": 180.0,
        "yaw": 0.0,
        "pitch": 0.0,
        "roll": 0.0,
        "occlusion_risk": 0.05,
    }
    values.update(changes)
    return FaceObservation(**values)  # type: ignore[arg-type]


def test_clear_frontal_closed_eye_is_decisive() -> None:
    evidence = evaluate_eye_evidence(face())

    assert evidence.decisive_closed is True
    assert evidence.uncertain is False
    assert evidence.open_score < 50.0
    assert evidence.reliability > 0.5


def test_clear_frontal_open_eye_is_reliable_and_not_uncertain() -> None:
    evidence = evaluate_eye_evidence(face(eye_state="Open", open_probability=0.96))

    assert evidence.decisive_closed is False
    assert evidence.uncertain is False
    assert evidence.open_score > 90.0


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"area_ratio": 0.0010}, "face_too_small"),
        ({"confidence": 0.71}, "low_detection_confidence"),
        ({"yaw": 44.0}, "side_pose"),
        ({"pitch": 22.0}, "pitch_out_of_range"),
        ({"roll": 44.0}, "roll_out_of_range"),
        ({"eye_sharpness": 18.0}, "eyes_not_sharp"),
        ({"occlusion_risk": 0.72}, "occluded"),
        ({"open_probability": None}, "missing_open_probability"),
    ],
)
def test_weak_closed_eye_evidence_never_hard_rejects(
    changes: dict[str, object],
    reason: str,
) -> None:
    evidence = evaluate_eye_evidence(face(**changes))

    assert evidence.decisive_closed is False
    assert evidence.uncertain is True
    assert reason in evidence.reasons


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("area_ratio", math.nan),
        ("confidence", math.inf),
        ("yaw", math.nan),
        ("pitch", -math.inf),
        ("roll", math.inf),
        ("eye_sharpness", math.nan),
        ("occlusion_risk", math.inf),
        ("open_probability", math.nan),
    ],
)
def test_non_finite_input_safely_disables_hard_eye_decision(field: str, invalid_value: float) -> None:
    evidence = evaluate_eye_evidence(face(**{field: invalid_value}))

    assert evidence.decisive_closed is False
    assert evidence.uncertain is True
    assert evidence.reliability == 0.0
    assert math.isfinite(evidence.open_score)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("area_ratio", -0.01),
        ("confidence", 1.01),
        ("yaw", 91.0),
        ("pitch", -91.0),
        ("roll", 91.0),
        ("eye_sharpness", -1.0),
        ("occlusion_risk", -0.01),
        ("open_probability", 1.01),
    ],
)
def test_out_of_range_input_safely_disables_hard_eye_decision(
    field: str,
    invalid_value: float,
) -> None:
    evidence = evaluate_eye_evidence(face(**{field: invalid_value}))

    assert evidence.decisive_closed is False
    assert evidence.uncertain is True
    assert evidence.reliability == 0.0
    assert f"invalid_{field}" in evidence.reasons


def test_profiles_are_limited_to_the_three_predeclared_threshold_sets() -> None:
    thresholds = {
        name: (profile.min_area_ratio, profile.min_confidence)
        for name, profile in EYE_EVIDENCE_PROFILES.items()
    }

    assert thresholds == {
        "default": (0.0030, 0.82),
        "wide-hard": (0.0025, 0.80),
        "conservative-hard": (0.0040, 0.86),
    }


def test_active_profile_uses_environment_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHOTOCULL_EYE_EVIDENCE_PROFILE", "conservative-hard")

    assert active_eye_evidence_profile().name == "conservative-hard"


def test_unknown_environment_profile_falls_back_to_frozen_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHOTOCULL_EYE_EVIDENCE_PROFILE", "not-a-profile")

    assert active_eye_evidence_profile().name == DEFAULT_EYE_EVIDENCE_PROFILE_NAME


def test_unknown_eye_state_is_neutral_without_hard_decision() -> None:
    evidence = evaluate_eye_evidence(face(eye_state="Unknown", open_probability=None))

    assert evidence.open_score == 72.0
    assert evidence.reliability == 0.0
    assert evidence.decisive_closed is False
    assert evidence.uncertain is False
