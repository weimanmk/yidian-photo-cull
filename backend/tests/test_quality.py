import numpy as np

from photocull.internal_models import FaceObservation
from photocull.quality import analyze_quality


def test_closed_eye_penalty_is_visible_in_score_and_issues() -> None:
    rng = np.random.default_rng(4)
    image = rng.integers(25, 225, size=(500, 700, 3), dtype=np.uint8)
    open_face = FaceObservation(
        face_id="open", bbox=(0.2, 0.2, 0.6, 0.8), confidence=0.95, area_ratio=0.18,
        embedding=None, eye_state="Open", open_probability=0.94, sharpness=190.0,
    )
    closed_face = FaceObservation(
        face_id="closed", bbox=(0.2, 0.2, 0.6, 0.8), confidence=0.95, area_ratio=0.18,
        embedding=None, eye_state="Closed", open_probability=0.08, sharpness=190.0,
    )

    _, open_score, open_issues = analyze_quality(image, [open_face])
    _, closed_score, closed_issues = analyze_quality(image, [closed_face])

    assert "主要人物闭眼" not in open_issues
    assert "主要人物闭眼" in closed_issues
    assert closed_score < open_score - 20


def test_one_bad_face_penalizes_a_group_photo() -> None:
    rng = np.random.default_rng(9)
    image = rng.integers(35, 220, size=(600, 900, 3), dtype=np.uint8)
    good = FaceObservation(
        face_id="good", bbox=(0.1, 0.2, 0.35, 0.75), confidence=0.98, area_ratio=0.12,
        embedding=None, eye_state="Open", open_probability=0.96, sharpness=320.0, eye_sharpness=260.0,
    )
    weak = FaceObservation(
        face_id="weak", bbox=(0.55, 0.2, 0.8, 0.75), confidence=0.96, area_ratio=0.12,
        embedding=None, eye_state="Closed", open_probability=0.04, sharpness=18.0, eye_sharpness=9.0,
    )

    good_metrics, good_score, _ = analyze_quality(image, [good])
    group_metrics, group_score, group_issues = analyze_quality(image, [good, weak])

    assert group_metrics["min_face_score"] < good_metrics["min_face_score"]
    assert group_metrics["bad_face_count"] == 1
    assert "主要人物闭眼" in group_issues
    assert group_score < good_score - 20


def test_tiny_event_face_eye_state_is_not_a_hard_reject() -> None:
    rng = np.random.default_rng(11)
    image = rng.integers(35, 220, size=(600, 900, 3), dtype=np.uint8)
    tiny = FaceObservation(
        face_id="tiny", bbox=(0.7, 0.3, 0.74, 0.38), confidence=0.87, area_ratio=0.00081,
        embedding=None, eye_state="Closed", open_probability=0.05, sharpness=110.0, eye_sharpness=92.0,
    )

    metrics, _, issues = analyze_quality(image, [tiny])

    assert "主要人物闭眼" not in issues
    assert metrics["eye_score"] > 50.0


def test_tiny_crowd_faces_do_not_trigger_worst_face_penalty() -> None:
    rng = np.random.default_rng(15)
    image = rng.integers(35, 220, size=(600, 900, 3), dtype=np.uint8)
    crowd = [
        FaceObservation(
            face_id=f"crowd-{index}", bbox=(0.05 * index, 0.3, 0.05 * index + 0.03, 0.36),
            confidence=0.80, area_ratio=0.0015, embedding=None, eye_state="Closed",
            open_probability=0.05, sharpness=40.0, eye_sharpness=30.0,
        )
        for index in range(8)
    ]

    metrics, _, issues = analyze_quality(image, crowd)

    assert metrics["min_face_score"] == 0.0
    assert metrics["bad_face_count"] == 0.0
    assert "主要人物闭眼" not in issues


def test_looking_down_is_not_treated_as_a_blink() -> None:
    rng = np.random.default_rng(21)
    image = rng.integers(35, 220, size=(600, 900, 3), dtype=np.uint8)
    looking_down = FaceObservation(
        face_id="down", bbox=(0.25, 0.2, 0.55, 0.8), confidence=0.94, area_ratio=0.08,
        embedding=None, eye_state="Closed", open_probability=0.02, sharpness=210.0,
        eye_sharpness=175.0, pitch=22.0,
    )

    metrics, _, issues = analyze_quality(image, [looking_down])

    assert metrics["eye_score"] >= 55.0
    assert "主要人物闭眼" not in issues
