from pathlib import Path

import numpy as np
from PIL import Image

from photocull.internal_models import PoseObservation
from photocull.pose_engine import (
    LANDMARK_INPUT_SIZE,
    PoseEngine,
    _PoseRoi,
    _deduplicate_poses,
    _pose_descriptor,
    _refine_landmarks_from_heatmap,
)


def _skeleton() -> np.ndarray:
    landmarks = np.zeros((33, 3), dtype=np.float32)
    landmarks[11] = (-0.28, 0.55, 0.02)
    landmarks[12] = (0.28, 0.55, -0.02)
    landmarks[13] = (-0.48, 0.28, 0.10)
    landmarks[14] = (0.42, 0.22, -0.08)
    landmarks[15] = (-0.62, 0.05, 0.18)
    landmarks[16] = (0.58, 0.02, -0.12)
    landmarks[17:23] = landmarks[[15, 16, 15, 16, 15, 16]]
    landmarks[23] = (-0.18, 0.0, 0.0)
    landmarks[24] = (0.18, 0.0, 0.0)
    landmarks[25] = (-0.24, -0.50, 0.12)
    landmarks[26] = (0.30, -0.42, -0.05)
    landmarks[27] = (-0.32, -0.95, 0.18)
    landmarks[28] = (0.42, -0.82, -0.08)
    landmarks[29] = (-0.34, -1.00, 0.10)
    landmarks[30] = (0.44, -0.88, -0.12)
    landmarks[31] = (-0.28, -1.08, 0.28)
    landmarks[32] = (0.52, -0.96, 0.08)
    return landmarks


def test_pose_descriptor_is_invariant_to_global_similarity_transform() -> None:
    original = _skeleton()
    angle = np.deg2rad(37.0)
    rotation = np.asarray(
        (
            (np.cos(angle), -np.sin(angle), 0.0),
            (np.sin(angle), np.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float32,
    )
    transformed = original @ rotation.T * 2.7 + np.asarray((1.2, -0.4, 3.0), dtype=np.float32)

    left = _pose_descriptor(original)
    right = _pose_descriptor(transformed)

    assert left is not None and right is not None
    assert np.dot(left, right) > 0.999


def test_heatmap_refinement_moves_landmark_toward_local_peak() -> None:
    landmarks = np.zeros((39, 3), dtype=np.float32)
    landmarks[:, :2] = 0.5
    heatmap = np.full((1, 64, 64, 39), -12.0, dtype=np.float32)
    heatmap[0, 34, 30, 0] = 12.0

    refined = _refine_landmarks_from_heatmap(landmarks, heatmap)

    assert np.allclose(refined[0, :2], (30 / 64, 34 / 64), atol=1e-3)
    assert np.allclose(refined[1, :2], (0.5, 0.5), atol=1e-6)


def test_packaged_heavy_pose_model_loads_without_tasks_sdk() -> None:
    model_path = Path(__file__).resolve().parents[2] / "models" / "pose_landmarker_heavy.task"
    assert model_path.is_file()
    engine = PoseEngine()
    engine.model_path = model_path

    status = engine.status()
    observations = engine.analyze(np.full((480, 640, 3), 127, dtype=np.uint8))

    assert status["available"] is True
    assert status["telemetry"] is False
    assert observations == []


def test_duplicate_pose_rois_collapse_to_one_skeleton() -> None:
    descriptor = _pose_descriptor(_skeleton())
    assert descriptor is not None
    landmarks = np.tile(np.asarray((0.5, 0.5, 0.92), dtype=np.float32), (33, 1))
    loose = PoseObservation((0.10, 0.05, 0.90, 0.98), 0.92, 0.98, 0.74, landmarks, descriptor, 0.91, "test")
    tight = PoseObservation((0.20, 0.10, 0.80, 0.95), 0.96, 0.99, 0.51, landmarks, descriptor, 0.96, "test")

    deduplicated = _deduplicate_poses([loose, tight])

    assert len(deduplicated) == 1
    assert deduplicated[0] is tight


def test_pose_crop_clips_extreme_roi_before_pillow_allocation() -> None:
    source = Image.new("RGB", (64, 48), "white")
    false_positive_roi = _PoseRoi(
        center_x=32.0,
        center_y=24.0,
        side_pixels=100_000.0,
        rotation=0.35,
        detection_confidence=0.9,
    )

    crop = PoseEngine._extract_crop(source, false_positive_roi)

    assert crop.shape == (LANDMARK_INPUT_SIZE, LANDMARK_INPUT_SIZE, 3)
    assert crop.dtype == np.uint8


def test_successful_pose_analysis_clears_previous_runtime_error() -> None:
    engine = PoseEngine()
    engine._error = "stale error"
    engine._ensure_loaded = lambda: True  # type: ignore[method-assign]
    engine._detect_rois = lambda rgb: []  # type: ignore[method-assign]

    assert engine.analyze(np.zeros((48, 64, 3), dtype=np.uint8)) == []
    assert engine._error == ""
