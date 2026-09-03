from pathlib import Path

import numpy as np

from photocull.depth_engine import DepthEngine, _depth_descriptor, _foreground_score


def test_foreground_score_separates_person_from_flat_backdrop() -> None:
    depth = np.full((518, 518), 0.18, dtype=np.float32)
    depth[120:450, 225:340] = 0.82
    person_points = np.asarray(((0.50, 0.30), (0.48, 0.50), (0.52, 0.72)), dtype=np.float32)
    poster_points = np.asarray(((0.16, 0.30), (0.18, 0.50), (0.20, 0.72)), dtype=np.float32)

    person = _foreground_score(depth, person_points)
    poster = _foreground_score(depth, poster_points)

    assert person is not None and poster is not None
    assert person[0] > poster[0] + 0.50
    assert person[1] > poster[1] + 0.20


def test_depth_descriptor_is_normalized_and_keeps_layout() -> None:
    depth = np.tile(np.linspace(0.0, 1.0, 518, dtype=np.float32), (518, 1))

    descriptor = _depth_descriptor(depth)

    assert descriptor is not None
    assert descriptor.shape == (18 * 18,)
    assert np.isclose(np.linalg.norm(descriptor), 1.0, atol=1e-5)
    assert float(descriptor[0]) < float(descriptor[-1])


def test_packaged_depth_model_has_expected_local_runtime_contract() -> None:
    model_path = Path(__file__).resolve().parents[2] / "models" / "depth_anything_v2_vitl.onnx"
    assert model_path.is_file()
    assert model_path.stat().st_size > 1_300_000_000
    engine = DepthEngine(use_gpu=False)
    engine.model_path = model_path

    status = engine.status()

    assert status["available"] is True
    assert status["relative_depth"] is True
    assert status["metric_depth"] is False
    assert status["local_only"] is True
