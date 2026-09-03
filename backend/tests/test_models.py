from pathlib import Path

import numpy as np
from PIL import Image

from photocull.face_engine import ARCFACE_TEMPLATE, FaceEngine
from photocull.scene_engine import SceneEmbeddingEngine


class _FakeInput:
    name = "input.1"


class _FakeEyeSession:
    def __init__(self, output: tuple[float, float]) -> None:
        self.output = np.asarray(output, dtype=np.float32).reshape(1, 2, 1, 1)
        self.tensors: list[np.ndarray] = []

    def get_inputs(self) -> list[_FakeInput]:
        return [_FakeInput()]

    def run(self, _outputs: object, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.tensors.append(inputs["input.1"])
        return [self.output]


class _FakeExpressionSession:
    def __init__(self, output: list[float]) -> None:
        self.output = np.asarray(output, dtype=np.float32).reshape(1, 7)
        self.tensor: np.ndarray | None = None

    def get_inputs(self) -> list[_FakeInput]:
        return [_FakeInput()]

    def run(self, _outputs: object, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.tensor = inputs["input.1"]
        return [self.output]


class _FakeLandmarkSession:
    def __init__(self, dimensions: int) -> None:
        self.output = np.zeros((1, dimensions), dtype=np.float32)
        self.tensor: np.ndarray | None = None

    def get_inputs(self) -> list[_FakeInput]:
        return [_FakeInput()]

    def run(self, _outputs: object, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.tensor = inputs["input.1"]
        return [self.output]


class _FakeFiqaSession:
    def __init__(self, score: float) -> None:
        self.output = np.asarray([[score]], dtype=np.float32)
        self.tensor: np.ndarray | None = None

    def get_inputs(self) -> list[_FakeInput]:
        return [_FakeInput()]

    def run(self, _outputs: object, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.tensor = inputs["input.1"]
        return [self.output]


def test_scene_model_produces_finite_embedding_when_installed() -> None:
    engine = SceneEmbeddingEngine(use_gpu=False)
    assert engine.available
    image = np.zeros((320, 480, 3), dtype=np.uint8)
    image[:, :, 1] = np.linspace(0, 255, 480, dtype=np.uint8)
    embedding = engine.embed(image)
    assert embedding is not None
    status = engine.status()
    expected_dimensions = 384 if status["model"] == "dinov2-small" else 1000
    assert embedding.shape == (expected_dimensions,)
    assert status["embedding_dimensions"] == expected_dimensions
    assert np.isfinite(embedding).all()
    assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-4)


def test_face_models_load_and_blank_image_has_no_faces() -> None:
    engine = FaceEngine(use_gpu=False)
    assert engine.available
    assert engine.status()["eye_model"]["available"]
    assert engine.eye_model_name == "openvino-open-closed-eye-0001"
    assert engine.status()["landmark_3d_model"]["available"]
    assert engine.status()["face_quality_model"]["available"]
    observations = engine.analyze(np.full((640, 640, 3), 127, dtype=np.uint8), "blank")
    assert observations == []


def test_original_face_alignment_maps_analysis_landmarks_to_full_resolution() -> None:
    original = np.zeros((224, 224, 3), dtype=np.uint8)
    nose_x, nose_y = np.rint(ARCFACE_TEMPLATE[2] * 2.0).astype(int)
    original[nose_y - 5 : nose_y + 6, nose_x - 5 : nose_x + 6, 0] = 255

    aligned = FaceEngine._align_original_face(
        Image.fromarray(original),
        ARCFACE_TEMPLATE.copy(),
        analysis_width=112,
        analysis_height=112,
    )

    assert aligned is not None and aligned.shape == (112, 112, 3)
    target_x, target_y = np.rint(ARCFACE_TEMPLATE[2]).astype(int)
    assert aligned[target_y, target_x, 0] >= 240


def test_insightface_landmark_model_receives_official_rgb_0_to_255_input() -> None:
    engine = FaceEngine(use_gpu=False)
    session = _FakeLandmarkSession(212)
    engine._landmark_session = session
    image = np.empty((240, 240, 3), dtype=np.uint8)
    image[:, :, 0] = 12
    image[:, :, 1] = 80
    image[:, :, 2] = 240

    points = engine._landmarks106(image, (40, 40, 200, 200))

    assert points is not None and points.shape == (106, 2)
    assert session.tensor is not None and session.tensor.shape == (1, 3, 192, 192)
    assert session.tensor.dtype == np.float32
    assert np.isclose(session.tensor[0, 0, 96, 96], 12.0)
    assert np.isclose(session.tensor[0, 2, 96, 96], 240.0)


def test_openvino_eye_model_uses_bgr_32px_and_open_class_zero() -> None:
    engine = FaceEngine(use_gpu=False)
    session = _FakeEyeSession((0.82, 0.18))
    engine.eye_model_name = "openvino-open-closed-eye-0001"
    engine._eye_session = session
    aligned = np.empty((112, 112, 3), dtype=np.uint8)
    aligned[:, :, 0] = 10
    aligned[:, :, 1] = 20
    aligned[:, :, 2] = 30

    state, open_probability = engine._eye_state(aligned, None)

    assert state == "Open"
    assert np.isclose(open_probability, 0.82)
    assert len(session.tensors) == 2
    assert session.tensors[0].shape == (1, 3, 32, 32)
    assert np.isclose(session.tensors[0][0, 0, 0, 0], (30.0 - 127.0) / 255.0)


def test_openvino_eye_model_marks_low_open_probability_closed() -> None:
    engine = FaceEngine(use_gpu=False)
    engine.eye_model_name = "openvino-open-closed-eye-0001"
    engine._eye_session = _FakeEyeSession((0.08, 0.92))

    state, open_probability = engine._eye_state(np.full((112, 112, 3), 127, dtype=np.uint8), None)

    assert state == "Closed"
    assert np.isclose(open_probability, 0.08)


def test_packaged_openvino_eye_model_runs_with_onnx_runtime() -> None:
    model_path = Path(__file__).resolve().parents[2] / "models" / "open-closed-eye.onnx"
    assert model_path.is_file()
    engine = FaceEngine(use_gpu=False)
    engine.eye_path = model_path
    engine.eye_model_name = "openvino-open-closed-eye-0001"
    engine._eye_session = engine._session(model_path)

    state, open_probability = engine._eye_state(np.full((112, 112, 3), 127, dtype=np.uint8), None)

    assert state in {"Open", "Partial", "Closed"}
    assert open_probability is not None
    assert 0.0 <= open_probability <= 1.0


def test_expression_model_uses_rgb_112px_and_maps_happy_class() -> None:
    engine = FaceEngine(use_gpu=False)
    session = _FakeExpressionSession([0.01, 0.01, 0.01, 0.92, 0.02, 0.01, 0.02])
    engine._expression_session = session
    aligned = np.empty((112, 112, 3), dtype=np.uint8)
    aligned[:, :, 0] = 255
    aligned[:, :, 1] = 128
    aligned[:, :, 2] = 0

    label, confidence, score, happy_probability = engine._expression(aligned)

    assert label == "happy"
    assert confidence is not None and confidence > 0.90
    assert happy_probability is not None and happy_probability > 0.90
    assert score > 90.0
    assert session.tensor is not None
    assert session.tensor.shape == (1, 3, 112, 112)
    assert np.isclose(session.tensor[0, 0, 0, 0], 1.0)
    assert np.isclose(session.tensor[0, 2, 0, 0], -1.0)


def test_packaged_expression_model_runs_with_onnx_runtime() -> None:
    model_path = Path(__file__).resolve().parents[2] / "models" / "facial_expression_mobilefacenet.onnx"
    assert model_path.is_file()
    engine = FaceEngine(use_gpu=False)
    engine._expression_session = engine._session(model_path)

    label, confidence, score, happy_probability = engine._expression(
        np.full((112, 112, 3), 127, dtype=np.uint8)
    )

    assert label in {"angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised"}
    assert confidence is not None and 0.0 <= confidence <= 1.0
    assert 0.0 <= score <= 100.0
    assert happy_probability is not None and 0.0 <= happy_probability <= 1.0


def test_ediffiqa_uses_aligned_rgb_and_returns_percent_score() -> None:
    engine = FaceEngine(use_gpu=False)
    session = _FakeFiqaSession(0.73)
    engine._fiqa_session = session
    aligned = np.empty((112, 112, 3), dtype=np.uint8)
    aligned[:, :, 0] = 255
    aligned[:, :, 1] = 128
    aligned[:, :, 2] = 0

    score = engine._fiqa(aligned)

    assert score is not None and np.isclose(score, 73.0)
    assert session.tensor is not None and session.tensor.shape == (1, 3, 112, 112)
    assert np.isclose(session.tensor[0, 0, 0, 0], 1.0)
    assert np.isclose(session.tensor[0, 2, 0, 0], -1.0)


def test_packaged_3d_landmark_and_fiqa_models_run_with_onnx_runtime() -> None:
    engine = FaceEngine(use_gpu=False)
    assert engine._ensure_loaded()
    image = np.full((240, 240, 3), 127, dtype=np.uint8)

    pose = engine._pose3d(image, (40, 40, 200, 200))
    score = engine._fiqa(np.full((112, 112, 3), 127, dtype=np.uint8))

    assert pose is not None and len(pose) == 3 and np.isfinite(pose).all()
    assert all(-90.0 <= value <= 90.0 for value in pose)
    assert score is not None and 0.0 <= score <= 100.0
