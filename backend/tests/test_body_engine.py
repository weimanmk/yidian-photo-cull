from pathlib import Path

import numpy as np

from photocull.body_engine import BodyEngine, _mediapipe_anchors


class _FakeInput:
    name = "images"


class _FakeYoloSession:
    def get_inputs(self) -> list[_FakeInput]:
        return [_FakeInput()]

    def run(self, _outputs: object, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        assert inputs["images"].shape == (1, 3, 640, 640)
        output = np.zeros((1, 84, 8400), dtype=np.float32)
        output[0, 0:4, 0] = (320.0, 320.0, 200.0, 300.0)
        output[0, 4, 0] = 0.93
        return [output]


def test_mediapipe_anchor_layout_has_expected_count() -> None:
    anchors = _mediapipe_anchors()

    assert anchors.shape == (2254, 2)
    assert np.allclose(anchors[0], (0.5 / 28.0, 0.5 / 28.0))
    assert np.all((anchors > 0.0) & (anchors < 1.0))


def test_yolo_person_parser_maps_letterbox_back_to_image() -> None:
    engine = BodyEngine(use_gpu=False)
    engine.detector_name = "yolov8n-person"
    engine._detector_session = _FakeYoloSession()
    engine._reid_session = None
    engine._load_attempted = True

    observations = engine.analyze(np.zeros((480, 640, 3), dtype=np.uint8))

    assert len(observations) == 1
    observation = observations[0]
    assert observation.detector == "yolov8n-person"
    assert np.isclose(observation.confidence, 0.93)
    assert np.allclose(observation.bbox, (220 / 640, 90 / 480, 420 / 640, 390 / 480), atol=1e-5)


def test_packaged_yolo_model_loads_and_blank_image_has_no_people() -> None:
    model_path = Path(__file__).resolve().parents[2] / "models" / "yolov8n.onnx"
    assert model_path.is_file()
    engine = BodyEngine(use_gpu=False)
    engine.detector_candidates = [("yolov8n-person", model_path)]
    engine.detector_name = "yolov8n-person"
    engine.detector_path = model_path

    observations = engine.analyze(np.full((640, 640, 3), 127, dtype=np.uint8))

    assert observations == []
    assert engine.status()["errors"] == []


def test_packaged_osnet_model_produces_unit_embedding() -> None:
    model_path = Path(__file__).resolve().parents[2] / "models" / "osnet_x0_25_msmt17.onnx"
    assert model_path.is_file()
    engine = BodyEngine(use_gpu=False)
    engine._reid_session = engine.runtime.create_session(model_path)
    image = np.zeros((300, 160, 3), dtype=np.uint8)
    image[:, :, 0] = np.linspace(0, 255, 160, dtype=np.uint8)

    embeddings = engine._reid_embeddings(image, [(0, 0, 160, 300)])

    assert len(embeddings) == 1
    assert embeddings[0] is not None
    assert embeddings[0].shape == (512,)
    assert np.isfinite(embeddings[0]).all()
    assert np.isclose(np.linalg.norm(embeddings[0]), 1.0, atol=1e-4)
