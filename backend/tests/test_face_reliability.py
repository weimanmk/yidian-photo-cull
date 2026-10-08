import numpy as np
from PIL import Image

from photocull.face_engine import FaceEngine
from photocull.face_quality import select_quality_faces
from photocull.internal_models import FaceObservation


def detections(boxes):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    points = np.array([[[x+8, y+8], [x+22, y+8], [x+15, y+15], [x+9, y+23], [x+21, y+23]]
                       for x, y, _, _ in boxes], dtype=np.float32).reshape(-1, 5, 2)
    return boxes, points, np.full(len(boxes), 0.95, dtype=np.float32)


def engine_without_models(monkeypatch):
    engine = FaceEngine(use_gpu=False)
    monkeypatch.setattr(engine, "_ensure_loaded", lambda: True)
    monkeypatch.setattr(engine, "_embedding", lambda rgb: np.array([1., 0.]))
    return engine


def test_analyze_preserves_more_than_24_valid_faces(monkeypatch):
    engine = engine_without_models(monkeypatch)
    boxes = [[10 + col*70, 10+row*70, 40+col*70, 40+row*70] for row in range(5) for col in range(6)]
    monkeypatch.setattr(engine, "_detect", lambda *args, **kwargs: detections(boxes))
    faces = engine.analyze(np.zeros((480, 640, 3), np.uint8), "crowd")
    assert len(faces) == 30
    assert len({face.face_id for face in faces}) == 30


def test_quality_selection_preserves_all_relevant_faces_and_filters_background():
    subjects = [FaceObservation(str(i), (0.1, 0.1, 0.2, 0.2), .95, .006, None) for i in range(30)]
    background = FaceObservation("background", (0, 0, .01, .01), .95, .0006, None)
    assert select_quality_faces(subjects + [background]) == subjects
    assert select_quality_faces([background, background]) == []


def test_retry_finds_small_faces_and_maps_original_coordinates_without_duplicates(monkeypatch):
    engine = engine_without_models(monkeypatch)
    def detect(rgb, input_size=(640, 640)):
        if input_size == (640, 640):
            return detections([[80, 80, 110, 110]])
        height, width = rgb.shape[:2]
        scale = np.array([width/800, height/400, width/800, height/400])
        boxes, points, scores = detections([[80, 80, 110, 110], [300, 100, 330, 130]])
        return boxes * scale, points * scale[:2], scores
    monkeypatch.setattr(engine, "_detect", detect)
    faces = engine.analyze(np.zeros((400, 800, 3), np.uint8), "retry", Image.new("RGB", (3200, 1600)))
    assert len(faces) == 2
    assert sorted(tuple(round(v, 4) for v in f.bbox) for f in faces) == [
        (.1, .2, .1375, .275), (.375, .25, .4125, .325)]
    diagnostics = engine.status()["detection"]
    assert diagnostics["retry_attempted"] is True
    assert diagnostics["returned_faces"] == 2


def test_failed_retry_keeps_first_pass_faces_and_reports_failure(monkeypatch):
    engine = engine_without_models(monkeypatch)
    def detect(rgb, input_size=(640, 640)):
        if input_size != (640, 640):
            raise RuntimeError("fixed input shape")
        return detections([[80, 80, 110, 110]])
    monkeypatch.setattr(engine, "_detect", detect)
    faces = engine.analyze(np.zeros((800, 1600, 3), np.uint8), "retry")
    assert len(faces) == 1
    assert "fixed input shape" in engine.status()["detection"]["retry_error"]
