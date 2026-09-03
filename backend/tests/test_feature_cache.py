from datetime import datetime
from pathlib import Path

import numpy as np

from photocull.feature_cache import FeatureCache
from photocull.imaging import photo_id
from photocull.internal_models import (
    BodyObservation,
    DepthObservation,
    FaceObservation,
    PhotoObservation,
    PoseObservation,
    VisualDescriptor,
)


def test_feature_cache_round_trip_preserves_embeddings_and_face_quality(tmp_path: Path) -> None:
    source = tmp_path / "IMG_0001.jpg"
    source.write_bytes(b"cache-key-fixture")
    embedding = np.linspace(-1.0, 1.0, 16, dtype=np.float32)
    embedding /= np.linalg.norm(embedding)
    photo = PhotoObservation(
        id=photo_id(source),
        path=source,
        source_root=tmp_path,
        filename=source.name,
        relative_path=source.name,
        width=6000,
        height=4000,
        capture_time=datetime(2026, 8, 23, 13, 18, 0, 441000),
        file_sequence=1,
        descriptor=VisualDescriptor(
            phash=0xAA55AA55AA55AA55,
            dhash=0xF0F0F0F0F0F0F0F0,
            layout=np.arange(8, dtype=np.float16),
            color=np.arange(6, dtype=np.float16),
            edge=np.arange(4, dtype=np.float16),
            semantic=embedding,
        ),
        faces=[
            FaceObservation(
                face_id="face-1",
                bbox=(0.2, 0.1, 0.5, 0.7),
                confidence=0.97,
                area_ratio=0.18,
                embedding=embedding,
                eye_state="Open",
                open_probability=0.94,
                sharpness=312.0,
                high_res_sharpness=312.0,
                eye_sharpness=221.0,
                yaw=4.0,
                pitch=-2.0,
                roll=1.0,
                occlusion_risk=0.04,
                expression="happy",
                expression_confidence=0.91,
                expression_score=94.0,
                fiqa_score=82.5,
            )
        ],
        bodies=[
            BodyObservation(
                bbox=(0.1, 0.05, 0.7, 0.95),
                confidence=0.93,
                area_ratio=0.54,
                embedding=embedding,
                detector="yolov8n-person",
            )
        ],
        poses=[
            PoseObservation(
                bbox=(0.08, 0.03, 0.72, 0.98),
                detection_confidence=0.96,
                presence_confidence=0.99,
                area_ratio=0.61,
                landmarks_2d=np.tile(np.asarray((0.5, 0.5, 0.9), dtype=np.float32), (33, 1)),
                descriptor=embedding,
                visibility=0.91,
                model="mediapipe-blazepose-ghum-heavy",
                foreground_score=0.93,
            )
        ],
        depth=DepthObservation(
            descriptor=embedding,
            subject_depth=0.78,
            background_depth=0.31,
            foreground_separation=0.88,
            subject_focus_score=81.5,
            background_blur_score=74.0,
            occlusion_risk=0.04,
            subject_confidence=0.93,
            model="depth-anything-v2-large",
        ),
        metrics={"face_quality_score": 88.0, "bad_face_count": 0.0},
        score=86.5,
        issues=[],
    )
    cache = FeatureCache(tmp_path / "cache.db")

    cache.save(photo, "pipeline-v1")
    restored = cache.load(source, tmp_path, "pipeline-v1")

    assert restored is not None
    assert restored.capture_time == photo.capture_time
    assert restored.descriptor.phash == photo.descriptor.phash
    assert restored.descriptor.dhash == photo.descriptor.dhash
    assert np.allclose(restored.descriptor.semantic, embedding, atol=1e-3)
    assert np.allclose(restored.faces[0].embedding, embedding, atol=1e-3)
    assert restored.faces[0].eye_sharpness == 221.0
    assert restored.faces[0].expression == "happy"
    assert restored.faces[0].expression_confidence == 0.91
    assert restored.faces[0].expression_score == 94.0
    assert restored.faces[0].fiqa_score == 82.5
    assert len(restored.bodies) == 1
    assert np.allclose(restored.bodies[0].embedding, embedding, atol=1e-3)
    assert restored.bodies[0].detector == "yolov8n-person"
    assert len(restored.poses) == 1
    assert restored.poses[0].landmarks_2d.shape == (33, 3)
    assert np.allclose(restored.poses[0].descriptor, embedding, atol=1e-3)
    assert restored.poses[0].visibility == 0.91
    assert restored.poses[0].foreground_score == 0.93
    assert restored.depth is not None
    assert np.allclose(restored.depth.descriptor, embedding, atol=1e-3)
    assert restored.depth.subject_focus_score == 81.5
    assert restored.depth.foreground_separation == 0.88
    assert cache.load(source, tmp_path, "pipeline-v2") is None
    assert cache.dominant_pipeline_signature(tmp_path) == "pipeline-v1"

    source.write_bytes(b"source-has-changed")
    assert cache.load(source, tmp_path, "pipeline-v1") is None
