from pathlib import Path

import numpy as np

from photocull.identity import IdentityClusterer
from photocull.internal_models import FaceObservation, PhotoObservation, VisualDescriptor


def unit(*values: float) -> np.ndarray:
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def face(identifier: str, embedding: np.ndarray) -> FaceObservation:
    return FaceObservation(
        face_id=identifier,
        bbox=(0.2, 0.2, 0.5, 0.7),
        confidence=0.95,
        area_ratio=0.12,
        embedding=embedding,
    )


def photo(identifier: str, faces: list[FaceObservation]) -> PhotoObservation:
    descriptor = VisualDescriptor(
        phash=0,
        layout=np.ones(4, dtype=np.float16),
        color=np.ones(4, dtype=np.float16),
        edge=np.ones(4, dtype=np.float16),
    )
    return PhotoObservation(
        id=identifier,
        path=Path(f"{identifier}.jpg"),
        source_root=Path("."),
        filename=f"{identifier}.jpg",
        relative_path=f"{identifier}.jpg",
        width=100,
        height=100,
        capture_time=None,
        file_sequence=1,
        descriptor=descriptor,
        faces=faces,
    )


def test_same_identity_is_stable_across_photos() -> None:
    alice = unit(1.0, 0.0, 0.0, 0.0)
    alice_variant = unit(0.97, 0.08, 0.0, 0.0)
    bob = unit(0.0, 1.0, 0.0, 0.0)
    photos = [
        photo("a", [face("a1", alice), face("b1", bob)]),
        photo("b", [face("a2", alice_variant)]),
    ]

    clusters = IdentityClusterer(threshold=0.42).assign(photos)

    assert photos[0].faces[0].person_id == photos[1].faces[0].person_id
    assert photos[0].faces[1].person_id != photos[0].faces[0].person_id
    assert len(clusters) == 2


def test_two_faces_in_one_photo_never_share_identity() -> None:
    nearly_equal_left = unit(1.0, 0.01, 0.0, 0.0)
    nearly_equal_right = unit(0.99, 0.02, 0.0, 0.0)
    current = photo("group", [face("left", nearly_equal_left), face("right", nearly_equal_right)])

    IdentityClusterer(threshold=0.32).assign([current])

    assert current.faces[0].person_id != current.faces[1].person_id
