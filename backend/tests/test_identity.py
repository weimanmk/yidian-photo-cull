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
        sharpness=120.0,
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


def test_identity_labels_are_stable_when_photo_and_face_order_changes() -> None:
    from copy import deepcopy

    originals = [photo("a", [face("a1", unit(1, 0)), face("b1", unit(0, 1))]),
                 photo("b", [face("a2", unit(1, 0)), face("b2", unit(0, 1))])]
    reordered = deepcopy(list(reversed(originals)))
    for item in reordered:
        item.faces.reverse()
    IdentityClusterer().assign(originals)
    IdentityClusterer().assign(reordered)
    assert {f.face_id: f.person_id for p in originals for f in p.faces} == {
        f.face_id: f.person_id for p in reordered for f in p.faces}


def test_ambiguous_face_remains_unknown_even_at_high_similarity() -> None:
    anchors = photo("anchors", [face("alice", unit(1, 0)), face("bob", unit(0, 1))])
    uncertain = photo("uncertain", [face("mixed", unit(1, 1))])
    clusters = IdentityClusterer().assign([anchors, uncertain])
    assert uncertain.faces[0].person_id is None
    assert len(clusters) == 2


def test_weak_face_cannot_seed_or_move_identity_centroid() -> None:
    weak = face("weak", unit(0.8, 0.6))
    weak.confidence = 0.72
    weak.sharpness = 25.0
    strong = face("strong", unit(1, 0))
    strong.sharpness = 150.0
    photos = [photo("a-weak", [weak]), photo("z-strong", [strong])]
    clusters = IdentityClusterer().assign(photos)
    assert len(clusters) == 1
    assert weak.person_id == strong.person_id
    np.testing.assert_allclose(clusters[0].centroid, unit(1, 0))


def test_only_weak_or_invalid_faces_remain_unknown_and_clear_stale_ids() -> None:
    weak = face("weak", unit(1, 0))
    weak.confidence = 0.50
    invalid = face("invalid", np.array([np.nan, 1]))
    for item in (weak, invalid):
        item.person_id = "stale"
    current = photo("photo", [weak, invalid])
    clusters = IdentityClusterer().assign([current])
    assert clusters == []
    assert current.person_ids == []
    assert all(item.person_id is None for item in current.faces)


def test_weak_faces_respect_same_photo_constraint_without_new_identities() -> None:
    seed = face("seed", unit(1, 0))
    seed.sharpness = 120
    first = face("first", unit(1, 0))
    second = face("second", unit(1, 0))
    first.confidence = second.confidence = 0.72
    first.sharpness = second.sharpness = 25
    group = photo("group", [first, second])
    clusters = IdentityClusterer().assign([group, photo("seed", [seed])])
    assert len(clusters) == 1
    assert sum(item.person_id is not None for item in group.faces) == 1


def test_zero_sharpness_face_cannot_seed_an_identity() -> None:
    blurred = face("blurred", unit(1, 0))
    blurred.sharpness = 0.0
    assert IdentityClusterer().assign([photo("blurred", [blurred])]) == []
    assert blurred.person_id is None


def test_poor_pose_size_occlusion_and_fiqa_cannot_contaminate_centroids() -> None:
    from copy import deepcopy

    anchor = face("anchor", unit(1, 0))
    observations = []
    for key, value in (("yaw", 60), ("area_ratio", .0002), ("occlusion_risk", .8), ("fiqa_score", 10)):
        weak = deepcopy(anchor)
        weak.face_id = key
        weak.embedding = unit(.8, .6)
        setattr(weak, key, value)
        observations.append(photo(key, [weak]))
    clusters = IdentityClusterer().assign(observations + [photo("anchor", [anchor])])
    assert len(clusters) == 1
    np.testing.assert_allclose(clusters[0].centroid, unit(1, 0))


def test_zero_embeddings_and_missing_embeddings_clear_stale_identity() -> None:
    invalid = [face("zero", np.zeros(2)), face("missing", None)]
    for item in invalid:
        item.person_id = "stale"
    item = photo("invalid", invalid)
    assert IdentityClusterer().assign([item]) == []
    assert item.person_ids == []
    assert all(f.person_id is None for f in invalid)
