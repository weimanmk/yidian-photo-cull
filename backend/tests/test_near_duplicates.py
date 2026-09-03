from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from photocull.internal_models import (
    FaceObservation,
    PhotoGroupInternal,
    PhotoObservation,
    SimilarityEvidence,
    VisualDescriptor,
)
from photocull.near_duplicates import build_duplicate_layers


def make_photo(identifier: str, sequence: int, person: str = "人物 01") -> PhotoObservation:
    face = FaceObservation(
        face_id=f"{identifier}-face",
        bbox=(0.2, 0.2, 0.7, 0.9),
        confidence=0.98,
        area_ratio=0.18,
        embedding=None,
        person_id=person,
    )
    return PhotoObservation(
        id=identifier,
        path=Path(f"{identifier}.jpg"),
        source_root=Path("."),
        filename=f"IMG_{sequence:04d}.jpg",
        relative_path=f"IMG_{sequence:04d}.jpg",
        width=1600,
        height=1000,
        capture_time=datetime(2026, 8, 30, 9, 0) + timedelta(seconds=sequence),
        file_sequence=sequence,
        descriptor=VisualDescriptor(
            phash=1,
            dhash=1,
            layout=np.ones(4, dtype=np.float32),
            color=np.ones(4, dtype=np.float32),
            edge=np.ones(4, dtype=np.float32),
        ),
        faces=[face],
        person_ids=[person],
    )


def make_group(*photos: PhotoObservation) -> PhotoGroupInternal:
    return PhotoGroupInternal(
        id="group-00001",
        photos=list(photos),
        confidence=0.95,
        reason="test",
    )


def evidence(*, scene: float = 0.95, pose: float | None = 0.95, compatible: bool = True,
             strong: bool = False) -> SimilarityEvidence:
    return SimilarityEvidence(
        total=0.95,
        scene=scene,
        semantic=0.95,
        phash=0.95,
        dhash=0.95,
        layout=0.95,
        color=0.95,
        edge=0.95,
        composition=0.95,
        people=0.95 if compatible else 0.0,
        body=None,
        pose=pose,
        depth=None,
        temporal=0.95,
        compatible_people=compatible,
        strong_duplicate=strong,
    )


def test_strong_duplicates_share_one_strict_cluster(monkeypatch) -> None:
    group = make_group(make_photo("a", 1), make_photo("b", 2), make_photo("c", 3))
    monkeypatch.setattr(
        "photocull.near_duplicates.compare_photos",
        lambda _left, _right: evidence(strong=True),
    )

    layers = build_duplicate_layers([group])

    cluster_ids = {layers.strict_cluster_by_photo[photo.id] for photo in group.photos}
    assert cluster_ids == {"group-00001:strict-0001"}


def test_same_moment_requires_endpoint_coherence(monkeypatch) -> None:
    group = make_group(make_photo("a", 1), make_photo("b", 2), make_photo("c", 3))

    def chain_only(left: PhotoObservation, right: PhotoObservation) -> SimilarityEvidence:
        pair = frozenset((left.id, right.id))
        return evidence(scene=0.95 if pair in {frozenset(("a", "b")), frozenset(("b", "c"))} else 0.50)

    monkeypatch.setattr("photocull.near_duplicates.compare_photos", chain_only)

    layers = build_duplicate_layers([group])

    assert layers.beat_by_photo["a"] == layers.beat_by_photo["b"]
    assert layers.beat_by_photo["c"] != layers.beat_by_photo["a"]


def test_different_reliable_people_never_share_a_cluster_or_beat() -> None:
    group = make_group(
        make_photo("person-a", 1, "人物 01"),
        make_photo("person-b", 2, "人物 02"),
    )

    layers = build_duplicate_layers([group])

    assert layers.strict_cluster_by_photo["person-a"] != layers.strict_cluster_by_photo["person-b"]
    assert layers.beat_by_photo["person-a"] != layers.beat_by_photo["person-b"]


def test_layer_ids_follow_capture_order_not_input_order(monkeypatch) -> None:
    first = make_photo("first", 1)
    second = make_photo("second", 2)
    third = make_photo("third", 3)
    monkeypatch.setattr(
        "photocull.near_duplicates.compare_photos",
        lambda _left, _right: evidence(scene=0.40),
    )

    layers = build_duplicate_layers([make_group(third, first, second)])

    assert layers.strict_cluster_by_photo == {
        "first": "group-00001:strict-0001",
        "second": "group-00001:strict-0002",
        "third": "group-00001:strict-0003",
    }
    assert layers.beat_by_photo == {
        "first": "group-00001:beat-0001",
        "second": "group-00001:beat-0002",
        "third": "group-00001:beat-0003",
    }
