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


def test_identical_descriptors_across_distant_groups_share_global_cluster() -> None:
    first = make_photo('original', 1)
    copy = make_photo('copy', 10000)
    groups = [make_group(first), make_group(copy)]
    groups[1].id = 'distant-group'
    layers = build_duplicate_layers(groups)
    assert layers.strict_cluster_by_photo['original'] == layers.strict_cluster_by_photo['copy']
    assert layers.beat_by_photo['original'] != layers.beat_by_photo['copy']


def test_strict_clusters_do_not_merge_transitive_hash_chain() -> None:
    a, b, c = [make_photo(name, index) for index, name in enumerate('abc')]
    for photo, bits in [(a, 0), (b, 15), (c, 255)]:
        photo.descriptor.phash = bits
        photo.descriptor.dhash = bits
    layers = build_duplicate_layers([make_group(a, b, c)])
    assert layers.strict_cluster_by_photo['a'] == layers.strict_cluster_by_photo['b']
    assert layers.strict_cluster_by_photo['a'] != layers.strict_cluster_by_photo['c']


def test_unknown_person_cannot_bridge_conflicting_people() -> None:
    a, b, c = make_photo('a', 1, 'one'), make_photo('b', 2), make_photo('c', 3, 'two')
    b.faces = []
    b.person_ids = []
    layers = build_duplicate_layers([make_group(a, b, c)])
    assert layers.strict_cluster_by_photo['a'] != layers.strict_cluster_by_photo['c']


def test_similar_stage_at_different_moments_is_not_strict_duplicate() -> None:
    a, b = make_photo('a', 1), make_photo('b', 30)
    b.descriptor.phash = 3  # Very similar scene, but not the identical descriptor.
    layers = build_duplicate_layers([make_group(a, b)])
    assert layers.strict_cluster_by_photo['a'] != layers.strict_cluster_by_photo['b']


def test_visible_eye_change_is_a_distinct_moment_even_with_matching_scene() -> None:
    a, b = make_photo('a', 1), make_photo('b', 2)
    a.faces[0].eye_state = 'Open'
    b.faces[0].eye_state = 'Closed'
    layers = build_duplicate_layers([make_group(a, b)])
    assert layers.strict_cluster_by_photo['a'] != layers.strict_cluster_by_photo['b']


def test_near_duplicates_cross_group_boundary_without_exact_hash_match() -> None:
    a, b = make_photo('a', 1), make_photo('b', 2)
    b.descriptor.phash = 3
    groups = [make_group(a), make_group(b)]
    groups[1].id = 'second'
    layers = build_duplicate_layers(groups)
    assert layers.strict_cluster_by_photo['a'] == layers.strict_cluster_by_photo['b']
    assert layers == build_duplicate_layers(list(reversed(groups)))


def test_distant_distinct_descriptors_do_not_require_all_pairs(monkeypatch) -> None:
    import photocull.near_duplicates as duplicates
    real_compare = duplicates.compare_photos
    comparisons = []

    def count(left, right):
        comparisons.append((left.id, right.id))
        return real_compare(left, right)

    monkeypatch.setattr(duplicates, 'compare_photos', count)
    photos = [make_photo(str(index), index * 100) for index in range(100)]
    for index, photo in enumerate(photos):
        photo.descriptor.phash = index
    groups = [make_group(photo) for photo in photos]
    for index, group in enumerate(groups):
        group.id = str(index)
    layers = build_duplicate_layers(groups)
    assert len(set(layers.strict_cluster_by_photo.values())) == 100
    assert len(comparisons) < 200
