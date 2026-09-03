from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

import photocull.grouping as grouping
from photocull.grouping import compare_photos, consolidate_split_groups, group_similar_photos
from photocull.imaging import file_sequence
from photocull.internal_models import (
    BodyObservation,
    DepthObservation,
    FaceObservation,
    PhotoGroupInternal,
    PhotoObservation,
    PoseObservation,
    VisualDescriptor,
)
from photocull.scoring import rank_groups


def normalized(seed: int, length: int = 32) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=length).astype(np.float32)
    return vector / np.linalg.norm(vector)


def photo(
    identifier: str,
    sequence: int,
    scene_seed: int,
    person: str | None,
    score: float = 80.0,
    eye_state: str = "Open",
    semantic_seed: int | None = None,
) -> PhotoObservation:
    layout = normalized(scene_seed)
    color = normalized(scene_seed + 100)
    color = np.abs(color) / np.linalg.norm(np.abs(color))
    edge = normalized(scene_seed + 200)
    semantic = normalized(semantic_seed if semantic_seed is not None else scene_seed + 300)
    descriptor = VisualDescriptor(
        phash=(0xAA55AA55AA55AA55 ^ (sequence & 0b111)) if scene_seed == 1 else 0x0F0F0F0F0F0F0F0F,
        layout=layout,
        color=color,
        edge=edge,
        semantic=semantic,
    )
    faces = []
    if person:
        faces = [
            FaceObservation(
                face_id=f"{identifier}-face",
                bbox=(0.2, 0.2, 0.6, 0.8),
                confidence=0.98,
                area_ratio=0.16,
                embedding=None,
                person_id=person,
                eye_state=eye_state,
                open_probability=0.95 if eye_state == "Open" else 0.08,
                sharpness=180.0,
            )
        ]
    return PhotoObservation(
        id=identifier,
        path=Path(f"{identifier}.jpg"),
        source_root=Path("."),
        filename=f"IMG_{sequence:04d}.jpg",
        relative_path=f"IMG_{sequence:04d}.jpg",
        width=1200,
        height=800,
        capture_time=datetime(2026, 1, 1, 12, 0, 0) + timedelta(seconds=sequence),
        file_sequence=sequence,
        descriptor=descriptor,
        faces=faces,
        person_ids=[person] if person else [],
        metrics={
            "sharpness": 180.0,
            "sharpness_score": 82.0,
            "face_sharpness_score": 84.0,
            "eye_score": 95.0 if eye_state == "Open" else 10.0,
            "exposure_score": 88.0,
            "composition_score": 80.0,
            "contrast_score": 78.0,
            "brightness": 0.48,
            "highlight_clip": 0.0,
            "shadow_clip": 0.0,
        },
        score=score,
        issues=["主要人物闭眼"] if eye_state == "Closed" else [],
    )


def test_same_scene_same_person_forms_one_group_and_removes_duplicates() -> None:
    photos = [photo(f"frame-{index}", index, 1, "人物 01", score=80 + index) for index in range(1, 7)]

    groups = group_similar_photos(photos, "balanced")
    rank_groups(groups, keep_per_group=1)

    assert len(groups) == 1
    assert sum(item.category == "selected" for item in photos) == 1
    assert sum(item.category == "duplicate" for item in photos) == 5


def test_identical_scene_with_different_people_is_blocked() -> None:
    left = photo("alice", 1, 1, "人物 01")
    right = photo("bob", 2, 1, "人物 02")

    evidence = compare_photos(left, right)
    groups = group_similar_photos([left, right], "aggressive")

    assert evidence.compatible_people is False
    assert len(groups) == 2


def test_body_reid_adds_soft_people_evidence_when_faces_are_missing() -> None:
    left = photo("body-left", 1, 1, None)
    right = photo("body-right", 2, 1, None)
    shared_embedding = normalized(700, 512)
    left.bodies = [BodyObservation((0.2, 0.1, 0.7, 0.95), 0.95, 0.42, shared_embedding, "test")]
    right.bodies = [BodyObservation((0.22, 0.1, 0.72, 0.95), 0.93, 0.42, shared_embedding, "test")]

    evidence = compare_photos(left, right)

    assert evidence.body is not None and evidence.body > 0.95
    assert evidence.people > 0.85
    assert evidence.compatible_people is True


def test_body_reid_never_overrides_reliable_face_conflict() -> None:
    left = photo("body-alice", 1, 1, "人物 01")
    right = photo("body-bob", 2, 1, "人物 02")
    shared_embedding = normalized(701, 512)
    left.bodies = [BodyObservation((0.2, 0.1, 0.7, 0.95), 0.95, 0.42, shared_embedding, "test")]
    right.bodies = [BodyObservation((0.2, 0.1, 0.7, 0.95), 0.95, 0.42, shared_embedding, "test")]

    evidence = compare_photos(left, right)

    assert evidence.body is not None and evidence.body > 0.95
    assert evidence.compatible_people is False
    assert evidence.people == 0.0
    assert evidence.total <= 0.40


def test_3d_pose_is_reported_as_action_evidence_without_becoming_identity() -> None:
    left = photo("pose-left", 1, 1, None)
    right = photo("pose-right", 2, 1, None)
    shared_pose = normalized(740, 66)
    landmarks = np.tile(np.asarray((0.5, 0.5, 0.9), dtype=np.float32), (33, 1))
    left.poses = [PoseObservation((0.2, 0.1, 0.8, 0.95), 0.95, 0.99, 0.51, landmarks, shared_pose, 0.90, "test")]
    right.poses = [PoseObservation((0.21, 0.1, 0.81, 0.95), 0.94, 0.98, 0.51, landmarks, shared_pose, 0.91, "test")]

    evidence = compare_photos(left, right)

    assert evidence.pose is not None and evidence.pose > 0.99
    assert evidence.body is None
    assert evidence.people == 0.58


def test_depth_foreground_filter_ignores_matching_stage_poster_pose() -> None:
    left = photo("pose-real-left", 1, 1, None)
    right = photo("pose-real-right", 2, 1, None)
    landmarks = np.tile(np.asarray((0.5, 0.5, 0.9), dtype=np.float32), (33, 1))
    left_action = np.eye(1, 66, 0, dtype=np.float32).reshape(-1)
    right_action = np.eye(1, 66, 1, dtype=np.float32).reshape(-1)
    poster = np.eye(1, 66, 2, dtype=np.float32).reshape(-1)
    left.poses = [
        PoseObservation((0.35, 0.1, 0.72, 0.95), 0.96, 0.99, 0.31, landmarks, left_action, 0.91, "test", 0.92),
        PoseObservation((0.05, 0.1, 0.32, 0.90), 0.94, 0.98, 0.22, landmarks, poster, 0.90, "test", 0.25),
    ]
    right.poses = [
        PoseObservation((0.36, 0.1, 0.73, 0.95), 0.96, 0.99, 0.31, landmarks, right_action, 0.91, "test", 0.90),
        PoseObservation((0.05, 0.1, 0.32, 0.90), 0.94, 0.98, 0.22, landmarks, poster, 0.90, "test", 0.24),
    ]

    evidence = compare_photos(left, right)

    assert evidence.pose is not None
    assert evidence.pose < 0.05


def test_depth_layout_is_only_soft_scene_evidence() -> None:
    left = photo("depth-left", 1, 1, None)
    right = photo("depth-right", 2, 1, None)
    descriptor = normalized(999, 18 * 18)
    observation = DepthObservation(descriptor, 0.8, 0.2, 0.9, 82.0, 75.0, 0.0, 0.9, "test")
    left.depth = observation
    right.depth = DepthObservation(descriptor.copy(), 0.8, 0.2, 0.9, 82.0, 75.0, 0.0, 0.9, "test")

    evidence = compare_photos(left, right)

    assert evidence.depth is not None and evidence.depth > 0.999
    assert evidence.people == 0.58


def test_different_scene_with_same_person_is_not_merged() -> None:
    first = photo("ceremony", 1, 1, "人物 01")
    second = photo("banquet", 2, 99, "人物 01")

    evidence = compare_photos(first, second)
    groups = group_similar_photos([first, second], "balanced")

    assert evidence.scene < 0.75
    assert len(groups) == 2


def test_open_eyes_beat_higher_raw_score_closed_eyes() -> None:
    closed = photo("closed", 1, 1, "人物 01", score=96.0, eye_state="Closed")
    open_photo = photo("open", 2, 1, "人物 01", score=82.0, eye_state="Open")

    groups = group_similar_photos([closed, open_photo], "balanced")
    rank_groups(groups, keep_per_group=1)

    assert len(groups) == 1
    assert open_photo.is_best_pick is True
    assert closed.category == "duplicate"


def test_second_pass_merges_a_split_same_person_burst() -> None:
    first = photo("intro", 10, 1, "人物 01")
    second = photo("action", 11, 1, "人物 01")
    second.descriptor.phash ^= 0xFFFF
    groups = [
        PhotoGroupInternal("group-a", [first], 1.0, "独立画面"),
        PhotoGroupInternal("group-b", [second], 1.0, "独立画面"),
    ]

    consolidated = consolidate_split_groups(groups, "balanced")

    assert len(consolidated) == 1
    assert first.group_id == second.group_id == "group-00001"


def test_second_pass_keeps_different_people_apart() -> None:
    first = photo("alice", 10, 1, "人物 01")
    second = photo("bob", 11, 1, "人物 02")
    second.descriptor.phash ^= 0xFFFF
    groups = [
        PhotoGroupInternal("group-a", [first], 1.0, "独立画面"),
        PhotoGroupInternal("group-b", [second], 1.0, "独立画面"),
    ]

    consolidated = consolidate_split_groups(groups, "aggressive")

    assert len(consolidated) == 2


def test_second_pass_stops_after_ordered_time_window(monkeypatch) -> None:
    groups = []
    for index in range(100):
        item = photo(f"far-{index}", index * 60, index + 10, None)
        groups.append(PhotoGroupInternal(f"group-{index}", [item], 1.0, "独立画面"))

    comparisons = 0
    original_compare = grouping.compare_photos

    def counted_compare(left: PhotoObservation, right: PhotoObservation):
        nonlocal comparisons
        comparisons += 1
        return original_compare(left, right)

    monkeypatch.setattr(grouping, "compare_photos", counted_compare)

    consolidated = consolidate_split_groups(groups, "balanced")

    assert len(consolidated) == 100
    assert comparisons == 0


def test_small_low_confidence_face_cannot_split_an_obvious_burst() -> None:
    first = photo("clear-face", 10, 1, "人物 01")
    second = photo("tiny-face", 11, 1, "人物 99")
    second.faces[0].area_ratio = 0.0012
    second.faces[0].confidence = 0.66
    second.descriptor.phash ^= 0xFFFF

    evidence = compare_photos(first, second)
    groups = [
        PhotoGroupInternal("group-a", [first], 1.0, "独立画面"),
        PhotoGroupInternal("group-b", [second], 1.0, "独立画面"),
    ]

    assert evidence.compatible_people is True
    assert len(consolidate_split_groups(groups, "balanced")) == 1


def test_tiny_high_confidence_event_face_participates_without_becoming_hard_constraint() -> None:
    event = photo("event", 12, 1, "人物 01")
    event.faces[0].area_ratio = 0.00081
    event.faces[0].confidence = 0.868

    assert event.significant_person_ids == {"人物 01"}
    assert event.reliable_person_ids == set()


def test_long_similar_frame_chain_still_has_a_bounded_burst_span() -> None:
    photos = [photo(f"moment-{index}", index, 1, "人物 01", score=80 + index / 10) for index in range(1, 42)]
    for index, item in enumerate(photos):
        item.capture_time = datetime(2026, 1, 1, 12, 0, 0) + timedelta(seconds=index * 0.62)
        angle = index / 40 * 1.2
        drifting = np.zeros(32, dtype=np.float32)
        drifting[0] = np.cos(angle)
        drifting[1] = np.sin(angle)
        item.descriptor.semantic = drifting.copy()
        item.descriptor.layout = drifting.copy()
        item.descriptor.phash = (1 << (index + 1)) - 1
        item.descriptor.dhash = (1 << (index + 1)) - 1

    groups = group_similar_photos(photos, "balanced")

    assert len(groups) >= 2
    assert max(len(group.photos) for group in groups) <= 27
    assert all(
        (max(photo.capture_time for photo in group.photos) - min(photo.capture_time for photo in group.photos)).total_seconds() <= 16.5
        for group in groups
    )


def test_panasonic_folder_prefix_change_does_not_split_a_timed_burst() -> None:
    first = photo("panasonic-a", 2697, 1, "人物 01")
    second = photo("panasonic-b", 2698, 1, "人物 01")
    first.file_sequence = file_sequence(Path("P1132697.RW2"))
    second.file_sequence = file_sequence(Path("P1142698.RW2"))
    first.capture_time = datetime(2026, 8, 23, 17, 0, 18)
    second.capture_time = datetime(2026, 8, 23, 17, 0, 21, 928000)

    groups = group_similar_photos([first, second], "balanced")

    assert first.file_sequence == 2697
    assert second.file_sequence == 2698
    assert len(groups) == 1
