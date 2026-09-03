from __future__ import annotations

from collections import Counter

import numpy as np

from .face_quality import select_quality_faces
from .internal_models import PhotoGroupInternal, PhotoObservation


def _dominant_people(group: PhotoGroupInternal) -> set[str]:
    counts = Counter(person for photo in group.photos for person in photo.significant_person_ids)
    return {person for person, count in counts.items() if count >= max(2, round(len(group.photos) * 0.45))}


def _generic_group_score(photo: PhotoObservation, expected_people: set[str]) -> float:
    metrics = photo.metrics
    subject_sharpness = metrics.get(
        "subject_sharpness_score",
        metrics.get("motion_blur_score", metrics.get("sharpness_score", 0.0)),
    )
    has_faces = bool(select_quality_faces(photo.faces))
    coverage = len(photo.significant_person_ids & expected_people) / len(expected_people) if expected_people else 1.0
    if has_faces:
        score = (
            0.25 * metrics.get("face_quality_score", 0.0)
            + 0.15 * metrics.get("eye_score", 0.0)
            + 0.15 * subject_sharpness
            + 0.10 * metrics.get("exposure_score", 0.0)
            + 0.10 * metrics.get("expression_score", 0.0)
            + 0.10 * (0.55 * metrics.get("composition_score", 0.0) + 0.45 * metrics.get("contrast_score", 0.0))
            + 0.10 * metrics.get("technical_score", 0.0)
            + 0.05 * metrics.get("composition_score", 0.0)
        )
        score += 7.0 * coverage
        score -= 8.0 * metrics.get("bad_face_count", 0.0)
    else:
        score = (
            0.45 * metrics.get("technical_score", photo.score)
            + 0.20 * subject_sharpness
            + 0.15 * metrics.get("exposure_score", 0.0)
            + 0.10 * metrics.get("composition_score", 0.0)
            + 0.10 * metrics.get("contrast_score", 0.0)
        )
    return float(np.clip(score, 0.0, 100.0))


def prepare_group_ranking_features(groups: list[PhotoGroupInternal]) -> None:
    """在不使用人工标签的前提下，为偏好模型补充组内相对特征。"""
    relative_metrics = {
        "relative_face_quality": "face_quality_score",
        "relative_eye": "eye_score",
        "relative_motion": "subject_sharpness_score",
        "relative_exposure": "exposure_score",
        "relative_composition": "composition_score",
    }
    for group in groups:
        expected_people = _dominant_people(group)
        generic_scores = [_generic_group_score(photo, expected_people) for photo in group.photos]
        best_generic = max(generic_scores, default=0.0)
        ordered_scores = sorted(generic_scores)
        maxima = {
            output_name: max((photo.metrics.get(source_name, 0.0) for photo in group.photos), default=0.0)
            for output_name, source_name in relative_metrics.items()
        }
        for photo, generic_score in zip(group.photos, generic_scores, strict=True):
            photo.metrics["generic_group_score"] = generic_score
            photo.metrics["group_relative_score"] = generic_score - best_generic
            photo.metrics["group_quality_percentile"] = (
                sum(score <= generic_score for score in ordered_scores) / max(1, len(ordered_scores)) * 100.0
            )
            photo.metrics["group_size"] = float(len(group.photos))
            for output_name, source_name in relative_metrics.items():
                photo.metrics[output_name] = photo.metrics.get(source_name, 0.0) - maxima[output_name]


def _group_score(photo: PhotoObservation, expected_people: set[str]) -> float:
    metrics = photo.metrics
    generic_score = metrics.get("generic_group_score")
    if generic_score is None:
        generic_score = _generic_group_score(photo, expected_people)
    preference_score = metrics.get("preference_score")
    if preference_score is None or not np.isfinite(preference_score):
        return generic_score
    # 偏好只修正合格候选之间的软排序，不能覆盖闭眼、缺人等硬约束。
    preference_weight = float(
        np.clip(
            metrics.get("preference_weight", 0.28 * metrics.get("preference_strength", 0.0)),
            0.0,
            0.65,
        )
    )
    return float(np.clip((1.0 - preference_weight) * generic_score + preference_weight * preference_score, 0.0, 100.0))


def _ranking_key(photo: PhotoObservation, expected_people: set[str]) -> tuple[float, ...]:
    significant_faces = select_quality_faces(photo.faces)
    eye_reliable = [face for face in significant_faces if face.area_ratio >= 0.0012 and face.confidence >= 0.58]
    closed = sum(face.eye_state == "Closed" for face in eye_reliable)
    partial = sum(face.eye_state == "Partial" for face in eye_reliable)
    coverage = len(photo.significant_person_ids & expected_people) / len(expected_people) if expected_people else 1.0
    missing_expected = int(bool(expected_people) and coverage < 0.999)
    min_open = min((face.open_probability for face in significant_faces if face.open_probability is not None), default=0.72)
    min_face_sharpness = min((face.sharpness for face in significant_faces), default=photo.metrics.get("sharpness", 0.0))
    severe_issue_count = sum(
        issue in photo.issues
        for issue in ("主要人物闭眼", "主体清晰度不足", "主要人物可能被遮挡", "曝光偏差明显")
    )
    group_score = _group_score(photo, expected_people)
    photo.metrics["group_ranking_score"] = group_score
    return (
        -closed,
        -missing_expected,
        -partial,
        -photo.metrics.get("bad_face_count", 0.0),
        -severe_issue_count,
        coverage,
        group_score,
        photo.metrics.get("learned_face_quality_score", 0.0),
        photo.metrics.get("min_face_score", 0.0),
        min_open,
        photo.metrics.get("eye_sharpness_score", 0.0),
        photo.metrics.get("face_sharpness_score", 0.0),
        min_face_sharpness,
        photo.score,
    )


def _issue_category(photo: PhotoObservation) -> str:
    if "主要人物闭眼" in photo.issues:
        return "closed_eyes"
    if "主体清晰度不足" in photo.issues:
        return "blurred"
    if "曝光偏差明显" in photo.issues:
        return "exposure"
    if any(issue in photo.issues for issue in ("主要人物可能被遮挡", "主要人物姿态异常", "高感噪声明显")):
        return "rejected"
    preference_score = photo.metrics.get("preference_score")
    preference_threshold = photo.metrics.get("preference_threshold")
    if (
        photo.metrics.get("preference_selection_enabled", 0.0) >= 0.5
        and preference_score is not None
        and preference_threshold is not None
        and np.isfinite(preference_score)
        and preference_score < preference_threshold
    ):
        return "rejected"
    return "selected"


def _winner_reasons(photo: PhotoObservation, group_size: int, expected_people: set[str]) -> list[str]:
    reasons: list[str] = []
    preference_score = photo.metrics.get("preference_score")
    preference_threshold = photo.metrics.get("preference_threshold")
    if (
        photo.metrics.get("preference_strength", 0.0) > 0.0
        and preference_score is not None
        and preference_threshold is not None
    ):
        if preference_score >= preference_threshold:
            reasons.append("符合当前摄影师的历史保留偏好")
        else:
            reasons.append("组内排名最佳，但未达到个人保留偏好阈值")
    if photo.metrics.get("eye_score", 0.0) >= 85:
        reasons.append("主要人物均处于良好睁眼状态")
    if photo.metrics.get("min_face_score", 0.0) >= 68:
        reasons.append("多人照片中最差人脸质量仍然良好")
    elif photo.metrics.get("face_sharpness_score", 0.0) >= 70:
        reasons.append("原分辨率人脸细节清晰")
    if expected_people and expected_people.issubset(photo.significant_person_ids):
        reasons.append("包含本组全部主要人物")
    if photo.metrics.get("subject_sharpness_score", photo.metrics.get("motion_blur_score", 0.0)) >= 72:
        reasons.append("未发现明显运动模糊")
    if photo.metrics.get("exposure_score", 0.0) >= 75:
        reasons.append("曝光与高光控制稳定")
    if group_size > 1:
        reasons.append(f"在 {group_size} 张同场景照片中综合排名第 1")
    elif not reasons:
        reasons.append("独立画面，未发现重复")
    return reasons[:4]


def _rejection_reasons(photo: PhotoObservation, winner: PhotoObservation, rank: int) -> list[str]:
    reasons: list[str] = []
    if (
        photo.metrics.get("preference_strength", 0.0) > 0.0
        and photo.metrics.get("preference_score", 50.0) + 10.0 < winner.metrics.get("preference_score", 50.0)
    ):
        reasons.append("历史选片偏好匹配度低于同组推荐")
    if photo.metrics.get("eye_score", 0.0) + 8.0 < winner.metrics.get("eye_score", 0.0):
        reasons.append("主要人物睁眼状态不如同组推荐")
    if photo.metrics.get("min_face_score", 0.0) + 8.0 < winner.metrics.get("min_face_score", 0.0):
        reasons.append("多人照片中的最差人脸质量较低")
    elif photo.metrics.get("face_sharpness_score", 0.0) + 9.0 < winner.metrics.get("face_sharpness_score", 0.0):
        reasons.append("原分辨率人脸清晰度低于同组推荐")
    if photo.metrics.get("subject_sharpness_score", photo.metrics.get("motion_blur_score", 0.0)) + 10.0 < winner.metrics.get(
        "subject_sharpness_score", winner.metrics.get("motion_blur_score", 0.0)
    ):
        reasons.append("运动模糊控制不如同组推荐")
    reasons.extend(issue for issue in photo.issues if issue not in reasons)
    reasons.append(f"与 {winner.filename} 高度相似，组内排名第 {rank}")
    return reasons[:4]


def apply_group_order(
    group: PhotoGroupInternal,
    ordered: list[PhotoObservation],
    keep_per_group: int,
    reason_overrides: dict[str, list[str]] | None = None,
) -> None:
    keep_per_group = min(5, max(1, keep_per_group))
    expected_people = _dominant_people(group)
    winner = ordered[0]
    reason_overrides = reason_overrides or {}
    for rank, photo in enumerate(ordered, start=1):
        photo.rank_in_group = rank
        photo.is_best_pick = rank <= keep_per_group
        if photo.is_best_pick:
            photo.category = _issue_category(photo)
            base_reasons = _winner_reasons(photo, len(group.photos), expected_people)
        else:
            photo.category = "duplicate"
            base_reasons = _rejection_reasons(photo, winner, rank)
        model_reasons = [f"视觉大模型复核：{reason}" for reason in reason_overrides.get(photo.id, [])]
        photo.selection_reasons = (model_reasons + base_reasons)[:5]
        photo.stars = 3 if photo.score >= 86 else 2 if photo.score >= 72 else 1 if photo.score >= 55 else 0
    group.photos = ordered


def rank_groups(groups: list[PhotoGroupInternal], keep_per_group: int) -> None:
    prepare_group_ranking_features(groups)
    for group in groups:
        expected_people = _dominant_people(group)
        ordered = sorted(group.photos, key=lambda photo: _ranking_key(photo, expected_people), reverse=True)
        apply_group_order(group, ordered, keep_per_group)
