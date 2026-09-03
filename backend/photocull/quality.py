from __future__ import annotations

import math

import cv2
import numpy as np

from .face_quality import select_quality_faces
from .internal_models import DepthObservation, FaceObservation


def _clamp(value: float, minimum: float = 0.0, maximum: float = 100.0) -> float:
    return max(minimum, min(maximum, float(value)))


def _log_score(value: float, low: float, high: float) -> float:
    logged = math.log1p(max(0.0, value))
    return _clamp((logged - math.log1p(low)) / max(math.log1p(high) - math.log1p(low), 1e-6) * 100.0)


def _eye_open_score(face: FaceObservation) -> float:
    looking_down = face.pitch >= 15.0 and abs(face.yaw) <= 45.0
    if face.eye_state == "Open":
        state_score = 100.0
    elif face.eye_state == "Partial":
        state_score = 70.0 if looking_down else 52.0
    elif face.eye_state == "Closed":
        state_score = 58.0 if looking_down else 5.0
    else:
        state_score = 72.0
    if face.open_probability is not None and not looking_down:
        state_score = _clamp(0.68 * state_score + 32.0 * face.open_probability)
    reliability = float(np.clip((face.area_ratio - 0.0007) / 0.0007, 0.0, 1.0))
    return _clamp(reliability * state_score + (1.0 - reliability) * 72.0)


def _eye_state_reliable(face: FaceObservation) -> bool:
    return (
        face.area_ratio >= 0.0012
        and face.confidence >= 0.70
        and face.pitch < 15.0
        and abs(face.yaw) <= 45.0
    )


def _face_scores(face: FaceObservation) -> dict[str, float]:
    sharpness = _log_score(face.sharpness, 8.0, 900.0)
    eye_sharpness = _log_score(face.eye_sharpness or face.sharpness, 7.0, 760.0)
    eye_open = _eye_open_score(face)
    pose = _clamp(100.0 - max(0.0, abs(face.yaw) - 12.0) * 1.15 - max(0.0, abs(face.roll) - 10.0) * 0.55)
    visibility = _clamp((1.0 - face.occlusion_risk) * 100.0)
    if face.fiqa_score is not None:
        learned_quality = _clamp(face.fiqa_score)
        quality = (
            0.24 * sharpness
            + 0.13 * eye_sharpness
            + 0.22 * eye_open
            + 0.08 * pose
            + 0.11 * visibility
            + 0.22 * learned_quality
        )
    else:
        learned_quality = 0.0
        quality = 0.32 * sharpness + 0.20 * eye_sharpness + 0.25 * eye_open + 0.10 * pose + 0.13 * visibility
    return {
        "sharpness": sharpness,
        "eye_sharpness": eye_sharpness,
        "eye_open": eye_open,
        "pose": pose,
        "visibility": visibility,
        "learned_quality": learned_quality,
        "quality": _clamp(quality),
    }


def _face_composition(faces: list[FaceObservation], contrast_score: float) -> float:
    if not faces:
        return _clamp(62.0 + (contrast_score - 50.0) * 0.16)
    thirds = ((1 / 3, 1 / 3), (2 / 3, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 2 / 3), (0.5, 0.5))
    composition_values: list[float] = []
    crop_penalties: list[float] = []
    for face in faces:
        x1, y1, x2, y2 = face.bbox
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        distance = min(math.dist(center, target) for target in thirds)
        composition_values.append(_clamp(100.0 - distance * 165.0))
        crop_penalties.append(30.0 if min(x1, y1, 1.0 - x2, 1.0 - y2) < 0.008 else 0.0)
    return _clamp(float(np.mean(composition_values)) - max(crop_penalties, default=0.0))


def rescore_quality(
    extracted_metrics: dict[str, float],
    faces: list[FaceObservation],
) -> tuple[dict[str, float], float, list[str]]:
    """从已缓存底层特征重算摄影规则，调权重时无需重跑神经网络。"""

    metrics = {key: float(value) for key, value in extracted_metrics.items()}
    sharpness_score = metrics.get("sharpness_score", 0.0)
    motion_blur_score = metrics.get("motion_blur_score", sharpness_score)
    depth_focus_score = metrics.get("depth_focus_score")
    depth_confidence = float(np.clip(metrics.get("depth_subject_confidence", 0.0), 0.0, 1.0))
    depth_weight = 0.20 * depth_confidence if depth_focus_score is not None else 0.0
    depth_focus_value = motion_blur_score if depth_focus_score is None else float(depth_focus_score)
    subject_sharpness_score = (1.0 - depth_weight) * motion_blur_score + depth_weight * depth_focus_value
    noise_score = metrics.get("noise_score", 100.0)
    exposure_score = metrics.get("exposure_score", 50.0)
    contrast_score = metrics.get("contrast_score", 50.0)
    significant = select_quality_faces(faces)
    composition_score = _face_composition(significant, contrast_score)

    if significant:
        per_face = [_face_scores(face) for face in significant]
        face_sharpness_score = 0.65 * min(item["sharpness"] for item in per_face) + 0.35 * float(
            np.mean([item["sharpness"] for item in per_face])
        )
        eye_sharpness_score = 0.65 * min(item["eye_sharpness"] for item in per_face) + 0.35 * float(
            np.mean([item["eye_sharpness"] for item in per_face])
        )
        eye_score = 0.72 * min(item["eye_open"] for item in per_face) + 0.28 * float(
            np.mean([item["eye_open"] for item in per_face])
        )
        mean_face_score = float(np.mean([item["quality"] for item in per_face]))
        min_face_score = min(item["quality"] for item in per_face)
        main_index = max(range(len(significant)), key=lambda index: significant[index].area_ratio)
        main_subject_score = per_face[main_index]["quality"]
        face_quality_score = 0.35 * mean_face_score + 0.40 * min_face_score + 0.25 * main_subject_score
        learned_face_values = [item["learned_quality"] for item in per_face if item["learned_quality"] > 0.0]
        learned_face_quality_score = (
            0.65 * min(learned_face_values) + 0.35 * float(np.mean(learned_face_values))
            if learned_face_values
            else 0.0
        )
        bad_face_count = sum(item["quality"] < 45.0 for item in per_face)
        visibility_score = float(np.mean([item["visibility"] for item in per_face]))
        pose_score = float(np.mean([item["pose"] for item in per_face]))
        learned_expressions = [
            face.expression_score for face in significant if face.expression_confidence is not None
        ]
        if learned_expressions:
            learned_expression_score = float(np.mean(learned_expressions))
            expression_score = (
                0.52 * eye_score
                + 0.16 * visibility_score
                + 0.14 * pose_score
                + 0.18 * learned_expression_score
            )
        else:
            expression_score = 0.62 * eye_score + 0.20 * visibility_score + 0.18 * pose_score
        aesthetic_proxy = 0.55 * composition_score + 0.45 * contrast_score
        technical_score = 0.45 * subject_sharpness_score + 0.35 * exposure_score + 0.20 * noise_score
        overall = (
            0.25 * face_quality_score
            + 0.15 * eye_score
            + 0.15 * subject_sharpness_score
            + 0.10 * exposure_score
            + 0.10 * expression_score
            + 0.10 * aesthetic_proxy
            + 0.10 * technical_score
            + 0.05 * composition_score
        )
    else:
        face_sharpness_score = sharpness_score
        eye_sharpness_score = sharpness_score
        eye_score = 78.0
        face_quality_score = sharpness_score
        learned_face_quality_score = 0.0
        mean_face_score = 0.0
        min_face_score = 0.0
        main_subject_score = 0.0
        bad_face_count = 0
        expression_score = 0.0
        technical_score = (
            0.46 * subject_sharpness_score
            + 0.30 * exposure_score
            + 0.14 * noise_score
            + 0.10 * contrast_score
        )
        overall = 0.58 * technical_score + 0.22 * composition_score + 0.12 * contrast_score + 0.08 * sharpness_score

    issues: list[str] = []
    if subject_sharpness_score < 34.0 or (significant and face_sharpness_score < 32.0):
        issues.append("主体清晰度不足")
    if significant and any(face.eye_state == "Closed" and _eye_state_reliable(face) for face in significant):
        issues.append("主要人物闭眼")
    elif significant and any(face.eye_state == "Partial" and _eye_state_reliable(face) for face in significant):
        issues.append("检测到半闭眼")

    dominant_detections = [face for face in faces if face.area_ratio >= 0.020 and face.confidence >= 0.82]
    if any(face.occlusion_risk >= 0.78 for face in dominant_detections):
        issues.append("主要人物可能被遮挡")
    if any(abs(face.yaw) >= 60.0 or abs(face.roll) >= 42.0 for face in dominant_detections):
        issues.append("主要人物姿态异常")
    if exposure_score < 43.0:
        issues.append("曝光偏差明显")
    if noise_score < 28.0:
        issues.append("高感噪声明显")
    if any(
        face.area_ratio >= 0.010
        and min(face.bbox[0], face.bbox[1], 1.0 - face.bbox[2], 1.0 - face.bbox[3]) < 0.008
        for face in significant
    ):
        issues.append("主要人物靠近画面裁切边缘")

    penalties = {
        "主要人物闭眼": 25.0,
        "主体清晰度不足": 35.0,
        "主要人物可能被遮挡": 15.0,
        "主要人物姿态异常": 10.0,
        "曝光偏差明显": 10.0,
        "高感噪声明显": 6.0,
    }
    overall -= sum(penalty for issue, penalty in penalties.items() if issue in issues)
    metrics.update(
        {
            "face_sharpness_score": face_sharpness_score,
            "eye_sharpness_score": eye_sharpness_score,
            "composition_score": composition_score,
            "eye_score": eye_score,
            "face_quality_score": face_quality_score,
            "learned_face_quality_score": learned_face_quality_score,
            "mean_face_score": mean_face_score,
            "min_face_score": min_face_score,
            "main_subject_score": main_subject_score,
            "bad_face_count": float(bad_face_count),
            "expression_score": expression_score,
            "technical_score": technical_score,
            "subject_sharpness_score": subject_sharpness_score,
        }
    )
    return metrics, _clamp(overall), issues


def analyze_quality(
    rgb: np.ndarray,
    faces: list[FaceObservation],
    depth: DepthObservation | None = None,
) -> tuple[dict[str, float], float, list[str]]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if max(gray.shape) > 1100:
        scale = 1100.0 / max(gray.shape)
        gray = cv2.resize(
            gray,
            (max(1, round(gray.shape[1] * scale)), max(1, round(gray.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )

    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness_score = _log_score(laplacian_variance, 12.0, 850.0)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    energy_x = float(np.mean(np.square(grad_x)))
    energy_y = float(np.mean(np.square(grad_y)))
    tenengrad = energy_x + energy_y
    tenengrad_score = _log_score(tenengrad, 90.0, 11500.0)
    directional_anisotropy = abs(energy_x - energy_y) / max(energy_x + energy_y, 1e-6)
    motion_blur_score = _clamp(0.56 * sharpness_score + 0.34 * tenengrad_score + 10.0 - 18.0 * directional_anisotropy)

    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    noise_estimate = float(np.median(np.abs(gray.astype(np.float32) - denoised.astype(np.float32))) / 255.0)
    noise_score = _clamp(100.0 - max(0.0, noise_estimate - 0.006) * 1700.0)
    brightness = float(gray.mean() / 255.0)
    highlight_clip = float(np.mean(gray >= 250))
    shadow_clip = float(np.mean(gray <= 5))
    percentile_5, percentile_95 = np.percentile(gray, [5, 95])
    contrast = float((percentile_95 - percentile_5) / 255.0)
    contrast_score = _clamp((contrast - 0.18) / 0.58 * 100.0)
    luminance_penalty = abs(brightness - 0.48) * 105.0
    clipping_penalty = max(0.0, highlight_clip - 0.008) * 440.0 + max(0.0, shadow_clip - 0.012) * 300.0
    exposure_score = _clamp(100.0 - luminance_penalty - clipping_penalty)

    extracted_metrics = {
        "sharpness": laplacian_variance,
        "sharpness_score": sharpness_score,
        "tenengrad": tenengrad,
        "tenengrad_score": tenengrad_score,
        "motion_blur_score": motion_blur_score,
        "directional_anisotropy": directional_anisotropy,
        "noise_estimate": noise_estimate,
        "noise_score": noise_score,
        "exposure_score": exposure_score,
        "contrast_score": contrast_score,
        "brightness": brightness,
        "highlight_clip": highlight_clip,
        "shadow_clip": shadow_clip,
    }
    if depth is not None:
        extracted_metrics.update(
            {
                "depth_focus_score": (
                    depth.subject_focus_score if depth.subject_focus_score is not None else motion_blur_score
                ),
                "depth_background_blur_score": (
                    depth.background_blur_score if depth.background_blur_score is not None else 50.0
                ),
                "depth_separation_score": depth.foreground_separation * 100.0,
                "depth_occlusion_risk": depth.occlusion_risk,
                "depth_subject_confidence": depth.subject_confidence,
            }
        )
    return rescore_quality(extracted_metrics, faces)
