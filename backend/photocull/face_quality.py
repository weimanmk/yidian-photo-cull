from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .internal_models import FaceObservation


def select_quality_faces(faces: list[FaceObservation]) -> list[FaceObservation]:
    """只返回足以代表画面主体的可靠脸，避免观众席小脸触发最差脸惩罚。"""

    candidates = [
        face
        for face in faces
        if (
            (face.area_ratio >= 0.0006 and face.confidence >= 0.82)
            or (face.area_ratio >= 0.0010 and face.confidence >= 0.70)
        )
        and abs(face.roll) <= 42.0
        and abs(face.yaw) <= 55.0
        and face.occlusion_risk < 0.68
    ]
    if not candidates:
        return []
    largest = max(face.area_ratio for face in candidates)
    # 多个脸都非常小时通常是观众席或背景路人，不应按合影规则惩罚最差脸。
    if len(candidates) >= 2 and largest < 0.0025:
        return []
    relative_floor = largest * 0.24 if largest >= 0.0035 else 0.0
    return sorted(
        (face for face in candidates if face.area_ratio >= relative_floor),
        key=lambda face: (face.area_ratio, face.confidence),
        reverse=True,
    )[:12]


def _analysis_gray(image: Image.Image, maximum: int) -> np.ndarray:
    width, height = image.size
    scale = min(1.0, maximum / max(1, width, height))
    if scale < 1.0:
        image = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.LANCZOS,
        )
    return cv2.cvtColor(np.asarray(image.convert("RGB"), dtype=np.uint8), cv2.COLOR_RGB2GRAY)


def _laplacian_variance(gray: np.ndarray) -> float:
    if gray.size == 0 or min(gray.shape[:2]) < 8:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def refine_face_quality(image: Image.Image, faces: list[FaceObservation]) -> None:
    """在原图坐标上二次裁脸，避免 1600px 预览掩盖轻微失焦。"""

    image_width, image_height = image.size
    for face in faces:
        x1, y1, x2, y2 = face.bbox
        face_width = max(1.0, (x2 - x1) * image_width)
        face_height = max(1.0, (y2 - y1) * image_height)
        margin_x = face_width * 0.08
        margin_y = face_height * 0.08
        box = (
            max(0, round(x1 * image_width - margin_x)),
            max(0, round(y1 * image_height - margin_y)),
            min(image_width, round(x2 * image_width + margin_x)),
            min(image_height, round(y2 * image_height + margin_y)),
        )
        if box[2] - box[0] < 12 or box[3] - box[1] < 12:
            continue
        gray = _analysis_gray(image.crop(box), maximum=720)
        high_res = _laplacian_variance(gray)
        eye_top = max(0, round(gray.shape[0] * 0.18))
        eye_bottom = min(gray.shape[0], round(gray.shape[0] * 0.55))
        eye_left = max(0, round(gray.shape[1] * 0.10))
        eye_right = min(gray.shape[1], round(gray.shape[1] * 0.90))
        eye_sharpness = _laplacian_variance(gray[eye_top:eye_bottom, eye_left:eye_right])
        face.high_res_sharpness = high_res
        face.eye_sharpness = eye_sharpness
        if high_res > 0.0:
            face.sharpness = high_res
