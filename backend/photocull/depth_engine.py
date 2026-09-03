from __future__ import annotations

import math
from pathlib import Path
from threading import RLock
from typing import Any

import cv2
import numpy as np

from .config import model_candidates
from .internal_models import BodyObservation, DepthObservation, FaceObservation, PoseObservation
from .runtime import InferenceRuntime


MODEL_FILENAME = "depth_anything_v2_vitl.onnx"
MODEL_NAME = "depth-anything-v2-large"
INPUT_SIZE = 518
BODY_LANDMARK_INDICES = np.arange(11, 33, dtype=np.int32)
POSE_CONNECTIONS = (
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (27, 29),
    (29, 31),
    (24, 26),
    (26, 28),
    (28, 30),
    (30, 32),
)


def _find_model() -> Path | None:
    for root in model_candidates():
        candidate = (root / MODEL_FILENAME).resolve()
        if candidate.is_file():
            return candidate
    return None


def _unit(values: np.ndarray) -> np.ndarray | None:
    flattened = np.asarray(values, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(flattened))
    return flattened / norm if norm > 1e-8 else None


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + math.exp(-float(np.clip(value, -30.0, 30.0)))))


def _normalize_depth(raw_depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(raw_depth, dtype=np.float32).squeeze()
    if depth.shape != (INPUT_SIZE, INPUT_SIZE) or not np.isfinite(depth).all():
        raise ValueError(f"景深输出结构不匹配: {depth.shape}")
    low, high = np.quantile(depth, (0.02, 0.98))
    if high - low <= 1e-6:
        return np.zeros_like(depth)
    return np.clip((depth - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _depth_descriptor(depth: np.ndarray) -> np.ndarray | None:
    small = cv2.resize(depth, (18, 18), interpolation=cv2.INTER_AREA)
    median = float(np.median(small))
    low, high = np.quantile(small, (0.20, 0.80))
    normalized = np.clip((small - median) / max(0.06, float(high - low)), -3.0, 3.0)
    return _unit(normalized)


def _pose_points(pose: PoseObservation) -> tuple[np.ndarray, np.ndarray]:
    landmarks = np.asarray(pose.landmarks_2d, dtype=np.float32).reshape(-1, 3)
    if landmarks.shape[0] < 33:
        return np.empty((0, 2), dtype=np.float32), np.empty(0, dtype=np.int32)
    reliable = (
        (landmarks[BODY_LANDMARK_INDICES, 2] >= 0.36)
        & np.isfinite(landmarks[BODY_LANDMARK_INDICES]).all(axis=1)
        & (landmarks[BODY_LANDMARK_INDICES, 0] >= -0.05)
        & (landmarks[BODY_LANDMARK_INDICES, 0] <= 1.05)
        & (landmarks[BODY_LANDMARK_INDICES, 1] >= -0.05)
        & (landmarks[BODY_LANDMARK_INDICES, 1] <= 1.05)
    )
    indices = BODY_LANDMARK_INDICES[reliable]
    return landmarks[indices, :2], indices


def _sample_points(depth: np.ndarray, points: np.ndarray, radius: int = 4) -> np.ndarray:
    values: list[np.ndarray] = []
    for x_value, y_value in np.asarray(points, dtype=np.float32):
        x = int(np.clip(round(float(x_value) * (INPUT_SIZE - 1)), 0, INPUT_SIZE - 1))
        y = int(np.clip(round(float(y_value) * (INPUT_SIZE - 1)), 0, INPUT_SIZE - 1))
        values.append(
            depth[
                max(0, y - radius) : min(INPUT_SIZE, y + radius + 1),
                max(0, x - radius) : min(INPUT_SIZE, x + radius + 1),
            ].reshape(-1)
        )
    return np.concatenate(values) if values else np.empty(0, dtype=np.float32)


def _expanded_point_region(depth: np.ndarray, points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return depth.reshape(-1)
    minimum = np.clip(points.min(axis=0) - 0.08, 0.0, 1.0)
    maximum = np.clip(points.max(axis=0) + 0.08, 0.0, 1.0)
    x1, y1 = np.floor(minimum * (INPUT_SIZE - 1)).astype(int)
    x2, y2 = np.ceil(maximum * (INPUT_SIZE - 1)).astype(int)
    return depth[y1 : max(y1 + 1, y2 + 1), x1 : max(x1 + 1, x2 + 1)].reshape(-1)


def _foreground_score(depth: np.ndarray, points: np.ndarray) -> tuple[float, float] | None:
    sampled = _sample_points(depth, points)
    if sampled.size < 40:
        return None
    subject_depth = float(np.median(sampled))
    lower = float(np.mean(depth < subject_depth - 1e-5))
    tied = float(np.mean(np.abs(depth - subject_depth) <= 1e-5))
    global_rank = lower + 0.5 * tied
    local = _expanded_point_region(depth, points)
    local_median = float(np.median(local))
    local_low, local_high = np.quantile(local, (0.25, 0.75))
    local_scale = max(0.035, float(local_high - local_low))
    local_score = _sigmoid((subject_depth - local_median) / local_scale * 1.8)
    return subject_depth, float(np.clip(0.48 * global_rank + 0.52 * local_score, 0.0, 1.0))


def _box_points(box: tuple[float, float, float, float]) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.asarray(
        (
            ((x1 + x2) * 0.5, (y1 + y2) * 0.5),
            ((2.0 * x1 + x2) / 3.0, (2.0 * y1 + y2) / 3.0),
            ((x1 + 2.0 * x2) / 3.0, (y1 + 2.0 * y2) / 3.0),
            ((2.0 * x1 + x2) / 3.0, (y1 + 2.0 * y2) / 3.0),
            ((x1 + 2.0 * x2) / 3.0, (2.0 * y1 + y2) / 3.0),
        ),
        dtype=np.float32,
    )


def _draw_pose_mask(mask: np.ndarray, depth: np.ndarray, pose: PoseObservation, subject_depth: float) -> None:
    points, indices = _pose_points(pose)
    if points.shape[0] < 5:
        return
    pixel_points = np.rint(points * (INPUT_SIZE - 1)).astype(np.int32)
    index_to_point = {int(index): tuple(pixel_points[offset]) for offset, index in enumerate(indices)}
    span = np.ptp(pixel_points, axis=0)
    thickness = max(5, int(round(max(10.0, float(span.max())) * 0.055)))
    for left, right in POSE_CONNECTIONS:
        if left in index_to_point and right in index_to_point:
            cv2.line(mask, index_to_point[left], index_to_point[right], 255, thickness, cv2.LINE_AA)
    for point in pixel_points:
        cv2.circle(mask, tuple(point), max(4, thickness // 2), 255, -1, cv2.LINE_AA)
    hull = cv2.convexHull(pixel_points)
    hull_mask = np.zeros_like(mask)
    cv2.fillConvexPoly(hull_mask, hull, 255, cv2.LINE_AA)
    hull_mask = cv2.dilate(hull_mask, np.ones((9, 9), dtype=np.uint8), iterations=1)
    depth_band = (depth >= subject_depth - 0.10) & (depth <= subject_depth + 0.14)
    mask[(hull_mask > 0) & depth_band] = 255


def _draw_box_mask(mask: np.ndarray, depth: np.ndarray, box: tuple[float, float, float, float], center_depth: float) -> None:
    x1, y1, x2, y2 = box
    left = int(np.clip(round(x1 * (INPUT_SIZE - 1)), 0, INPUT_SIZE - 1))
    top = int(np.clip(round(y1 * (INPUT_SIZE - 1)), 0, INPUT_SIZE - 1))
    right = int(np.clip(round(x2 * (INPUT_SIZE - 1)), left + 1, INPUT_SIZE))
    bottom = int(np.clip(round(y2 * (INPUT_SIZE - 1)), top + 1, INPUT_SIZE))
    region = depth[top:bottom, left:right]
    band = (region >= center_depth - 0.10) & (region <= center_depth + 0.14)
    mask_region = mask[top:bottom, left:right]
    mask_region[band] = 255


def _focus_metrics(rgb: np.ndarray, subject_mask: np.ndarray) -> tuple[float, float] | None:
    subject = subject_mask > 0
    if int(subject.sum()) < 300:
        return None
    gray = cv2.cvtColor(cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    gray_float = gray.astype(np.float32)
    grad_x = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(grad_x, grad_y)
    laplacian = np.abs(cv2.Laplacian(gray_float, cv2.CV_32F, ksize=3))
    energy = cv2.GaussianBlur(np.log1p(0.65 * gradient + 0.35 * laplacian), (0, 0), 1.1)
    exclusion = cv2.dilate(subject_mask, np.ones((31, 31), dtype=np.uint8), iterations=1) > 0
    background = ~exclusion
    background[: round(INPUT_SIZE * 0.03)] = False
    background[round(INPUT_SIZE * 0.97) :] = False
    if int(background.sum()) < 1000:
        background = ~subject
    subject_energy = float(np.quantile(energy[subject], 0.72))
    background_energy = float(np.quantile(energy[background], 0.72))
    detail_rank = float(np.mean(energy <= subject_energy))
    log_ratio = math.log((subject_energy + 0.08) / (background_energy + 0.08))
    relative_score = float(np.clip(50.0 + 42.0 * log_ratio, 0.0, 100.0))
    focus_score = float(np.clip(0.62 * detail_rank * 100.0 + 0.38 * relative_score, 0.0, 100.0))
    background_blur_score = float(np.clip(50.0 + 55.0 * log_ratio, 0.0, 100.0))
    return focus_score, background_blur_score


class DepthEngine:
    """Depth Anything V2 Large 相对景深；用于合焦、前后景与假人体过滤。"""

    def __init__(self, use_gpu: bool = True) -> None:
        self.model_path = _find_model()
        self.runtime = InferenceRuntime(use_gpu)
        self._session: Any | None = None
        self._input_name = ""
        self._output_name = ""
        self._load_attempted = False
        self._error = ""
        self._lock = RLock()

    @property
    def available(self) -> bool:
        return self.model_path is not None and self.runtime.available

    def _ensure_loaded(self) -> bool:
        with self._lock:
            if self._load_attempted:
                return self._session is not None
            self._load_attempted = True
            if self.model_path is None:
                self._error = f"缺少 {MODEL_FILENAME}"
                return False
            try:
                self._session = self.runtime.create_session(self.model_path)
                model_input = self._session.get_inputs()[0]
                model_output = self._session.get_outputs()[0]
                if tuple(model_input.shape) != (1, 3, INPUT_SIZE, INPUT_SIZE):
                    raise RuntimeError(f"景深输入结构不匹配: {model_input.shape}")
                self._input_name = model_input.name
                self._output_name = model_output.name
                return True
            except Exception as exc:
                self._session = None
                self._error = str(exc)
                return False

    def status(self) -> dict[str, object]:
        loaded = self._ensure_loaded() if self.available else False
        actual_providers = self.runtime.actual_providers(self._session)
        return {
            "available": loaded,
            "backend": actual_providers[0] if actual_providers else self.runtime.primary_provider if loaded else "unavailable",
            "providers": actual_providers or self.runtime.providers if loaded else [],
            "provider_source": "actual" if actual_providers else "configured",
            "cuda_preload_error": self.runtime.cuda_preload_error,
            "model": MODEL_NAME,
            "path": str(self.model_path) if self.model_path else None,
            "relative_depth": True,
            "metric_depth": False,
            "local_only": True,
            "roles": ["subject-focus", "background-separation", "pose-foreground-filter"],
            "error": self._error or None,
        }

    def signature(self) -> str:
        if self.model_path is None:
            return "depth-unavailable"
        stat = self.model_path.stat()
        return f"{self.model_path.name}:{stat.st_size}:{stat.st_mtime_ns}"

    def analyze(
        self,
        rgb: np.ndarray,
        faces: list[FaceObservation],
        bodies: list[BodyObservation],
        poses: list[PoseObservation],
    ) -> DepthObservation | None:
        if not self._ensure_loaded():
            return None
        try:
            resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_CUBIC).astype(np.float32)
            tensor = resized / 255.0
            tensor = (tensor - np.asarray((0.485, 0.456, 0.406), dtype=np.float32)) / np.asarray(
                (0.229, 0.224, 0.225), dtype=np.float32
            )
            tensor = tensor.transpose(2, 0, 1)[None]
            with self._lock:
                raw_depth = self._session.run([self._output_name], {self._input_name: tensor})[0]
            depth = _normalize_depth(raw_depth)

            pose_depths: list[tuple[PoseObservation, float, float]] = []
            for pose in poses:
                points, _indices = _pose_points(pose)
                result = _foreground_score(depth, points)
                if result is None:
                    pose.foreground_score = None
                    continue
                subject_depth, foreground = result
                pose.foreground_score = foreground
                pose_depths.append((pose, subject_depth, foreground))

            subject_mask = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.uint8)
            subject_depth_samples: list[float] = []
            confidence_samples: list[float] = []
            if pose_depths:
                maximum = max(item[2] for item in pose_depths)
                primary = [
                    item
                    for item in pose_depths
                    if item[2] >= max(0.48, maximum - 0.16)
                    and item[0].visibility >= 0.38
                    and item[0].presence_confidence >= 0.60
                ]
                for pose, subject_depth, foreground in primary[:4]:
                    _draw_pose_mask(subject_mask, depth, pose, subject_depth)
                    subject_depth_samples.append(subject_depth)
                    confidence_samples.append(foreground)

            if not subject_depth_samples:
                body_candidates: list[tuple[float, float, BodyObservation]] = []
                for body in bodies:
                    result = _foreground_score(depth, _box_points(body.bbox))
                    if result is not None:
                        body_candidates.append((result[1] * math.sqrt(max(body.area_ratio, 1e-5)), result[0], body))
                if body_candidates:
                    best_salience = max(item[0] for item in body_candidates)
                    for salience, body_depth, body in sorted(body_candidates, reverse=True, key=lambda item: item[0]):
                        if salience < max(0.04, best_salience * 0.72):
                            continue
                        _draw_box_mask(subject_mask, depth, body.bbox, body_depth)
                        subject_depth_samples.append(body_depth)
                        confidence_samples.append(float(np.clip(salience / max(best_salience, 1e-6), 0.0, 1.0)))

            for face in faces:
                if face.area_ratio < 0.0010 or face.confidence < 0.65:
                    continue
                result = _foreground_score(depth, _box_points(face.bbox))
                if result is None or result[1] < 0.42:
                    continue
                _draw_box_mask(subject_mask, depth, face.bbox, result[0])

            subject = subject_mask > 0
            subject_depth = float(np.median(depth[subject])) if int(subject.sum()) >= 300 else None
            expanded = cv2.dilate(subject_mask, np.ones((31, 31), dtype=np.uint8), iterations=1) > 0
            background_values = depth[~expanded]
            background_depth = float(np.median(background_values)) if background_values.size >= 1000 else None
            if subject_depth is not None and background_depth is not None:
                separation = float(np.clip((subject_depth - background_depth + 0.015) / 0.34, 0.0, 1.0))
                closer = depth[subject] > subject_depth + 0.14
                occlusion_risk = float(np.clip((float(np.mean(closer)) - 0.07) / 0.28, 0.0, 1.0))
            else:
                separation = 0.0
                occlusion_risk = 0.0
            focus = _focus_metrics(rgb, subject_mask)
            focus_score, background_blur_score = focus if focus is not None else (None, None)
            subject_confidence = float(np.mean(confidence_samples)) if confidence_samples else 0.0
            return DepthObservation(
                descriptor=_depth_descriptor(depth),
                subject_depth=subject_depth,
                background_depth=background_depth,
                foreground_separation=separation,
                subject_focus_score=focus_score,
                background_blur_score=background_blur_score,
                occlusion_risk=occlusion_risk,
                subject_confidence=subject_confidence,
                model=MODEL_NAME,
            )
        except Exception as exc:
            self._error = str(exc)
            return None
