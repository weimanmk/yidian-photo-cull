from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from threading import RLock
from typing import Any
import zipfile

import cv2
import numpy as np
from PIL import Image

from .body_engine import _mediapipe_anchors, _nms
from .config import model_candidates
from .internal_models import PoseObservation


MODEL_FILENAME = "pose_landmarker_heavy.task"
DETECTOR_INPUT_SIZE = 224
LANDMARK_INPUT_SIZE = 256
POSE_LANDMARK_COUNT = 33
MODEL_LANDMARK_COUNT = 39
BODY_LANDMARK_INDICES = np.arange(11, 33, dtype=np.int32)


@dataclass(slots=True)
class _PoseRoi:
    center_x: float
    center_y: float
    side_pixels: float
    rotation: float
    detection_confidence: float


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -80.0, 80.0)))


def _unit(vector: np.ndarray) -> np.ndarray | None:
    flattened = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(flattened))
    return flattened / norm if norm > 1e-8 else None


def _find_model() -> Path | None:
    for root in model_candidates():
        candidate = (root / MODEL_FILENAME).resolve()
        if candidate.is_file():
            return candidate
    return None


def _refine_landmarks_from_heatmap(
    normalized_landmarks: np.ndarray,
    heatmap: np.ndarray,
    kernel_size: int = 7,
    minimum_confidence: float = 0.5,
) -> np.ndarray:
    """复现 MediaPipe 的 7×7 热力图局部加权细化。"""
    refined = np.asarray(normalized_landmarks, dtype=np.float32).copy()
    values = np.asarray(heatmap, dtype=np.float32)
    if values.ndim == 4:
        values = values[0]
    if values.ndim != 3 or values.shape[2] != refined.shape[0]:
        return refined
    height, width, _ = values.shape
    offset = (kernel_size - 1) // 2
    for index in range(refined.shape[0]):
        center_x = int(refined[index, 0] * width)
        center_y = int(refined[index, 1] * height)
        if center_x < 0 or center_x >= width or center_y < 0 or center_y >= height:
            continue
        start_x = max(0, center_x - offset)
        end_x = min(width, center_x + offset + 1)
        start_y = max(0, center_y - offset)
        end_y = min(height, center_y + offset + 1)
        confidence = _sigmoid(values[start_y:end_y, start_x:end_x, index])
        total = float(confidence.sum())
        if total <= 0.0 or float(confidence.max(initial=0.0)) < minimum_confidence:
            continue
        columns = np.arange(start_x, end_x, dtype=np.float32)[None, :]
        rows = np.arange(start_y, end_y, dtype=np.float32)[:, None]
        refined[index, 0] = float((confidence * columns).sum() / total / width)
        refined[index, 1] = float((confidence * rows).sum() / total / height)
    return refined


def _pose_descriptor(world_landmarks: np.ndarray) -> np.ndarray | None:
    """将世界坐标归一到人体自身坐标系，保留动作而弱化平移、尺度和机位。"""
    landmarks = np.asarray(world_landmarks, dtype=np.float32)
    if landmarks.shape != (POSE_LANDMARK_COUNT, 3) or not np.isfinite(landmarks).all():
        return None
    hip_center = (landmarks[23] + landmarks[24]) * 0.5
    shoulder_center = (landmarks[11] + landmarks[12]) * 0.5
    right_axis = (landmarks[12] - landmarks[11]) + (landmarks[24] - landmarks[23])
    up_axis = shoulder_center - hip_center
    right_axis = _unit(right_axis)
    up_axis = _unit(up_axis)
    if right_axis is None or up_axis is None:
        return None
    forward_axis = _unit(np.cross(right_axis, up_axis))
    if forward_axis is None:
        return None
    right_axis = _unit(np.cross(up_axis, forward_axis))
    if right_axis is None:
        return None
    axes = np.stack((right_axis, up_axis, forward_axis), axis=1)
    centered = landmarks - hip_center
    local = centered @ axes
    scale_candidates = np.asarray(
        [
            np.linalg.norm(landmarks[12] - landmarks[11]),
            np.linalg.norm(landmarks[24] - landmarks[23]),
            np.linalg.norm(shoulder_center - hip_center),
        ],
        dtype=np.float32,
    )
    valid_scales = scale_candidates[scale_candidates > 1e-5]
    if valid_scales.size == 0:
        return None
    scale = float(np.median(valid_scales))
    body = np.clip(local[BODY_LANDMARK_INDICES] / scale, -4.0, 4.0)
    return _unit(body)


def _project_landmarks(
    normalized_landmarks: np.ndarray,
    roi: _PoseRoi,
    width: int,
    height: int,
) -> np.ndarray:
    local = (np.asarray(normalized_landmarks, dtype=np.float32)[:, :2] - 0.5) * roi.side_pixels
    cosine = math.cos(roi.rotation)
    sine = math.sin(roi.rotation)
    x = roi.center_x + cosine * local[:, 0] - sine * local[:, 1]
    y = roi.center_y + sine * local[:, 0] + cosine * local[:, 1]
    return np.column_stack((x / max(1, width), y / max(1, height))).astype(np.float32)


def _same_projected_pose(left: PoseObservation, right: PoseObservation) -> bool:
    left_landmarks = np.asarray(left.landmarks_2d, dtype=np.float32).reshape(-1, 3)
    right_landmarks = np.asarray(right.landmarks_2d, dtype=np.float32).reshape(-1, 3)
    if left_landmarks.shape[0] < POSE_LANDMARK_COUNT or right_landmarks.shape[0] < POSE_LANDMARK_COUNT:
        return False
    reliable = (
        (left_landmarks[BODY_LANDMARK_INDICES, 2] >= 0.42)
        & (right_landmarks[BODY_LANDMARK_INDICES, 2] >= 0.42)
    )
    if int(reliable.sum()) < 7:
        return False
    left_points = left_landmarks[BODY_LANDMARK_INDICES, :2][reliable]
    right_points = right_landmarks[BODY_LANDMARK_INDICES, :2][reliable]
    combined = np.vstack((left_points, right_points))
    scale = max(0.025, float(np.linalg.norm(np.ptp(combined, axis=0))))
    normalized_distance = float(np.median(np.linalg.norm(left_points - right_points, axis=1))) / scale
    descriptor_similarity = (
        float(np.dot(left.descriptor, right.descriptor))
        if left.descriptor is not None and right.descriptor is not None
        else 0.0
    )
    return normalized_distance <= 0.075 and descriptor_similarity >= 0.94


def _deduplicate_poses(observations: list[PoseObservation]) -> list[PoseObservation]:
    ordered = sorted(
        observations,
        key=lambda pose: (
            pose.presence_confidence,
            pose.visibility,
            pose.detection_confidence,
            -pose.area_ratio,
        ),
        reverse=True,
    )
    kept: list[PoseObservation] = []
    for candidate in ordered:
        if any(_same_projected_pose(candidate, existing) for existing in kept):
            continue
        kept.append(candidate)
    return kept


class PoseEngine:
    """完全本地的 BlazePose GHUM Heavy 3D 姿态引擎。"""

    def __init__(self) -> None:
        self.model_path = _find_model()
        self._detector: Any | None = None
        self._landmarker: Any | None = None
        self._detector_input: dict[str, Any] | None = None
        self._detector_box_output: dict[str, Any] | None = None
        self._detector_score_output: dict[str, Any] | None = None
        self._landmarker_input: dict[str, Any] | None = None
        self._landmark_outputs: dict[str, dict[str, Any]] = {}
        self._load_attempted = False
        self._lock = RLock()
        self._error = ""
        self._anchors = _mediapipe_anchors(DETECTOR_INPUT_SIZE)

    @property
    def available(self) -> bool:
        return self.model_path is not None

    def _ensure_loaded(self) -> bool:
        with self._lock:
            if self._load_attempted:
                return self._detector is not None and self._landmarker is not None
            self._load_attempted = True
            if self.model_path is None:
                self._error = f"缺少 {MODEL_FILENAME}"
                return False
            try:
                from ai_edge_litert.interpreter import Interpreter

                with zipfile.ZipFile(self.model_path) as archive:
                    detector_model = archive.read("pose_detector.tflite")
                    landmark_model = archive.read("pose_landmarks_detector.tflite")
                threads = max(1, min(8, os.cpu_count() or 1))
                self._detector = Interpreter(model_content=detector_model, num_threads=threads)
                self._landmarker = Interpreter(model_content=landmark_model, num_threads=threads)
                self._detector.allocate_tensors()
                self._landmarker.allocate_tensors()
                self._detector_input = self._detector.get_input_details()[0]
                detector_outputs = self._detector.get_output_details()
                self._detector_box_output = next(
                    detail for detail in detector_outputs if tuple(detail["shape"])[-1] == 12
                )
                self._detector_score_output = next(
                    detail for detail in detector_outputs if tuple(detail["shape"])[-1] == 1
                )
                self._landmarker_input = self._landmarker.get_input_details()[0]
                for detail in self._landmarker.get_output_details():
                    shape = tuple(int(value) for value in detail["shape"])
                    if shape[-1] == 195:
                        self._landmark_outputs["landmarks"] = detail
                    elif shape[-1] == 117:
                        self._landmark_outputs["world"] = detail
                    elif shape[-1] == 39 and len(shape) == 4:
                        self._landmark_outputs["heatmap"] = detail
                    elif int(np.prod(shape)) == 1:
                        self._landmark_outputs["presence"] = detail
                required = {"landmarks", "world", "heatmap", "presence"}
                if set(self._landmark_outputs) != required:
                    raise RuntimeError(f"姿态输出结构不匹配: {sorted(self._landmark_outputs)}")
                return True
            except Exception as exc:
                self._detector = None
                self._landmarker = None
                self._error = str(exc)
                return False

    def status(self) -> dict[str, object]:
        loaded = self._ensure_loaded() if self.available else False
        return {
            "available": loaded,
            "backend": "LiteRT XNNPACK CPU" if loaded else "unavailable",
            "model": "mediapipe-blazepose-ghum-heavy",
            "path": str(self.model_path) if self.model_path else None,
            "landmarks": 33,
            "world_coordinates": True,
            "telemetry": False,
            "runtime": "raw TFLite via ai-edge-litert",
            "error": self._error or None,
        }

    def signature(self) -> str:
        if self.model_path is None:
            return "pose-unavailable"
        stat = self.model_path.stat()
        return f"{self.model_path.name}:{stat.st_size}:{stat.st_mtime_ns}"

    def analyze(self, rgb: np.ndarray, original_image: Image.Image | None = None) -> list[PoseObservation]:
        if not self._ensure_loaded():
            return []
        try:
            rois = self._detect_rois(rgb)
            source: Image.Image | np.ndarray = original_image if original_image is not None else rgb
            width, height = original_image.size if original_image is not None else (rgb.shape[1], rgb.shape[0])
            observations: list[PoseObservation] = []
            for roi in rois:
                scaled_roi = _PoseRoi(
                    center_x=roi.center_x * width / rgb.shape[1],
                    center_y=roi.center_y * height / rgb.shape[0],
                    side_pixels=roi.side_pixels * max(width / rgb.shape[1], height / rgb.shape[0]),
                    rotation=roi.rotation,
                    detection_confidence=roi.detection_confidence,
                )
                observation = self._infer_pose(source, scaled_roi, width, height)
                if observation is not None:
                    observations.append(observation)
            deduplicated = _deduplicate_poses(observations)
            self._error = ""
            return deduplicated
        except Exception as exc:
            self._error = str(exc)
            return []

    def _detect_rois(self, rgb: np.ndarray) -> list[_PoseRoi]:
        height, width = rgb.shape[:2]
        scale = min(DETECTOR_INPUT_SIZE / width, DETECTOR_INPUT_SIZE / height)
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        resized = cv2.resize(rgb, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
        pad_x = (DETECTOR_INPUT_SIZE - resized_width) // 2
        pad_y = (DETECTOR_INPUT_SIZE - resized_height) // 2
        canvas = np.zeros((DETECTOR_INPUT_SIZE, DETECTOR_INPUT_SIZE, 3), dtype=np.uint8)
        canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
        tensor = ((canvas.astype(np.float32) / 255.0 - 0.5) * 2.0)[None]
        with self._lock:
            self._detector.set_tensor(self._detector_input["index"], tensor)
            self._detector.invoke()
            raw_boxes = self._detector.get_tensor(self._detector_box_output["index"])[0]
            logits = self._detector.get_tensor(self._detector_score_output["index"])[0, :, 0]
        scores = _sigmoid(np.asarray(logits, dtype=np.float32))
        positive = np.where(scores >= 0.45)[0]
        if positive.size == 0:
            return []
        raw = np.asarray(raw_boxes[positive], dtype=np.float32)
        anchors = self._anchors[positive]
        center_x = raw[:, 0] / DETECTOR_INPUT_SIZE + anchors[:, 0]
        center_y = raw[:, 1] / DETECTOR_INPUT_SIZE + anchors[:, 1]
        box_width = raw[:, 2] / DETECTOR_INPUT_SIZE
        box_height = raw[:, 3] / DETECTOR_INPUT_SIZE
        keypoints = np.empty((raw.shape[0], 4, 2), dtype=np.float32)
        for index in range(4):
            keypoints[:, index, 0] = raw[:, 4 + index * 2] / DETECTOR_INPUT_SIZE + anchors[:, 0]
            keypoints[:, index, 1] = raw[:, 5 + index * 2] / DETECTOR_INPUT_SIZE + anchors[:, 1]

        def remove_letterbox_x(values: np.ndarray) -> np.ndarray:
            return (values * DETECTOR_INPUT_SIZE - pad_x) / resized_width

        def remove_letterbox_y(values: np.ndarray) -> np.ndarray:
            return (values * DETECTOR_INPUT_SIZE - pad_y) / resized_height

        center_x = remove_letterbox_x(center_x)
        center_y = remove_letterbox_y(center_y)
        box_width = box_width * DETECTOR_INPUT_SIZE / resized_width
        box_height = box_height * DETECTOR_INPUT_SIZE / resized_height
        keypoints[:, :, 0] = remove_letterbox_x(keypoints[:, :, 0])
        keypoints[:, :, 1] = remove_letterbox_y(keypoints[:, :, 1])
        boxes = np.column_stack(
            (
                center_x - box_width * 0.5,
                center_y - box_height * 0.5,
                center_x + box_width * 0.5,
                center_y + box_height * 0.5,
            )
        )
        keep = _nms(boxes, scores[positive], 0.30)[:8]
        rois: list[_PoseRoi] = []
        for selected_index in keep:
            center = keypoints[selected_index, 0] * (width, height)
            scale_point = keypoints[selected_index, 1] * (width, height)
            delta = scale_point - center
            side = float(np.linalg.norm(delta) * 2.0 * 1.25)
            if side < 40.0 or side * side / max(1.0, width * height) < 0.0015:
                continue
            rotation = math.pi / 2.0 - math.atan2(-float(delta[1]), float(delta[0]))
            rotation = (rotation + math.pi) % (2.0 * math.pi) - math.pi
            rois.append(
                _PoseRoi(
                    center_x=float(center[0]),
                    center_y=float(center[1]),
                    side_pixels=side,
                    rotation=rotation,
                    detection_confidence=float(scores[positive[selected_index]]),
                )
            )
        return rois

    @staticmethod
    def _extract_crop(source: Image.Image | np.ndarray, roi: _PoseRoi) -> np.ndarray:
        half = roi.side_pixels * 0.5
        local = np.asarray(((-half, -half), (half, -half), (half, half), (-half, half)), dtype=np.float32)
        cosine = math.cos(roi.rotation)
        sine = math.sin(roi.rotation)
        rotation = np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float32)
        corners = local @ rotation.T + (roi.center_x, roi.center_y)
        left = int(math.floor(float(corners[:, 0].min())))
        top = int(math.floor(float(corners[:, 1].min())))
        right = int(math.ceil(float(corners[:, 0].max()))) + 1
        bottom = int(math.ceil(float(corners[:, 1].max()))) + 1
        if isinstance(source, Image.Image):
            source_width, source_height = source.size
        else:
            source_height, source_width = source.shape[:2]
        crop_left = max(0, left)
        crop_top = max(0, top)
        crop_right = min(source_width, right)
        crop_bottom = min(source_height, bottom)
        if crop_right <= crop_left or crop_bottom <= crop_top:
            return np.zeros((LANDMARK_INPUT_SIZE, LANDMARK_INPUT_SIZE, 3), dtype=np.uint8)
        if isinstance(source, Image.Image):
            patch = np.asarray(
                source.crop((crop_left, crop_top, crop_right, crop_bottom)).convert("RGB"),
                dtype=np.uint8,
            )
        else:
            patch = np.asarray(source[crop_top:crop_bottom, crop_left:crop_right], dtype=np.uint8)
        source_points = corners - (crop_left, crop_top)
        destination_points = np.asarray(
            ((0.0, 0.0), (255.0, 0.0), (255.0, 255.0), (0.0, 255.0)),
            dtype=np.float32,
        )
        transform = cv2.getPerspectiveTransform(source_points.astype(np.float32), destination_points)
        return cv2.warpPerspective(
            patch,
            transform,
            (LANDMARK_INPUT_SIZE, LANDMARK_INPUT_SIZE),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

    def _infer_pose(
        self,
        source: Image.Image | np.ndarray,
        roi: _PoseRoi,
        width: int,
        height: int,
    ) -> PoseObservation | None:
        crop = self._extract_crop(source, roi)
        tensor = (crop.astype(np.float32) / 255.0)[None]
        with self._lock:
            self._landmarker.set_tensor(self._landmarker_input["index"], tensor)
            self._landmarker.invoke()
            raw = self._landmarker.get_tensor(self._landmark_outputs["landmarks"]["index"])[0]
            world = self._landmarker.get_tensor(self._landmark_outputs["world"]["index"])[0]
            heatmap = self._landmarker.get_tensor(self._landmark_outputs["heatmap"]["index"])
            presence = float(
                self._landmarker.get_tensor(self._landmark_outputs["presence"]["index"]).reshape(-1)[0]
            )
        if presence < 0.50:
            return None
        raw_landmarks = np.asarray(raw, dtype=np.float32).reshape(MODEL_LANDMARK_COUNT, 5)
        normalized = raw_landmarks[:, :3].copy()
        normalized[:, :3] /= LANDMARK_INPUT_SIZE
        normalized = _refine_landmarks_from_heatmap(normalized, heatmap)
        visibility = _sigmoid(raw_landmarks[:, 3])
        joint_presence = _sigmoid(raw_landmarks[:, 4])
        reliability = np.minimum(visibility, joint_presence)
        world_landmarks = np.asarray(world, dtype=np.float32).reshape(MODEL_LANDMARK_COUNT, 3)[:POSE_LANDMARK_COUNT]
        descriptor = _pose_descriptor(world_landmarks)
        projected = _project_landmarks(normalized[:POSE_LANDMARK_COUNT], roi, width, height)
        landmarks_2d = np.column_stack((projected, reliability[:POSE_LANDMARK_COUNT])).astype(np.float32)
        half = roi.side_pixels * 0.5
        bbox = (
            max(0.0, (roi.center_x - half) / width),
            max(0.0, (roi.center_y - half) / height),
            min(1.0, (roi.center_x + half) / width),
            min(1.0, (roi.center_y + half) / height),
        )
        area_ratio = float(max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1]))
        pose_visibility = float(np.mean(reliability[BODY_LANDMARK_INDICES]))
        return PoseObservation(
            bbox=bbox,
            detection_confidence=roi.detection_confidence,
            presence_confidence=presence,
            area_ratio=area_ratio,
            landmarks_2d=landmarks_2d,
            descriptor=descriptor,
            visibility=pose_visibility,
            model="mediapipe-blazepose-ghum-heavy",
        )
