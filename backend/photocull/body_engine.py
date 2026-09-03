from __future__ import annotations

import math
from pathlib import Path
from threading import RLock

import cv2
import numpy as np

from .config import model_candidates
from .internal_models import BodyObservation
from .runtime import InferenceRuntime


def _unit(vector: np.ndarray) -> np.ndarray | None:
    flattened = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(flattened))
    return flattened / norm if norm > 1e-8 else None


def _find_models(filename: str) -> list[Path]:
    found: list[Path] = []
    for root in model_candidates():
        path = (root / filename).resolve()
        if path.is_file() and path not in found:
            found.append(path)
    return found


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        index = int(order[0])
        keep.append(index)
        if order.size == 1:
            break
        remaining = order[1:]
        xx1 = np.maximum(x1[index], x1[remaining])
        yy1 = np.maximum(y1[index], y1[remaining])
        xx2 = np.minimum(x2[index], x2[remaining])
        yy2 = np.minimum(y2[index], y2[remaining])
        overlap = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[index] + areas[remaining] - overlap
        iou = overlap / np.maximum(union, 1e-8)
        order = remaining[iou <= threshold]
    return keep


def _mediapipe_anchors(input_size: int = 224) -> np.ndarray:
    """生成 MediaPipe BlazePose detector 的 2254 个固定中心锚点。"""
    anchors: list[tuple[float, float]] = []
    for stride, repetitions in ((8, 2), (16, 2), (32, 6)):
        feature_size = math.ceil(input_size / stride)
        for y in range(feature_size):
            for x in range(feature_size):
                center = ((x + 0.5) / feature_size, (y + 0.5) / feature_size)
                anchors.extend([center] * repetitions)
    return np.asarray(anchors, dtype=np.float32)


class BodyEngine:
    """人体检测与服饰外观 ReID；只提供分组软证据，不产生硬身份。"""

    def __init__(self, use_gpu: bool = True) -> None:
        self.runtime = InferenceRuntime(use_gpu=use_gpu)
        self.detector_candidates: list[tuple[str, Path]] = [
            *(('yolov8n-person', path) for path in _find_models('yolov8n.onnx')),
            *(('opencv-mediapipe-person', path) for path in _find_models('person_detection_mediapipe.onnx')),
        ]
        reid_paths = _find_models("osnet_x0_25_msmt17.onnx")
        self.reid_path = reid_paths[0] if reid_paths else None
        self.detector_name = self.detector_candidates[0][0] if self.detector_candidates else "unavailable"
        self.detector_path = self.detector_candidates[0][1] if self.detector_candidates else None
        self._detector_session = None
        self._reid_session = None
        self._load_attempted = False
        self._lock = RLock()
        self._errors: list[str] = []
        self._anchors = _mediapipe_anchors()

    @property
    def available(self) -> bool:
        return self.runtime.available and bool(self.detector_candidates)

    def _ensure_loaded(self) -> bool:
        with self._lock:
            if self._load_attempted:
                return self._detector_session is not None
            self._load_attempted = True
            for name, path in self.detector_candidates:
                try:
                    self._detector_session = self.runtime.create_session(path)
                    self.detector_name = name
                    self.detector_path = path
                    break
                except Exception as exc:
                    self._errors.append(f"{name}: {exc}")
            if self.reid_path is not None:
                try:
                    self._reid_session = self.runtime.create_session(self.reid_path)
                except Exception as exc:
                    self._errors.append(f"osnet: {exc}")
            return self._detector_session is not None

    def status(self) -> dict[str, object]:
        actual_providers = self.runtime.actual_providers(self._detector_session, self._reid_session)
        return {
            "available": self.available,
            "backend": actual_providers[0] if actual_providers else self.runtime.primary_provider,
            "providers": actual_providers or self.runtime.providers,
            "provider_source": "actual" if actual_providers else "configured",
            "cuda_preload_error": self.runtime.cuda_preload_error,
            "detector": {
                "available": bool(self.detector_path and self.detector_path.is_file()),
                "name": self.detector_name,
                "path": str(self.detector_path) if self.detector_path else None,
                "fallbacks": [name for name, _ in self.detector_candidates[1:]],
            },
            "reid_model": {
                "available": bool(self.reid_path and self.reid_path.is_file()),
                "name": "osnet-x0.25-msmt17",
                "path": str(self.reid_path) if self.reid_path else None,
                "role": "soft-evidence-only",
            },
            "errors": list(self._errors),
        }

    def signature(self) -> str:
        paths = [path for _, path in self.detector_candidates]
        if self.reid_path is not None:
            paths.append(self.reid_path)
        parts: list[str] = []
        for path in paths:
            stat = path.stat()
            parts.append(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}")
        return "|".join(parts) or "body-unavailable"

    def analyze(self, rgb: np.ndarray) -> list[BodyObservation]:
        if not self._ensure_loaded():
            return []
        try:
            if self.detector_name == "yolov8n-person":
                boxes, scores = self._detect_yolo(rgb)
            else:
                boxes, scores = self._detect_mediapipe(rgb)
        except Exception as exc:
            self._errors.append(f"detect: {exc}")
            return []

        height, width = rgb.shape[:2]
        image_area = float(max(1, height * width))
        candidates: list[tuple[tuple[int, int, int, int], float, float]] = []
        for box, score in zip(boxes[:16], scores[:16], strict=False):
            x1 = max(0, min(width - 1, int(math.floor(float(box[0])))))
            y1 = max(0, min(height - 1, int(math.floor(float(box[1])))))
            x2 = max(x1 + 1, min(width, int(math.ceil(float(box[2])))))
            y2 = max(y1 + 1, min(height, int(math.ceil(float(box[3])))))
            area_ratio = float((x2 - x1) * (y2 - y1) / image_area)
            if x2 - x1 < 18 or y2 - y1 < 28 or area_ratio < 0.0012:
                continue
            candidates.append(((x1, y1, x2, y2), float(score), area_ratio))

        embeddings = self._reid_embeddings(rgb, [item[0] for item in candidates])
        observations: list[BodyObservation] = []
        for index, (box, score, area_ratio) in enumerate(candidates):
            x1, y1, x2, y2 = box
            observations.append(
                BodyObservation(
                    bbox=(x1 / width, y1 / height, x2 / width, y2 / height),
                    confidence=score,
                    area_ratio=area_ratio,
                    embedding=embeddings[index] if index < len(embeddings) else None,
                    detector=self.detector_name,
                )
            )
        return observations

    def _detect_yolo(self, rgb: np.ndarray, input_size: int = 640) -> tuple[np.ndarray, np.ndarray]:
        height, width = rgb.shape[:2]
        scale = min(input_size / width, input_size / height)
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        resized = cv2.resize(rgb, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
        pad_x = (input_size - resized_width) // 2
        pad_y = (input_size - resized_height) // 2
        canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
        canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
        tensor = (canvas.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        session = self._detector_session
        raw = np.asarray(session.run(None, {session.get_inputs()[0].name: tensor})[0], dtype=np.float32).squeeze()
        if raw.ndim != 2:
            raise RuntimeError(f"YOLO 输出维度异常: {raw.shape}")
        rows = raw.T if raw.shape[0] <= 100 and raw.shape[1] > raw.shape[0] else raw
        if rows.shape[1] < 5:
            raise RuntimeError(f"YOLO 输出列数异常: {rows.shape}")
        person_scores = rows[:, 4]
        selected = rows[person_scores >= 0.35]
        scores = person_scores[person_scores >= 0.35]
        if selected.size == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32)
        center_x, center_y, box_width, box_height = selected[:, :4].T
        boxes = np.column_stack(
            (
                (center_x - box_width / 2.0 - pad_x) / scale,
                (center_y - box_height / 2.0 - pad_y) / scale,
                (center_x + box_width / 2.0 - pad_x) / scale,
                (center_y + box_height / 2.0 - pad_y) / scale,
            )
        ).astype(np.float32)
        keep = _nms(boxes, scores, 0.50)[:16]
        return boxes[keep], scores[keep]

    def _detect_mediapipe(self, rgb: np.ndarray, input_size: int = 224) -> tuple[np.ndarray, np.ndarray]:
        height, width = rgb.shape[:2]
        scale = min(input_size / width, input_size / height)
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        resized = cv2.resize(rgb, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
        pad_x = (input_size - resized_width) // 2
        pad_y = (input_size - resized_height) // 2
        canvas = np.zeros((input_size, input_size, 3), dtype=np.uint8)
        canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
        tensor = ((canvas.astype(np.float32) / 255.0 - 0.5) * 2.0).transpose(2, 0, 1)[None]
        session = self._detector_session
        outputs = session.run(None, {session.get_inputs()[0].name: tensor})
        score_output = next((value for value in outputs if np.asarray(value).shape[-1] == 1), None)
        box_output = next((value for value in outputs if np.asarray(value).shape[-1] == 12), None)
        if score_output is None or box_output is None:
            raise RuntimeError("MediaPipe 人体检测输出结构不匹配")
        logits = np.asarray(score_output, dtype=np.float32).reshape(-1)
        raw_boxes = np.asarray(box_output, dtype=np.float32).reshape(-1, 12)
        length = min(logits.size, raw_boxes.shape[0], self._anchors.shape[0])
        scores = 1.0 / (1.0 + np.exp(-np.clip(logits[:length], -80.0, 80.0)))
        # 远景活动照中的人物常被裁到画面边缘，采用略低于官方演示值的候选阈值，
        # 后续仍由 NMS、最小尺寸和跨照片相似度共同约束。
        positive = np.where(scores >= 0.42)[0]
        if positive.size == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32)
        raw = raw_boxes[positive]
        anchors = self._anchors[positive]
        center_x = raw[:, 0] / input_size + anchors[:, 0]
        center_y = raw[:, 1] / input_size + anchors[:, 1]
        box_width = raw[:, 2] / input_size
        box_height = raw[:, 3] / input_size
        boxes = np.column_stack(
            (
                (center_x * input_size - box_width * input_size / 2.0 - pad_x) / scale,
                (center_y * input_size - box_height * input_size / 2.0 - pad_y) / scale,
                (center_x * input_size + box_width * input_size / 2.0 - pad_x) / scale,
                (center_y * input_size + box_height * input_size / 2.0 - pad_y) / scale,
            )
        ).astype(np.float32)
        selected_scores = scores[positive].astype(np.float32)
        keep = _nms(boxes, selected_scores, 0.30)[:16]
        return boxes[keep], selected_scores[keep]

    def _reid_embeddings(
        self,
        rgb: np.ndarray,
        boxes: list[tuple[int, int, int, int]],
    ) -> list[np.ndarray | None]:
        session = self._reid_session
        if session is None or not boxes:
            return [None] * len(boxes)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        crops: list[np.ndarray] = []
        for x1, y1, x2, y2 in boxes:
            crop = rgb[y1:y2, x1:x2]
            resized = cv2.resize(crop, (128, 256), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            crops.append(((resized - mean) / std).transpose(2, 0, 1))
        input_info = session.get_inputs()[0]
        configured_batch = input_info.shape[0]
        batch_size = int(configured_batch) if isinstance(configured_batch, int) and configured_batch > 0 else len(crops)
        selected = crops[:batch_size]
        valid_count = len(selected)
        while len(selected) < batch_size:
            selected.append(np.zeros_like(crops[0]))
        tensor = np.stack(selected).astype(np.float32)
        try:
            output = np.asarray(session.run(None, {input_info.name: tensor})[0], dtype=np.float32)
        except Exception as exc:
            self._errors.append(f"reid: {exc}")
            return [None] * len(boxes)
        embeddings = [_unit(output[index]) for index in range(min(valid_count, output.shape[0]))]
        return [*embeddings, *([None] * (len(boxes) - len(embeddings)))]
