from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from threading import RLock

import cv2
import numpy as np
from PIL import Image

from .config import find_face_model_dir, model_candidates
from .internal_models import FaceObservation
from .runtime import InferenceRuntime


ARCFACE_TEMPLATE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)
EXPRESSION_LABELS = ("angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised")


def _unit(vector: np.ndarray) -> np.ndarray | None:
    flattened = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(flattened))
    return flattened / norm if norm > 1e-8 else None


def _softmax(vector: np.ndarray) -> np.ndarray:
    shifted = vector.astype(np.float32) - float(np.max(vector))
    exponent = np.exp(shifted)
    return exponent / max(float(exponent.sum()), 1e-8)


def _find_eye_model() -> tuple[Path | None, str]:
    """Find an optional eye-state model independently from the InsightFace pack."""
    candidates = (
        ("open-closed-eye.onnx", "openvino-open-closed-eye-0001"),
        ("open_closed_eye.onnx", "openvino-open-closed-eye-0001"),
        ("eye_state_mobilenet.onnx", "eye-state-mobilenet"),
    )
    roots = model_candidates()
    for filename, model_name in candidates:
        for root in roots:
            path = (root / filename).resolve()
            if path.is_file():
                return path, model_name
    return None, "unavailable"


def _find_optional_model(*filenames: str) -> Path | None:
    for filename in filenames:
        for root in model_candidates():
            path = (root / filename).resolve()
            if path.is_file():
                return path
    return None


def _find_optional_pack_asset(filename: str) -> Path | None:
    for root in model_candidates():
        for path in (root / "buffalo_l" / filename, root / filename):
            resolved = path.resolve()
            if resolved.is_file():
                return resolved
    return None


class FaceEngine:
    def __init__(self, use_gpu: bool = True, detection_threshold: float = 0.42) -> None:
        self.use_gpu = use_gpu
        self.runtime = InferenceRuntime(use_gpu=use_gpu)
        self.detection_threshold = detection_threshold
        self.model_root = find_face_model_dir()
        self.pack_dir = self._pack_dir(self.model_root)
        self.det_path = self.pack_dir / "det_10g.onnx" if self.pack_dir else None
        self.rec_path = self.pack_dir / "w600k_r50.onnx" if self.pack_dir else None
        self.landmark_path = self.pack_dir / "2d106det.onnx" if self.pack_dir else None
        self.landmark3d_path = self.pack_dir / "1k3d68.onnx" if self.pack_dir else None
        self.mean_shape_path = _find_optional_pack_asset("meanshape_68.json")
        self.eye_path, self.eye_model_name = _find_eye_model()
        self.expression_path = _find_optional_model(
            "facial_expression_mobilefacenet.onnx",
            "facial_expression_recognition_mobilefacenet_2022july.onnx",
        )
        self.fiqa_path = _find_optional_model("ediffiqa_tiny.onnx", "ediffiqa_tiny_jun2024.onnx")
        self._det_session = None
        self._rec_session = None
        self._landmark_session = None
        self._landmark3d_session = None
        self._eye_session = None
        self._expression_session = None
        self._fiqa_session = None
        self._mean_shape: np.ndarray | None = None
        self._lock = RLock()
        self._error = ""

    @staticmethod
    def _pack_dir(root: Path | None) -> Path | None:
        if root is None:
            return None
        nested = root / "buffalo_l"
        return nested if nested.is_dir() else root

    @property
    def available(self) -> bool:
        return bool(self.runtime.available and self.det_path and self.det_path.is_file() and self.rec_path and self.rec_path.is_file())

    def _session(self, path: Path | None):
        if not self.runtime.available or path is None or not path.is_file():
            return None
        return self.runtime.create_session(path)

    def _ensure_loaded(self) -> bool:
        if not self.available:
            return False
        with self._lock:
            if self._det_session is not None and self._rec_session is not None:
                return True
            try:
                self._det_session = self._session(self.det_path)
                self._rec_session = self._session(self.rec_path)
            except Exception as exc:
                self._error = str(exc)
                self._det_session = None
                self._rec_session = None
                return False
            optional_errors: list[str] = []
            for name, path, attribute in (
                ("landmark", self.landmark_path, "_landmark_session"),
                ("landmark3d", self.landmark3d_path, "_landmark3d_session"),
                ("eye", self.eye_path, "_eye_session"),
                ("expression", self.expression_path, "_expression_session"),
                ("fiqa", self.fiqa_path, "_fiqa_session"),
            ):
                try:
                    setattr(self, attribute, self._session(path))
                except Exception as exc:
                    optional_errors.append(f"{name}: {exc}")
            if self._landmark3d_session is not None:
                try:
                    if self.mean_shape_path is None:
                        raise FileNotFoundError("meanshape_68.json missing")
                    mean_shape = np.asarray(
                        json.loads(self.mean_shape_path.read_text(encoding="utf-8")), dtype=np.float32
                    )
                    if mean_shape.shape != (68, 3) or not np.isfinite(mean_shape).all():
                        raise ValueError(f"invalid mean shape: {mean_shape.shape}")
                    self._mean_shape = mean_shape
                except Exception as exc:
                    self._landmark3d_session = None
                    self._mean_shape = None
                    optional_errors.append(f"landmark3d-mean: {exc}")
            if optional_errors:
                self._error = "; ".join(optional_errors)
            return True

    def status(self) -> dict[str, object]:
        required = {
            "buffalo_l/det_10g.onnx": self.det_path,
            "buffalo_l/w600k_r50.onnx": self.rec_path,
            "buffalo_l/2d106det.onnx": self.landmark_path,
        }
        missing = [name for name, path in required.items() if path is None or not path.is_file()]
        actual_providers = self.runtime.actual_providers(
            self._det_session,
            self._rec_session,
            self._landmark_session,
            self._landmark3d_session,
            self._eye_session,
            self._expression_session,
            self._fiqa_session,
        )
        providers = actual_providers or self.runtime.providers
        return {
            "available": self.available,
            "backend": providers[0] if providers else "unavailable",
            "providers": providers,
            "provider_source": "actual" if actual_providers else "configured",
            "cuda_preload_error": self.runtime.cuda_preload_error,
            "model_dir": str(self.model_root) if self.model_root else None,
            "eye_model": {
                "available": bool(self.eye_path and self.eye_path.is_file()),
                "name": self.eye_model_name,
                "path": str(self.eye_path) if self.eye_path else None,
            },
            "expression_model": {
                "available": bool(self.expression_path and self.expression_path.is_file()),
                "name": "opencv-zoo-mobilefacenet-fer",
                "path": str(self.expression_path) if self.expression_path else None,
                "labels": list(EXPRESSION_LABELS),
            },
            "landmark_3d_model": {
                "available": bool(
                    self.landmark3d_path
                    and self.landmark3d_path.is_file()
                    and self.mean_shape_path
                    and self.mean_shape_path.is_file()
                ),
                "name": "insightface-1k3d68",
                "path": str(self.landmark3d_path) if self.landmark3d_path else None,
                "role": "3D head pose and profile estimation",
            },
            "face_quality_model": {
                "available": bool(self.fiqa_path and self.fiqa_path.is_file()),
                "name": "opencv-zoo-ediffiqa-tiny",
                "path": str(self.fiqa_path) if self.fiqa_path else None,
                "role": "aligned-face perceptual quality",
            },
            "missing_models": missing,
            "error": self._error or None,
        }

    def signature(self) -> str:
        paths = (
            self.det_path,
            self.rec_path,
            self.landmark_path,
            self.landmark3d_path,
            self.mean_shape_path,
            self.eye_path,
            self.expression_path,
            self.fiqa_path,
        )
        parts = []
        for path in paths:
            if path is None or not path.is_file():
                continue
            stat = path.stat()
            parts.append(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}")
        return "|".join(parts) or "face-unavailable"

    def analyze(
        self,
        rgb: np.ndarray,
        photo_identifier: str,
        original_image: Image.Image | None = None,
    ) -> list[FaceObservation]:
        if not self._ensure_loaded():
            return []
        try:
            boxes, landmarks, scores = self._detect(rgb)
        except Exception as exc:
            self._error = f"人脸检测失败: {exc}"
            return []

        image_height, image_width = rgb.shape[:2]
        image_area = float(image_height * image_width)
        observations: list[FaceObservation] = []
        for index, (bbox, points5, confidence) in enumerate(zip(boxes[:24], landmarks[:24], scores[:24], strict=False)):
            x1, y1, x2, y2 = self._clip_box(bbox, image_width, image_height)
            if x2 - x1 < 12 or y2 - y1 < 12:
                continue
            area_ratio = float((x2 - x1) * (y2 - y1) / image_area)
            if area_ratio < 0.00018:
                continue
            aligned = self._align_face(rgb, points5)
            use_detailed_face_models = area_ratio >= 0.0006 and float(confidence) >= 0.70
            detailed_aligned = (
                self._align_original_face(original_image, points5, image_width, image_height)
                if use_detailed_face_models and original_image is not None
                else aligned
            )
            embedding_source = detailed_aligned if use_detailed_face_models else aligned
            embedding = self._embedding(embedding_source) if embedding_source is not None else None
            landmarks106 = self._landmarks106(rgb, (x1, y1, x2, y2))
            eye_state, open_probability = (
                self._eye_state(detailed_aligned, landmarks106)
                if use_detailed_face_models
                else ("Unknown", None)
            )
            pose3d = self._pose3d(rgb, (x1, y1, x2, y2)) if use_detailed_face_models else None
            yaw, pitch, roll = pose3d if pose3d is not None else self._pose(points5)
            profile = self._is_profile(points5) or abs(yaw) >= 38.0
            face_crop = rgb[y1:y2, x1:x2]
            sharpness = self._sharpness(face_crop)
            landmark_smile = self._smile_score(landmarks106, x2 - x1)
            expression, expression_confidence, expression_score, happy_probability = self._expression(
                detailed_aligned if use_detailed_face_models else None
            )
            fiqa_score = self._fiqa(detailed_aligned) if use_detailed_face_models else None
            smile = (
                0.62 * happy_probability * 100.0 + 0.38 * landmark_smile
                if happy_probability is not None
                else landmark_smile
            )
            normalized_bbox = (x1 / image_width, y1 / image_height, x2 / image_width, y2 / image_height)
            occlusion_risk = self._occlusion_risk(float(confidence), normalized_bbox, yaw, profile)
            digest = hashlib.sha1(f"{photo_identifier}:{index}:{normalized_bbox}".encode()).hexdigest()[:16]
            observations.append(
                FaceObservation(
                    face_id=digest,
                    bbox=normalized_bbox,
                    confidence=float(confidence),
                    area_ratio=area_ratio,
                    embedding=embedding,
                    eye_state=eye_state,
                    open_probability=open_probability,
                    sharpness=sharpness,
                    profile=profile,
                    smile_score=smile,
                    yaw=yaw,
                    pitch=pitch,
                    roll=roll,
                    occlusion_risk=occlusion_risk,
                    expression=expression,
                    expression_confidence=expression_confidence,
                    expression_score=expression_score,
                    fiqa_score=fiqa_score,
                )
            )
        return observations

    def _expression(self, aligned_rgb: np.ndarray | None) -> tuple[str, float | None, float, float | None]:
        session = self._expression_session
        if aligned_rgb is None or session is None:
            return "unknown", None, 0.0, None
        resized = cv2.resize(aligned_rgb, (112, 112), interpolation=cv2.INTER_AREA).astype(np.float32)
        tensor = ((resized / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)[None]
        try:
            output = session.run(None, {session.get_inputs()[0].name: tensor})[0]
        except Exception as exc:
            self._error = f"expression: {exc}"
            return "unknown", None, 0.0, None
        values = np.asarray(output, dtype=np.float32).reshape(-1)
        if values.size < len(EXPRESSION_LABELS) or not np.isfinite(values[: len(EXPRESSION_LABELS)]).all():
            return "unknown", None, 0.0, None
        values = values[: len(EXPRESSION_LABELS)]
        values_sum = float(values.sum())
        normalized = bool(values.min() >= 0.0 and values.max() <= 1.0 and abs(values_sum - 1.0) <= 0.02)
        probabilities = values / max(values_sum, 1e-8) if normalized else _softmax(values)
        index = int(np.argmax(probabilities))
        # “最佳表情”不等同于只选笑脸：惊喜和自然中性仍保留较高分，负面表情只做温和降权。
        desirability = np.array([0.62, 0.32, 0.38, 1.00, 0.72, 0.46, 0.84], dtype=np.float32)
        score = float(np.dot(probabilities, desirability) * 100.0)
        return (
            EXPRESSION_LABELS[index],
            float(np.clip(probabilities[index], 0.0, 1.0)),
            score,
            float(np.clip(probabilities[3], 0.0, 1.0)),
        )

    def _fiqa(self, aligned_rgb: np.ndarray | None) -> float | None:
        session = self._fiqa_session
        if aligned_rgb is None or session is None:
            return None
        resized = cv2.resize(aligned_rgb, (112, 112), interpolation=cv2.INTER_AREA).astype(np.float32)
        tensor = ((resized / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)[None]
        try:
            output = session.run(None, {session.get_inputs()[0].name: tensor})[0]
        except Exception as exc:
            self._error = f"fiqa: {exc}"
            return None
        values = np.asarray(output, dtype=np.float32).reshape(-1)
        if values.size != 1 or not np.isfinite(values[0]):
            return None
        return float(np.clip(values[0], 0.0, 1.0) * 100.0)

    @staticmethod
    def _clip_box(bbox: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
        x1 = max(0, min(width - 1, int(math.floor(float(bbox[0])))))
        y1 = max(0, min(height - 1, int(math.floor(float(bbox[1])))))
        x2 = max(x1 + 1, min(width, int(math.ceil(float(bbox[2])))))
        y2 = max(y1 + 1, min(height, int(math.ceil(float(bbox[3])))))
        return x1, y1, x2, y2

    def _detect(self, rgb: np.ndarray, input_size: tuple[int, int] = (640, 640)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        height, width = rgb.shape[:2]
        scale = min(input_size[0] / width, input_size[1] / height)
        resized_width, resized_height = max(1, round(width * scale)), max(1, round(height * scale))
        resized = cv2.resize(rgb, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
        canvas = np.full((input_size[1], input_size[0], 3), 127, dtype=np.uint8)
        canvas[:resized_height, :resized_width] = resized
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128.0, input_size, (127.5, 127.5, 127.5), swapRB=False)
        session = self._det_session
        outputs = session.run(None, {session.get_inputs()[0].name: blob})
        strides = (8, 16, 32)
        feature_count = len(strides)
        if len(outputs) < feature_count * 3:
            raise RuntimeError(f"SCRFD 输出数量异常: {len(outputs)}")

        score_outputs = outputs[:feature_count]
        bbox_outputs = outputs[feature_count : feature_count * 2]
        landmark_outputs = outputs[feature_count * 2 : feature_count * 3]
        all_boxes: list[np.ndarray] = []
        all_landmarks: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []
        for stride, score_output, bbox_output, landmark_output in zip(strides, score_outputs, bbox_outputs, landmark_outputs, strict=True):
            scores = np.asarray(score_output).reshape(-1)
            bbox_predictions = np.asarray(bbox_output).reshape(-1, 4) * stride
            landmark_predictions = np.asarray(landmark_output).reshape(-1, 10) * stride
            feature_height, feature_width = input_size[1] // stride, input_size[0] // stride
            anchors = max(1, scores.size // (feature_height * feature_width))
            grid_x, grid_y = np.meshgrid(np.arange(feature_width), np.arange(feature_height))
            centers = np.stack((grid_x, grid_y), axis=-1).astype(np.float32).reshape(-1, 2) * stride
            if anchors > 1:
                centers = np.repeat(centers, anchors, axis=0)
            length = min(scores.size, bbox_predictions.shape[0], landmark_predictions.shape[0], centers.shape[0])
            scores = scores[:length]
            bbox_predictions = bbox_predictions[:length]
            landmark_predictions = landmark_predictions[:length]
            centers = centers[:length]
            positive = np.where(scores >= self.detection_threshold)[0]
            if positive.size == 0:
                continue
            distances = bbox_predictions[positive]
            selected_centers = centers[positive]
            boxes = np.column_stack(
                (
                    selected_centers[:, 0] - distances[:, 0],
                    selected_centers[:, 1] - distances[:, 1],
                    selected_centers[:, 0] + distances[:, 2],
                    selected_centers[:, 1] + distances[:, 3],
                )
            ) / scale
            points = landmark_predictions[positive].reshape(-1, 5, 2)
            points = (points + selected_centers[:, None, :]) / scale
            all_boxes.append(boxes)
            all_landmarks.append(points)
            all_scores.append(scores[positive])

        if not all_boxes:
            return np.empty((0, 4), dtype=np.float32), np.empty((0, 5, 2), dtype=np.float32), np.empty(0, dtype=np.float32)
        boxes = np.vstack(all_boxes).astype(np.float32)
        landmarks = np.vstack(all_landmarks).astype(np.float32)
        scores = np.concatenate(all_scores).astype(np.float32)
        keep = self._nms(boxes, scores, 0.4)
        order = sorted(keep, key=lambda position: float(scores[position]), reverse=True)
        return boxes[order], landmarks[order], scores[order]

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
        x1, y1, x2, y2 = boxes.T
        areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        order = scores.argsort()[::-1]
        keep: list[int] = []
        while order.size:
            index = int(order[0])
            keep.append(index)
            if order.size == 1:
                break
            xx1 = np.maximum(x1[index], x1[order[1:]])
            yy1 = np.maximum(y1[index], y1[order[1:]])
            xx2 = np.minimum(x2[index], x2[order[1:]])
            yy2 = np.minimum(y2[index], y2[order[1:]])
            overlap = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            union = areas[index] + areas[order[1:]] - overlap
            iou = overlap / np.maximum(union, 1e-8)
            order = order[np.where(iou <= threshold)[0] + 1]
        return keep

    @staticmethod
    def _align_face(rgb: np.ndarray, landmarks5: np.ndarray) -> np.ndarray | None:
        matrix, _ = cv2.estimateAffinePartial2D(landmarks5.astype(np.float32), ARCFACE_TEMPLATE, method=cv2.LMEDS)
        if matrix is None:
            return None
        return cv2.warpAffine(rgb, matrix, (112, 112), borderValue=0)

    @staticmethod
    def _align_original_face(
        original_image: Image.Image,
        landmarks5: np.ndarray,
        analysis_width: int,
        analysis_height: int,
    ) -> np.ndarray | None:
        """从原分辨率局部裁切对齐人脸，避免小脸先缩小再放大造成睁眼误判。"""

        original_width, original_height = original_image.size
        if min(original_width, original_height, analysis_width, analysis_height) <= 0:
            return None
        scale = np.array(
            [original_width / float(analysis_width), original_height / float(analysis_height)],
            dtype=np.float32,
        )
        source_points = landmarks5.astype(np.float32) * scale
        matrix, _ = cv2.estimateAffinePartial2D(source_points, ARCFACE_TEMPLATE, method=cv2.LMEDS)
        if matrix is None:
            return None
        inverse = cv2.invertAffineTransform(matrix)
        target_corners = np.float32([[0, 0, 1], [111, 0, 1], [111, 111, 1], [0, 111, 1]])
        source_corners = target_corners @ inverse.T
        left = max(0, int(math.floor(float(source_corners[:, 0].min()))) - 3)
        top = max(0, int(math.floor(float(source_corners[:, 1].min()))) - 3)
        right = min(original_width, int(math.ceil(float(source_corners[:, 0].max()))) + 4)
        bottom = min(original_height, int(math.ceil(float(source_corners[:, 1].max()))) + 4)
        if right - left < 12 or bottom - top < 12:
            return None
        crop = np.asarray(original_image.crop((left, top, right, bottom)).convert("RGB"), dtype=np.uint8)
        local_points = source_points - np.array([left, top], dtype=np.float32)
        local_matrix, _ = cv2.estimateAffinePartial2D(local_points, ARCFACE_TEMPLATE, method=cv2.LMEDS)
        if local_matrix is None:
            return None
        return cv2.warpAffine(crop, local_matrix, (112, 112), borderValue=0)

    def _embedding(self, aligned_rgb: np.ndarray) -> np.ndarray | None:
        blob = ((aligned_rgb.astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None]
        session = self._rec_session
        output = session.run(None, {session.get_inputs()[0].name: blob})[0]
        return _unit(output)

    @staticmethod
    def _landmark_crop(
        rgb: np.ndarray, bbox: tuple[int, int, int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        x1, y1, x2, y2 = bbox
        width, height = x2 - x1, y2 - y1
        center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        side = max(width, height) * 1.5
        source = np.float32([[center_x - side / 2, center_y - side / 2], [center_x + side / 2, center_y - side / 2], [center_x - side / 2, center_y + side / 2]])
        target = np.float32([[0, 0], [192, 0], [0, 192]])
        matrix = cv2.getAffineTransform(source, target)
        crop = cv2.warpAffine(rgb, matrix, (192, 192), borderValue=0)
        return crop, matrix

    def _landmarks106(self, rgb: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
        session = self._landmark_session
        if session is None:
            return None
        crop, matrix = self._landmark_crop(rgb, bbox)
        # buffalo_l 的坐标回归 ONNX 已内置归一化；官方实现直接输入 RGB 0..255。
        tensor = crop.astype(np.float32).transpose(2, 0, 1)[None]
        prediction = session.run(None, {session.get_inputs()[0].name: tensor})[0]
        points = np.asarray(prediction, dtype=np.float32).reshape(-1, 2)
        points = (points + 1.0) * 96.0
        inverse = cv2.invertAffineTransform(matrix)
        homogeneous = np.column_stack((points, np.ones(points.shape[0], dtype=np.float32)))
        return homogeneous @ inverse.T

    def _pose3d(
        self, rgb: np.ndarray, bbox: tuple[int, int, int, int]
    ) -> tuple[float, float, float] | None:
        session = self._landmark3d_session
        mean_shape = self._mean_shape
        if session is None or mean_shape is None:
            return None
        crop, matrix = self._landmark_crop(rgb, bbox)
        tensor = crop.astype(np.float32).transpose(2, 0, 1)[None]
        try:
            output = session.run(None, {session.get_inputs()[0].name: tensor})[0]
        except Exception as exc:
            self._error = f"landmark3d: {exc}"
            return None
        values = np.asarray(output, dtype=np.float32).reshape(-1)
        if values.size < 68 * 3 or values.size % 3 or not np.isfinite(values).all():
            return None
        points = values.reshape(-1, 3)[-68:].copy()
        points[:, :2] = (points[:, :2] + 1.0) * 96.0
        points[:, 2] *= 96.0
        inverse = cv2.invertAffineTransform(matrix)
        homogeneous = np.column_stack((points[:, :2], np.ones(points.shape[0], dtype=np.float32)))
        transformed = np.empty_like(points)
        transformed[:, :2] = homogeneous @ inverse.T
        inverse_scale = math.hypot(float(inverse[0, 0]), float(inverse[0, 1]))
        transformed[:, 2] = points[:, 2] * inverse_scale

        source = np.column_stack((mean_shape, np.ones(mean_shape.shape[0], dtype=np.float32)))
        try:
            projection = np.linalg.lstsq(source, transformed, rcond=None)[0].T
        except np.linalg.LinAlgError:
            return None
        first = projection[0, :3]
        second = projection[1, :3]
        first_norm = float(np.linalg.norm(first))
        second_norm = float(np.linalg.norm(second))
        if first_norm <= 1e-8 or second_norm <= 1e-8:
            return None
        rotation = np.vstack((first / first_norm, second / second_norm, np.cross(first / first_norm, second / second_norm)))
        sy = math.hypot(float(rotation[0, 0]), float(rotation[1, 0]))
        if sy >= 1e-6:
            pitch = math.degrees(math.atan2(float(rotation[2, 1]), float(rotation[2, 2])))
            yaw = math.degrees(math.atan2(float(-rotation[2, 0]), sy))
            roll = math.degrees(math.atan2(float(rotation[1, 0]), float(rotation[0, 0])))
        else:
            pitch = math.degrees(math.atan2(float(-rotation[1, 2]), float(rotation[1, 1])))
            yaw = math.degrees(math.atan2(float(-rotation[2, 0]), sy))
            roll = 0.0
        if not np.isfinite((pitch, yaw, roll)).all():
            return None
        return (
            float(np.clip(yaw, -90.0, 90.0)),
            float(np.clip(pitch, -90.0, 90.0)),
            float(np.clip(roll, -90.0, 90.0)),
        )

    def _eye_state(self, aligned_rgb: np.ndarray | None, landmarks106: np.ndarray | None) -> tuple[str, float | None]:
        cnn_probabilities: list[float] = []
        if aligned_rgb is not None and self._eye_session is not None:
            for center_x in (38, 74):
                crop = aligned_rgb[39:63, max(0, center_x - 17) : min(112, center_x + 17)]
                if crop.size == 0:
                    continue
                if self.eye_model_name == "openvino-open-closed-eye-0001":
                    resized = cv2.resize(crop, (32, 32), interpolation=cv2.INTER_CUBIC)
                    bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR).astype(np.float32)
                    tensor = ((bgr - 127.0) / 255.0).transpose(2, 0, 1)[None]
                    open_class = 0
                else:
                    resized = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
                    normalized = (resized - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
                        [0.229, 0.224, 0.225], dtype=np.float32
                    )
                    tensor = normalized.transpose(2, 0, 1)[None]
                    open_class = 1
                output = self._eye_session.run(None, {self._eye_session.get_inputs()[0].name: tensor})[0]
                values = np.asarray(output, dtype=np.float32).reshape(-1)
                if values.size >= 2:
                    values_sum = float(values.sum())
                    looks_normalized = bool(
                        np.isfinite(values).all()
                        and values.min() >= 0.0
                        and values.max() <= 1.0
                        and abs(values_sum - 1.0) <= 0.02
                    )
                    probabilities = values / max(values_sum, 1e-8) if looks_normalized else _softmax(values)
                    cnn_probabilities.append(float(np.clip(probabilities[open_class], 0.0, 1.0)))
                elif values.size == 1 and np.isfinite(values[0]):
                    cnn_probabilities.append(float(np.clip(values[0], 0.0, 1.0)))

        ear_values = self._ear_values(landmarks106)
        cnn_min = min(cnn_probabilities) if cnn_probabilities else None
        cnn_avg = float(np.mean(cnn_probabilities)) if cnn_probabilities else None
        ear_min = min(ear_values) if ear_values else None
        if cnn_avg is not None:
            if cnn_min is not None and cnn_min < 0.24 and cnn_avg < 0.48:
                state = "Closed"
            elif cnn_min is not None and (cnn_min < 0.58 or cnn_avg < 0.67):
                state = "Partial"
            else:
                state = "Open"
            if ear_min is not None:
                if state == "Closed" and ear_min >= 0.25:
                    state = "Partial"
                elif state == "Open" and ear_min < 0.13:
                    state = "Partial"
            return state, cnn_avg
        if ear_min is not None:
            if ear_min >= 0.25:
                return "Open", min(1.0, ear_min / 0.32)
            if ear_min >= 0.13:
                return "Partial", min(0.75, ear_min / 0.32)
            return "Closed", min(0.35, ear_min / 0.32)
        return "Unknown", None

    @staticmethod
    def _ear_values(points: np.ndarray | None) -> list[float]:
        if points is None or points.shape[0] < 106:
            return []
        groups = (list(range(62, 72)), [74, 76, 77, 78, 79, 80, 82, 83, 84, 85])
        values: list[float] = []
        for indices in groups:
            eye = points[indices]
            mean_y = float(eye[:, 1].mean())
            upper = eye[eye[:, 1] < mean_y]
            lower = eye[eye[:, 1] >= mean_y]
            width = float(eye[:, 0].max() - eye[:, 0].min())
            if upper.size and lower.size and width > 1e-5:
                values.append(abs(float(lower[:, 1].mean() - upper[:, 1].mean())) / width)
        return values

    @staticmethod
    def _is_profile(points5: np.ndarray) -> bool:
        left_eye, right_eye, nose = points5[:3]
        eye_distance = float(np.linalg.norm(right_eye - left_eye))
        if eye_distance < 1e-5:
            return False
        midpoint = (left_eye + right_eye) / 2.0
        nose_offset = abs(float(nose[0] - midpoint[0])) / eye_distance
        left_distance = float(np.linalg.norm(nose - left_eye))
        right_distance = float(np.linalg.norm(nose - right_eye))
        asymmetry = min(left_distance, right_distance) / max(left_distance, right_distance, 1e-5)
        return nose_offset > 0.42 or asymmetry < 0.60

    @staticmethod
    def _pose(points5: np.ndarray) -> tuple[float, float, float]:
        left_eye, right_eye, nose, left_mouth, right_mouth = points5
        eye_vector = right_eye - left_eye
        eye_distance = max(float(np.linalg.norm(eye_vector)), 1e-5)
        roll = math.degrees(math.atan2(float(eye_vector[1]), float(eye_vector[0])))
        eye_midpoint = (left_eye + right_eye) / 2.0
        mouth_midpoint = (left_mouth + right_mouth) / 2.0
        yaw = float(np.clip((nose[0] - eye_midpoint[0]) / eye_distance * 78.0, -65.0, 65.0))
        vertical_span = max(float(mouth_midpoint[1] - eye_midpoint[1]), 1e-5)
        nose_ratio = float((nose[1] - eye_midpoint[1]) / vertical_span)
        pitch = float(np.clip((nose_ratio - 0.48) * 75.0, -40.0, 40.0))
        return yaw, pitch, roll

    @staticmethod
    def _occlusion_risk(
        confidence: float,
        bbox: tuple[float, float, float, float],
        yaw: float,
        profile: bool,
    ) -> float:
        boundary = min(bbox[0], bbox[1], 1.0 - bbox[2], 1.0 - bbox[3])
        boundary_risk = float(np.clip((0.015 - boundary) / 0.015, 0.0, 1.0))
        confidence_risk = float(np.clip((0.82 - confidence) / 0.40, 0.0, 1.0))
        pose_risk = float(np.clip((abs(yaw) - 30.0) / 35.0, 0.0, 1.0))
        if profile:
            pose_risk = max(pose_risk, 0.35)
        return float(np.clip(0.52 * boundary_risk + 0.30 * confidence_risk + 0.18 * pose_risk, 0.0, 1.0))

    @staticmethod
    def _sharpness(face_crop: np.ndarray) -> float:
        if face_crop.size == 0 or min(face_crop.shape[:2]) < 10:
            return 0.0
        gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
        if max(gray.shape) > 240:
            scale = 240.0 / max(gray.shape)
            gray = cv2.resize(gray, (max(1, round(gray.shape[1] * scale)), max(1, round(gray.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _smile_score(points: np.ndarray | None, face_width: float) -> float:
        if points is None or points.shape[0] < 98 or face_width <= 0:
            return 0.0
        mouth = points[86:98]
        width_ratio = float((mouth[:, 0].max() - mouth[:, 0].min()) / face_width)
        height_ratio = float((mouth[:, 1].max() - mouth[:, 1].min()) / face_width)
        width_signal = np.clip((width_ratio - 0.30) / 0.18, 0.0, 1.0)
        height_signal = np.clip((height_ratio - 0.07) / 0.12, 0.0, 1.0)
        return float((0.72 * width_signal + 0.28 * height_signal) * 100.0)
