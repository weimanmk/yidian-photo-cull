from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import DATA_DIR
from .face_quality import select_quality_faces
from .internal_models import PhotoObservation


PREFERENCE_MODEL_VERSION = "1.1"
PREFERENCE_MODEL_PATH = DATA_DIR / "preference-model.json"
PREFERENCE_FEATURES = (
    "sharpness_score",
    "motion_blur_score",
    "exposure_score",
    "contrast_score",
    "noise_score",
    "composition_score",
    "face_quality_score",
    "min_face_score",
    "face_sharpness_score",
    "eye_sharpness_score",
    "eye_score",
    "expression_score",
    "technical_score",
    "generic_group_score",
    "group_relative_score",
    "group_quality_percentile",
    "group_size",
    "relative_face_quality",
    "relative_eye",
    "relative_motion",
    "relative_exposure",
    "relative_composition",
    "bad_face_count",
    "face_count",
    "largest_face_ratio",
    "mean_occlusion_risk",
    "mean_abs_yaw",
    "mean_smile_score",
    "portrait_present",
)


def preference_features(photo: PhotoObservation) -> np.ndarray:
    metrics = photo.metrics
    faces = photo.faces
    quality_faces = select_quality_faces(faces)
    values = {
        **metrics,
        "face_count": float(len(faces)),
        "largest_face_ratio": max((face.area_ratio for face in quality_faces), default=0.0) * 1000.0,
        "mean_occlusion_risk": (
            float(np.mean([face.occlusion_risk for face in quality_faces])) * 100.0 if quality_faces else 0.0
        ),
        "mean_abs_yaw": float(np.mean([abs(face.yaw) for face in quality_faces])) if quality_faces else 0.0,
        "mean_smile_score": float(np.mean([face.smile_score for face in quality_faces])) if quality_faces else 0.0,
        "portrait_present": float(bool(quality_faces)),
    }
    return np.array([float(values.get(name, 0.0)) for name in PREFERENCE_FEATURES], dtype=np.float64)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _best_threshold(probabilities: np.ndarray, labels: np.ndarray) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    for threshold in np.linspace(0.12, 0.88, 153):
        predicted = probabilities >= threshold
        true_positive = int(np.sum(predicted & (labels == 1)))
        false_positive = int(np.sum(predicted & (labels == 0)))
        false_negative = int(np.sum(~predicted & (labels == 1)))
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
        candidate = (f1, precision, -abs(float(threshold) - 0.5))
        current = (best_metrics["f1"], best_metrics["precision"], -abs(best_threshold - 0.5))
        if candidate > current:
            best_threshold = float(threshold)
            best_metrics = {"precision": precision, "recall": recall, "f1": f1}
    return best_threshold, best_metrics


def best_preference_threshold(probabilities: np.ndarray, labels: np.ndarray) -> tuple[float, dict[str, float]]:
    """从独立验证预测中选择 F1 最优阈值，供训练脚本复用。"""
    return _best_threshold(
        np.asarray(probabilities, dtype=np.float64).reshape(-1),
        np.asarray(labels, dtype=np.int8).reshape(-1),
    )


@dataclass(slots=True)
class PreferenceModel:
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float
    threshold: float
    metadata: dict[str, object]

    @property
    def available(self) -> bool:
        expected = len(PREFERENCE_FEATURES)
        return bool(
            self.feature_names == PREFERENCE_FEATURES
            and self.mean.size == expected
            and self.scale.size == expected
            and self.weights.size == expected
            and np.isfinite(self.mean).all()
            and np.isfinite(self.scale).all()
            and np.all(self.scale > 0.0)
            and np.isfinite(self.weights).all()
            and np.isfinite(self.bias)
            and np.isfinite(self.threshold)
        )

    def predict_probability(self, photo: PhotoObservation) -> float:
        vector = preference_features(photo)
        standardized = (vector - self.mean) / self.scale
        return float(_sigmoid(np.array([standardized @ self.weights + self.bias]))[0])

    def predict_probabilities(self, photos: Iterable[PhotoObservation]) -> np.ndarray:
        return np.asarray([self.predict_probability(photo) for photo in photos], dtype=np.float64)

    def applies_to_source(self, source_root: Path) -> bool:
        if self.metadata.get("scope") == "global":
            return True
        trained_source = self.metadata.get("source_dir")
        if not isinstance(trained_source, str) or not trained_source.strip():
            return False
        try:
            return Path(trained_source).expanduser().resolve() == source_root.expanduser().resolve()
        except OSError:
            return False

    def apply(self, photos: Iterable[PhotoObservation]) -> None:
        threshold_score = self.threshold * 100.0
        strength = float(np.clip(self.metadata.get("ranking_strength", 0.0), 0.0, 1.0))
        blend_weight = float(np.clip(self.metadata.get("blend_weight", 0.0), 0.0, 0.65))
        selection_enabled = bool(self.metadata.get("selection_filter_enabled", False))
        for photo in photos:
            photo.metrics["preference_score"] = self.predict_probability(photo) * 100.0
            photo.metrics["preference_threshold"] = threshold_score
            photo.metrics["preference_strength"] = strength
            photo.metrics["preference_weight"] = blend_weight
            photo.metrics["preference_selection_enabled"] = float(selection_enabled)

    def save(self, path: Path | None = None) -> Path:
        destination = path or PREFERENCE_MODEL_PATH
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": PREFERENCE_MODEL_VERSION,
            "feature_names": list(self.feature_names),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "threshold": self.threshold,
            "metadata": self.metadata,
        }
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(destination)
        return destination

    def status(self) -> dict[str, object]:
        return {
            "available": self.available,
            "version": PREFERENCE_MODEL_VERSION,
            "threshold": round(self.threshold, 4),
            "feature_count": len(self.feature_names),
            "ranking_strength": round(float(self.metadata.get("ranking_strength", 0.0)), 4),
            "blend_weight": round(float(self.metadata.get("blend_weight", 0.0)), 4),
            "selection_filter_enabled": bool(self.metadata.get("selection_filter_enabled", False)),
            "metadata": self.metadata,
        }

    @classmethod
    def load(cls, path: Path | None = None) -> "PreferenceModel | None":
        source = path or PREFERENCE_MODEL_PATH
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
            if payload.get("version") != PREFERENCE_MODEL_VERSION:
                return None
            feature_names = tuple(payload["feature_names"])
            if feature_names != PREFERENCE_FEATURES:
                return None
            model = cls(
                feature_names=feature_names,
                mean=np.asarray(payload["mean"], dtype=np.float64),
                scale=np.asarray(payload["scale"], dtype=np.float64),
                weights=np.asarray(payload["weights"], dtype=np.float64),
                bias=float(payload["bias"]),
                threshold=float(payload["threshold"]),
                metadata=dict(payload.get("metadata", {})),
            )
            return model if model.available else None
        except (OSError, ValueError, TypeError, KeyError):
            return None


def fit_preference_model(
    photos: list[PhotoObservation],
    labels: np.ndarray,
    *,
    iterations: int = 1400,
    l2: float = 0.045,
) -> PreferenceModel:
    if len(photos) != len(labels) or len(photos) < 20:
        raise ValueError("偏好模型至少需要 20 张带标签照片")
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    positive = int(np.sum(labels == 1))
    negative = int(np.sum(labels == 0))
    if positive < 5 or negative < 5:
        raise ValueError("偏好模型至少需要 5 张保留和 5 张淘汰样本")

    matrix = np.vstack([preference_features(photo) for photo in photos])
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-6] = 1.0
    standardized = (matrix - mean) / scale
    sample_weights = np.where(labels == 1, len(labels) / (2.0 * positive), len(labels) / (2.0 * negative))
    weights, bias = _fit_logistic(standardized, labels, sample_weights, iterations=iterations, l2=l2)

    probabilities = _sigmoid(standardized @ weights + bias)
    threshold, training_metrics = _best_threshold(probabilities, labels.astype(np.int8))
    metadata: dict[str, object] = {
        "trained_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "samples": len(photos),
        "positive": positive,
        "negative": negative,
        "objective": "pointwise",
        "training_metrics": {key: round(value, 4) for key, value in training_metrics.items()},
    }
    return PreferenceModel(PREFERENCE_FEATURES, mean, scale, weights, bias, threshold, metadata)


def _fit_logistic(
    matrix: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
    *,
    iterations: int,
    l2: float,
) -> tuple[np.ndarray, float]:
    weights = np.zeros(matrix.shape[1], dtype=np.float64)
    bias = 0.0
    normalizer = float(sample_weights.sum())
    for step in range(iterations):
        probabilities = _sigmoid(matrix @ weights + bias)
        error = (probabilities - labels) * sample_weights
        gradient = matrix.T @ error / normalizer + l2 * weights
        bias_gradient = float(error.sum() / normalizer)
        learning_rate = 0.11 / (1.0 + step / 520.0)
        weights -= learning_rate * gradient
        bias -= learning_rate * bias_gradient
    return weights, bias


def fit_pairwise_preference_model(
    photos: list[PhotoObservation],
    labels: np.ndarray,
    *,
    iterations: int = 1400,
    l2: float = 0.045,
) -> PreferenceModel:
    """从同组“人工保留 > 人工淘汰”对中学习可比较的组内效用。"""
    if len(photos) != len(labels) or len(photos) < 20:
        raise ValueError("成对偏好模型至少需要 20 张带标签照片")
    labels = np.asarray(labels, dtype=np.int8).reshape(-1)
    matrix = np.vstack([preference_features(photo) for photo in photos])
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-6] = 1.0
    standardized = (matrix - mean) / scale
    groups: dict[str, list[int]] = {}
    for index, photo in enumerate(photos):
        groups.setdefault(photo.group_id, []).append(index)

    pair_vectors: list[np.ndarray] = []
    pair_labels: list[float] = []
    pair_weights: list[float] = []
    comparable_groups = 0
    for members in groups.values():
        positive = [index for index in members if labels[index] == 1]
        negative = [index for index in members if labels[index] == 0]
        if not positive or not negative:
            continue
        comparable_groups += 1
        group_weight = 1.0 / (len(positive) * len(negative))
        for preferred in positive:
            for rejected in negative:
                difference = standardized[preferred] - standardized[rejected]
                pair_vectors.extend((difference, -difference))
                pair_labels.extend((1.0, 0.0))
                pair_weights.extend((group_weight, group_weight))
    if comparable_groups < 10 or len(pair_vectors) < 40:
        raise ValueError("可比较的人工保留/淘汰照片组不足")

    pair_matrix = np.vstack(pair_vectors)
    pair_label_array = np.asarray(pair_labels, dtype=np.float64)
    weights, bias = _fit_logistic(
        pair_matrix,
        pair_label_array,
        np.asarray(pair_weights, dtype=np.float64),
        iterations=iterations,
        l2=l2,
    )
    metadata: dict[str, object] = {
        "trained_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "samples": len(photos),
        "positive": int(np.sum(labels == 1)),
        "negative": int(np.sum(labels == 0)),
        "objective": "pairwise",
        "comparable_groups": comparable_groups,
        "training_pairs": len(pair_vectors) // 2,
    }
    return PreferenceModel(PREFERENCE_FEATURES, mean, scale, weights, bias, 0.5, metadata)
