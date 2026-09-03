from __future__ import annotations

from pathlib import Path
from threading import RLock

import cv2
import numpy as np

from .config import PROJECT_ROOT, model_candidates
from .runtime import InferenceRuntime


DINO_MODEL_NAME = "dinov2_small.onnx"
MOBILENET_MODEL_NAME = "scene_mobilenetv2.onnx"


def _find_models() -> list[tuple[str, Path]]:
    roots = [PROJECT_ROOT / "models", *model_candidates()]
    found: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for kind, filename in (("dinov2-small", DINO_MODEL_NAME), ("mobilenetv2", MOBILENET_MODEL_NAME)):
        for root in roots:
            path = (root / filename).resolve()
            if path in seen or not path.is_file():
                continue
            found.append((kind, path))
            seen.add(path)
            break
    return found


def _unit(vector: np.ndarray) -> np.ndarray | None:
    flattened = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(flattened))
    return flattened / norm if norm > 1e-8 else None


def _center_crop_tensor(rgb: np.ndarray) -> np.ndarray:
    height, width = rgb.shape[:2]
    scale = 256.0 / max(1, min(height, width))
    resized = cv2.resize(
        rgb,
        (max(224, round(width * scale)), max(224, round(height * scale))),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
    )
    top = max(0, (resized.shape[0] - 224) // 2)
    left = max(0, (resized.shape[1] - 224) // 2)
    crop = resized[top : top + 224, left : left + 224].astype(np.float32) / 255.0
    crop = (crop - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return crop.transpose(2, 0, 1)[None].astype(np.float32)


class SceneEmbeddingEngine:
    """DINOv2 优先的图像向量 Provider，并提供可验证的本地降级链。"""

    def __init__(self, use_gpu: bool = True) -> None:
        self.runtime = InferenceRuntime(use_gpu=use_gpu)
        self._models = _find_models()
        self._model_index = 0
        self._session = None
        self._lock = RLock()
        self._errors: list[str] = []
        self._has_succeeded = False

    @property
    def model_kind(self) -> str:
        return self._models[self._model_index][0] if self._model_index < len(self._models) else "handcrafted"

    @property
    def model_path(self) -> Path | None:
        return self._models[self._model_index][1] if self._model_index < len(self._models) else None

    @property
    def available(self) -> bool:
        return bool(self._models and self.runtime.available)

    @property
    def backend(self) -> str:
        if self._session is not None:
            return self._session.get_providers()[0]
        return self.runtime.primary_provider if self.available else "handcrafted-fallback"

    @property
    def error(self) -> str:
        return "; ".join(self._errors)

    def signature(self) -> str:
        parts = []
        for kind, path in self._models:
            stat = path.stat()
            parts.append(f"{kind}:{stat.st_size}:{stat.st_mtime_ns}")
        return "|".join(parts) or "handcrafted"

    def _advance_model(self, message: str) -> None:
        current = self.model_path
        self._errors.append(f"{current.name if current else self.model_kind}: {message}")
        self._session = None
        self._model_index += 1

    def _ensure_session(self):
        if not self.runtime.available:
            return None
        with self._lock:
            while self._session is None and self.model_path is not None:
                try:
                    self._session = self.runtime.create_session(self.model_path)
                except Exception as exc:  # 单个模型失败时继续尝试下一 Provider。
                    self._advance_model(str(exc))
            return self._session

    def embed(self, rgb: np.ndarray) -> np.ndarray | None:
        tensor = _center_crop_tensor(rgb)
        while True:
            session = self._ensure_session()
            if session is None:
                return None
            kind = self.model_kind
            try:
                output = np.asarray(session.run(None, {session.get_inputs()[0].name: tensor})[0], dtype=np.float32)
                if kind == "dinov2-small":
                    if output.ndim != 3 or output.shape[1] < 2:
                        raise RuntimeError(f"DINOv2 输出形状异常: {output.shape}")
                    cls_token = output[0, 0]
                    patch_mean = output[0, 1:].mean(axis=0)
                    vector = 0.72 * cls_token + 0.28 * patch_mean
                else:
                    vector = output.reshape(-1)
                    vector -= float(vector.mean())
                normalized = _unit(vector)
                self._has_succeeded = normalized is not None
                return normalized
            except Exception as exc:
                with self._lock:
                    if self._has_succeeded:
                        self._errors.append(f"{self.model_kind} 单张推理失败: {exc}")
                        return None
                    self._advance_model(str(exc))

    def status(self) -> dict[str, object]:
        actual_providers = self.runtime.actual_providers(self._session)
        return {
            "available": self.available,
            "backend": self.backend,
            "providers": actual_providers or self.runtime.providers,
            "provider_source": "actual" if actual_providers else "configured",
            "cuda_preload_error": self.runtime.cuda_preload_error,
            "model": self.model_kind,
            "model_path": str(self.model_path) if self.model_path else None,
            "fallback_chain": [kind for kind, _ in self._models],
            "embedding_dimensions": 384 if self.model_kind == "dinov2-small" else 1000 if self.model_kind == "mobilenetv2" else 0,
            "errors": list(self._errors),
        }
