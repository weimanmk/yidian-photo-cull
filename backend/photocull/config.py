from __future__ import annotations

import json
import os
import platform
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from threading import RLock
from typing import Any
from urllib.parse import urlparse


APP_NAME = "YidianPhotoCull"
HOST = "127.0.0.1"
PORT = int(os.getenv("PHOTOCULL_PORT", "8767"))
VLM_DISABLED = os.getenv("PHOTOCULL_DISABLE_VLM", "0").strip().casefold() in {"1", "true", "yes", "on"}

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
    ".heic", ".heif", ".cr2", ".cr3", ".nef", ".arw", ".raf",
    ".dng", ".rw2", ".orf", ".srw", ".pef",
}
RAW_EXTENSIONS = {".cr2", ".cr3", ".nef", ".arw", ".raf", ".dng", ".rw2", ".orf", ".srw", ".pef"}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VLM_SERVER_URL = "http://127.0.0.1:8768"


def lightroom_bridge_root() -> Path:
    configured = os.getenv("PHOTOCULL_LIGHTROOM_BRIDGE_DIR")
    if configured:
        return Path(configured).expanduser().resolve(strict=False)
    if platform.system() == "Windows":
        base = Path(os.getenv("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif platform.system() == "Darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path.home() / ".local" / "share"
    return (base / "Adobe" / "Lightroom" / APP_NAME / "lightroom-bridge").resolve(strict=False)


def _loopback_url(value: str) -> str:
    try:
        parsed = urlparse(str(value).strip())
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
            return DEFAULT_VLM_SERVER_URL
        port = parsed.port or 8768
        if not 1 <= port <= 65535:
            return DEFAULT_VLM_SERVER_URL
        return f"http://{parsed.hostname}:{port}"
    except (TypeError, ValueError):
        return DEFAULT_VLM_SERVER_URL


def _data_dir() -> Path:
    configured = os.getenv("PHOTOCULL_DATA_DIR")
    if configured:
        target = Path(configured).expanduser().resolve()
    elif platform.system() == "Windows":
        target = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / APP_NAME
    elif platform.system() == "Darwin":
        target = Path.home() / "Library" / "Application Support" / APP_NAME
    else:
        target = Path.home() / ".local" / "share" / APP_NAME
    target.mkdir(parents=True, exist_ok=True)
    return target


DATA_DIR = _data_dir()
CACHE_DIR = DATA_DIR / "cache"
THUMBNAIL_DIR = CACHE_DIR / "thumbnails"
PREVIEW_DIR = CACHE_DIR / "previews"
CACHE_DB = DATA_DIR / "cache.db"
PROJECTS_DIR = DATA_DIR / "projects"
for directory in (CACHE_DIR, THUMBNAIL_DIR, PREVIEW_DIR, PROJECTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def model_candidates() -> list[Path]:
    configured = os.getenv("PHOTOCULL_MODEL_DIR")
    candidates = [
        Path(configured).expanduser() if configured else None,
        PROJECT_ROOT / "models",
    ]
    unique: list[Path] = []
    for candidate in candidates:
        if candidate is None:
            continue
        resolved = candidate.resolve()
        if resolved not in unique:
            unique.append(resolved)
    return unique


def find_model_dir() -> Path | None:
    for candidate in model_candidates():
        if candidate.is_dir():
            return candidate
    return None


def find_face_model_dir() -> Path | None:
    """查找包含 InsightFace 主模型的目录，避免场景模型目录遮蔽旧模型。"""
    for root in model_candidates():
        pack = root / "buffalo_l"
        if (pack / "det_10g.onnx").is_file() and (pack / "w600k_r50.onnx").is_file():
            return root
        if (root / "det_10g.onnx").is_file() and (root / "w600k_r50.onnx").is_file():
            return root.parent if root.name == "buffalo_l" else root
    return None


@dataclass(slots=True)
class EngineSettings:
    grouping_preset: str = "balanced"
    keep_per_group: int = 1
    coverage_enabled: bool = False
    coverage_window_minutes: int = 15
    face_identity_threshold: float = 0.42
    use_gpu: bool = True
    recursive: bool = True
    jpeg_preview_quality: int = 80
    vlm_enabled: bool = False
    vlm_server_url: str = DEFAULT_VLM_SERVER_URL
    vlm_executable_path: str = ""
    vlm_model_path: str = ""
    vlm_mmproj_path: str = ""
    vlm_model_id: str = "Qwen3.8-27B"
    vlm_quantization: str = "Q4_K_M"
    vlm_context_size: int = 8192
    vlm_gpu_layers: int = 16
    vlm_max_groups: int = 12
    vlm_max_candidates: int = 4
    vlm_ambiguity_margin: float = 8.0
    vlm_min_confidence: float = 0.65
    vlm_timeout_seconds: int = 300

    def validated(self) -> "EngineSettings":
        if self.grouping_preset not in {"cautious", "balanced", "aggressive"}:
            self.grouping_preset = "balanced"
        self.keep_per_group = min(5, max(1, int(self.keep_per_group)))
        self.coverage_enabled = bool(self.coverage_enabled)
        self.coverage_window_minutes = min(60, max(5, int(self.coverage_window_minutes)))
        self.face_identity_threshold = min(0.62, max(0.32, float(self.face_identity_threshold)))
        self.jpeg_preview_quality = min(95, max(60, int(self.jpeg_preview_quality)))
        self.use_gpu = bool(self.use_gpu)
        self.recursive = bool(self.recursive)
        self.vlm_enabled = bool(self.vlm_enabled)
        self.vlm_server_url = _loopback_url(self.vlm_server_url)
        self.vlm_executable_path = str(self.vlm_executable_path).strip()
        self.vlm_model_path = str(self.vlm_model_path).strip()
        self.vlm_mmproj_path = str(self.vlm_mmproj_path).strip()
        self.vlm_model_id = str(self.vlm_model_id).strip()[:160] or "Qwen3.8-27B"
        self.vlm_quantization = str(self.vlm_quantization).strip()[:40] or "Q4_K_M"
        self.vlm_context_size = min(32768, max(4096, int(self.vlm_context_size)))
        self.vlm_gpu_layers = min(256, max(0, int(self.vlm_gpu_layers)))
        self.vlm_max_groups = min(500, max(1, int(self.vlm_max_groups)))
        self.vlm_max_candidates = min(8, max(2, int(self.vlm_max_candidates)))
        self.vlm_ambiguity_margin = min(30.0, max(0.5, float(self.vlm_ambiguity_margin)))
        self.vlm_min_confidence = min(0.95, max(0.5, float(self.vlm_min_confidence)))
        self.vlm_timeout_seconds = min(900, max(30, int(self.vlm_timeout_seconds)))
        if VLM_DISABLED:
            self.vlm_enabled = False
            self.vlm_executable_path = ""
            self.vlm_model_path = ""
            self.vlm_mmproj_path = ""
            self.vlm_model_id = "disabled"
            self.vlm_quantization = "none"
        return self


class SettingsStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DATA_DIR / "settings.json"
        self._lock = RLock()
        self._settings = self._load()

    def _load(self) -> EngineSettings:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            allowed = {field.name for field in fields(EngineSettings)}
            return EngineSettings(**{key: value for key, value in payload.items() if key in allowed}).validated()
        except (OSError, ValueError, TypeError):
            return EngineSettings().validated()

    def get(self) -> EngineSettings:
        with self._lock:
            return EngineSettings(**asdict(self._settings))

    def update(self, changes: dict[str, Any]) -> EngineSettings:
        allowed = {field.name for field in fields(EngineSettings)}
        with self._lock:
            payload = asdict(self._settings)
            payload.update({key: value for key, value in changes.items() if key in allowed})
            self._settings = EngineSettings(**payload).validated()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(asdict(self._settings), ensure_ascii=False, indent=2), encoding="utf-8")
            return EngineSettings(**asdict(self._settings))


settings_store = SettingsStore()
