from __future__ import annotations

import hashlib
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageOps

from .config import PREVIEW_DIR, RAW_EXTENSIONS, SUPPORTED_EXTENSIONS, THUMBNAIL_DIR
from .internal_models import VisualDescriptor

try:
    import exifread
except ImportError:
    exifread = None

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:
    pillow_heif = None

try:
    import rawpy
except ImportError:
    rawpy = None


_SEQUENCE_PATTERN = re.compile(r"(\d{3,})(?!.*\d)")
_PANASONIC_SEQUENCE_PATTERN = re.compile(r"^[Pp]\d{3}(\d{4})$")
_RAW_COMPANION_EXTENSIONS = {".jpg", ".jpeg"}


def _capture_key(path: Path) -> tuple[str, str]:
    return str(path.parent).casefold(), path.stem.casefold()


def discover_images(root: Path, recursive: bool) -> list[Path]:
    iterator: Iterable[Path] = root.rglob("*") if recursive else root.glob("*")
    candidates = [
        path.resolve()
        for path in iterator
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    raw_captures = {
        _capture_key(path)
        for path in candidates
        if path.suffix.lower() in RAW_EXTENSIONS
    }
    return sorted(
        (
            path
            for path in candidates
            if not (
                path.suffix.lower() in _RAW_COMPANION_EXTENSIONS
                and _capture_key(path) in raw_captures
            )
        ),
        key=lambda path: path.name.casefold(),
    )


def photo_id(path: Path) -> str:
    stat = path.stat()
    source = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8", errors="surrogatepass")
    return hashlib.sha256(source).hexdigest()[:24]


def file_sequence(path: Path) -> int:
    panasonic = _PANASONIC_SEQUENCE_PATTERN.fullmatch(path.stem)
    if panasonic:
        return int(panasonic.group(1))
    match = _SEQUENCE_PATTERN.search(path.stem)
    return int(match.group(1)) if match else -1


def raw_companion_preview(path: Path) -> Path | None:
    if path.suffix.lower() not in RAW_EXTENSIONS:
        return None
    for extension in (".JPG", ".JPEG", ".jpg", ".jpeg"):
        candidate = path.with_suffix(extension)
        if candidate.is_file():
            return candidate
    return None


def load_image(path: Path) -> Image.Image:
    extension = path.suffix.lower()
    if extension in RAW_EXTENSIONS:
        companion = raw_companion_preview(path)
        if companion is not None:
            with Image.open(companion) as opened:
                return ImageOps.exif_transpose(opened).convert("RGB")
        if rawpy is None:
            raise RuntimeError(f"缺少 rawpy，无法读取 {extension} 文件")
        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=False,
                output_bps=8,
                half_size=max(raw.sizes.width, raw.sizes.height) > 7000,
            )
        image = Image.fromarray(rgb, mode="RGB")
    else:
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
    return image


def resize_for_analysis(image: Image.Image, max_dimension: int = 1600) -> np.ndarray:
    width, height = image.size
    scale = min(1.0, max_dimension / max(width, height))
    if scale < 1:
        image = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.uint8)


def capture_time(image: Image.Image, path: Path) -> datetime | None:
    if path.suffix.lower() in RAW_EXTENSIONS and exifread is not None:
        try:
            with path.open("rb") as handle:
                tags = exifread.process_file(handle, details=False, extract_thumbnail=False)
            value = tags.get("EXIF DateTimeOriginal") or tags.get("Image DateTime")
            if value:
                parsed = datetime.strptime(str(value).strip()[:19], "%Y:%m:%d %H:%M:%S")
                subseconds = str(tags.get("EXIF SubSecTimeOriginal", "")).strip()
                digits = "".join(character for character in subseconds if character.isdigit())[:6]
                if digits:
                    parsed = parsed.replace(microsecond=int(digits.ljust(6, "0")))
                return parsed
        except (OSError, ValueError, TypeError):
            pass
    try:
        exif = image.getexif()
        for tag in (36867, 36868, 306):
            value = exif.get(tag)
            if not value:
                continue
            text = str(value).strip().replace("\x00", "")
            for pattern in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
                try:
                    return datetime.strptime(text[:19], pattern)
                except ValueError:
                    pass
    except (OSError, ValueError, TypeError):
        pass
    try:
        return datetime.fromtimestamp(path.stat().st_mtime)
    except OSError:
        return None


def ensure_cached_images(image: Image.Image, identifier: str, jpeg_quality: int) -> None:
    thumbnail = THUMBNAIL_DIR / f"{identifier}.jpg"
    preview = PREVIEW_DIR / f"{identifier}.jpg"
    if not thumbnail.exists():
        thumb = image.copy()
        thumb.thumbnail((480, 360), Image.Resampling.LANCZOS)
        thumb.save(thumbnail, "JPEG", quality=jpeg_quality, optimize=True)
    if not preview.exists():
        large = image.copy()
        large.thumbnail((2200, 1800), Image.Resampling.LANCZOS)
        large.save(preview, "JPEG", quality=max(76, jpeg_quality), optimize=True)


def cached_images_exist(identifier: str) -> bool:
    return (THUMBNAIL_DIR / f"{identifier}.jpg").is_file() and (PREVIEW_DIR / f"{identifier}.jpg").is_file()


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = vector.astype(np.float32, copy=False).reshape(-1)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-8 else np.zeros_like(vector)


def _perceptual_hash(rgb: np.ndarray) -> int:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)
    low = dct[:8, :8].reshape(-1)
    median = float(np.median(low[1:]))
    bits = low > median
    result = 0
    for bit in bits:
        result = (result << 1) | int(bit)
    return result


def _difference_hash(rgb: np.ndarray) -> int:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    result = 0
    for bit in bits.reshape(-1):
        result = (result << 1) | int(bit)
    return result


def _layout_descriptor(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    layout = cv2.resize(gray, (24, 24), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    layout -= float(layout.mean())
    return _normalize(layout)


def _color_descriptor(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [16, 6], [0, 180, 0, 256]).reshape(-1)
    histogram = np.sqrt(histogram / max(float(histogram.sum()), 1.0))
    return _normalize(histogram)


def _edge_descriptor(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    grad_x = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    blocks = magnitude.reshape(8, 4, 8, 4).mean(axis=(1, 3))
    return _normalize(blocks)


def build_descriptor(rgb: np.ndarray, semantic: np.ndarray | None = None) -> VisualDescriptor:
    return VisualDescriptor(
        phash=_perceptual_hash(rgb),
        dhash=_difference_hash(rgb),
        layout=_layout_descriptor(rgb).astype(np.float16),
        color=_color_descriptor(rgb).astype(np.float16),
        edge=_edge_descriptor(rgb).astype(np.float16),
        semantic=None if semantic is None else _normalize(semantic).astype(np.float16),
    )


def cosine_similarity(left: np.ndarray | None, right: np.ndarray | None) -> float | None:
    if left is None or right is None or left.size != right.size:
        return None
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator < 1e-8:
        return None
    return float(np.clip(np.dot(left.astype(np.float32), right.astype(np.float32)) / denominator, -1.0, 1.0))


def phash_similarity(left: int, right: int) -> float:
    distance = (left ^ right).bit_count()
    return 1.0 - distance / 64.0


def dhash_similarity(left: int, right: int) -> float:
    distance = (left ^ right).bit_count()
    return 1.0 - distance / 64.0


def temporal_similarity(left: datetime | None, right: datetime | None, left_seq: int, right_seq: int) -> float:
    if left and right:
        delta = abs((left - right).total_seconds())
        return math.exp(-delta / 18.0)
    if left_seq >= 0 and right_seq >= 0:
        return math.exp(-abs(left_seq - right_seq) / 6.0)
    return 0.45
