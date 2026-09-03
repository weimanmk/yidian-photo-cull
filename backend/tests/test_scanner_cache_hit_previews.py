from pathlib import Path

import photocull.scanner as scanner_module


def test_cache_hit_preview_skip_does_not_decode_source(monkeypatch, tmp_path: Path) -> None:
    decoded: list[Path] = []
    monkeypatch.setattr(scanner_module, "cached_images_exist", lambda _identifier: False)
    monkeypatch.setattr(scanner_module, "load_image", lambda path: decoded.append(path))

    scanner_module.ensure_cache_hit_images(
        tmp_path / "photo.raw",
        "photo-id",
        jpeg_quality=80,
        enabled=False,
    )

    assert decoded == []


def test_cache_hit_preview_default_still_generates_missing_images(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "photo.jpg"
    image = object()
    generated: list[tuple[object, str, int]] = []
    monkeypatch.setattr(scanner_module, "cached_images_exist", lambda _identifier: False)
    monkeypatch.setattr(scanner_module, "load_image", lambda path: image if path == source else None)
    monkeypatch.setattr(
        scanner_module,
        "ensure_cached_images",
        lambda loaded, identifier, quality: generated.append((loaded, identifier, quality)),
    )

    scanner_module.ensure_cache_hit_images(
        source,
        "photo-id",
        jpeg_quality=82,
        enabled=True,
    )

    assert generated == [(image, "photo-id", 82)]
