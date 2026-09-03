import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance

from photocull.config import EngineSettings
from photocull.project_store import ProjectStore
from photocull.scanner import ScannerService


def draw_burst_frame(path: Path, offset: int, contrast: float) -> None:
    image = Image.new("RGB", (900, 600), (181, 139, 106))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 360, 900, 600), fill=(82, 66, 58))
    draw.rectangle((90, 70, 810, 350), fill=(205, 181, 147), outline=(238, 224, 198), width=8)
    draw.ellipse((290 + offset, 125, 430 + offset, 265), fill=(78, 52, 43))
    draw.rectangle((315 + offset, 250, 405 + offset, 480), fill=(43, 63, 87))
    draw.ellipse((485 - offset, 130, 625 - offset, 270), fill=(98, 65, 50))
    draw.rectangle((505 - offset, 255, 605 - offset, 480), fill=(126, 43, 53))
    draw.line((145, 92, 760, 322), fill=(123, 91, 70), width=5)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    image.save(path, quality=92)


def draw_other_scene(path: Path, seed: int) -> None:
    image = Image.new("RGB", (900, 600), (42, 77 + seed * 5, 118))
    draw = ImageDraw.Draw(image)
    for x in range(0, 900, 90):
        draw.rectangle((x, 0, x + 44, 600), fill=(24 + seed, 35, 61))
    draw.ellipse((320, 140, 580, 400), outline=(235, 209, 109), width=18)
    image.save(path, quality=92)


def test_full_pipeline_collapses_a_visual_burst(tmp_path, monkeypatch) -> None:
    class StubDepthEngine:
        def __init__(self, use_gpu: bool = True) -> None:
            self.use_gpu = use_gpu

        @staticmethod
        def signature() -> str:
            return "depth-disabled-for-orchestration-test"

        @staticmethod
        def status() -> dict[str, object]:
            return {"available": False, "model": "test-stub", "error": None}

        @staticmethod
        def analyze(*_args, **_kwargs):
            return None

    source = tmp_path / "source"
    source.mkdir()
    thumbnail_directory = tmp_path / "cache" / "thumbnails"
    preview_directory = tmp_path / "cache" / "previews"
    thumbnail_directory.mkdir(parents=True)
    preview_directory.mkdir(parents=True)
    monkeypatch.setattr("photocull.imaging.THUMBNAIL_DIR", thumbnail_directory)
    monkeypatch.setattr("photocull.imaging.PREVIEW_DIR", preview_directory)
    monkeypatch.setattr("photocull.scanner.DepthEngine", StubDepthEngine)
    for index in range(1, 7):
        draw_burst_frame(source / f"IMG_{index:04d}.jpg", offset=index - 3, contrast=0.96 + index * 0.012)
    draw_other_scene(source / "IMG_0101.jpg", 1)
    draw_other_scene(source / "IMG_0201.jpg", 4)

    monkeypatch.setattr(
        "photocull.scanner.settings_store.get",
        lambda: EngineSettings(grouping_preset="balanced", keep_per_group=1, use_gpu=False),
    )
    service = ScannerService(ProjectStore(tmp_path / "projects"))
    service.start(str(source), "balanced", 1, False)
    deadline = time.monotonic() + 40
    while service.status()["status"] not in {"completed", "failed", "cancelled"} and time.monotonic() < deadline:
        time.sleep(0.05)

    status = service.status()
    assert status["status"] == "completed", status
    results = service.results()
    assert results is not None
    assert results["summary"]["total"] == 8
    assert max(group["size"] for group in results["groups"]) >= 5
    assert results["summary"]["duplicates"] >= 4
    assert results["schema_version"] == 2
    assert results["rating_migration_status"] == "native"
    assert results["lightroom_ready"] is True
    assert results["summary"]["selected"] == sum(photo["stars"] >= 2 for photo in results["photos"])
    assert sum(results["summary"][f"stars_{stars}"] for stars in range(4)) == 8
    assert results["rating_policy"]["primary_duplicate_leaks"] == 0
    assert all(0 <= photo["stars"] <= 3 for photo in results["photos"])
    assert results["engine"]["eye_evidence"]["name"] in {
        "default",
        "wide-hard",
        "conservative-hard",
    }
    assert results["engine"]["eye_evidence"]["max_closed_probability"] == 0.18
    assert results["engine"]["eye_evidence"]["ranking_applied"] is False
    assert results["engine"]["eye_evidence"]["validation_status"] == "development-gates-failed"
