from pathlib import Path

import pytest

from photocull.exporter import (
    ExportPlanChangedError,
    build_export_plan,
    execute_export_plan,
)


def write_source(directory: Path, name: str, content: bytes) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(content)
    return path


def semantic_results(*photos: dict) -> dict:
    return {
        "schema_version": 2,
        "rating_migration_status": "native",
        "lightroom_ready": True,
        "project_id": "export-project",
        "photos": list(photos),
    }


def test_plan_routes_each_rating_to_its_semantic_directory_without_writing(tmp_path) -> None:
    source_dir = tmp_path / "source"
    files = {
        "primary": str(write_source(source_dir, "primary.jpg", b"primary")),
        "coverage": str(write_source(source_dir, "coverage.jpg", b"coverage")),
        "valuable": str(write_source(source_dir, "valuable.jpg", b"valuable")),
    }
    results = semantic_results(
        {"id": "primary", "relative_path": "primary.jpg", "stars": 3},
        {"id": "coverage", "relative_path": "coverage.jpg", "stars": 2},
        {"id": "valuable", "relative_path": "valuable.jpg", "stars": 1},
        {"id": "waste", "relative_path": "waste.jpg", "stars": 0},
    )
    destination = tmp_path / "destination"

    plan = build_export_plan(results, files, destination, minimum_stars=1)

    assert [item.relative_target.parts[0] for item in plan.items] == [
        "3星精选",
        "2星补位",
        "1星有价值",
    ]
    assert plan.copy_count == 3
    assert not destination.exists()


def test_existing_different_file_is_conflict_and_never_overwritten(tmp_path) -> None:
    source = write_source(tmp_path / "source", "photo.jpg", b"source")
    destination = tmp_path / "destination"
    target = destination / "2星补位" / "photo.jpg"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"different")
    results = semantic_results({"id": "photo", "relative_path": "photo.jpg", "stars": 2})

    plan = build_export_plan(results, {"photo": str(source)}, destination, minimum_stars=2)
    receipt = execute_export_plan(plan)

    assert plan.conflict_count == 1
    assert receipt.conflicts == 1
    assert receipt.copied == 0
    assert target.read_bytes() == b"different"


def test_identical_existing_file_is_skipped(tmp_path) -> None:
    source = write_source(tmp_path / "source", "photo.jpg", b"same")
    destination = tmp_path / "destination"
    target = destination / "3星精选" / "photo.jpg"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"same")
    results = semantic_results({"id": "photo", "relative_path": "photo.jpg", "stars": 3})

    receipt = execute_export_plan(
        build_export_plan(results, {"photo": str(source)}, destination, minimum_stars=3)
    )

    assert receipt.skipped == 1
    assert receipt.copied == 0


def test_export_never_creates_xmp(tmp_path) -> None:
    source = write_source(tmp_path / "source", "photo.raw", b"raw")
    destination = tmp_path / "destination"
    results = semantic_results({"id": "photo", "relative_path": "photo.raw", "stars": 3})

    receipt = execute_export_plan(
        build_export_plan(results, {"photo": str(source)}, destination, minimum_stars=2)
    )

    assert receipt.copied == 1
    assert not list(destination.rglob("*.xmp"))


def test_changed_source_fingerprint_aborts_before_copy(tmp_path) -> None:
    source = write_source(tmp_path / "source", "photo.jpg", b"before")
    destination = tmp_path / "destination"
    results = semantic_results({"id": "photo", "relative_path": "photo.jpg", "stars": 3})
    plan = build_export_plan(results, {"photo": str(source)}, destination, minimum_stars=3)
    source.write_bytes(b"after-change")

    with pytest.raises(ExportPlanChangedError, match="源文件"):
        execute_export_plan(plan)

    assert not destination.exists()


def test_legacy_project_cannot_be_exported_as_semantic_stars(tmp_path) -> None:
    source = write_source(tmp_path / "source", "photo.jpg", b"photo")
    results = semantic_results({"id": "photo", "relative_path": "photo.jpg", "stars": 3})
    results["lightroom_ready"] = False
    results["rating_migration_status"] = "rescan_required"

    with pytest.raises(ValueError, match="重新扫描"):
        build_export_plan(results, {"photo": str(source)}, tmp_path / "destination", minimum_stars=1)
