from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from photocull.lightroom_bridge import LightroomBridge
from photocull.lightroom_protocol import (
    LightroomItem,
    LightroomPolicy,
    LightroomPreflightManifest,
    LightroomProject,
    plan_hash,
)
from photocull.lightroom_service import LightroomOperation, LightroomService
from photocull.project_store import ProjectStore


PLUGIN_ROOT = Path(__file__).resolve().parents[2] / "lightroom" / "YidianPhotoCull.lrplugin"
LIGHTROOM_FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "lightroom" / "tests" / "fixtures"


def _path_hash(path: Path) -> str:
    normalized = os.path.normcase(str(path.resolve()))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def manifest(
    tmp_path: Path,
    *,
    created_at: str = "2026-08-30T10:00:00+08:00",
    rating: int = 3,
    request_id: str = "1" * 32,
    operation_id: str = "2" * 32,
) -> LightroomPreflightManifest:
    source = tmp_path / "source.jpg"
    if not source.exists():
        source.write_bytes(b"photo")
    stat = source.stat()
    draft = LightroomPreflightManifest(
        request_id=request_id,
        operation_id=operation_id,
        created_at=created_at,
        plan_hash="0" * 64,
        app_version="0.2.1",
        project=LightroomProject(
            id="project-1",
            name="项目 1",
            source_root=str(tmp_path.resolve()),
        ),
        policy=LightroomPolicy(),
        items=[
            LightroomItem(
                item_id="photo-1",
                source_path=str(source.resolve()),
                path_hash=_path_hash(source),
                file_size=stat.st_size,
                modified_ns=stat.st_mtime_ns,
                target_rating=rating,
                rating_tier="primary" if rating == 3 else "coverage",
                rating_reason="primary_rank" if rating == 3 else "person_stage_gap",
            )
        ],
    )
    return draft.model_copy(update={"plan_hash": plan_hash(draft)})


def service_with_project(
    tmp_path: Path,
    *,
    lightroom_ready: bool,
    missing_file_mapping: bool = False,
) -> LightroomService:
    project_id = "v2-project" if lightroom_ready else "legacy-project"
    source_root = tmp_path / "source"
    source_root.mkdir(parents=True, exist_ok=True)
    photos: list[dict[str, Any]] = []
    files: dict[str, str] = {}
    definitions = (
        ("primary", 3, "primary", "primary_rank"),
        ("coverage", 2, "coverage", "person_stage_gap"),
        ("waste", 0, "waste", "redundant_reject"),
    )
    for photo_id, stars, tier, reason in definitions:
        source = source_root / f"{photo_id}.jpg"
        source.write_bytes(f"photo-{photo_id}".encode("utf-8"))
        photos.append(
            {
                "id": photo_id,
                "relative_path": source.name,
                "stars": stars,
                "rating_tier": tier,
                "rating_origin": "ai",
                "rating_reason": reason,
                "rating_locked": False,
                "needs_review": False,
                "coverage_keys": [],
                "strict_duplicate_cluster_id": "",
                "beat_id": "beat-1",
            }
        )
        if not (missing_file_mapping and photo_id == "waste"):
            files[photo_id] = str(source)

    results = {
        "schema_version": 2,
        "rating_migration_status": "native" if lightroom_ready else "rescan_required",
        "lightroom_ready": lightroom_ready,
        "project_id": project_id,
        "project_name": "一点筛图测试项目",
        "source_name": source_root.name,
        "created_at": "2026-08-30T10:00:00+08:00",
        "photos": photos,
        "groups": [],
        "summary": {"total": len(photos)},
    }
    store = ProjectStore(tmp_path / "projects")
    store.save(project_id, results, files)
    bridge = LightroomBridge(tmp_path / "bridge")
    return LightroomService(store, bridge, app_version="0.2.1")


def write_preflight_receipt(
    bridge: LightroomBridge,
    operation: LightroomOperation,
    *,
    counts: dict[str, int] | None = None,
    baseline_hash: str = "b" * 64,
    catalog_identity_hash: str = "c" * 64,
) -> Path:
    manifest_value = operation.preflight_manifest
    requested = dict(counts or {})
    invalid = int(requested.get("invalid", 0))
    protected = int(requested.get("protected", 0))
    unchanged = int(requested.get("unchanged", 0))
    update = int(requested.get("update", 0))
    new = int(requested.get("new", len(manifest_value.items) - invalid - protected - unchanged - update))
    actions = (["invalid"] * invalid) + (["protected"] * protected) + (["unchanged"] * unchanged)
    actions += (["update"] * update) + (["new"] * new)
    assert len(actions) == len(manifest_value.items)
    receipt_counts = {
        "total": len(manifest_value.items),
        "new": new,
        "update": update,
        "unchanged": unchanged,
        "protected": protected,
        "invalid": invalid,
        "catalog_added": 0,
        "pending_rating": 0,
        "verified": 0,
        "rolled_back": 0,
    }
    receipt_counts.update(requested)
    payload = {
        "schema_version": 1,
        "request_id": operation.preflight_request_id,
        "operation_id": operation.id,
        "plan_hash": operation.plan_hash,
        "baseline_hash": baseline_hash,
        "catalog_name": "Yidian Test Catalog",
        "catalog_identity_hash": catalog_identity_hash,
        "started_at": "2026-08-30T10:00:00+08:00",
        "finished_at": "2026-08-30T10:00:01+08:00",
        "status": "awaiting_confirmation",
        "counts": receipt_counts,
        "chunks": [],
        "items": [
            {
                "item_id": item.item_id,
                "path_hash": item.path_hash,
                "action": action,
                "previous_rating": 1 if action in {"update", "unchanged", "protected"} else None,
                "target_rating": item.target_rating,
                "final_rating": None,
                "status": "planned" if action != "invalid" else "invalid",
            }
            for item, action in zip(manifest_value.items, actions, strict=True)
        ],
    }
    target = bridge.outbox / f"{operation.preflight_request_id}.json"
    target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return target


def write_execute_receipt(
    bridge: LightroomBridge,
    operation: LightroomOperation,
    *,
    status: str = "complete",
    pending_rating: int = 0,
) -> Path:
    assert operation.execute_request_id is not None
    assert operation.preflight_receipt is not None
    preflight = operation.preflight_receipt
    mutated = preflight.counts.new + preflight.counts.update
    counts = preflight.counts.model_dump(mode="json")
    counts["catalog_added"] = preflight.counts.new
    counts["pending_rating"] = pending_rating
    counts["verified"] = mutated - pending_rating
    payload = {
        "schema_version": 1,
        "request_id": operation.execute_request_id,
        "operation_id": operation.id,
        "plan_hash": operation.plan_hash,
        "baseline_hash": preflight.baseline_hash,
        "catalog_name": preflight.catalog_name,
        "catalog_identity_hash": preflight.catalog_identity_hash,
        "started_at": "2026-08-30T10:01:00+08:00",
        "finished_at": "2026-08-30T10:01:02+08:00",
        "status": status,
        "counts": counts,
        "chunks": [],
        "items": [
            {
                **item.model_dump(mode="json"),
                "final_rating": (
                    item.target_rating
                    if item.action in {"new", "update"}
                    else item.previous_rating
                ),
                "status": "verified" if item.action in {"new", "update"} else item.action,
            }
            for item in preflight.items
        ],
    }
    if pending_rating:
        for item in payload["items"]:
            if item["action"] in {"new", "update"}:
                item["final_rating"] = None
                item["status"] = "pending_rating"
                break
    target = bridge.outbox / f"{operation.execute_request_id}.json"
    target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return target


def plugin_file(name: str) -> str:
    return (PLUGIN_ROOT / name).read_text(encoding="utf-8")


def all_plugin_source() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in sorted(PLUGIN_ROOT.glob("*.lua")))


def fixture(name: str) -> bytes:
    return (LIGHTROOM_FIXTURE_ROOT / name).read_bytes()
