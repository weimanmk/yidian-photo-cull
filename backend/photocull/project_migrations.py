from __future__ import annotations

from copy import deepcopy
from typing import Any


SCHEMA_VERSION = 2
SEMANTIC_PHOTO_FIELDS = frozenset(
    {
        "rating_tier",
        "rating_origin",
        "rating_reason",
        "rating_locked",
        "needs_review",
        "coverage_keys",
        "strict_duplicate_cluster_id",
        "beat_id",
    }
)


def migrate_project_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """把项目包装负载规范化为 schema v2，同时明确旧星级不可直接导入 LR。"""
    normalized = deepcopy(payload)
    results = normalized.get("results")
    if not isinstance(results, dict):
        raise ValueError("项目缺少 results 对象")
    photos = results.get("photos", [])
    if not isinstance(photos, list) or not all(isinstance(photo, dict) for photo in photos):
        raise ValueError("项目照片列表无效")

    has_semantic_contract = all(SEMANTIC_PHOTO_FIELDS.issubset(photo) for photo in photos)
    if int(results.get("schema_version", 0) or 0) == SCHEMA_VERSION and has_semantic_contract:
        status = str(results.get("rating_migration_status", "native"))
        if status not in {"native", "rescan_required"}:
            status = "rescan_required"
        results["rating_migration_status"] = status
        results["lightroom_ready"] = bool(results.get("lightroom_ready", status == "native")) and status == "native"
        return normalized

    for photo in photos:
        photo.setdefault("rating_tier", "legacy")
        photo.setdefault("rating_origin", "legacy")
        photo.setdefault("rating_reason", "legacy_score")
        photo.setdefault("rating_locked", False)
        photo.setdefault("needs_review", True)
        photo.setdefault("coverage_keys", [])
        photo.setdefault("strict_duplicate_cluster_id", "")
        photo.setdefault("beat_id", "")
    results["schema_version"] = SCHEMA_VERSION
    results["rating_migration_status"] = "rescan_required"
    results["lightroom_ready"] = False
    return normalized
