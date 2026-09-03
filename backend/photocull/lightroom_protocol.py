from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SCHEMA_VERSION = 1
SHA256_PATTERN = r"^[0-9a-f]{64}$"
RatingTier = Literal["waste", "valuable", "coverage", "primary"]
ReceiptAction = Literal["new", "update", "unchanged", "protected", "invalid", "rollback", "conflict"]
ReceiptStatus = Literal[
    "awaiting_confirmation",
    "complete",
    "pending_rating",
    "replan_required",
    "manual_recovery_required",
    "failed",
    "quarantined",
    "rollback_awaiting_confirmation",
    "rolled_back",
]


class LightroomProtocolError(ValueError):
    """Lightroom 桥接载荷不满足协议约束。"""


class LightroomProject(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1, max_length=160)
    name: str = Field(min_length=1, max_length=500)
    source_root: str = Field(min_length=3, max_length=4096)


class LightroomItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    item_id: str = Field(min_length=1, max_length=128)
    source_path: str = Field(min_length=3, max_length=4096)
    path_hash: str = Field(pattern=SHA256_PATTERN)
    file_size: int = Field(ge=0)
    modified_ns: int = Field(ge=0)
    target_rating: Literal[0, 1, 2, 3]
    rating_tier: RatingTier
    rating_reason: str = Field(min_length=1, max_length=256)


class LightroomPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    import_mode: Literal["in_place"] = "in_place"
    collection_mode: Literal["none"] = "none"
    write_xmp: Literal[False] = False
    protect_existing_rating_above: Literal[3] = 3


class _ManifestBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = SCHEMA_VERSION
    request_id: str = Field(min_length=16, max_length=64, pattern=r"^[0-9A-Fa-f-]+$")
    operation_id: str = Field(min_length=16, max_length=64, pattern=r"^[0-9A-Fa-f-]+$")
    plan_hash: str = Field(pattern=SHA256_PATTERN)
    app_version: str = Field(min_length=1, max_length=40)
    created_at: str = Field(min_length=10, max_length=80)
    project: LightroomProject
    policy: LightroomPolicy
    items: list[LightroomItem] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_items(self) -> "_ManifestBase":
        item_ids = [item.item_id for item in self.items]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("Lightroom 计划包含重复照片 ID")
        path_hashes = [item.path_hash for item in self.items]
        if len(path_hashes) != len(set(path_hashes)):
            raise ValueError("Lightroom 计划包含重复源路径")
        return self


class LightroomPreflightManifest(_ManifestBase):
    operation: Literal["preflight"] = "preflight"


class LightroomExecuteManifest(_ManifestBase):
    operation: Literal["execute"] = "execute"
    preflight_request_id: str = Field(min_length=16, max_length=64, pattern=r"^[0-9A-Fa-f-]+$")
    baseline_hash: str = Field(pattern=SHA256_PATTERN)
    catalog_identity_hash: str = Field(pattern=SHA256_PATTERN)


LightroomManifest = LightroomPreflightManifest | LightroomExecuteManifest


class LightroomReceiptCounts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    total: int = Field(default=0, ge=0)
    new: int = Field(default=0, ge=0)
    update: int = Field(default=0, ge=0)
    unchanged: int = Field(default=0, ge=0)
    protected: int = Field(default=0, ge=0)
    invalid: int = Field(default=0, ge=0)
    catalog_added: int = Field(default=0, ge=0)
    pending_rating: int = Field(default=0, ge=0)
    verified: int = Field(default=0, ge=0)
    rolled_back: int = Field(default=0, ge=0)


class LightroomReceiptChunk(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str = Field(min_length=1, max_length=160)
    status: str = Field(min_length=1, max_length=80)
    counts: dict[str, int] = Field(default_factory=dict)
    error_code: str | None = Field(default=None, max_length=160)


class LightroomReceiptItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    item_id: str = Field(min_length=1, max_length=128)
    path_hash: str = Field(pattern=SHA256_PATTERN)
    action: ReceiptAction
    previous_rating: int | None = Field(default=None, ge=0, le=5)
    target_rating: Literal[0, 1, 2, 3]
    final_rating: int | None = Field(default=None, ge=0, le=5)
    status: str = Field(min_length=1, max_length=80)


class LightroomReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = SCHEMA_VERSION
    request_id: str = Field(min_length=16, max_length=64, pattern=r"^[0-9A-Fa-f-]+$")
    operation_id: str = Field(min_length=16, max_length=64, pattern=r"^[0-9A-Fa-f-]+$")
    plan_hash: str = Field(pattern=SHA256_PATTERN)
    baseline_hash: str | None = None
    catalog_name: str = Field(min_length=1, max_length=500)
    catalog_identity_hash: str = Field(pattern=SHA256_PATTERN)
    started_at: str = Field(min_length=10, max_length=80)
    finished_at: str = Field(min_length=10, max_length=80)
    status: ReceiptStatus
    counts: LightroomReceiptCounts
    chunks: list[LightroomReceiptChunk] = Field(default_factory=list)
    items: list[LightroomReceiptItem] = Field(default_factory=list)
    error_code: str | None = Field(default=None, max_length=160)
    error_message: str | None = Field(default=None, max_length=2000)

    @field_validator("baseline_hash")
    @classmethod
    def validate_optional_hash(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("baseline_hash 必须是小写 SHA-256")
        return value

    @model_validator(mode="after")
    def validate_completion_invariant(self) -> "LightroomReceipt":
        if self.status == "complete":
            expected = self.counts.new + self.counts.update
            if self.counts.verified != expected or self.counts.pending_rating != 0:
                raise ValueError("完成收据必须验证全部新增和更新项目")
        return self


def canonical_json(payload: BaseModel | dict[str, Any] | list[Any]) -> bytes:
    if isinstance(payload, BaseModel):
        value: Any = payload.model_dump(mode="json")
    else:
        value = payload
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def normalized_source_path(path: str) -> str:
    return os.path.normcase(str(Path(path).expanduser().resolve(strict=False)))


def source_path_hash(path: str) -> str:
    return hashlib.sha256(normalized_source_path(path).encode("utf-8")).hexdigest()


def _plan_payload(manifest: LightroomManifest) -> dict[str, Any]:
    items = sorted(
        (
            {
                "item_id": item.item_id,
                "source_path": normalized_source_path(item.source_path),
                "path_hash": item.path_hash,
                "file_size": item.file_size,
                "modified_ns": item.modified_ns,
                "target_rating": item.target_rating,
                "rating_tier": item.rating_tier,
                "rating_reason": item.rating_reason,
            }
            for item in manifest.items
        ),
        key=lambda item: (item["source_path"], item["item_id"]),
    )
    return {
        "schema_version": manifest.schema_version,
        "project_id": manifest.project.id,
        "policy": manifest.policy.model_dump(mode="json"),
        "items": items,
    }


def plan_hash(manifest: LightroomManifest) -> str:
    return hashlib.sha256(canonical_json(_plan_payload(manifest))).hexdigest()
