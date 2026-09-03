from __future__ import annotations

import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from . import __version__
from .lightroom_bridge import LightroomBridge
from .lightroom_protocol import (
    LightroomExecuteManifest,
    LightroomItem,
    LightroomPolicy,
    LightroomPreflightManifest,
    LightroomProject,
    LightroomProtocolError,
    LightroomReceipt,
    LightroomReceiptCounts,
    canonical_json,
    normalized_source_path,
    plan_hash,
    source_path_hash,
)
from .project_store import ProjectStore


OperationStatus = Literal[
    "created",
    "waiting_for_plugin",
    "preflighting",
    "awaiting_confirmation",
    "executing",
    "verifying",
    "pending_rating",
    "complete",
    "failed",
    "quarantined",
    "manual_recovery_required",
    "rollback_preflight",
    "rollback_awaiting_confirmation",
    "rolling_back",
    "rolled_back",
]

TERMINAL_STATUSES = frozenset({"complete", "failed", "quarantined", "manual_recovery_required", "rolled_back"})
ERROR_RECEIPT_STATUSES = frozenset({"failed", "quarantined", "manual_recovery_required"})
RATING_TIERS = {0: "waste", 1: "valuable", 2: "coverage", 3: "primary"}


class LightroomPlanError(ValueError):
    """项目无法生成安全的 Lightroom 计划。"""


class LightroomStateError(RuntimeError):
    """Lightroom 操作状态或权威基线已发生变化。"""


class LightroomOperationNotFound(FileNotFoundError):
    """Lightroom 操作不存在。"""


class LightroomOperation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    id: str = Field(min_length=32, max_length=32, pattern=r"^[0-9a-f]{32}$")
    project_id: str = Field(min_length=1, max_length=160)
    status: OperationStatus
    created_at: str = Field(min_length=10, max_length=80)
    updated_at: str = Field(min_length=10, max_length=80)
    plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    preflight_request_id: str = Field(min_length=32, max_length=32, pattern=r"^[0-9a-f]{32}$")
    execute_request_id: str | None = Field(default=None, min_length=32, max_length=32, pattern=r"^[0-9a-f]{32}$")
    preflight_manifest: LightroomPreflightManifest
    execute_manifest: LightroomExecuteManifest | None = None
    preflight_receipt: LightroomReceipt | None = None
    execute_receipt: LightroomReceipt | None = None
    progress_counts: LightroomReceiptCounts | None = None
    can_execute: bool = False
    error_code: str | None = Field(default=None, max_length=160)
    error_message: str | None = Field(default=None, max_length=2000)

    def public_dict(self) -> dict[str, Any]:
        receipt = self.execute_receipt or self.preflight_receipt
        counts = receipt.counts if receipt is not None else None
        if self.status in {"executing", "verifying"} and self.progress_counts is not None:
            counts = self.progress_counts
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "project_id": self.project_id,
            "project_name": self.preflight_manifest.project.name,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "plan_hash": self.plan_hash,
            "preflight_request_id": self.preflight_request_id,
            "execute_request_id": self.execute_request_id,
            "item_count": len(self.preflight_manifest.items),
            "can_execute": self.can_execute,
            "counts": counts.model_dump(mode="json") if counts is not None else None,
            "catalog_name": receipt.catalog_name if receipt is not None else None,
            "catalog_identity_hash": receipt.catalog_identity_hash if receipt is not None else None,
            "baseline_hash": receipt.baseline_hash if receipt is not None else None,
            "error_code": self.error_code,
            "error_message": self.error_message,
        }


class LightroomService:
    def __init__(
        self,
        projects: ProjectStore,
        bridge: LightroomBridge | None = None,
        *,
        app_version: str = __version__,
    ) -> None:
        self.projects = projects
        self.bridge = bridge or LightroomBridge()
        self.app_version = app_version
        self._lock = RLock()

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).astimezone().isoformat(timespec="seconds")

    @staticmethod
    def _operation_id(value: str) -> str:
        operation_id = str(value).strip()
        if len(operation_id) != 32 or any(character not in "0123456789abcdef" for character in operation_id):
            raise LightroomOperationNotFound("Lightroom 操作不存在")
        return operation_id

    def _path(self, operation_id: str) -> Path:
        return self.bridge.operations / f"{self._operation_id(operation_id)}.json"

    def _save(self, operation: LightroomOperation) -> LightroomOperation:
        payload = canonical_json(operation)
        target = self._path(operation.id)
        temporary = target.parent / f".{operation.id}.{uuid4().hex}.tmp"
        try:
            with temporary.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return operation

    def _replace(self, operation: LightroomOperation, **changes: Any) -> LightroomOperation:
        payload = operation.model_dump(mode="json")
        payload.update(changes)
        payload["updated_at"] = self._now()
        return self._save(LightroomOperation.model_validate(payload))

    def get(self, operation_id: str) -> LightroomOperation:
        path = self._path(operation_id)
        with self._lock:
            if not path.is_file():
                raise LightroomOperationNotFound("Lightroom 操作不存在")
            try:
                return LightroomOperation.model_validate_json(path.read_bytes())
            except (OSError, ValueError) as exc:
                raise LightroomStateError("Lightroom 操作记录已损坏") from exc

    @staticmethod
    def _validate_semantic_project(results: dict[str, Any], project_id: str) -> None:
        if (
            int(results.get("schema_version", 0) or 0) != 2
            or results.get("rating_migration_status") != "native"
            or results.get("lightroom_ready") is not True
        ):
            raise LightroomPlanError("当前项目不是 v0.2.1 原生语义星级，请重新扫描后再导入 Lightroom")
        if str(results.get("project_id", "")).strip() != project_id:
            raise LightroomPlanError("项目 ID 与保存结果不一致")

    def _build_manifest(self, project_id: str, operation_id: str, request_id: str) -> LightroomPreflightManifest:
        try:
            results, files = self.projects.load(project_id)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise LightroomPlanError("项目不存在或已损坏") from exc
        self._validate_semantic_project(results, project_id)
        photos = results.get("photos")
        if not isinstance(photos, list) or not photos:
            raise LightroomPlanError("项目没有可导入的照片")

        items: list[LightroomItem] = []
        seen_ids: set[str] = set()
        for photo in photos:
            if not isinstance(photo, dict):
                raise LightroomPlanError("项目照片记录无效")
            item_id = str(photo.get("id", "")).strip()
            if not item_id or item_id in seen_ids:
                raise LightroomPlanError("项目包含空白或重复的照片 ID")
            seen_ids.add(item_id)
            source_value = files.get(item_id)
            if not source_value:
                raise LightroomPlanError(f"照片缺少权威源文件映射：{item_id}")
            source = Path(source_value).expanduser().resolve(strict=False)
            if not source.is_file():
                raise LightroomPlanError(f"源文件不存在：{item_id}")
            try:
                target_rating = int(photo.get("stars", -1))
            except (TypeError, ValueError) as exc:
                raise LightroomPlanError(f"照片星级无效：{item_id}") from exc
            expected_tier = RATING_TIERS.get(target_rating)
            if expected_tier is None or photo.get("rating_tier") != expected_tier:
                raise LightroomPlanError(f"照片语义星级不一致：{item_id}")
            rating_reason = str(photo.get("rating_reason", "")).strip()
            if not rating_reason:
                raise LightroomPlanError(f"照片缺少星级原因：{item_id}")
            stat = source.stat()
            items.append(
                LightroomItem(
                    item_id=item_id,
                    source_path=normalized_source_path(str(source)),
                    path_hash=source_path_hash(str(source)),
                    file_size=stat.st_size,
                    modified_ns=stat.st_mtime_ns,
                    target_rating=target_rating,  # type: ignore[arg-type]
                    rating_tier=expected_tier,  # type: ignore[arg-type]
                    rating_reason=rating_reason,
                )
            )
        items.sort(key=lambda item: (normalized_source_path(item.source_path), item.item_id))
        try:
            source_root = os.path.commonpath([str(Path(item.source_path).parent) for item in items])
        except ValueError as exc:
            raise LightroomPlanError("项目源文件不在同一文件系统中") from exc
        project = LightroomProject(
            id=project_id,
            name=str(results.get("project_name") or results.get("source_name") or project_id),
            source_root=source_root,
        )
        draft = LightroomPreflightManifest(
            request_id=request_id,
            operation_id=operation_id,
            created_at=self._now(),
            plan_hash="0" * 64,
            app_version=self.app_version,
            project=project,
            policy=LightroomPolicy(),
            items=items,
        )
        return draft.model_copy(update={"plan_hash": plan_hash(draft)})

    def create_preflight(self, project_id: str) -> LightroomOperation:
        project_value = str(project_id).strip()
        if not project_value:
            raise LightroomPlanError("项目 ID 不能为空")
        operation_id = uuid4().hex
        request_id = uuid4().hex
        manifest = self._build_manifest(project_value, operation_id, request_id)
        now = self._now()
        operation = LightroomOperation(
            id=operation_id,
            project_id=project_value,
            status="created",
            created_at=now,
            updated_at=now,
            plan_hash=manifest.plan_hash,
            preflight_request_id=request_id,
            preflight_manifest=manifest,
        )
        with self._lock:
            self._save(operation)
            try:
                self.bridge.submit(manifest)
            except (OSError, LightroomProtocolError) as exc:
                self._replace(operation, status="failed", error_code="queue_submit_failed", error_message=str(exc))
                raise LightroomStateError("Lightroom 预检请求投递失败") from exc
            return self._replace(operation, status="waiting_for_plugin")

    @staticmethod
    def _validate_receipt(
        operation: LightroomOperation,
        receipt: LightroomReceipt,
        *,
        request_id: str,
    ) -> None:
        if (
            receipt.request_id != request_id
            or receipt.operation_id != operation.id
            or receipt.plan_hash != operation.plan_hash
        ):
            raise LightroomStateError("Lightroom 收据与当前计划不匹配")
        manifest_items = {item.item_id: item for item in operation.preflight_manifest.items}
        if receipt.counts.total != len(manifest_items) or len(receipt.items) != len(manifest_items):
            raise LightroomStateError("Lightroom 收据项目总数与计划不匹配")
        actions = Counter(item.action for item in receipt.items)
        counted = (
            receipt.counts.new
            + receipt.counts.update
            + receipt.counts.unchanged
            + receipt.counts.protected
            + receipt.counts.invalid
        )
        if counted != receipt.counts.total:
            raise LightroomStateError("Lightroom 收据计数不完整")
        for action in ("new", "update", "unchanged", "protected", "invalid"):
            if actions[action] != getattr(receipt.counts, action):
                raise LightroomStateError("Lightroom 收据动作计数不一致")
        seen: set[str] = set()
        for item in receipt.items:
            planned = manifest_items.get(item.item_id)
            if planned is None or item.item_id in seen:
                raise LightroomStateError("Lightroom 收据包含未知或重复照片")
            seen.add(item.item_id)
            if item.path_hash != planned.path_hash or item.target_rating != planned.target_rating:
                raise LightroomStateError("Lightroom 收据照片指纹或目标星级不匹配")

    @staticmethod
    def _validate_source_fingerprints(operation: LightroomOperation) -> None:
        for item in operation.preflight_manifest.items:
            source = Path(item.source_path)
            if not source.is_file():
                raise LightroomStateError(f"源文件已不存在：{item.item_id}")
            stat = source.stat()
            if stat.st_size != item.file_size or stat.st_mtime_ns != item.modified_ns:
                raise LightroomStateError(f"源文件在预检后发生变化：{item.item_id}")
            if source_path_hash(str(source)) != item.path_hash:
                raise LightroomStateError(f"源文件路径在预检后发生变化：{item.item_id}")

    def _apply_error_receipt(self, operation: LightroomOperation, receipt: LightroomReceipt) -> LightroomOperation:
        status: OperationStatus
        if receipt.status in ERROR_RECEIPT_STATUSES:
            status = receipt.status  # type: ignore[assignment]
        else:
            status = "failed"
        return self._replace(
            operation,
            status=status,
            can_execute=False,
            error_code=receipt.error_code or receipt.status,
            error_message=receipt.error_message or "Lightroom 要求重新预检",
            execute_receipt=receipt if operation.execute_request_id == receipt.request_id else operation.execute_receipt,
            preflight_receipt=receipt if operation.preflight_request_id == receipt.request_id else operation.preflight_receipt,
        )

    def _refresh_preflight(self, operation: LightroomOperation) -> LightroomOperation:
        processing = self.bridge.processing / f"{operation.preflight_request_id}.json"
        if operation.status == "waiting_for_plugin" and processing.is_file():
            operation = self._replace(operation, status="preflighting")
        receipt = self.bridge.read_receipt(operation.preflight_request_id)
        if receipt is None:
            return operation
        self._validate_receipt(operation, receipt, request_id=operation.preflight_request_id)
        if (
            operation.preflight_receipt is not None
            and operation.preflight_receipt.model_dump(mode="json") != receipt.model_dump(mode="json")
        ):
            raise LightroomStateError("Lightroom 当前目录或预检基线已发生变化")
        if receipt.status != "awaiting_confirmation":
            return self._apply_error_receipt(operation, receipt)
        if operation.status == "waiting_for_plugin":
            operation = self._replace(operation, status="preflighting")
        can_execute = receipt.counts.invalid == 0 and receipt.baseline_hash is not None
        return self._replace(
            operation,
            status="awaiting_confirmation",
            preflight_receipt=receipt,
            can_execute=can_execute,
            error_code=None,
            error_message=None,
        )

    def _reconcile_completed_journal(
        self,
        operation: LightroomOperation,
        receipt: LightroomReceipt,
    ) -> LightroomReceipt | None:
        """Recover a completed write when a concurrent consumer published a stale drift receipt."""
        manifest = operation.execute_manifest
        if (
            receipt.status != "replan_required"
            or receipt.error_code != "catalog_membership_drift"
            or manifest is None
            or receipt.baseline_hash is None
        ):
            return None
        try:
            journal = self.bridge.read_latest_journal(operation.id)
        except (OSError, ValueError):
            return None
        if journal is None:
            return None
        if (
            journal.get("schema_version") != 1
            or journal.get("operation_id") != operation.id
            or journal.get("preflight_request_id") != manifest.preflight_request_id
            or journal.get("plan_hash") != operation.plan_hash
            or journal.get("baseline_hash") != receipt.baseline_hash
            or journal.get("catalog_identity_hash") != receipt.catalog_identity_hash
            or journal.get("catalog_name") != receipt.catalog_name
        ):
            return None

        journal_items = journal.get("items")
        progress = journal.get("progress")
        if not isinstance(journal_items, list) or not isinstance(progress, dict):
            return None
        expected_items = {item.item_id: item for item in receipt.items}
        journal_by_id: dict[str, dict[str, object]] = {}
        for journal_item in journal_items:
            if not isinstance(journal_item, dict):
                return None
            item_id = journal_item.get("item_id")
            if not isinstance(item_id, str) or item_id in journal_by_id:
                return None
            journal_by_id[item_id] = journal_item
        if set(journal_by_id) != set(expected_items):
            return None

        catalog_added = 0
        verified = 0
        corrected_items = []
        for item_id, expected in expected_items.items():
            journal_item = journal_by_id[item_id]
            if (
                journal_item.get("path_hash") != expected.path_hash
                or journal_item.get("action") != expected.action
                or journal_item.get("target_rating") != expected.target_rating
                or journal_item.get("previous_rating") != expected.previous_rating
            ):
                return None
            if expected.action in {"new", "update"}:
                item_progress = progress.get(item_id)
                if not isinstance(item_progress, dict) or item_progress.get("verified") is not True:
                    return None
                if expected.action == "new":
                    if item_progress.get("catalog_added") is not True:
                        return None
                    catalog_added += 1
                verified += 1
                corrected_items.append(
                    expected.model_copy(
                        update={
                            "final_rating": expected.target_rating,
                            "status": "verified",
                        }
                    )
                )
            else:
                corrected_items.append(
                    expected.model_copy(
                        update={
                            "final_rating": expected.previous_rating,
                            "status": expected.action,
                        }
                    )
                )

        corrected_counts = receipt.counts.model_copy(
            update={
                "catalog_added": catalog_added,
                "pending_rating": 0,
                "verified": verified,
            }
        )
        reconciled = receipt.model_copy(
            update={
                "finished_at": self._now(),
                "status": "complete",
                "counts": corrected_counts,
                "items": corrected_items,
                "error_code": None,
                "error_message": None,
            }
        )
        return LightroomReceipt.model_validate(reconciled.model_dump(mode="json"))

    def _running_progress_counts(self, operation: LightroomOperation) -> LightroomReceiptCounts | None:
        manifest = operation.execute_manifest
        baseline = operation.preflight_receipt
        if manifest is None or baseline is None:
            return None
        try:
            journal = self.bridge.read_latest_journal(operation.id)
        except (OSError, ValueError):
            return None
        if journal is None or (
            journal.get("schema_version") != 1
            or journal.get("operation_id") != operation.id
            or journal.get("preflight_request_id") != manifest.preflight_request_id
            or journal.get("plan_hash") != operation.plan_hash
            or journal.get("baseline_hash") != baseline.baseline_hash
            or journal.get("catalog_identity_hash") != baseline.catalog_identity_hash
            or journal.get("catalog_name") != baseline.catalog_name
        ):
            return None

        journal_items = journal.get("items")
        progress = journal.get("progress")
        if not isinstance(journal_items, list) or not isinstance(progress, dict):
            return None
        expected_ids = {item.item_id for item in baseline.items}
        journal_ids = {
            item.get("item_id")
            for item in journal_items
            if isinstance(item, dict) and isinstance(item.get("item_id"), str)
        }
        if journal_ids != expected_ids or len(journal_items) != len(expected_ids):
            return None

        catalog_added = 0
        pending_rating = 0
        verified = 0
        for item in baseline.items:
            if item.action not in {"new", "update"}:
                continue
            item_progress = progress.get(item.item_id)
            if not isinstance(item_progress, dict):
                continue
            if item.action == "new" and item_progress.get("catalog_added") is True:
                catalog_added += 1
            if item_progress.get("verified") is True:
                verified += 1
            elif item_progress.get("pending_rating") is True:
                pending_rating += 1
        return baseline.counts.model_copy(
            update={
                "catalog_added": catalog_added,
                "pending_rating": pending_rating,
                "verified": verified,
            }
        )

    def _refresh_execute(self, operation: LightroomOperation) -> LightroomOperation:
        if operation.execute_request_id is None or operation.execute_manifest is None:
            raise LightroomStateError("Lightroom 执行请求记录不完整")
        processing = self.bridge.processing / f"{operation.execute_request_id}.json"
        if operation.status == "executing" and processing.is_file():
            operation = self._replace(operation, status="verifying")
        receipt = self.bridge.read_receipt(operation.execute_request_id)
        if receipt is None:
            progress_counts = self._running_progress_counts(operation)
            if progress_counts is not None and progress_counts != operation.progress_counts:
                return self._replace(
                    operation,
                    progress_counts=progress_counts.model_dump(mode="json"),
                )
            return operation
        self._validate_receipt(operation, receipt, request_id=operation.execute_request_id)
        baseline = operation.preflight_receipt
        if (
            baseline is None
            or baseline.baseline_hash is None
            or receipt.baseline_hash != baseline.baseline_hash
            or receipt.catalog_identity_hash != baseline.catalog_identity_hash
        ):
            raise LightroomStateError("Lightroom 当前目录或预检基线已发生变化")
        if receipt.status == "complete":
            if operation.status == "executing":
                operation = self._replace(operation, status="verifying")
            elif operation.status == "pending_rating":
                operation = self._replace(operation, status="executing")
                operation = self._replace(operation, status="verifying")
            return self._replace(
                operation,
                status="complete",
                execute_receipt=receipt,
                progress_counts=None,
                can_execute=False,
            )
        if receipt.status == "pending_rating":
            return self._replace(operation, status="pending_rating", execute_receipt=receipt, can_execute=False)
        reconciled = self._reconcile_completed_journal(operation, receipt)
        if reconciled is not None:
            return self._replace(
                operation,
                status="complete",
                execute_receipt=reconciled,
                can_execute=False,
                error_code=None,
                error_message=None,
            )
        return self._apply_error_receipt(operation, receipt)

    def refresh(self, operation_id: str) -> LightroomOperation:
        with self._lock:
            operation = self.get(operation_id)
            if operation.status in {"waiting_for_plugin", "preflighting", "awaiting_confirmation"}:
                return self._refresh_preflight(operation)
            if operation.status in {"executing", "verifying", "pending_rating"}:
                return self._refresh_execute(operation)
            if (
                operation.status == "failed"
                and operation.error_code == "catalog_membership_drift"
                and operation.execute_request_id is not None
                and operation.execute_manifest is not None
            ):
                return self._refresh_execute(operation)
            return operation

    def _validate_current_preflight(
        self,
        operation: LightroomOperation,
        receipt: LightroomReceipt,
    ) -> None:
        fresh_receipt = self.bridge.read_receipt(operation.preflight_request_id)
        if fresh_receipt is None:
            raise LightroomStateError("Lightroom 预检收据已不可用")
        self._validate_receipt(operation, fresh_receipt, request_id=operation.preflight_request_id)
        if fresh_receipt.model_dump(mode="json") != receipt.model_dump(mode="json"):
            raise LightroomStateError("Lightroom 当前目录或预检基线已发生变化")

    def confirm_execute(self, operation_id: str) -> LightroomOperation:
        with self._lock:
            operation = self.refresh(operation_id)
            if operation.status != "awaiting_confirmation":
                raise LightroomStateError("Lightroom 预检尚未完成，不能执行")
            receipt = operation.preflight_receipt
            if receipt is None or receipt.baseline_hash is None:
                raise LightroomStateError("Lightroom 预检缺少可验证基线")
            if receipt.counts.invalid:
                raise LightroomStateError("Lightroom 预检包含无效项目，不能执行")
            if not operation.can_execute:
                raise LightroomStateError("Lightroom 预检未通过，不能执行")
            self._validate_current_preflight(operation, receipt)
            self._validate_source_fingerprints(operation)

            request_id = uuid4().hex
            execute_manifest = LightroomExecuteManifest(
                request_id=request_id,
                operation_id=operation.id,
                created_at=self._now(),
                plan_hash=operation.plan_hash,
                app_version=self.app_version,
                project=operation.preflight_manifest.project,
                policy=operation.preflight_manifest.policy,
                items=operation.preflight_manifest.items,
                preflight_request_id=operation.preflight_request_id,
                baseline_hash=receipt.baseline_hash,
                catalog_identity_hash=receipt.catalog_identity_hash,
            )
            if plan_hash(execute_manifest) != operation.plan_hash:
                raise LightroomStateError("Lightroom 执行计划哈希漂移")
            executing = self._replace(
                operation,
                status="executing",
                execute_request_id=request_id,
                execute_manifest=execute_manifest,
                can_execute=False,
            )
            try:
                self.bridge.submit(execute_manifest)
            except (OSError, LightroomProtocolError) as exc:
                self._replace(
                    executing,
                    status="failed",
                    error_code="queue_submit_failed",
                    error_message=str(exc),
                )
                raise LightroomStateError("Lightroom 执行请求投递失败") from exc
            return executing

    def retry_pending_rating(self, operation_id: str) -> LightroomOperation:
        """为已记录的 pending_rating 操作生成新的执行请求，不改变原计划。"""
        with self._lock:
            operation = self.refresh(operation_id)
            if operation.status != "pending_rating":
                raise LightroomStateError("只有等待星级回读的 Lightroom 操作可以重试")
            previous_receipt = operation.execute_receipt
            previous_manifest = operation.execute_manifest
            baseline = operation.preflight_receipt
            if previous_receipt is None or previous_manifest is None or baseline is None:
                raise LightroomStateError("Lightroom 等待星级操作缺少可恢复的执行记录")
            if previous_receipt.request_id != operation.execute_request_id:
                raise LightroomStateError("Lightroom 等待星级操作的执行收据不匹配")
            has_pending_evidence = (
                previous_receipt.counts.pending_rating > 0
                or any(
                    chunk.status == "pending_rating" or chunk.error_code == "pending_rating"
                    for chunk in previous_receipt.chunks
                )
                or any(item.status == "pending_rating" for item in previous_receipt.items)
            )
            if previous_receipt.status != "pending_rating" or not has_pending_evidence:
                raise LightroomStateError("Lightroom 操作没有可重试的等待星级项目")
            self._validate_receipt(operation, previous_receipt, request_id=operation.execute_request_id)
            if baseline.baseline_hash is None:
                raise LightroomStateError("Lightroom 预检缺少可验证基线")
            self._validate_current_preflight(operation, baseline)
            self._validate_source_fingerprints(operation)
            if (
                previous_manifest.operation_id != operation.id
                or previous_manifest.plan_hash != operation.plan_hash
                or previous_manifest.preflight_request_id != operation.preflight_request_id
                or previous_manifest.baseline_hash != baseline.baseline_hash
                or previous_manifest.catalog_identity_hash != baseline.catalog_identity_hash
            ):
                raise LightroomStateError("Lightroom 等待星级操作的执行计划不匹配")

            request_id = uuid4().hex
            retry_manifest = previous_manifest.model_copy(
                update={
                    "request_id": request_id,
                    "created_at": self._now(),
                }
            )
            if plan_hash(retry_manifest) != operation.plan_hash:
                raise LightroomStateError("Lightroom 重试计划哈希漂移")
            retrying = self._replace(
                operation,
                status="executing",
                execute_request_id=request_id,
                execute_manifest=retry_manifest,
                can_execute=False,
                error_code=None,
                error_message=None,
            )
            try:
                self.bridge.submit(retry_manifest)
            except (OSError, LightroomProtocolError) as exc:
                self._replace(
                    retrying,
                    status="pending_rating",
                    execute_request_id=operation.execute_request_id,
                    execute_manifest=previous_manifest,
                    execute_receipt=previous_receipt,
                    error_code="queue_submit_failed",
                    error_message=str(exc),
                )
                raise LightroomStateError("Lightroom 重试请求投递失败") from exc
            return retrying

    def request_rollback(self, operation_id: str) -> LightroomOperation:
        with self._lock:
            operation = self.refresh(operation_id)
            if operation.status != "complete" or operation.execute_receipt is None:
                raise LightroomStateError("只有已完成且已验证的 Lightroom 操作可以撤销")
            return self._replace(operation, status="rollback_preflight", can_execute=False)

    def status(self) -> dict[str, Any]:
        heartbeat: dict[str, Any] | None = None
        for candidate in (self.bridge.root / "plugin-heartbeat.json", self.bridge.outbox / "plugin-heartbeat.json"):
            if not candidate.is_file():
                continue
            try:
                value = json.loads(candidate.read_text(encoding="utf-8"))
                heartbeat = value if isinstance(value, dict) else None
            except (OSError, ValueError):
                heartbeat = None
            break
        return {
            "bridge_root": str(self.bridge.root),
            "plugin_heartbeat": heartbeat,
            "queue": {
                "inbox": len(list(self.bridge.inbox.glob("*.json"))),
                "processing": len(list(self.bridge.processing.glob("*.json"))),
                "outbox": len(list(self.bridge.outbox.glob("*.json"))),
                "quarantine": len(list(self.bridge.quarantine.glob("*.json"))),
            },
        }
