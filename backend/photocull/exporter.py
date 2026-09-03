from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any, Literal
from uuid import uuid4


EXPORT_PLAN_VERSION = 1
SEMANTIC_DIRECTORIES = {3: "3星精选", 2: "2星补位", 1: "1星有价值"}
ExportAction = Literal["copy", "skip", "conflict", "invalid"]


class ExportPlanChangedError(RuntimeError):
    """预检后文件系统状态发生变化。"""


class ExportVerificationError(RuntimeError):
    """导出副本无法通过源文件哈希校验。"""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _validate_operation_id(operation_id: str) -> str:
    if len(operation_id) != 32 or any(character not in "0123456789abcdef" for character in operation_id):
        raise ValueError("导出操作 ID 非法")
    return operation_id


def _validate_destination(destination: Path) -> Path:
    resolved = destination.expanduser().resolve(strict=False)
    anchor = Path(resolved.anchor) if resolved.anchor else None
    if anchor is not None and resolved == anchor:
        raise ValueError("不能把磁盘根目录作为导出目标")
    if resolved.exists() and not resolved.is_dir():
        raise ValueError("导出目标必须是文件夹")
    return resolved


def _safe_relative_path(value: object, source: Path) -> Path:
    text = str(value or source.name).strip()
    relative = Path(text)
    if not text or relative.is_absolute() or relative.drive or relative.anchor:
        raise ValueError("照片相对路径非法")
    clean_parts = tuple(part for part in relative.parts if part not in {"", "."})
    if not clean_parts or any(part == ".." for part in clean_parts):
        raise ValueError("照片相对路径不能越过导出目录")
    return Path(*clean_parts)


def _target_for(destination: Path, relative_target: Path) -> Path:
    target = (destination / relative_target).resolve(strict=False)
    try:
        target.relative_to(destination)
    except ValueError as exc:
        raise ValueError("导出目标越过了目标文件夹") from exc
    return target


def _has_non_directory_ancestor(target: Path, destination: Path) -> bool:
    current = target.parent
    while current != destination and current != current.parent:
        if current.exists() and not current.is_dir():
            return True
        current = current.parent
    return destination.exists() and not destination.is_dir()


@dataclass(frozen=True)
class ExportItem:
    photo_id: str
    stars: int
    source: Path | None
    relative_target: Path
    source_size: int | None
    source_mtime_ns: int | None
    source_sha256: str | None
    action: ExportAction
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "photo_id": self.photo_id,
            "stars": self.stars,
            "source": str(self.source) if self.source is not None else None,
            "relative_target": self.relative_target.as_posix(),
            "source_size": self.source_size,
            "source_mtime_ns": self.source_mtime_ns,
            "source_sha256": self.source_sha256,
            "action": self.action,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExportItem":
        source = payload.get("source")
        action = str(payload["action"])
        if action not in {"copy", "skip", "conflict", "invalid"}:
            raise ExportPlanChangedError("导出项目动作非法")
        return cls(
            photo_id=str(payload["photo_id"]),
            stars=int(payload["stars"]),
            source=Path(source) if source else None,
            relative_target=Path(str(payload["relative_target"])),
            source_size=int(payload["source_size"]) if payload.get("source_size") is not None else None,
            source_mtime_ns=(int(payload["source_mtime_ns"]) if payload.get("source_mtime_ns") is not None else None),
            source_sha256=str(payload["source_sha256"]) if payload.get("source_sha256") is not None else None,
            action=action,  # type: ignore[arg-type]
            reason=str(payload.get("reason", "")),
        )


@dataclass(frozen=True)
class ExportPlan:
    operation_id: str
    plan_hash: str
    project_id: str
    destination: Path
    minimum_stars: int
    items: tuple[ExportItem, ...]
    created_at: str
    version: int = EXPORT_PLAN_VERSION

    @property
    def copy_count(self) -> int:
        return sum(item.action == "copy" for item in self.items)

    @property
    def skip_count(self) -> int:
        return sum(item.action == "skip" for item in self.items)

    @property
    def conflict_count(self) -> int:
        return sum(item.action == "conflict" for item in self.items)

    @property
    def invalid_count(self) -> int:
        return sum(item.action == "invalid" for item in self.items)

    def hash_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "project_id": self.project_id,
            "destination": str(self.destination),
            "minimum_stars": self.minimum_stars,
            "items": [item.to_dict() for item in self.items],
        }

    def expected_hash(self) -> str:
        return hashlib.sha256(_canonical_json(self.hash_payload()).encode("utf-8")).hexdigest().upper()

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "plan_hash": self.plan_hash,
            "project_id": self.project_id,
            "destination": str(self.destination),
            "minimum_stars": self.minimum_stars,
            "created_at": self.created_at,
            "version": self.version,
            "copy_count": self.copy_count,
            "skip_count": self.skip_count,
            "conflict_count": self.conflict_count,
            "invalid_count": self.invalid_count,
            "items": [item.to_dict() for item in self.items],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExportPlan":
        plan = cls(
            operation_id=_validate_operation_id(str(payload["operation_id"])),
            plan_hash=str(payload["plan_hash"]),
            project_id=str(payload["project_id"]),
            destination=_validate_destination(Path(str(payload["destination"]))),
            minimum_stars=int(payload["minimum_stars"]),
            items=tuple(ExportItem.from_dict(item) for item in payload.get("items", [])),
            created_at=str(payload.get("created_at", "")),
            version=int(payload.get("version", 0)),
        )
        if plan.version != EXPORT_PLAN_VERSION or plan.plan_hash != plan.expected_hash():
            raise ExportPlanChangedError("导出计划文件已损坏或被修改")
        return plan


@dataclass(frozen=True)
class ExportReceipt:
    operation_id: str
    plan_hash: str
    destination: Path
    copied: int
    skipped: int
    conflicts: int
    invalid: int
    verification_passed: bool
    completed_at: str
    items: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "plan_hash": self.plan_hash,
            "destination": str(self.destination),
            "copied": self.copied,
            "skipped": self.skipped,
            "conflicts": self.conflicts,
            "invalid": self.invalid,
            "verification_passed": self.verification_passed,
            "completed_at": self.completed_at,
            "items": list(self.items),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExportReceipt":
        return cls(
            operation_id=_validate_operation_id(str(payload["operation_id"])),
            plan_hash=str(payload["plan_hash"]),
            destination=Path(str(payload["destination"])),
            copied=int(payload["copied"]),
            skipped=int(payload["skipped"]),
            conflicts=int(payload["conflicts"]),
            invalid=int(payload.get("invalid", 0)),
            verification_passed=bool(payload["verification_passed"]),
            completed_at=str(payload["completed_at"]),
            items=tuple(dict(item) for item in payload.get("items", [])),
        )


def _semantic_project_id(results: dict[str, Any]) -> str:
    if (
        int(results.get("schema_version", 0)) != 2
        or results.get("rating_migration_status") != "native"
        or results.get("lightroom_ready") is not True
    ):
        raise ValueError("当前项目不是 v0.2.1 原生语义星级，请重新扫描后再导出")
    project_id = str(results.get("project_id", "")).strip()
    if not project_id:
        raise ValueError("项目缺少有效 ID")
    return project_id


def _invalid_item(photo_id: str, stars: int, relative_target: Path, reason: str) -> ExportItem:
    return ExportItem(photo_id, stars, None, relative_target, None, None, None, "invalid", reason)


def build_export_plan(
    results: dict[str, Any],
    files: dict[str, str],
    destination: Path,
    minimum_stars: int,
) -> ExportPlan:
    project_id = _semantic_project_id(results)
    if minimum_stars not in SEMANTIC_DIRECTORIES:
        raise ValueError("最低星级必须是 1、2 或 3")
    resolved_destination = _validate_destination(destination)
    candidates = sorted(
        (photo for photo in results.get("photos", []) if minimum_stars <= int(photo.get("stars", 0)) <= 3),
        key=lambda photo: (
            -int(photo.get("stars", 0)),
            str(photo.get("relative_path", "")).casefold(),
            str(photo.get("id", "")),
        ),
    )
    items: list[ExportItem] = []
    planned_targets: set[str] = set()
    for photo in candidates:
        photo_id = str(photo.get("id", "")).strip()
        stars = int(photo.get("stars", 0))
        source_text = files.get(photo_id)
        source = Path(source_text).expanduser().resolve(strict=False) if source_text else None
        fallback_source = source if source is not None else Path(f"{photo_id or 'unknown'}.jpg")
        try:
            relative_source = _safe_relative_path(photo.get("relative_path"), fallback_source)
            relative_target = Path(SEMANTIC_DIRECTORIES[stars]) / relative_source
            target = _target_for(resolved_destination, relative_target)
        except ValueError as exc:
            items.append(_invalid_item(photo_id, stars, Path(SEMANTIC_DIRECTORIES[stars]) / fallback_source.name, str(exc)))
            continue
        target_key = os.path.normcase(str(target))
        if target_key in planned_targets:
            items.append(_invalid_item(photo_id, stars, relative_target, "多个源文件映射到同一目标"))
            continue
        planned_targets.add(target_key)
        if source is None or not source.is_file():
            items.append(_invalid_item(photo_id, stars, relative_target, "源文件不存在"))
            continue
        if _has_non_directory_ancestor(target, resolved_destination):
            items.append(_invalid_item(photo_id, stars, relative_target, "目标路径的父级不是文件夹"))
            continue
        stat = source.stat()
        source_hash = _sha256(source)
        action: ExportAction = "copy"
        reason = "目标不存在，可以安全复制"
        if target.exists():
            if target.is_file() and target.stat().st_size == stat.st_size and _sha256(target) == source_hash:
                action, reason = "skip", "目标已有完全相同的文件"
            else:
                action, reason = "conflict", "目标已有不同内容，不会覆盖"
        items.append(
            ExportItem(photo_id, stars, source, relative_target, stat.st_size, stat.st_mtime_ns, source_hash, action, reason)
        )
    draft = ExportPlan(uuid4().hex, "", project_id, resolved_destination, minimum_stars, tuple(items), _utc_now())
    return ExportPlan(
        draft.operation_id,
        draft.expected_hash(),
        draft.project_id,
        draft.destination,
        draft.minimum_stars,
        draft.items,
        draft.created_at,
    )


def _validate_source(item: ExportItem) -> None:
    if item.source is None or item.source_size is None or item.source_mtime_ns is None or item.source_sha256 is None:
        raise ExportPlanChangedError(f"源文件信息不完整：{item.photo_id}")
    if not item.source.is_file():
        raise ExportPlanChangedError(f"源文件已不存在：{item.source}")
    stat = item.source.stat()
    if stat.st_size != item.source_size or stat.st_mtime_ns != item.source_mtime_ns:
        raise ExportPlanChangedError(f"源文件在预检后发生变化：{item.source}")
    if _sha256(item.source) != item.source_sha256:
        raise ExportPlanChangedError(f"源文件内容在预检后发生变化：{item.source}")


def _validate_target_state(plan: ExportPlan, item: ExportItem) -> Path:
    target = _target_for(plan.destination, item.relative_target)
    if item.action == "copy":
        if target.exists() or _has_non_directory_ancestor(target, plan.destination):
            raise ExportPlanChangedError(f"目标路径在预检后发生变化：{target}")
    elif item.action == "skip":
        if not target.is_file() or _sha256(target) != item.source_sha256:
            raise ExportPlanChangedError(f"相同文件在预检后发生变化：{target}")
    elif item.action == "conflict":
        if not target.exists():
            raise ExportPlanChangedError(f"冲突文件在预检后发生变化：{target}")
        if target.is_file() and _sha256(target) == item.source_sha256:
            raise ExportPlanChangedError(f"冲突文件在预检后发生变化：{target}")
    return target


def _copy_exclusive(item: ExportItem, target: Path, operation_id: str) -> None:
    if item.source is None or item.source_sha256 is None:
        raise ExportPlanChangedError(f"源文件信息不完整：{item.photo_id}")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.parent / f".{target.name}.{operation_id}.partial"
    created_target = False
    try:
        with item.source.open("rb") as source_stream, partial.open("xb") as partial_stream:
            shutil.copyfileobj(source_stream, partial_stream, length=1024 * 1024)
            partial_stream.flush()
            os.fsync(partial_stream.fileno())
        if _sha256(partial) != item.source_sha256:
            raise ExportVerificationError(f"临时副本校验失败：{target}")
        try:
            os.link(partial, target)
            created_target = True
        except FileExistsError as exc:
            raise ExportPlanChangedError(f"目标文件在执行期间出现：{target}") from exc
        except OSError:
            try:
                with partial.open("rb") as partial_stream, target.open("xb") as target_stream:
                    created_target = True
                    shutil.copyfileobj(partial_stream, target_stream, length=1024 * 1024)
                    target_stream.flush()
                    os.fsync(target_stream.fileno())
            except FileExistsError as exc:
                raise ExportPlanChangedError(f"目标文件在执行期间出现：{target}") from exc
        if not target.is_file() or _sha256(target) != item.source_sha256:
            raise ExportVerificationError(f"目标文件校验失败：{target}")
    except Exception:
        if created_target and target.exists():
            target.unlink(missing_ok=True)
        raise
    finally:
        partial.unlink(missing_ok=True)


def execute_export_plan(plan: ExportPlan) -> ExportReceipt:
    if plan.version != EXPORT_PLAN_VERSION or plan.plan_hash != plan.expected_hash():
        raise ExportPlanChangedError("导出计划哈希不匹配")
    targets: dict[str, Path] = {}
    for item in plan.items:
        if item.action == "invalid":
            continue
        _validate_source(item)
        targets[item.photo_id] = _validate_target_state(plan, item)

    outcomes: list[dict[str, Any]] = []
    copied = 0
    for item in plan.items:
        target = _target_for(plan.destination, item.relative_target)
        if item.action == "copy":
            _copy_exclusive(item, targets[item.photo_id], plan.operation_id)
            copied += 1
            status = "copied"
        elif item.action == "skip":
            status = "skipped"
        elif item.action == "conflict":
            status = "conflict"
        else:
            status = "invalid"
        outcomes.append(
            {"photo_id": item.photo_id, "stars": item.stars, "target": str(target), "status": status, "source_sha256": item.source_sha256}
        )
    verification_passed = all(
        item.action != "copy"
        or (
            _target_for(plan.destination, item.relative_target).is_file()
            and _sha256(_target_for(plan.destination, item.relative_target)) == item.source_sha256
        )
        for item in plan.items
    )
    if not verification_passed:
        raise ExportVerificationError("导出后的权威读回校验失败")
    return ExportReceipt(
        plan.operation_id,
        plan.plan_hash,
        plan.destination,
        copied,
        plan.skip_count,
        plan.conflict_count,
        plan.invalid_count,
        True,
        _utc_now(),
        tuple(outcomes),
    )


class ExportOperationStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._lock = RLock()

    def _directory(self, operation_id: str) -> Path:
        return self.root / _validate_operation_id(operation_id)

    @staticmethod
    def _write_once(path: Path, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        try:
            with path.open("xb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != payload:
                raise ExportPlanChangedError(f"操作记录已存在且内容不同：{path}")

    def save_plan(self, plan: ExportPlan) -> None:
        if plan.plan_hash != plan.expected_hash():
            raise ExportPlanChangedError("导出计划哈希不匹配")
        with self._lock:
            directory = self._directory(plan.operation_id)
            directory.mkdir(parents=True, exist_ok=True)
            self._write_once(directory / "plan.json", plan.to_dict())

    def load_plan(self, operation_id: str) -> ExportPlan:
        with self._lock:
            path = self._directory(operation_id) / "plan.json"
            if not path.is_file():
                raise FileNotFoundError("导出预检不存在")
            return ExportPlan.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def load_receipt(self, operation_id: str) -> ExportReceipt | None:
        with self._lock:
            path = self._directory(operation_id) / "receipt.json"
            if not path.is_file():
                return None
            return ExportReceipt.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def save_receipt(self, receipt: ExportReceipt) -> None:
        with self._lock:
            directory = self._directory(receipt.operation_id)
            directory.mkdir(parents=True, exist_ok=True)
            self._write_once(directory / "receipt.json", receipt.to_dict())

    def execute(self, operation_id: str, plan_hash: str) -> ExportReceipt:
        with self._lock:
            existing = self.load_receipt(operation_id)
            if existing is not None:
                if existing.plan_hash != plan_hash:
                    raise ExportPlanChangedError("确认的计划哈希与已执行记录不一致")
                return existing
            plan = self.load_plan(operation_id)
            if plan.plan_hash != plan_hash:
                raise ExportPlanChangedError("确认的计划哈希与预检结果不一致")
            receipt = execute_export_plan(plan)
            self.save_receipt(receipt)
            return receipt
