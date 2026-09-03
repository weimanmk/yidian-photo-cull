from __future__ import annotations

import json
import os
from pathlib import Path
from threading import RLock
from uuid import uuid4

from .config import lightroom_bridge_root
from .lightroom_protocol import (
    LightroomManifest,
    LightroomProtocolError,
    LightroomReceipt,
    canonical_json,
    plan_hash,
)


class LightroomBridge:
    DIRECTORY_NAMES = (
        "inbox",
        "processing",
        "outbox",
        "archive",
        "quarantine",
        "journals",
        "operations",
    )

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or lightroom_bridge_root()).expanduser().resolve(strict=False)
        self._lock = RLock()
        for name in self.DIRECTORY_NAMES:
            directory = self.root / name
            directory.mkdir(parents=True, exist_ok=True)
            setattr(self, name, directory)

    @staticmethod
    def _safe_request_id(request_id: str) -> str:
        value = str(request_id).strip()
        if not 16 <= len(value) <= 64 or any(character not in "0123456789abcdefABCDEF-" for character in value):
            raise LightroomProtocolError("Lightroom 请求 ID 非法")
        return value

    @staticmethod
    def _write_atomic(directory: Path, filename: str, payload: bytes) -> Path:
        target = directory / filename
        temporary = directory / f".{uuid4().hex}.tmp"
        try:
            with temporary.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            if target.exists():
                if target.read_bytes() == payload:
                    return target
                raise LightroomProtocolError(f"Lightroom 队列请求已存在且内容不同：{target.name}")
            os.replace(temporary, target)
            return target
        finally:
            temporary.unlink(missing_ok=True)

    def submit(self, manifest: LightroomManifest) -> Path:
        if manifest.plan_hash != plan_hash(manifest):
            raise LightroomProtocolError("Lightroom 计划哈希不匹配")
        request_id = self._safe_request_id(manifest.request_id)
        payload = canonical_json(manifest)
        with self._lock:
            return self._write_atomic(self.inbox, f"{request_id}.json", payload)

    def read_receipt(self, request_id: str) -> LightroomReceipt | None:
        safe_request_id = self._safe_request_id(request_id)
        path = self.outbox / f"{safe_request_id}.json"
        if not path.is_file():
            return None
        return LightroomReceipt.model_validate_json(path.read_bytes())

    def read_latest_journal(self, operation_id: str) -> dict[str, object] | None:
        safe_operation_id = self._safe_request_id(operation_id)
        prefix = f"{safe_operation_id}."
        candidates: list[tuple[int, Path]] = []
        for candidate in self.journals.glob(f"{safe_operation_id}.*.json"):
            suffix = candidate.name[len(prefix) : -len(".json")]
            if suffix.isdecimal():
                candidates.append((int(suffix), candidate))
        if not candidates:
            return None
        _, latest = max(candidates, key=lambda entry: entry[0])
        try:
            payload = json.loads(latest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LightroomProtocolError("Lightroom 执行 journal 已损坏") from exc
        if not isinstance(payload, dict):
            raise LightroomProtocolError("Lightroom 执行 journal 格式无效")
        return payload

    def recover_processing(self) -> list[Path]:
        recovered: list[Path] = []
        with self._lock:
            for source in sorted(self.processing.glob("*.json"), key=lambda path: path.name.casefold()):
                target = self.inbox / source.name
                if target.exists():
                    conflict = self.quarantine / f"{source.stem}.recovery-conflict-{uuid4().hex}.json"
                    os.replace(source, conflict)
                    continue
                os.replace(source, target)
                recovered.append(target)
        return recovered
