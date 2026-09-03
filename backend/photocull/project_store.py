from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import Any

from .config import PROJECTS_DIR
from .project_migrations import migrate_project_payload


class ProjectStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or PROJECTS_DIR
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def _path(self, project_id: str) -> Path:
        if not project_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in project_id):
            raise ValueError("项目 ID 非法")
        return self.root / f"{project_id}.json"

    def save(self, project_id: str, results: dict[str, Any], files: dict[str, str]) -> None:
        payload = {"results": results, "files": files}
        path = self._path(project_id)
        temporary = path.with_suffix(".json.tmp")
        with self._lock:
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(path)

    def load(self, project_id: str) -> tuple[dict[str, Any], dict[str, str]]:
        path = self._path(project_id)
        with self._lock:
            payload = json.loads(path.read_text(encoding="utf-8"))
            normalized = migrate_project_payload(payload)
            if normalized != payload:
                temporary = path.with_suffix(".json.migrate.tmp")
                temporary.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
                temporary.replace(path)
        return normalized["results"], normalized.get("files", {})

    def list(self) -> list[dict[str, Any]]:
        projects: list[dict[str, Any]] = []
        with self._lock:
            paths = sorted(self.root.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        for path in paths:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))["results"]
                summary = payload.get("summary", {})
                projects.append(
                    {
                        "id": payload["project_id"],
                        "name": payload.get("project_name", path.stem),
                        "source_name": payload.get("source_name", ""),
                        "created_at": payload.get("created_at", ""),
                        "total": int(summary.get("total", 0)),
                        "selected": int(summary.get("selected", 0)),
                    }
                )
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return projects
