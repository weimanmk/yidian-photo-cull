from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_script() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts" / "run-offline-scan.py"
    spec = importlib.util.spec_from_file_location("run_offline_scan", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_offline_scan_accepts_predeclared_eye_evidence_profile(monkeypatch) -> None:
    module = load_script()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run-offline-scan.py",
            "--source-dir",
            "source",
            "--data-dir",
            "data",
            "--eye-evidence-profile",
            "wide-hard",
        ],
    )

    args = module.parse_args()

    assert args.eye_evidence_profile == "wide-hard"


def test_offline_scan_leaves_profile_unset_when_not_explicit(monkeypatch) -> None:
    module = load_script()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run-offline-scan.py",
            "--source-dir",
            "source",
            "--data-dir",
            "data",
        ],
    )

    args = module.parse_args()

    assert args.eye_evidence_profile is None


def test_offline_scan_can_skip_cache_hit_previews(monkeypatch) -> None:
    module = load_script()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run-offline-scan.py",
            "--source-dir",
            "source",
            "--data-dir",
            "data",
            "--no-cache-hit-previews",
        ],
    )

    args = module.parse_args()

    assert args.cache_hit_previews is False
