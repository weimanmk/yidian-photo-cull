from __future__ import annotations

from pathlib import Path

import photocull.config as config


def test_missing_settings_still_enforces_packaged_vlm_disable(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(config, "VLM_DISABLED", True)

    settings = config.SettingsStore(tmp_path / "missing-settings.json").get()

    assert settings.vlm_enabled is False
    assert settings.vlm_executable_path == ""
    assert settings.vlm_model_path == ""
    assert settings.vlm_mmproj_path == ""
    assert settings.vlm_model_id == "disabled"
    assert settings.vlm_quantization == "none"
