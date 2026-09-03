from __future__ import annotations

from pathlib import Path


def test_pyinstaller_spec_includes_frozen_rating_asset() -> None:
    source = (Path(__file__).parents[1] / 'photocull-backend.spec').read_text(encoding='utf-8')

    assert 'rating_model_v1.json' in source
    assert 'photocull/assets' in source
