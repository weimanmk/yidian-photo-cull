from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from photocull.config import EngineSettings
from photocull.feature_cache import FeatureCache
from photocull.internal_models import PhotoGroupInternal, PhotoObservation, VisualDescriptor
from photocull.vlm import (
    LlamaServerManager,
    LlamaVlmClient,
    VlmGroupDecision,
    VlmRuntimeConfig,
    apply_vlm_decision,
    build_contact_sheet_data_url,
    build_group_prompt,
    review_groups_with_vlm,
    select_ambiguous_groups,
    validate_group_decision,
)


def make_photo(tmp_path: Path, name: str, score: float, color: tuple[int, int, int]) -> PhotoObservation:
    path = tmp_path / name
    Image.new("RGB", (900, 600), color).save(path, quality=90)
    descriptor = VisualDescriptor(
        phash=0,
        layout=np.zeros(576, dtype=np.float16),
        color=np.zeros(96, dtype=np.float16),
        edge=np.zeros(64, dtype=np.float16),
    )
    return PhotoObservation(
        id=path.stem,
        path=path,
        source_root=tmp_path,
        filename=path.name,
        relative_path=path.name,
        width=900,
        height=600,
        capture_time=None,
        file_sequence=-1,
        descriptor=descriptor,
        metrics={
            "generic_group_score": score,
            "group_ranking_score": score,
            "face_quality_score": 82.0,
            "eye_score": 94.0,
            "motion_blur_score": 86.0,
            "exposure_score": 80.0,
            "composition_score": 76.0,
            "technical_score": 84.0,
            "bad_face_count": 0.0,
        },
        score=score,
        rank_in_group=1 if score >= 80 else 2,
        is_best_pick=score >= 80,
        category="selected" if score >= 80 else "duplicate",
        selection_reasons=["原技术排序"],
    )


def test_llama_command_is_loopback_and_configurable(tmp_path: Path) -> None:
    settings = EngineSettings(
        vlm_enabled=True,
        vlm_server_url="http://127.0.0.1:18768",
        vlm_executable_path=str(tmp_path / "llama-server.exe"),
        vlm_model_path=str(tmp_path / "Qwen3.8-27B-Q4_K_M.gguf"),
        vlm_mmproj_path=str(tmp_path / "mmproj-Qwen3.8-27B-BF16.gguf"),
        vlm_context_size=8192,
        vlm_gpu_layers=16,
    ).validated()
    command = LlamaServerManager().build_command(VlmRuntimeConfig.from_settings(settings))

    assert command[0].endswith("llama-server.exe")
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--port") + 1] == "18768"
    assert command[command.index("--ctx-size") + 1] == "8192"
    assert command[command.index("--n-gpu-layers") + 1] == "16"
    assert "--mmproj" in command


def test_non_loopback_vlm_url_is_rejected() -> None:
    settings = EngineSettings(vlm_server_url="http://0.0.0.0:8768").validated()
    assert settings.vlm_server_url == "http://127.0.0.1:8768"


def test_contact_sheet_is_one_local_image_and_prompt_contains_facts(tmp_path: Path) -> None:
    photos = [
        make_photo(tmp_path, "A.jpg", 82.0, (190, 80, 70)),
        make_photo(tmp_path, "B.jpg", 79.0, (70, 120, 190)),
    ]

    data_url = build_contact_sheet_data_url(photos, image_loader=lambda photo: Image.open(photo.path))
    prompt = build_group_prompt("group-1", photos)

    assert data_url.startswith("data:image/jpeg;base64,")
    decoded = base64.b64decode(data_url.split(",", 1)[1])
    assert decoded.startswith(b"\xff\xd8")
    assert "A" in prompt and "B" in prompt
    assert "eye_score=94.0" in prompt
    assert "不要判断人物身份" in prompt


def test_group_decision_rejects_unknown_ids_and_duplicate_ranks() -> None:
    valid = VlmGroupDecision.model_validate(
        {
            "best_photo_id": "A",
            "ranking": [
                {"photo_id": "A", "rank": 1, "reasons": ["表情更自然"]},
                {"photo_id": "B", "rank": 2, "reasons": ["动作稍差"]},
            ],
            "rejected_photo_ids": ["B"],
            "confidence": 0.82,
            "best_reasons": ["人物互动最好"],
        }
    )
    validate_group_decision(valid, ["A", "B"])

    with pytest.raises(ValueError, match="未知照片"):
        validate_group_decision(valid, ["A", "C"])

    with pytest.raises(ValueError, match="排名"):
        VlmGroupDecision.model_validate(
            {
                "best_photo_id": "A",
                "ranking": [
                    {"photo_id": "A", "rank": 1, "reasons": ["好"]},
                    {"photo_id": "B", "rank": 1, "reasons": ["一般"]},
                ],
                "confidence": 0.8,
                "best_reasons": ["好"],
            }
        )


def test_high_confidence_vlm_decision_reranks_only_safe_candidates(tmp_path: Path) -> None:
    first = make_photo(tmp_path, "A.jpg", 82.0, (190, 80, 70))
    second = make_photo(tmp_path, "B.jpg", 79.0, (70, 120, 190))
    group = PhotoGroupInternal("group-1", [first, second], 0.94, "连拍")
    decision = VlmGroupDecision.model_validate(
        {
            "best_photo_id": "B",
            "ranking": [
                {"photo_id": "B", "rank": 1, "reasons": ["表情与动作更自然"]},
                {"photo_id": "A", "rank": 2, "reasons": ["瞬间感稍弱"]},
            ],
            "rejected_photo_ids": [],
            "confidence": 0.83,
            "best_reasons": ["人物互动更好"],
        }
    )

    changed = apply_vlm_decision(group, [first, second], decision, keep_per_group=1, model_id="Qwen3.8-27B", minimum_confidence=0.65)

    assert changed is True
    assert group.photos[0].id == "B"
    assert group.photos[0].is_best_pick is True
    assert group.photos[0].vlm_rank == 1
    assert group.photos[0].vlm_reasons == ["表情与动作更自然"]
    assert group.vlm_decision is not None
    assert group.vlm_decision["applied"] is True


def test_low_confidence_vlm_decision_keeps_technical_winner(tmp_path: Path) -> None:
    first = make_photo(tmp_path, "A.jpg", 82.0, (190, 80, 70))
    second = make_photo(tmp_path, "B.jpg", 79.0, (70, 120, 190))
    group = PhotoGroupInternal("group-1", [first, second], 0.94, "连拍")
    decision = VlmGroupDecision.model_validate(
        {
            "best_photo_id": "B",
            "ranking": [
                {"photo_id": "B", "rank": 1, "reasons": ["不确定"]},
                {"photo_id": "A", "rank": 2, "reasons": ["不确定"]},
            ],
            "confidence": 0.42,
            "best_reasons": ["不确定"],
        }
    )

    changed = apply_vlm_decision(group, [first, second], decision, keep_per_group=1, model_id="Qwen3.8-27B", minimum_confidence=0.65)

    assert changed is False
    assert group.photos[0].id == "A"
    assert group.vlm_decision is not None
    assert group.vlm_decision["applied"] is False


def test_client_repairs_invalid_json_once_and_uses_single_contact_sheet() -> None:
    calls: list[dict[str, object]] = []
    valid_content = json.dumps(
        {
            "best_photo_id": "A",
            "ranking": [
                {"photo_id": "A", "rank": 1, "reasons": ["表情更好"]},
                {"photo_id": "B", "rank": 2, "reasons": ["动作稍弱"]},
            ],
            "rejected_photo_ids": [],
            "confidence": 0.78,
            "best_reasons": ["表情更好"],
        },
        ensure_ascii=False,
    )

    def transport(_url: str, payload: dict[str, object], _timeout: float) -> dict[str, object]:
        calls.append(payload)
        content = "not-json" if len(calls) == 1 else valid_content
        return {"choices": [{"message": {"content": content}}]}

    config = VlmRuntimeConfig.from_settings(EngineSettings(vlm_enabled=True).validated())
    decision, _raw = LlamaVlmClient(config, transport=transport).rank_group(
        "data:image/jpeg;base64,/9j/2Q==",
        "prompt",
        ["A", "B"],
    )

    assert decision.best_photo_id == "A"
    assert len(calls) == 2
    for payload in calls:
        user_content = payload["messages"][1]["content"]  # type: ignore[index]
        assert sum(item["type"] == "image_url" for item in user_content) == 1  # type: ignore[union-attr]
        assert payload["chat_template_kwargs"] == {"enable_thinking": False, "preserve_thinking": False}


def test_ambiguous_selection_excludes_hard_technical_failure(tmp_path: Path) -> None:
    first = make_photo(tmp_path, "A.jpg", 82.0, (190, 80, 70))
    second = make_photo(tmp_path, "B.jpg", 80.0, (70, 120, 190))
    third = make_photo(tmp_path, "C.jpg", 79.5, (90, 170, 80))
    first.rank_in_group, second.rank_in_group, third.rank_in_group = 1, 2, 3
    second.issues = ["主要人物闭眼"]
    group = PhotoGroupInternal("group-1", [first, second, third], 0.94, "连拍")

    selected = select_ambiguous_groups([group], max_groups=10, max_candidates=8, ambiguity_margin=8.0)

    assert len(selected) == 1
    assert [photo.id for photo in selected[0][1]] == ["A", "C"]


def test_vlm_decision_cache_round_trip(tmp_path: Path) -> None:
    cache = FeatureCache(tmp_path / "cache.db")
    payload = {
        "best_photo_id": "A",
        "ranking": [
            {"photo_id": "A", "rank": 1, "reasons": ["表情更好"]},
            {"photo_id": "B", "rank": 2, "reasons": ["动作稍弱"]},
        ],
        "rejected_photo_ids": [],
        "confidence": 0.78,
        "best_reasons": ["表情更好"],
    }

    cache.save_vlm_decision("key-1", "Qwen3.8-27B", "prompt-v1", payload, "raw")

    assert cache.load_vlm_decision("key-1") == (payload, "raw")


def test_disabled_review_is_a_noop_without_server(tmp_path: Path) -> None:
    first = make_photo(tmp_path, "A.jpg", 82.0, (190, 80, 70))
    second = make_photo(tmp_path, "B.jpg", 80.0, (70, 120, 190))
    first.rank_in_group, second.rank_in_group = 1, 2
    group = PhotoGroupInternal("group-1", [first, second], 0.94, "连拍")

    report = review_groups_with_vlm(
        [group],
        EngineSettings(vlm_enabled=False).validated(),
        FeatureCache(tmp_path / "cache.db"),
        LlamaServerManager(),
    )

    assert report["applied"] is False
    assert report["reviewed_groups"] == 0
    assert group.photos[0].id == "A"


def test_unavailable_enabled_server_falls_back_to_technical_order(tmp_path: Path) -> None:
    first = make_photo(tmp_path, "A.jpg", 82.0, (190, 80, 70))
    second = make_photo(tmp_path, "B.jpg", 80.0, (70, 120, 190))
    first.rank_in_group, second.rank_in_group = 1, 2
    group = PhotoGroupInternal("group-1", [first, second], 0.94, "连拍")
    settings = EngineSettings(
        vlm_enabled=True,
        vlm_server_url="http://127.0.0.1:65534",
        vlm_executable_path="",
        vlm_model_path="",
        vlm_mmproj_path="",
    ).validated()

    report = review_groups_with_vlm(
        [group],
        settings,
        FeatureCache(tmp_path / "cache.db"),
        LlamaServerManager(),
    )

    assert report["available"] is False
    assert report["applied"] is False
    assert "未连接" in report["reason"]
    assert group.photos[0].id == "A"
