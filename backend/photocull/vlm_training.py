from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .internal_models import PhotoGroupInternal, PhotoObservation
from .vlm import build_group_prompt


TRAINING_SYSTEM_PROMPT = (
    "你是严谨的职业摄影选片师，只比较同一场景、同一人物组合的候选照片。"
    "专业人脸与技术检测是高优先级事实；不要因为想象出的氛围或故事推翻这些事实。"
    "输出必须是完整 JSON，不要输出思考过程或 Markdown。"
)


def _stem(photo: PhotoObservation) -> str:
    return photo.path.stem.casefold()


def _ranking_key(photo: PhotoObservation) -> tuple[int, float, str]:
    rank = photo.rank_in_group if photo.rank_in_group > 0 else 1_000_000
    score = float(photo.metrics.get("group_ranking_score", photo.score))
    return rank, -score, photo.filename.casefold()


def find_single_manual_photo(
    group: PhotoGroupInternal,
    manual_stems: set[str],
) -> PhotoObservation | None:
    matches = [photo for photo in group.photos if _stem(photo) in manual_stems]
    return matches[0] if len(matches) == 1 else None


def choose_training_candidates(
    group: PhotoGroupInternal,
    manual_photo: PhotoObservation,
    max_candidates: int,
    dataset_name: str,
    permutation: int,
) -> list[PhotoObservation]:
    if manual_photo not in group.photos:
        raise ValueError("人工首选不属于目标照片组")
    limit = max(2, int(max_candidates))
    negatives = sorted((photo for photo in group.photos if photo is not manual_photo), key=_ranking_key)
    selected = [manual_photo, *negatives[: limit - 1]]
    if len(selected) < 2:
        raise ValueError("训练候选组至少需要两张照片")

    def stable_position(photo: PhotoObservation) -> bytes:
        value = f"{dataset_name}|{group.id}|{photo.id}".encode("utf-8")
        return hashlib.sha256(value).digest()

    ordered = sorted(selected, key=stable_position)
    offset = max(0, int(permutation)) % len(ordered)
    return ordered[offset:] + ordered[:offset]


def deterministic_split(
    dataset_name: str,
    group_id: str,
    locked_test_groups: set[str],
    validation_ratio: float,
) -> str:
    if group_id in locked_test_groups:
        return "test"
    ratio = min(0.5, max(0.0, float(validation_ratio)))
    digest = hashlib.sha256(f"{dataset_name}|{group_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(2**64)
    return "validation" if bucket < ratio else "train"


def build_sft_record(
    group: PhotoGroupInternal,
    candidates: list[PhotoObservation],
    manual_photo: PhotoObservation,
    image_path: Path,
) -> dict[str, object]:
    if manual_photo not in candidates:
        raise ValueError("人工首选必须位于训练候选中")
    if len({photo.id for photo in candidates}) != len(candidates):
        raise ValueError("训练候选包含重复照片 ID")

    remaining = sorted((photo for photo in candidates if photo is not manual_photo), key=_ranking_key)
    target_order = [manual_photo, *remaining]
    answer = {
        "best_photo_id": manual_photo.id,
        "ranking": [
            {
                "photo_id": photo.id,
                "rank": index,
                "reasons": [
                    "更符合目标摄影师的同组保留偏好"
                    if index == 1
                    else "同组综合优先级低于首选"
                ],
            }
            for index, photo in enumerate(target_order, start=1)
        ],
        "rejected_photo_ids": [],
        "confidence": 0.8,
        "best_reasons": ["更符合目标摄影师的同组保留偏好"],
    }
    user_prompt = (
        "<image>\n"
        + build_group_prompt(group.id, candidates)
        + "\n人工偏好微调要求：只学习组内相对选择，不把未选照片虚构成技术废片；"
        "没有独立硬证据时，不得输出极端置信度。"
    )
    return {
        "messages": [
            {"role": "system", "content": TRAINING_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {
                "role": "assistant",
                "content": json.dumps(answer, ensure_ascii=False, separators=(",", ":")),
            },
        ],
        "images": [str(image_path)],
    }
