from __future__ import annotations

import argparse
import base64
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from photocull.internal_models import PhotoGroupInternal, PhotoObservation
from photocull.project_store import ProjectStore
from photocull.saved_project import build_saved_groups, reference_stems
from photocull.vlm import build_contact_sheet_data_url
from photocull.vlm_training import (
    build_sft_record,
    choose_training_candidates,
    deterministic_split,
    find_single_manual_photo,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 PhotoCull 多模态偏好微调数据集。")
    parser.add_argument("--spec", type=Path, required=True, help="本地数据集规范 JSON")
    parser.add_argument("--output-dir", type=Path, required=True, help="训练包输出目录")
    parser.add_argument("--dry-run", action="store_true", help="只统计样本，不生成联系表或训练文件")
    return parser.parse_args()


def read_spec(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("version", 0)) != 1:
        raise ValueError("数据集规范 version 必须为 1")
    datasets = payload.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("数据集规范至少需要一个 datasets 项")
    names: set[str] = set()
    for item in datasets:
        name = str(item.get("name", ""))
        if not name or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in name):
            raise ValueError(f"数据集名称非法: {name}")
        if name in names:
            raise ValueError(f"数据集名称重复: {name}")
        names.add(name)
    return payload


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    for attempt in range(5):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.2 * (attempt + 1))


def save_contact_sheet(path: Path, candidates: list[PhotoObservation]) -> None:
    data_url = build_contact_sheet_data_url(candidates)
    encoded = data_url.split(",", 1)[1]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(base64.b64decode(encoded))
    for attempt in range(5):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.2 * (attempt + 1))


def manual_photos(group: PhotoGroupInternal, stems: set[str]) -> list[PhotoObservation]:
    return [photo for photo in group.photos if photo.path.stem.casefold() in stems]


def ranking_key(photo: PhotoObservation) -> tuple[int, float, str]:
    rank = photo.rank_in_group if photo.rank_in_group > 0 else 1_000_000
    score = float(photo.metrics.get("group_ranking_score", photo.score))
    return rank, -score, photo.filename.casefold()


def choose_holdout_candidates(
    group: PhotoGroupInternal,
    required: list[PhotoObservation],
    max_candidates: int,
    dataset_name: str,
) -> list[PhotoObservation]:
    negatives = sorted((photo for photo in group.photos if photo not in required), key=ranking_key)
    selected = [*sorted(required, key=ranking_key), *negatives]
    selected = selected[: max(max_candidates, len(required), 2)]

    def stable_position(photo: PhotoObservation) -> bytes:
        return hashlib.sha256(f"{dataset_name}|{group.id}|holdout|{photo.id}".encode("utf-8")).digest()

    return sorted(selected, key=stable_position)


def json_lines(records: list[dict[str, object]]) -> str:
    return "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)


def training_yaml() -> str:
    return """ENV:
  PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
  MAX_PIXELS: '1003520'

model: Qwen/Qwen3.8-27B
tuner_type: lora
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
freeze_vit: true
freeze_aligner: true
freeze_llm: false

dataset:
  - train.jsonl
val_dataset:
  - validation.jsonl
load_from_cache_file: true
max_length: 4096
truncation_strategy: delete
loss_scale: ignore_empty_think
add_non_thinking_prefix: true

torch_dtype: bfloat16
num_train_epochs: 2
per_device_train_batch_size: 1
per_device_eval_batch_size: 1
gradient_accumulation_steps: 16
gradient_checkpointing: true
learning_rate: 5.0e-5
warmup_ratio: 0.1
lr_scheduler_type: cosine
eval_steps: 25
save_steps: 25
save_total_limit: 2
logging_steps: 1
dataloader_num_workers: 2
dataset_num_proc: 2
report_to: none
output_dir: adapter
save_only_model: true
"""


def training_readme(summary: dict[str, Any]) -> str:
    return f"""# PhotoCull Qwen3.8-27B 定向 LoRA 训练包

生成时间：{summary['completed_at']}

- 训练记录：{summary['records']['train']}
- 验证记录：{summary['records']['validation']}
- 锁定测试组：{summary['groups']['test']}
- 联系表：{summary['images']}

## 重要限制

当前 RTX 4070 Laptop 8GB 仅用于部署，不能可靠训练 Qwen3.8-27B。此配置使用 BF16 LoRA、冻结视觉塔和对齐器，目标是单张 80GB GPU；不要在 8GB 本机直接启动。

训练目录必须保持 `train.jsonl`、`validation.jsonl` 与 `images/` 的相对位置。复制到训练机后，在本目录执行：

```bash
pip install -U ms-swift transformers qwen_vl_utils
swift sft --config ms-swift-a100-80gb.yaml
```

训练完成后先在 `holdout.json` 的锁定组上评测。未经 Top-1 净改善与翻错率验收，不得替换现有 GGUF。
"""


def main() -> None:
    args = parse_args()
    spec = read_spec(args.spec.resolve())
    output_dir = args.output_dir.resolve()
    max_candidates = max(2, int(spec.get("max_candidates", 4)))
    train_permutations = max(1, int(spec.get("train_permutations", 2)))
    validation_ratio = float(spec.get("validation_ratio", 0.15))

    records: dict[str, list[dict[str, object]]] = {"train": [], "validation": []}
    manifest: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    dataset_stats: list[dict[str, Any]] = []
    used_images: set[str] = set()

    for dataset in spec["datasets"]:
        name = str(dataset["name"])
        project_id = str(dataset["project_id"])
        reference_dir = Path(str(dataset["reference_dir"])).expanduser().resolve()
        locked = {str(group_id) for group_id in dataset.get("locked_test_groups", [])}
        results, files = ProjectStore().load(project_id)
        groups = build_saved_groups(results, files)
        stems = reference_stems(reference_dir)
        stats = {
            "name": name,
            "project_id": project_id,
            "reference_files": len(stems),
            "groups": len(groups),
            "single_label_groups": 0,
            "multi_label_groups": 0,
            "unlabeled_groups": 0,
            "locked_test_groups": 0,
            "train_groups": 0,
            "validation_groups": 0,
        }

        for group in groups:
            if len(group.photos) < 2:
                continue
            matches = manual_photos(group, stems)
            if not matches:
                stats["unlabeled_groups"] += 1
                continue
            stats["single_label_groups" if len(matches) == 1 else "multi_label_groups"] += 1
            split = deterministic_split(name, group.id, locked, validation_ratio)
            if split == "test":
                candidates = choose_holdout_candidates(group, matches, max_candidates, name)
                image_name = f"{name}-{group.id}-test.jpg"
                image_ref = Path("images") / image_name
                if not args.dry_run:
                    save_contact_sheet(output_dir / image_ref, candidates)
                used_images.add(image_name)
                holdout.append(
                    {
                        "dataset": name,
                        "project_id": project_id,
                        "group_id": group.id,
                        "image": str(image_ref),
                        "candidate_ids": [photo.id for photo in candidates],
                        "candidate_filenames": [photo.filename for photo in candidates],
                        "acceptable_best_photo_ids": [photo.id for photo in matches if photo in candidates],
                        "acceptable_best_filenames": [photo.filename for photo in matches if photo in candidates],
                    }
                )
                stats["locked_test_groups"] += 1
                continue

            manual = find_single_manual_photo(group, stems)
            if manual is None:
                continue
            stats[f"{split}_groups"] += 1
            permutations = train_permutations if split == "train" else 1
            for permutation in range(permutations):
                candidates = choose_training_candidates(
                    group,
                    manual,
                    max_candidates,
                    name,
                    permutation,
                )
                image_name = f"{name}-{group.id}-p{permutation}.jpg"
                image_ref = Path("images") / image_name
                if not args.dry_run:
                    save_contact_sheet(output_dir / image_ref, candidates)
                used_images.add(image_name)
                record = build_sft_record(group, candidates, manual, image_ref)
                records[split].append(record)
                manifest.append(
                    {
                        "dataset": name,
                        "project_id": project_id,
                        "group_id": group.id,
                        "split": split,
                        "permutation": permutation,
                        "image": str(image_ref),
                        "candidate_ids": [photo.id for photo in candidates],
                        "manual_best_id": manual.id,
                        "manual_best_filename": manual.filename,
                    }
                )
        missing_locked = sorted(locked - {item["group_id"] for item in holdout if item["dataset"] == name})
        if missing_locked:
            raise ValueError(f"{name} 锁定测试组未进入 holdout: {missing_locked}")
        dataset_stats.append(stats)

    summary = {
        "status": "dry_run" if args.dry_run else "complete",
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "spec": str(args.spec.resolve()),
        "max_candidates": max_candidates,
        "train_permutations": train_permutations,
        "validation_ratio": validation_ratio,
        "datasets": dataset_stats,
        "groups": {
            "train": sum(item["train_groups"] for item in dataset_stats),
            "validation": sum(item["validation_groups"] for item in dataset_stats),
            "test": len(holdout),
        },
        "records": {key: len(value) for key, value in records.items()},
        "images": len(used_images),
        "holdout_group_ids": [f"{item['dataset']}:{item['group_id']}" for item in holdout],
    }
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    atomic_write_text(output_dir / "train.jsonl", json_lines(records["train"]))
    atomic_write_text(output_dir / "validation.jsonl", json_lines(records["validation"]))
    atomic_write_text(output_dir / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    atomic_write_text(output_dir / "holdout.json", json.dumps(holdout, ensure_ascii=False, indent=2))
    atomic_write_text(output_dir / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2))
    atomic_write_text(output_dir / "ms-swift-a100-80gb.yaml", training_yaml())
    atomic_write_text(output_dir / "README.md", training_readme(summary))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
