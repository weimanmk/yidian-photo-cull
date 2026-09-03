from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SPIKE_DIR = ROOT / "output" / "alternative-culling-spike"
SPIKE_PATH = SPIKE_DIR / "experiment.py"
IQA_PATH = SPIKE_DIR / "iqa-scores.json"
IQA_FEATURE_COUNT = 6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_spike() -> ModuleType:
    spec = importlib.util.spec_from_file_location("photocull_frozen_spike", SPIKE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载冻结实验: {SPIKE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def model_payload(model: Any, feature_names: tuple[str, ...]) -> dict[str, Any]:
    return {
        "feature_names": list(feature_names),
        "mean": model.mean.tolist(),
        "scale": model.scale.tolist(),
        "weights": model.weights.tolist(),
        "bias": float(model.bias),
    }


def export_model(frozen_config: Path, output: Path) -> str:
    frozen = json.loads(frozen_config.read_text(encoding="utf-8"))
    if frozen.get("holdout_reference_read_at_freeze") is not False:
        raise RuntimeError("冻结配置缺少未读取最终验收标签的来源标记")
    if frozen.get("no_post_holdout_tuning") is not True:
        raise RuntimeError("冻结配置缺少禁止验收后调参标记")
    if sha256(IQA_PATH) != str(frozen["training_iqa_scores_sha256"]).upper():
        raise RuntimeError("训练 IQA 分数哈希与冻结配置不一致")

    spike = load_spike()
    datasets = [spike.load_dataset(paths) for paths in spike.DATASET_PATHS]
    actual_hashes = {dataset.name: dataset.project_hash.upper() for dataset in datasets}
    expected_hashes = {
        str(name): str(value).upper() for name, value in dict(frozen["training_events"]).items()
    }
    if actual_hashes != expected_hashes:
        raise RuntimeError(
            "训练项目哈希与冻结配置不一致: "
            + json.dumps({"expected": expected_hashes, "actual": actual_hashes}, ensure_ascii=False)
        )

    feature_names = tuple(str(name) for name in spike.PREFERENCE_FEATURES)
    if len(feature_names) <= IQA_FEATURE_COUNT:
        raise RuntimeError("冻结实验特征数量异常")

    original_vectors = {
        photo.id: photo.vector.copy() for dataset in datasets for photo in dataset.photos
    }
    iqa_model = spike.fit_pointwise(datasets)
    try:
        for dataset in datasets:
            for photo in dataset.photos:
                photo.vector = original_vectors[photo.id][:-IQA_FEATURE_COUNT]
        base_model = spike.fit_pointwise(datasets)
    finally:
        for dataset in datasets:
            for photo in dataset.photos:
                photo.vector = original_vectors[photo.id]

    configuration = dict(frozen["configuration"])
    selection_parameters = {
        key: configuration[key]
        for key in (
            "target_reduction",
            "learned_alpha",
            "group_demote",
            "duplicate_demote",
            "duplicate_scene_floor",
            "duplicate_pose_floor",
            "duplicate_max_sequence_span",
            "duplicate_max_time_span_seconds",
        )
    }
    payload = {
        "version": "rating-pointwise-v1",
        "frozen_candidate_sha256": sha256(frozen_config),
        "training_hashes": expected_hashes,
        "training_iqa_scores_sha256": sha256(IQA_PATH),
        "frozen_implementation_sha256": str(frozen.get("implementation_sha256", "")).upper(),
        "export_implementation_sha256": sha256(SPIKE_PATH),
        "selection_parameters": selection_parameters,
        "profiles": {
            "base": model_payload(base_model, feature_names[:-IQA_FEATURE_COUNT]),
            "iqa": model_payload(iqa_model, feature_names),
        },
    }
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(output)
    return sha256(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出一点筛图冻结语义评分模型")
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    digest = export_model(args.frozen_config.resolve(), args.output.resolve())
    print(f"sha256: {digest}")
    print("profiles: base,iqa")


if __name__ == "__main__":
    main()
