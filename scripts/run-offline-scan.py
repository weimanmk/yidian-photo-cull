from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


TERMINAL_STATES = {"completed", "cancelled", "failed"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在独立数据目录运行一点筛图离线回归")
    parser.add_argument("--source-dir", type=Path, required=True, help="照片源目录")
    parser.add_argument("--data-dir", type=Path, required=True, help="项目、缓存与缩略图输出目录")
    parser.add_argument("--model-dir", type=Path, help="本地模型目录，默认使用项目 models")
    parser.add_argument("--preset", choices=("cautious", "balanced", "aggressive"), default="balanced")
    parser.add_argument("--keep-per-group", type=int, choices=range(1, 6), default=1)
    parser.add_argument(
        "--coverage",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="启用人物×环节交付保底",
    )
    parser.add_argument("--coverage-window", type=int, choices=(5, 10, 15, 20, 30, 60), default=15)
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPUExecutionProvider")
    parser.add_argument(
        "--eye-evidence-profile",
        choices=("default", "wide-hard", "conservative-hard"),
        default=None,
        help="眼态硬判证据档位；不传时使用产品冻结默认值",
    )
    parser.add_argument(
        "--cache-hit-previews",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="缓存命中时补生成缩略图和预览图；纯指标 A/B 可用 --no-cache-hit-previews 跳过",
    )
    parser.add_argument("--iqa-scores-file", type=Path, help="冻结的可选 IQA 分数 JSON")
    parser.add_argument("--iqa-dataset", help="IQA 分数 JSON 中的活动键")
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    return parser.parse_args()


def compact_status(status: dict[str, Any]) -> dict[str, Any]:
    return {
        key: status.get(key)
        for key in (
            "status",
            "phase",
            "processed",
            "total",
            "progress",
            "current_file",
            "elapsed_seconds",
            "eta_seconds",
            "cache_hits",
            "cache_misses",
            "error",
            "project_id",
        )
    }


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    source_dir = args.source_dir.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    model_dir = (args.model_dir or project_root / "models").expanduser().resolve()
    if not source_dir.is_dir():
        raise SystemExit(f"照片源目录不存在：{source_dir}")
    if not model_dir.is_dir():
        raise SystemExit(f"模型目录不存在：{model_dir}")
    if bool(args.iqa_scores_file) != bool(args.iqa_dataset):
        raise SystemExit("--iqa-scores-file 与 --iqa-dataset 必须同时提供")

    data_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PHOTOCULL_DATA_DIR"] = str(data_dir)
    os.environ["PHOTOCULL_MODEL_DIR"] = str(model_dir)
    os.environ["PHOTOCULL_DISABLE_VLM"] = "1"
    if args.eye_evidence_profile is not None:
        os.environ["PHOTOCULL_EYE_EVIDENCE_PROFILE"] = args.eye_evidence_profile
    sys.path.insert(0, str(project_root / "backend"))

    from photocull.config import settings_store
    from photocull.project_store import ProjectStore
    from photocull.rating_model import IqaRatingFeatures
    from photocull.scanner import ScannerService

    settings_store.update({"use_gpu": not args.cpu, "vlm_enabled": False})
    rating_feature_provider = None
    if args.iqa_scores_file is not None:
        score_file = args.iqa_scores_file.expanduser().resolve()
        score_payload = json.loads(score_file.read_text(encoding="utf-8"))
        try:
            scores_by_photo = score_payload["scores"][args.iqa_dataset]
        except (KeyError, TypeError) as exc:
            raise SystemExit(f"IQA 分数缺少活动 {args.iqa_dataset}：{score_file}") from exc
        rating_feature_provider = IqaRatingFeatures(scores_by_photo=scores_by_photo)
    scanner = ScannerService(
        ProjectStore(data_dir / "projects"),
        rating_feature_provider=rating_feature_provider,
    )
    scanner.start(
        str(source_dir),
        args.preset,
        args.keep_per_group,
        args.recursive,
        coverage_enabled=args.coverage,
        coverage_window_minutes=args.coverage_window,
        cache_hit_previews=args.cache_hit_previews,
    )

    previous_marker: tuple[Any, ...] | None = None
    while True:
        status = scanner.status()
        marker = (status.get("status"), status.get("phase"), status.get("processed"), status.get("progress"))
        if marker != previous_marker:
            print(json.dumps(compact_status(status), ensure_ascii=False), flush=True)
            previous_marker = marker
        if status.get("status") in TERMINAL_STATES:
            break
        time.sleep(max(0.2, args.poll_seconds))

    result = scanner.results()
    final_payload = {
        "status": compact_status(status),
        "summary": result.get("summary", {}) if result else {},
        "engine": result.get("engine", {}) if result else {},
    }
    print(json.dumps(final_payload, ensure_ascii=False), flush=True)
    return 0 if status.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
