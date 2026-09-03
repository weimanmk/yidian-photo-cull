from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对已有一点筛图项目预演人物×环节覆盖保底")
    parser.add_argument("--project", type=Path, required=True, help="projects/*.json 项目文件")
    parser.add_argument("--window", type=int, choices=(5, 10, 15, 20, 30, 60), default=15)
    parser.add_argument("--details-limit", type=int, default=20)
    parser.add_argument("--output", type=Path, help="可选的 JSON 收据路径")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_path = args.project.expanduser().resolve()
    if not project_path.is_file():
        raise SystemExit(f"项目文件不存在：{project_path}")

    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "backend"))
    from photocull.coverage import apply_person_stage_coverage
    from photocull.saved_project import build_saved_groups

    payload = json.loads(project_path.read_text(encoding="utf-8"))
    results = payload.get("results", payload)
    files = payload.get("files", {})
    groups = build_saved_groups(results, files)
    selected_before = sum(photo.category == "selected" for group in groups for photo in group.photos)
    report = apply_person_stage_coverage(groups, enabled=True, window_minutes=args.window)
    protected = [
        {
            "filename": photo.filename,
            "stage": photo.stage_label,
            "people": photo.coverage_person_ids,
            "original_category": photo.coverage_original_category,
            "issues": photo.issues,
            "score": round(photo.score, 2),
            "group_id": photo.group_id,
            "rank_in_group": photo.rank_in_group,
        }
        for group in groups
        for photo in group.photos
        if photo.coverage_protected
    ]
    receipt: dict[str, Any] = {
        "project": str(project_path),
        "window_minutes": args.window,
        "selected_before": selected_before,
        "selected_after": selected_before + len(protected),
        "protected_category_counts": dict(Counter(item["original_category"] for item in protected)),
        "protected_with_quality_warnings": sum(bool(item["issues"]) for item in protected),
        "report": report,
        "protected_preview": protected[: max(0, args.details_limit)],
    }
    text = json.dumps(receipt, ensure_ascii=False, indent=2)
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
