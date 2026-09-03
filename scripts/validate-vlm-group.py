from __future__ import annotations

import argparse
from datetime import datetime
import json
import time
from pathlib import Path
from typing import Any

from photocull.config import SettingsStore
from photocull.project_store import ProjectStore
from photocull.saved_project import build_saved_groups, reference_stems
from photocull.vlm import (
    VLM_PROMPT_VERSION,
    LlamaServerManager,
    LlamaVlmClient,
    VlmRuntimeConfig,
    build_contact_sheet_data_url,
    build_group_prompt,
    select_ambiguous_groups,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用当前本地 VLM 配置复核已保存的照片组。")
    parser.add_argument("project_id", help="PhotoCull 本机项目 ID")
    parser.add_argument("group_id", nargs="?", help="项目中的单个照片组 ID")
    parser.add_argument("--reference-dir", type=Path, help="人工参考目录；提供后自动选择疑难组批量 A/B")
    parser.add_argument("--max-groups", type=int, default=None, help="批量验证的最大组数，默认使用当前设置")
    parser.add_argument("--output", type=Path, help="批量验证 JSON 输出路径")
    parser.add_argument("--dry-run", action="store_true", help="只输出候选组清单，不启动模型")
    return parser.parse_args()


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [record for record in records if record.get("status") == "success"]
    return {
        "requested_groups": len(records),
        "successes": len(successes),
        "failures": len(records) - len(successes),
        "baseline_top1_hits": sum(bool(record.get("baseline_hit")) for record in successes),
        "vlm_raw_top1_hits": sum(bool(record.get("vlm_raw_hit")) for record in successes),
        "effective_top1_hits": sum(bool(record.get("effective_hit")) for record in successes),
        "effective_top2_hits": sum(bool(record.get("vlm_top2_hit")) for record in successes),
        "applied_changes": sum(bool(record.get("applied_changed_winner")) for record in successes),
        "improvements": sum(record.get("effect") == "improved" for record in successes),
        "regressions": sum(record.get("effect") == "regressed" for record in successes),
        "neutral": sum(record.get("effect") == "neutral" for record in successes),
        "average_group_seconds": round(
            sum(float(record.get("elapsed_seconds", 0.0)) for record in successes) / len(successes), 2
        )
        if successes
        else 0.0,
    }


def write_batch_output(path: Path, payload: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for attempt in range(5):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.2 * (attempt + 1))


def main() -> None:
    args = parse_args()
    if bool(args.group_id) == bool(args.reference_dir):
        raise SystemExit("请提供 group_id，或只提供 --reference-dir 进行批量验证")

    results, files = ProjectStore().load(args.project_id)
    groups = build_saved_groups(results, files)
    group_by_id = {group.id: group for group in groups}
    settings = SettingsStore().get()
    config = VlmRuntimeConfig.from_settings(settings)

    manual_stems: set[str] | None = None
    if args.group_id:
        group = group_by_id.get(args.group_id)
        if group is None:
            raise SystemExit(f"照片组不存在: {args.group_id}")
        if not 2 <= len(group.photos) <= 8:
            raise SystemExit("单组验证必须包含 2-8 张照片")
        selected = [(group, group.photos, 0.0)]
    else:
        manual_stems = reference_stems(args.reference_dir)
        limit = max(1, int(args.max_groups or settings.vlm_max_groups))
        ambiguous = select_ambiguous_groups(
            groups,
            max_groups=len(groups),
            max_candidates=settings.vlm_max_candidates,
            ambiguity_margin=settings.vlm_ambiguity_margin,
        )
        selected = [
            item
            for item in ambiguous
            if any(photo.path.stem.casefold() in manual_stems for photo in item[1])
        ][:limit]
        if not selected:
            raise SystemExit("没有找到人工参考落在候选集内的疑难组")

    output: dict[str, Any] = {
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "project_id": args.project_id,
        "project_name": results.get("project_name"),
        "reference_dir": str(args.reference_dir.resolve()) if args.reference_dir else None,
        "reference_count": len(manual_stems) if manual_stems is not None else None,
        "selection_method": "production ambiguity gates, then require human reference inside candidates"
        if manual_stems is not None
        else "explicit group",
        "prompt_version": VLM_PROMPT_VERSION,
        "model_id": config.model_id,
        "quantization": config.quantization,
        "minimum_confidence": settings.vlm_min_confidence,
        "max_candidates": settings.vlm_max_candidates,
        "selected_groups": [
            {
                "group_id": group.id,
                "gap": round(gap, 4),
                "candidates": [photo.filename for photo in candidates],
                "manual_candidates": [
                    photo.filename
                    for photo in candidates
                    if manual_stems is not None and photo.path.stem.casefold() in manual_stems
                ],
            }
            for group, candidates, gap in selected
        ],
        "records": [],
        "summary": {},
    }
    if args.output:
        write_batch_output(args.output, output)
    if args.dry_run:
        output["status"] = "dry_run"
        if args.output:
            write_batch_output(args.output, output)
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return

    runtime = LlamaServerManager()
    total_started = time.monotonic()
    try:
        ready, message = runtime.ensure_ready(config)
        if not ready:
            raise SystemExit(message)
        client = LlamaVlmClient(config)
        print(f"READY {message} groups={len(selected)}", flush=True)
        for index, (group, candidates, gap) in enumerate(selected, start=1):
            group_started = time.monotonic()
            candidate_ids = [photo.id for photo in candidates]
            filename_by_id = {photo.id: photo.filename for photo in candidates}
            manual_ids = {
                photo.id
                for photo in candidates
                if manual_stems is not None and photo.path.stem.casefold() in manual_stems
            }
            print(f"START {index}/{len(selected)} {group.id} candidates={len(candidates)} gap={gap:.4f}", flush=True)
            try:
                decision, raw_response = client.rank_group(
                    build_contact_sheet_data_url(candidates),
                    build_group_prompt(group.id, candidates),
                    candidate_ids,
                )
                ordered = sorted(decision.ranking, key=lambda item: item.rank)
                baseline_id = candidate_ids[0]
                raw_best_id = decision.best_photo_id
                applied = decision.confidence >= settings.vlm_min_confidence
                effective_id = raw_best_id if applied else baseline_id
                baseline_hit = bool(manual_ids and baseline_id in manual_ids)
                raw_hit = bool(manual_ids and raw_best_id in manual_ids)
                effective_hit = bool(manual_ids and effective_id in manual_ids)
                effect = "neutral"
                if not baseline_hit and effective_hit:
                    effect = "improved"
                elif baseline_hit and not effective_hit:
                    effect = "regressed"
                record = {
                    "status": "success",
                    "group_id": group.id,
                    "gap": round(gap, 4),
                    "candidate_filenames": [photo.filename for photo in candidates],
                    "manual_filenames": [filename_by_id[identifier] for identifier in manual_ids],
                    "baseline_filename": filename_by_id[baseline_id],
                    "vlm_best_filename": filename_by_id[raw_best_id],
                    "effective_best_filename": filename_by_id[effective_id],
                    "confidence": decision.confidence,
                    "applied": applied,
                    "applied_changed_winner": applied and raw_best_id != baseline_id,
                    "baseline_hit": baseline_hit,
                    "vlm_raw_hit": raw_hit,
                    "effective_hit": effective_hit,
                    "vlm_top2_hit": bool(manual_ids.intersection(item.photo_id for item in ordered[:2])),
                    "effect": effect,
                    "best_reasons": decision.best_reasons,
                    "ranking": [
                        {
                            "rank": item.rank,
                            "filename": filename_by_id[item.photo_id],
                            "reasons": item.reasons,
                        }
                        for item in ordered
                    ],
                    "raw_response": raw_response,
                    "elapsed_seconds": round(time.monotonic() - group_started, 2),
                }
                print(
                    f"DONE {index}/{len(selected)} {group.id} baseline={record['baseline_filename']} "
                    f"vlm={record['vlm_best_filename']} effect={effect} seconds={record['elapsed_seconds']}",
                    flush=True,
                )
            except Exception as exc:
                record = {
                    "status": "failed",
                    "group_id": group.id,
                    "gap": round(gap, 4),
                    "candidate_filenames": [photo.filename for photo in candidates],
                    "manual_filenames": [filename_by_id[identifier] for identifier in manual_ids],
                    "error": str(exc),
                    "elapsed_seconds": round(time.monotonic() - group_started, 2),
                }
                print(f"FAILED {index}/{len(selected)} {group.id} error={exc}", flush=True)
            output["records"].append(record)
            output["summary"] = summarize(output["records"])
            if args.output:
                write_batch_output(args.output, output)
    finally:
        runtime.stop()

    output["status"] = "complete" if not output["summary"].get("failures") else "complete_with_errors"
    output["completed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    output["total_elapsed_seconds"] = round(time.monotonic() - total_started, 2)
    if args.output:
        write_batch_output(args.output, output)
        print(f"OUTPUT {args.output.resolve()}", flush=True)
    elif len(output["records"]) == 1:
        print(json.dumps(output["records"][0], ensure_ascii=False, indent=2))
    else:
        print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
