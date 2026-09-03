"""把当前本地扫描结果与同名人工“已修”文件集进行可复查评测。"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


REFERENCE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
TOP_K_VALUES = (1, 2, 3, 5)


def fetch_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def fetch_image(url: str) -> Image.Image:
    with urllib.request.urlopen(url, timeout=30) as response:
        return Image.open(io.BytesIO(response.read())).convert("RGB")


def load_project_result(project_file: Path | None, api_base: str) -> dict[str, Any]:
    if project_file is None:
        return fetch_json(f"{api_base}/api/scan/results")
    payload = json.loads(project_file.read_text(encoding="utf-8"))
    result = payload.get("results", payload)
    if not isinstance(result, dict) or "photos" not in result:
        raise ValueError(f"项目结果格式无效: {project_file}")
    return result


def infer_thumbnail_dir(project_file: Path, result: dict[str, Any]) -> Path | None:
    data_root = project_file.resolve().parent.parent
    candidates = [
        data_root / "cache" / "thumbnails",
        data_root / "data" / "cache" / "thumbnails",
    ]
    database = result.get("engine", {}).get("feature_cache", {}).get("database")
    if database:
        candidates.append(Path(str(database)).expanduser().resolve().parent / "cache" / "thumbnails")
    return next((candidate for candidate in candidates if candidate.is_dir()), None)


def load_thumbnail(photo: dict[str, Any], api_base: str, thumbnail_dir: Path | None) -> Image.Image:
    if thumbnail_dir is not None:
        path = thumbnail_dir / f"{photo['id']}.jpg"
        return Image.open(path).convert("RGB")
    return fetch_image(f"{api_base}{photo['thumbnail_url']}")


def stem_key(filename: str) -> str:
    return Path(filename).stem.casefold()


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return round(float(numerator) / float(denominator), 4) if denominator else 0.0


def classification_metrics(predicted: set[str], expected: set[str], universe: set[str]) -> dict[str, Any]:
    true_positive = len(predicted & expected)
    false_positive = len(predicted - expected)
    false_negative = len(expected - predicted)
    true_negative = len(universe - predicted - expected)
    precision = safe_ratio(true_positive, true_positive + false_positive)
    recall = safe_ratio(true_positive, true_positive + false_negative)
    return {
        "predicted": len(predicted),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": precision,
        "recall": recall,
        "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
    }


def semantic_rating_metrics(
    photos: list[dict[str, Any]],
    matched_manual: set[str],
    universe: set[str],
) -> dict[str, Any]:
    if not photos or any("stars" not in photo for photo in photos):
        return {"available": False}

    stars_by_stem = {
        stem_key(str(photo["filename"])): max(0, min(3, int(photo.get("stars", 0))))
        for photo in photos
    }
    star_counts = Counter(stars_by_stem.values())
    primary = {stem for stem, stars in stars_by_stem.items() if stars >= 3}
    delivery = {stem for stem, stars in stars_by_stem.items() if stars >= 2}
    return {
        "available": True,
        "star_counts": {str(stars): star_counts.get(stars, 0) for stars in range(4)},
        "three_star": classification_metrics(primary, matched_manual, universe),
        "two_plus_three_star": classification_metrics(delivery, matched_manual, universe),
    }


def _metric_value(photo: dict[str, Any], name: str, default: float = 0.0) -> float:
    try:
        value = float(photo.get("metrics", {}).get(name, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _manual_has_decisive_closed(photo: dict[str, Any]) -> bool:
    metrics = photo.get("metrics", {})
    if "decisive_closed_face_count" in metrics:
        return _metric_value(photo, "decisive_closed_face_count") > 0.0
    return "主要人物闭眼" in photo.get("issues", [])


def eye_evidence_diagnostics(
    photos: list[dict[str, Any]],
    matched_manual: set[str],
) -> dict[str, Any]:
    manual_photos = [photo for photo in photos if stem_key(photo["filename"]) in matched_manual]
    manual_with_decisive_closed = sum(_manual_has_decisive_closed(photo) for photo in manual_photos)
    manual_with_uncertain_eye = sum(
        _metric_value(photo, "uncertain_eye_face_count") > 0.0
        for photo in manual_photos
    )

    by_group: dict[str, list[dict[str, Any]]] = {}
    for photo in photos:
        by_group.setdefault(str(photo.get("group_id", "")), []).append(photo)

    top1_disagreements = 0
    eye_related_top1_disagreements = 0
    eye_issue_names = {"主要人物闭眼", "检测到半闭眼"}
    for members in by_group.values():
        manual_members = [photo for photo in members if stem_key(photo["filename"]) in matched_manual]
        if not manual_members:
            continue
        winner = min(
            members,
            key=lambda photo: (int(photo.get("rank_in_group", 10**9)), str(photo.get("filename", "")).casefold()),
        )
        if stem_key(winner["filename"]) in matched_manual:
            continue
        top1_disagreements += 1
        manual = min(
            manual_members,
            key=lambda photo: (int(photo.get("rank_in_group", 10**9)), str(photo.get("filename", "")).casefold()),
        )
        manual_issues = set(manual.get("issues", []))
        winner_issues = set(winner.get("issues", []))
        eye_related = (
            _manual_has_decisive_closed(manual)
            or _metric_value(manual, "uncertain_eye_face_count") > 0.0
            or abs(_metric_value(winner, "eye_score", 72.0) - _metric_value(manual, "eye_score", 72.0)) >= 8.0
            or bool((manual_issues | winner_issues) & eye_issue_names)
        )
        eye_related_top1_disagreements += int(eye_related)

    return {
        "manual_with_decisive_closed": manual_with_decisive_closed,
        "manual_with_uncertain_eye": manual_with_uncertain_eye,
        "eye_related_top1_disagreements": eye_related_top1_disagreements,
        "top1_disagreements": top1_disagreements,
        "eye_related_disagreement_rate": safe_ratio(
            eye_related_top1_disagreements,
            top1_disagreements,
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def render_disagreement_sheets(
    entries: list[tuple[dict[str, Any], str]],
    manual_stems: set[str],
    api_base: str,
    thumbnail_dir: Path | None,
    output_dir: Path,
) -> list[str]:
    if not entries:
        return []
    columns = 5
    page_size = 40
    cell_width = 280
    image_height = 185
    label_height = 62
    font = ImageFont.load_default(size=13)
    outputs: list[str] = []
    for page_index in range(math.ceil(len(entries) / page_size)):
        page = entries[page_index * page_size : (page_index + 1) * page_size]
        rows = math.ceil(len(page) / columns)
        sheet = Image.new("RGB", (columns * cell_width, rows * (image_height + label_height)), "#111827")
        draw = ImageDraw.Draw(sheet)
        for index, (photo, role) in enumerate(page):
            x = index % columns * cell_width
            y = index // columns * (image_height + label_height)
            source = load_thumbnail(photo, api_base, thumbnail_dir)
            fitted = ImageOps.fit(source, (cell_width - 10, image_height - 10), Image.Resampling.LANCZOS)
            is_manual = stem_key(photo["filename"]) in manual_stems
            is_best = bool(photo.get("is_best_pick"))
            color = "#34d399" if is_manual and is_best else "#22d3ee" if is_manual else "#f59e0b"
            sheet.paste(fitted, (x + 5, y + 5))
            draw.rectangle((x + 3, y + 3, x + cell_width - 3, y + image_height - 3), outline=color, width=4)
            marker = "MANUAL+AI" if is_manual and is_best else "MANUAL" if is_manual else "AI TOP1"
            draw.text((x + 8, y + image_height + 4), f"{photo['filename']}  {marker}", fill="#f8fafc", font=font)
            draw.text(
                (x + 8, y + image_height + 24),
                f"{photo['group_id']}  R{photo['rank_in_group']}  score {photo['score']}",
                fill=color,
                font=font,
            )
            draw.text((x + 8, y + image_height + 43), role, fill="#cbd5e1", font=font)
        destination = output_dir / f"disagreements-{page_index + 1:02d}.jpg"
        sheet.save(destination, "JPEG", quality=91, optimize=True)
        outputs.append(destination.name)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="评测 PhotoCull 与人工已修结果")
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:8767")
    parser.add_argument("--project-file", type=Path, help="直接读取本地项目 JSON，不依赖正在运行的 API")
    parser.add_argument("--thumbnail-dir", type=Path, help="本地缩略图目录；默认从项目 JSON 位置自动推断")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-contact-sheets", action="store_true")
    args = parser.parse_args()

    result = load_project_result(args.project_file, args.api_base)
    thumbnail_dir = args.thumbnail_dir
    if args.project_file is not None and thumbnail_dir is None:
        thumbnail_dir = infer_thumbnail_dir(args.project_file, result)
    photos = list(result["photos"])
    photos_by_stem = {stem_key(photo["filename"]): photo for photo in photos}
    universe = set(photos_by_stem)
    references = sorted(
        (path for path in args.reference_dir.rglob("*") if path.is_file() and path.suffix.casefold() in REFERENCE_EXTENSIONS),
        key=lambda path: path.name.casefold(),
    )
    reference_by_stem = {path.stem.casefold(): path for path in references}
    manual_stems = set(reference_by_stem)
    unmatched_references = sorted(manual_stems - universe)
    matched_manual = manual_stems & universe
    eye_diagnostics = eye_evidence_diagnostics(photos, matched_manual)
    semantic_metrics = semantic_rating_metrics(photos, matched_manual, universe)

    ai_best = {stem_key(photo["filename"]) for photo in photos if photo.get("is_best_pick")}
    clean_selected = {stem_key(photo["filename"]) for photo in photos if photo.get("category") == "selected"}
    baseline_top1 = {
        stem_key(photo["filename"])
        for photo in photos
        if int(photo.get("rank_in_group", 10**9)) == 1 and int(photo.get("width", 0)) > 0
    }
    coverage_protected = {stem_key(photo["filename"]) for photo in photos if photo.get("coverage_protected")}
    clean_before_coverage = clean_selected - coverage_protected
    strict_metrics = classification_metrics(ai_best, matched_manual, universe)
    clean_metrics = classification_metrics(clean_selected, matched_manual, universe)
    baseline_top1_metrics = classification_metrics(baseline_top1, matched_manual, universe)
    clean_before_coverage_metrics = classification_metrics(clean_before_coverage, matched_manual, universe)
    coverage_original_categories = Counter(
        str(photo.get("coverage_original_category") or "unknown") for photo in photos if photo.get("coverage_protected")
    )
    coverage_original_manual_categories = Counter(
        str(photo.get("coverage_original_category") or "unknown")
        for photo in photos
        if photo.get("coverage_protected") and stem_key(photo["filename"]) in matched_manual
    )

    photos_by_group: dict[str, list[dict[str, Any]]] = {}
    for photo in photos:
        photos_by_group.setdefault(photo["group_id"], []).append(photo)
    group_metadata = {group["id"]: group for group in result["groups"]}
    group_rows: list[dict[str, Any]] = []
    missed_rows: list[dict[str, Any]] = []
    model_only_rows: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    disagreement_entries: list[tuple[dict[str, Any], str]] = []
    reference_group_ids: list[str] = []
    exact_group_hits = 0
    manual_ranks: list[int] = []
    first_manual_ranks: list[int] = []

    for group_id, members in photos_by_group.items():
        ordered = sorted(members, key=lambda photo: (int(photo.get("rank_in_group", 10**9)), photo["filename"].casefold()))
        manual_members = [photo for photo in ordered if stem_key(photo["filename"]) in matched_manual]
        best_members = [photo for photo in ordered if photo.get("is_best_pick")]
        winner = best_members[0] if best_members else ordered[0]
        metadata = group_metadata.get(group_id, {})
        ranks = [int(photo["rank_in_group"]) for photo in manual_members]
        exact_hit = stem_key(winner["filename"]) in matched_manual
        if manual_members:
            reference_group_ids.append(group_id)
            exact_group_hits += int(exact_hit)
            manual_ranks.extend(ranks)
            first_manual_ranks.append(min(ranks))
            if not exact_hit:
                disagreement_entries.append((winner, "AI TOP1"))
                disagreement_entries.extend((photo, f"MANUAL R{photo['rank_in_group']}") for photo in manual_members)
        group_rows.append(
            {
                "group_id": group_id,
                "group_size": len(ordered),
                "manual_count": len(manual_members),
                "manual_files": " | ".join(photo["filename"] for photo in manual_members),
                "manual_ranks": " | ".join(str(rank) for rank in ranks),
                "first_manual_rank": min(ranks) if ranks else "",
                "ai_winner": winner["filename"],
                "ai_winner_is_manual": exact_hit,
                "confidence": metadata.get("confidence", ""),
                "person_ids": " | ".join(metadata.get("person_ids", [])),
                "scene_reason": metadata.get("scene_reason", ""),
            }
        )

    for stem in sorted(matched_manual - ai_best):
        photo = photos_by_stem[stem]
        members = photos_by_group[photo["group_id"]]
        winner = next((member for member in members if member.get("is_best_pick")), min(members, key=lambda item: item["rank_in_group"]))
        missed_rows.append(
            {
                "manual_file": reference_by_stem[stem].name,
                "source_file": photo["filename"],
                "group_id": photo["group_id"],
                "group_size": len(members),
                "rank_in_group": photo["rank_in_group"],
                "score": photo["score"],
                "ai_winner": winner["filename"],
                "ai_winner_score": winner["score"],
                "score_gap": round(float(winner["score"]) - float(photo["score"]), 2),
                "category": photo["category"],
                "issues": " | ".join(photo.get("issues", [])),
                "selection_reasons": " | ".join(photo.get("selection_reasons", [])),
            }
        )

    for stem in sorted(ai_best - matched_manual):
        photo = photos_by_stem[stem]
        members = photos_by_group[photo["group_id"]]
        manual_members = [member for member in members if stem_key(member["filename"]) in matched_manual]
        model_only_rows.append(
            {
                "source_file": photo["filename"],
                "group_id": photo["group_id"],
                "group_size": len(members),
                "score": photo["score"],
                "category": photo["category"],
                "manual_in_same_group": " | ".join(member["filename"] for member in manual_members),
                "manual_ranks": " | ".join(str(member["rank_in_group"]) for member in manual_members),
                "issues": " | ".join(photo.get("issues", [])),
                "selection_reasons": " | ".join(photo.get("selection_reasons", [])),
            }
        )

    metric_keys = (
        "group_ranking_score", "generic_group_score", "preference_score", "preference_threshold",
        "group_relative_score", "group_quality_percentile", "face_quality_score", "min_face_score", "face_sharpness_score",
        "eye_sharpness_score", "eye_score", "motion_blur_score", "exposure_score",
        "composition_score", "technical_score", "bad_face_count",
        "decisive_closed_face_count", "uncertain_eye_face_count", "mean_eye_reliability",
    )
    for photo in sorted(photos, key=lambda item: item["filename"].casefold()):
        metrics = photo.get("metrics", {})
        row = {
            "source_file": photo["filename"],
            "manual_selected": stem_key(photo["filename"]) in matched_manual,
            "ai_best": bool(photo.get("is_best_pick")),
            "clean_selected": photo.get("category") == "selected",
            "stars": photo.get("stars", ""),
            "rating_tier": photo.get("rating_tier", ""),
            "rating_reason": photo.get("rating_reason", ""),
            "category": photo.get("category"),
            "group_id": photo.get("group_id"),
            "group_size": len(photos_by_group.get(photo.get("group_id"), [])),
            "rank_in_group": photo.get("rank_in_group"),
            "score": photo.get("score"),
            "issues": " | ".join(photo.get("issues", [])),
            "face_count": len(photo.get("faces", [])),
            "person_ids": " | ".join(photo.get("person_ids", [])),
        }
        row.update({key: metrics.get(key, "") for key in metric_keys})
        all_rows.append(row)

    reference_group_count = len(reference_group_ids)
    rank_metrics: dict[str, Any] = {}
    keep_table: list[dict[str, Any]] = []
    for k in TOP_K_VALUES:
        predicted_at_k = {
            stem_key(photo["filename"])
            for photo in photos
            if int(photo.get("rank_in_group", 10**9)) <= k and int(photo.get("width", 0)) > 0
        }
        metrics_at_k = classification_metrics(predicted_at_k, matched_manual, universe)
        group_hits = sum(
            any(stem_key(photo["filename"]) in matched_manual and int(photo["rank_in_group"]) <= k for photo in photos_by_group[group_id])
            for group_id in reference_group_ids
        )
        rank_metrics[f"manual_recall_at_{k}"] = safe_ratio(sum(rank <= k for rank in manual_ranks), len(manual_ranks))
        rank_metrics[f"group_hit_at_{k}"] = safe_ratio(group_hits, reference_group_count)
        keep_table.append({"keep_per_group": k, **metrics_at_k})

    category_counts = Counter(photo.get("category", "unknown") for photo in photos)
    manual_per_group = Counter(min(5, int(row["manual_count"])) for row in group_rows if row["manual_count"])
    group_sizes = sorted(len(members) for members in photos_by_group.values())
    grouping_summary = {
        "groups": len(group_sizes),
        "singletons": sum(size == 1 for size in group_sizes),
        "multi_photo_groups": sum(size > 1 for size in group_sizes),
        "largest_group": max(group_sizes, default=0),
        "p95_group_size": group_sizes[min(len(group_sizes) - 1, math.floor(len(group_sizes) * 0.95))] if group_sizes else 0,
        "top1_duplicate_reduction": len(photos) - len(group_sizes),
        "top1_reduction_rate": safe_ratio(len(photos) - len(group_sizes), len(photos)),
    }
    summary = {
        "project_id": result["project_id"],
        "engine": result.get("engine", {}),
        "dataset": {
            "scanned": len(photos),
            "manual_reference": len(manual_stems),
            "matched_manual_reference": len(matched_manual),
            "unmatched_reference": [reference_by_stem[stem].name for stem in unmatched_references],
        },
        "strict_ai_best": strict_metrics,
        "clean_selected": clean_metrics,
        "semantic_ratings": semantic_metrics,
        "coverage_evaluation": {
            "protected_photos": len(coverage_protected),
            "protected_manual_matches": len(coverage_protected & matched_manual),
            "protected_manual_match_rate": safe_ratio(len(coverage_protected & matched_manual), len(coverage_protected)),
            "added_best_candidates": len(ai_best - baseline_top1),
            "added_best_manual_matches": len((ai_best - baseline_top1) & matched_manual),
            "baseline_top1": baseline_top1_metrics,
            "after_coverage_best": strict_metrics,
            "best_recall_gain": round(strict_metrics["recall"] - baseline_top1_metrics["recall"], 4),
            "clean_before_coverage": clean_before_coverage_metrics,
            "clean_after_coverage": clean_metrics,
            "clean_recall_gain": round(clean_metrics["recall"] - clean_before_coverage_metrics["recall"], 4),
            "original_category_counts": dict(coverage_original_categories),
            "original_category_manual_matches": dict(coverage_original_manual_categories),
        },
        "grouping_summary": grouping_summary,
        "eye_evidence_evaluation": eye_diagnostics,
        "group_evaluation": {
            "groups": len(photos_by_group),
            "reference_groups": reference_group_count,
            "top1_exact_group_hits": exact_group_hits,
            "top1_exact_group_hit_rate": safe_ratio(exact_group_hits, reference_group_count),
            "mean_reciprocal_first_manual_rank": round(sum(1.0 / rank for rank in first_manual_ranks) / len(first_manual_ranks), 4) if first_manual_ranks else 0.0,
            "groups_with_multiple_manual_picks": sum(int(row["manual_count"]) > 1 for row in group_rows),
            "manual_picks_per_reference_group_capped_at_5": {str(key): value for key, value in sorted(manual_per_group.items())},
            **rank_metrics,
        },
        "keep_per_group_simulation": keep_table,
        "category_counts": dict(category_counts),
        "scan_summary": result.get("summary", {}),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "all-photos-evaluation.csv", all_rows)
    write_csv(args.output_dir / "missed-manual.csv", missed_rows)
    write_csv(args.output_dir / "model-only.csv", model_only_rows)
    write_csv(args.output_dir / "group-analysis.csv", group_rows)
    contact_sheets = [] if args.no_contact_sheets else render_disagreement_sheets(
        disagreement_entries, manual_stems, args.api_base, thumbnail_dir, args.output_dir
    )
    summary["artifacts"] = {
        "all_photos": "all-photos-evaluation.csv",
        "missed_manual": "missed-manual.csv",
        "model_only": "model-only.csv",
        "group_analysis": "group-analysis.csv",
        "contact_sheets": contact_sheets,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    preference_status = result.get("engine", {}).get("preference_ai", {})
    preference_applied = preference_status.get("applied")
    if preference_applied is None:
        preference_applied = float(preference_status.get("ranking_strength", 0)) > 0
    preference_reason = str(preference_status.get("reason", "")).strip()
    preference_report = (
        f"- 个性化 Ranking：{'已应用' if preference_applied else '未应用'}；"
        f"验证强度 {float(preference_status.get('ranking_strength', 0)):.2%}；"
        f"交叉验证混合权重 {float(preference_status.get('blend_weight', 0)):.2%}；"
        f"全局偏好过滤 "
        f"{'启用' if preference_applied and preference_status.get('selection_filter_enabled') else '关闭'}"
    )
    if preference_reason:
        preference_report += f"；原因：{preference_reason}"
    semantic_report_lines: list[str] = []
    if semantic_metrics.get("available"):
        primary_metrics = semantic_metrics["three_star"]
        delivery_metrics = semantic_metrics["two_plus_three_star"]
        semantic_report_lines = [
            "",
            "## 语义星级",
            "",
            f"- 3 星精选：{primary_metrics['predicted']} 张；命中 {primary_metrics['true_positive']}；Recall {primary_metrics['recall']:.2%}。",
            f"- 2+3 星交付：{delivery_metrics['predicted']} 张；命中 {delivery_metrics['true_positive']}；Recall {delivery_metrics['recall']:.2%}。",
            f"- 星级分布：{semantic_metrics['star_counts']}。",
        ]
    report_lines = [
        f"# {args.reference_dir.parent.name} 人工选片对照评测",
        "",
        f"- 扫描原片：{len(photos)} 张",
        f"- 人工参考：{len(manual_stems)} 张；成功匹配：{len(matched_manual)} 张",
        f"- AI 最佳候选（含覆盖补选）：{strict_metrics['predicted']} 张；命中 {strict_metrics['true_positive']}，多选 {strict_metrics['false_positive']}，漏选 {strict_metrics['false_negative']}",
        f"- 分组：{grouping_summary['groups']} 组，其中多图组 {grouping_summary['multi_photo_groups']}、单张组 {grouping_summary['singletons']}；每组留 1 张可压缩 {grouping_summary['top1_duplicate_reduction']} 张（{grouping_summary['top1_reduction_rate']:.2%}）",
        f"- 最大组 / P95 组大小：{grouping_summary['largest_group']} / {grouping_summary['p95_group_size']} 张",
        preference_report,
        f"- 严格 Precision / Recall / F1：{strict_metrics['precision']:.2%} / {strict_metrics['recall']:.2%} / {strict_metrics['f1']:.2%}",
        f"- 含人工参考的 AI 组：{reference_group_count}；Top-1 与人工完全一致：{exact_group_hits} 组（{safe_ratio(exact_group_hits, reference_group_count):.2%}）",
        *semantic_report_lines,
        "",
        "## 眼态证据诊断",
        "",
        f"- 人工参考被当前版本硬判为主要人物闭眼：{eye_diagnostics['manual_with_decisive_closed']} 张。",
        f"- 人工参考包含不确定眼态：{eye_diagnostics['manual_with_uncertain_eye']} 张。",
        f"- Top-1 分歧 {eye_diagnostics['top1_disagreements']} 组，其中眼态相关 {eye_diagnostics['eye_related_top1_disagreements']} 组（{eye_diagnostics['eye_related_disagreement_rate']:.2%}）。",
        "",
        "## 人物 / 环节覆盖保护增益",
        "",
        f"- 覆盖保护补选 {len(coverage_protected)} 张，其中 {len(coverage_protected & matched_manual)} 张也被人工保留（{safe_ratio(len(coverage_protected & matched_manual), len(coverage_protected)):.2%}）。",
        f"- 纯组内 Top-1 的人工单片 Recall 为 {baseline_top1_metrics['recall']:.2%}；加入覆盖补选后为 {strict_metrics['recall']:.2%}，提升 {strict_metrics['recall'] - baseline_top1_metrics['recall']:.2%}。",
        f"- 严格清洁输出在覆盖前为 {clean_before_coverage_metrics['predicted']} 张、Recall {clean_before_coverage_metrics['recall']:.2%}；覆盖后为 {clean_metrics['predicted']} 张、Recall {clean_metrics['recall']:.2%}，提升 {clean_metrics['recall'] - clean_before_coverage_metrics['recall']:.2%}。",
        f"- 覆盖前原分类：{dict(coverage_original_categories)}；其中人工保留：{dict(coverage_original_manual_categories)}。",
        f"- 覆盖单元：要求 {int(result.get('engine', {}).get('coverage_guard', {}).get('required_cells', 0))}，补齐 {int(result.get('engine', {}).get('coverage_guard', {}).get('protected_cells', 0))}，未解决 {int(result.get('engine', {}).get('coverage_guard', {}).get('unresolved_cells', 0))}。",
        "",
        "## 组内排名召回",
        "",
        "| 每组前 K 张 | 人工单片召回 | 至少命中一张的参考组 | 模拟选中数 | Precision | Recall | F1 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in keep_table:
        k = row["keep_per_group"]
        report_lines.append(
            f"| {k} | {rank_metrics[f'manual_recall_at_{k}']:.2%} | {rank_metrics[f'group_hit_at_{k}']:.2%} | {row['predicted']} | {row['precision']:.2%} | {row['recall']:.2%} | {row['f1']:.2%} |"
        )
    report_lines.extend(
        [
            "",
            "## 口径说明",
            "",
            "- `AI 最佳候选` 使用 `is_best_pick`；覆盖保护可能把非第一名补为候选，因此数量可以大于分组数。",
            "- `clean_selected` 只统计没有明显问题且分类为 `selected` 的照片。",
            "- 本报告把“已修”视为完整人工保留集；若它只是交付子集，严格 Precision 会偏低。",
            "- 联系表仅包含“AI 第一名不是人工保留”的分歧组，青色为人工、橙色为 AI 第一名。",
            "",
        ]
    )
    (args.output_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
