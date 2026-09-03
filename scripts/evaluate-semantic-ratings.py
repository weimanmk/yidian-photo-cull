"""只读评测 v0.2.1 语义星级与人工参考的一致性。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


REFERENCE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
    ".dng",
    ".rw2",
    ".arw",
    ".nef",
    ".cr2",
    ".cr3",
    ".raf",
    ".orf",
}
TIER_BY_STAR = {0: "waste", 1: "valuable", 2: "coverage", 3: "primary"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def canonical_hash(values: Iterable[str]) -> str:
    payload = json.dumps(sorted(values), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def stem_key(value: str) -> str:
    return Path(value).stem.casefold()


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return round(float(numerator) / float(denominator), 4) if denominator else 0.0


def load_project(project_file: Path) -> dict[str, Any]:
    payload = json.loads(project_file.read_text(encoding="utf-8"))
    result = payload.get("results", payload)
    if not isinstance(result, dict) or not isinstance(result.get("photos"), list):
        raise ValueError(f"项目结果格式无效：{project_file}")
    return result


def reference_stems(reference_dir: Path) -> set[str]:
    if not reference_dir.is_dir():
        raise ValueError(f"人工参考目录不存在：{reference_dir}")
    return {
        path.stem.casefold()
        for path in reference_dir.rglob("*")
        if path.is_file() and path.suffix.casefold() in REFERENCE_EXTENSIONS
    }


def validate_semantic_project(result: dict[str, Any]) -> None:
    if (
        int(result.get("schema_version", 0)) != 2
        or result.get("rating_migration_status") != "native"
        or result.get("lightroom_ready") is not True
    ):
        raise ValueError("项目不是 v0.2.1 原生语义星级，请重新扫描后再评测")


def selected_ids(result: dict[str, Any], *, minimum_stars: int) -> set[str]:
    validate_semantic_project(result)
    if minimum_stars not in {2, 3}:
        raise ValueError("评测最低星级只允许 2 或 3")
    return {
        str(photo["id"])
        for photo in result["photos"]
        if int(photo.get("stars", 0)) >= minimum_stars
    }


def _filename(photo: dict[str, Any]) -> str:
    return str(photo.get("filename") or photo.get("relative_path") or photo.get("id", ""))


def _is_decodable(photo: dict[str, Any]) -> bool:
    width = photo.get("width")
    height = photo.get("height")
    if width is not None and int(width) <= 0:
        return False
    if height is not None and int(height) <= 0:
        return False
    return not any(str(issue).startswith("文件读取失败") for issue in photo.get("issues", []))


def _strict_cluster_metrics(
    photos_by_id: dict[str, dict[str, Any]],
    selected: set[str],
) -> tuple[int, list[str], int]:
    members_by_cluster: dict[str, list[str]] = {}
    for photo_id in selected:
        photo = photos_by_id[photo_id]
        cluster_id = str(photo.get("strict_duplicate_cluster_id") or f"photo:{photo_id}")
        members_by_cluster.setdefault(cluster_id, []).append(photo_id)
    leak_clusters = sorted(
        cluster_id
        for cluster_id, members in members_by_cluster.items()
        if len(members) > 1 and not cluster_id.startswith("photo:")
    )
    leaks = sum(max(0, len(members) - 1) for members in members_by_cluster.values())
    pairs = sum(len(members) * (len(members) - 1) // 2 for members in members_by_cluster.values())
    return leaks, leak_clusters, pairs


def _semantic_contract_errors(photos: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for photo in photos:
        photo_id = str(photo.get("id", ""))
        stars = int(photo.get("stars", -1))
        expected = TIER_BY_STAR.get(stars)
        if expected is None:
            errors.append(f"{photo_id}:stars={stars}")
        elif str(photo.get("rating_tier", "")) != expected:
            errors.append(f"{photo_id}:stars={stars},tier={photo.get('rating_tier')}")
    return errors


def evaluate_rating_set(
    result: dict[str, Any],
    *,
    manual_stems: set[str],
    minimum_stars: int,
) -> dict[str, Any]:
    validate_semantic_project(result)
    photos = [dict(photo) for photo in result["photos"]]
    photos_by_id = {str(photo["id"]): photo for photo in photos}
    selected = selected_ids(result, minimum_stars=minimum_stars)
    normalized_manual = {value.casefold() for value in manual_stems}
    manual_ids = {
        photo_id
        for photo_id, photo in photos_by_id.items()
        if stem_key(_filename(photo)) in normalized_manual
    }
    decodable_ids = {
        photo_id for photo_id, photo in photos_by_id.items() if _is_decodable(photo)
    }
    reference_groups: dict[str, set[str]] = {}
    for photo_id in manual_ids:
        photo = photos_by_id[photo_id]
        group_id = str(photo.get("group_id") or f"photo:{photo_id}")
        reference_groups.setdefault(group_id, set()).add(photo_id)
    true_positive_ids = selected & manual_ids
    group_hits = sum(bool(members & selected) for members in reference_groups.values())
    precision = safe_ratio(len(true_positive_ids), len(selected))
    recall = safe_ratio(len(true_positive_ids), len(manual_ids))
    f1 = round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0
    leaks, leak_clusters, leak_pairs = _strict_cluster_metrics(photos_by_id, selected)
    rating_policy = result.get("rating_policy", {})
    required_coverage = int(rating_policy.get("required_coverage_keys", 0))
    unresolved_coverage = int(rating_policy.get("unresolved_coverage_keys", 0))
    person_stage_coverage = (
        round((required_coverage - unresolved_coverage) / required_coverage, 4)
        if required_coverage
        else 1.0
    )
    selected_coverage_keys = sorted(
        {
            str(key)
            for photo_id in selected
            for key in photos_by_id[photo_id].get("coverage_keys", [])
        }
    )
    eligible_total = len(decodable_ids)
    return {
        "minimum_stars": minimum_stars,
        "photo_ids": sorted(selected),
        "selected": len(selected),
        "eligible_total": eligible_total,
        "manual_reference": len(manual_ids),
        "manual_photo_ids": sorted(manual_ids),
        "true_positive": len(true_positive_ids),
        "true_positive_photo_ids": sorted(true_positive_ids),
        "precision": precision,
        "manual_recall": recall,
        "f1": f1,
        "reference_groups": len(reference_groups),
        "group_hits": group_hits,
        "group_hit": safe_ratio(group_hits, len(reference_groups)),
        "reduction": round(1.0 - len(selected) / eligible_total, 4) if eligible_total else 1.0,
        "strict_cluster_leaks": leaks,
        "strict_cluster_leak_pairs": leak_pairs,
        "strict_cluster_leak_clusters": leak_clusters,
        "required_person_stage_keys": required_coverage,
        "unresolved_person_stage_keys": unresolved_coverage,
        "person_stage_coverage": person_stage_coverage,
        "selected_coverage_keys": selected_coverage_keys,
    }


def _finite_metric(photo: dict[str, Any], name: str) -> bool:
    try:
        return math.isfinite(float(photo.get("metrics", {})[name]))
    except (KeyError, TypeError, ValueError):
        return False


def inferred_rating_profile(result: dict[str, Any]) -> str:
    reported = str(result.get("rating_policy", {}).get("rating_model_profile", ""))
    if reported:
        return reported
    decodable = [photo for photo in result["photos"] if _is_decodable(photo)]
    return (
        "iqa"
        if decodable and all(
            _finite_metric(photo, "musiq_score") and _finite_metric(photo, "qualiclip_score")
            for photo in decodable
        )
        else "base"
    )


def build_report(
    *,
    event_name: str,
    project_file: Path,
    reference_dir: Path,
    rating_model_file: Path,
) -> dict[str, Any]:
    resolved_project = project_file.resolve()
    resolved_reference = reference_dir.resolve()
    resolved_model = rating_model_file.resolve()
    result = load_project(resolved_project)
    validate_semantic_project(result)
    manual_stems = reference_stems(resolved_reference)
    model_payload = json.loads(resolved_model.read_text(encoding="utf-8"))
    matched_stems = {
        stem_key(_filename(photo))
        for photo in result["photos"]
        if stem_key(_filename(photo)) in manual_stems
    }
    contract_errors = _semantic_contract_errors(result["photos"])
    return {
        "format_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "source_event": event_name,
        "project_id": str(result.get("project_id", "")),
        "project_file": str(resolved_project),
        "project_sha256": sha256(resolved_project),
        "rating_model_file": str(resolved_model),
        "rating_model_sha256": sha256(resolved_model),
        "rating_model_version": str(model_payload.get("version", "")),
        "rating_model_profile": inferred_rating_profile(result),
        "training_hashes": dict(model_payload.get("training_hashes", {})),
        "selection_parameters": dict(model_payload.get("selection_parameters", {})),
        "runtime_fit": False,
        "per_event_overrides": {},
        "reference_dir": str(resolved_reference),
        "reference_names_sha256": canonical_hash(manual_stems),
        "reference_stems": sorted(manual_stems),
        "matched_reference_stems": sorted(matched_stems),
        "unmatched_reference_stems": sorted(manual_stems - matched_stems),
        "semantic_contract_valid": not contract_errors,
        "semantic_contract_errors": contract_errors,
        "rating_sets": {
            "three_star": evaluate_rating_set(result, manual_stems=manual_stems, minimum_stars=3),
            "two_plus_three_star": evaluate_rating_set(result, manual_stems=manual_stems, minimum_stars=2),
        },
    }


def aggregate_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    def aggregate_set(name: str) -> dict[str, Any]:
        rows = [report["rating_sets"][name] for report in reports]
        selected = sum(int(row["selected"]) for row in rows)
        eligible = sum(int(row["eligible_total"]) for row in rows)
        true_positive = sum(int(row["true_positive"]) for row in rows)
        manual = sum(int(row["manual_reference"]) for row in rows)
        group_hits = sum(int(row["group_hits"]) for row in rows)
        reference_groups = sum(int(row["reference_groups"]) for row in rows)
        required = sum(int(row["required_person_stage_keys"]) for row in rows)
        unresolved = sum(int(row["unresolved_person_stage_keys"]) for row in rows)
        precision = safe_ratio(true_positive, selected)
        recall = safe_ratio(true_positive, manual)
        return {
            "selected": selected,
            "eligible_total": eligible,
            "true_positive": true_positive,
            "manual_reference": manual,
            "precision": precision,
            "manual_recall": recall,
            "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
            "group_hit": safe_ratio(group_hits, reference_groups),
            "reduction": round(1.0 - selected / eligible, 4) if eligible else 1.0,
            "strict_cluster_leaks": sum(int(row["strict_cluster_leaks"]) for row in rows),
            "required_person_stage_keys": required,
            "unresolved_person_stage_keys": unresolved,
            "person_stage_coverage": round((required - unresolved) / required, 4) if required else 1.0,
        }

    model_hashes = sorted({str(report["rating_model_sha256"]) for report in reports})
    return {
        "format_version": 1,
        "events": [str(report["source_event"]) for report in reports],
        "project_hashes": {
            str(report["source_event"]): str(report["project_sha256"])
            for report in reports
        },
        "rating_model_hashes": model_hashes,
        "same_frozen_model": len(model_hashes) == 1,
        "no_runtime_fit": all(report.get("runtime_fit") is False for report in reports),
        "no_per_event_overrides": all(not report.get("per_event_overrides") for report in reports),
        "semantic_contract_valid": all(report.get("semantic_contract_valid") is True for report in reports),
        "rating_sets": {
            "three_star": aggregate_set("three_star"),
            "two_plus_three_star": aggregate_set("two_plus_three_star"),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    rows = []
    for label, key in (("3 星", "three_star"), ("2+3 星", "two_plus_three_star")):
        metrics = report["rating_sets"][key]
        rows.append(
            f"| {label} | {metrics['selected']} | {metrics['group_hit']:.2%} | "
            f"{metrics['manual_recall']:.2%} | {metrics['reduction']:.2%} | "
            f"{metrics['strict_cluster_leaks']} | {metrics['person_stage_coverage']:.2%} |"
        )
    return "\n".join(
        [
            f"# {report['source_event']} 语义星级评测",
            "",
            f"- 项目 SHA-256：`{report['project_sha256']}`",
            f"- 模型 SHA-256：`{report['rating_model_sha256']}`",
            f"- 模型档案：`{report['rating_model_profile']}`",
            f"- 语义契约：{'通过' if report['semantic_contract_valid'] else '失败'}",
            "",
            "| 集合 | 张数 | 组命中 | 人工召回 | 去除率 | 重复泄漏 | 人物×环节覆盖 |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
        ]
    )


def render_aggregate_markdown(report: dict[str, Any]) -> str:
    rows = []
    for label, key in (("3 星", "three_star"), ("2+3 星", "two_plus_three_star")):
        metrics = report["rating_sets"][key]
        rows.append(
            f"| {label} | {metrics['selected']} | {metrics['group_hit']:.2%} | "
            f"{metrics['manual_recall']:.2%} | {metrics['reduction']:.2%} | "
            f"{metrics['strict_cluster_leaks']} | {metrics['person_stage_coverage']:.2%} |"
        )
    return "\n".join(
        [
            "# v0.2.1 跨活动语义星级评测",
            "",
            f"- 活动：{'、'.join(report['events'])}",
            f"- 冻结模型一致：{'通过' if report['same_frozen_model'] else '失败'}",
            f"- 运行时拟合：{'无' if report['no_runtime_fit'] else '存在'}",
            f"- 单活动参数覆盖：{'无' if report['no_per_event_overrides'] else '存在'}",
            f"- 语义契约：{'通过' if report['semantic_contract_valid'] else '失败'}",
            "",
            "| 集合 | 张数 | 组命中 | 人工召回 | 去除率 | 重复泄漏 | 人物×环节覆盖 |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
        ]
    )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="评测 v0.2.1 语义星级")
    parser.add_argument("--event")
    parser.add_argument("--project-file", type=Path)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--aggregate-input", type=Path, nargs="+")
    parser.add_argument(
        "--rating-model-file",
        type=Path,
        default=root / "backend" / "photocull" / "assets" / "rating_model_v1.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.aggregate_input:
        if args.event or args.project_file or args.reference_dir:
            parser.error("聚合模式不能同时提供 --event、--project-file 或 --reference-dir")
        reports = [json.loads(path.resolve().read_text(encoding="utf-8")) for path in args.aggregate_input]
        report = aggregate_reports(reports)
        json_path = output_dir / "aggregate-semantic-ratings.json"
        markdown_path = output_dir / "aggregate-semantic-ratings.md"
        markdown = render_aggregate_markdown(report)
    else:
        missing = [
            name
            for name, value in (
                ("--event", args.event),
                ("--project-file", args.project_file),
                ("--reference-dir", args.reference_dir),
            )
            if value is None
        ]
        if missing:
            parser.error(f"单活动模式缺少：{', '.join(missing)}")
        report = build_report(
            event_name=args.event,
            project_file=args.project_file,
            reference_dir=args.reference_dir,
            rating_model_file=args.rating_model_file,
        )
        json_path = output_dir / "semantic-ratings.json"
        markdown_path = output_dir / "semantic-ratings.md"
        markdown = render_markdown(report)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(markdown, encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
