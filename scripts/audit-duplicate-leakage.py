"""分析已完成项目中跨组的高相似照片，定位仍会重复入选的组边界。"""

from __future__ import annotations

import argparse
import io
import json
import urllib.request
from datetime import datetime
from pathlib import Path

from photocull.audit import audit_project, evaluation_provenance, file_sha256, DEFAULT_MODEL


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def fetch_rgb(url: str) -> np.ndarray:
    import numpy as np
    from PIL import Image
    with urllib.request.urlopen(url, timeout=30) as response:
        return np.asarray(Image.open(io.BytesIO(response.read())).convert("RGB"), dtype=np.uint8)


def restore_photo(
    item: dict,
    api_base: str,
    scene_engine: SceneEmbeddingEngine,
    image_key: str,
) -> PhotoObservation:
    from photocull.imaging import build_descriptor, file_sequence
    from photocull.internal_models import FaceObservation, PhotoObservation
    rgb = fetch_rgb(f"{api_base}{item[image_key]}")
    faces = [
        FaceObservation(
            face_id=face["face_id"],
            bbox=tuple(face["bbox"]),
            confidence=float(face["confidence"]),
            area_ratio=float(face["area_ratio"]),
            embedding=None,
            person_id=face.get("person_id"),
        )
        for face in item["faces"]
    ]
    filename = item["filename"]
    return PhotoObservation(
        id=item["id"],
        path=Path(filename),
        source_root=Path("."),
        filename=filename,
        relative_path=item["relative_path"],
        width=int(item["width"]),
        height=int(item["height"]),
        capture_time=datetime.fromisoformat(item["capture_time"]) if item.get("capture_time") else None,
        file_sequence=file_sequence(Path(filename)),
        descriptor=build_descriptor(rgb, scene_engine.embed(rgb)),
        faces=faces,
        person_ids=list(item.get("person_ids", [])),
        score=float(item["score"]),
        group_id=item["group_id"],
        is_best_pick=bool(item["is_best_pick"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="检测跨组重复泄漏")
    parser.add_argument("--project-file", type=Path, help="离线审计已保存项目，无需图像或模型推理")
    parser.add_argument("--pair-labels", type=Path, help="独立人工照片对标注 JSON")
    parser.add_argument("--minimum-stars", type=int, choices=(2, 3), default=3)
    parser.add_argument("--rating-model-file", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--api-base", default="http://127.0.0.1:8767")
    parser.add_argument("--sequence-window", type=int, default=20)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--review-total", type=float, default=0.70)
    parser.add_argument("--review-scene", type=float, default=0.72)
    parser.add_argument("--strict-total", type=float, default=0.86)
    parser.add_argument("--strict-scene", type=float, default=0.86)
    parser.add_argument("--all-photos", action="store_true", help="审计全部照片；默认只检查最终干净优选")
    parser.add_argument("--full-preview", action="store_true", help="使用大预览而不是缩略图重算特征")
    parser.add_argument("--cpu", action="store_true", help="强制 CPU；默认优先使用 DirectML")
    parser.add_argument("--output", type=Path, help="可选 JSON 输出路径")
    args = parser.parse_args()

    if args.project_file:
        if args.all_photos:
            parser.error("离线审计按最终星级选择；不支持 --all-photos")
        saved = json.loads(args.project_file.read_text(encoding="utf-8"))
        result = saved.get("results", saved)
        pair_labels = json.loads(args.pair_labels.read_text(encoding="utf-8")) if args.pair_labels else None
        payload = audit_project(result, minimum_stars=args.minimum_stars, pair_labels=pair_labels)
        payload.update(project_sha256=file_sha256(args.project_file),
                       pair_labels_sha256=file_sha256(args.pair_labels) if args.pair_labels else None,
                       provenance=evaluation_provenance(result, args.rating_model_file))
        rendered = json.dumps(payload, ensure_ascii=False, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        print(rendered)
        return
    if args.pair_labels:
        parser.error("独立标注审计需要 --project-file")
    from photocull.grouping import compare_photos
    from photocull.scene_engine import SceneEmbeddingEngine
    result = fetch_json(f"{args.api_base}/api/scan/results")
    engine = SceneEmbeddingEngine(use_gpu=not args.cpu)
    image_key = "image_url" if args.full_preview else "thumbnail_url"
    source_items = result["photos"] if args.all_photos else [
        item for item in result["photos"] if int(item.get("stars", 0)) >= args.minimum_stars
    ]
    photos = [restore_photo(item, args.api_base, engine, image_key) for item in source_items]
    best_by_pair: dict[tuple[str, str], dict] = {}
    ordered = sorted(photos, key=lambda photo: photo.file_sequence)
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            sequence_gap = right.file_sequence - left.file_sequence
            if sequence_gap > args.sequence_window:
                break
            if left.group_id == right.group_id:
                continue
            evidence = compare_photos(left, right)
            group_pair = tuple(sorted((left.group_id, right.group_id)))
            candidate = {
                "left": left.filename,
                "right": right.filename,
                "left_group": left.group_id,
                "right_group": right.group_id,
                "sequence_gap": sequence_gap,
                "shared_people": sorted(left.significant_person_ids & right.significant_person_ids),
                "total": round(evidence.total, 4),
                "scene": round(evidence.scene, 4),
                "semantic": None if evidence.semantic is None else round(evidence.semantic, 4),
                "phash": round(evidence.phash, 4),
                "dhash": round(evidence.dhash, 4),
                "layout": round(evidence.layout, 4),
                "composition": round(evidence.composition, 4),
                "people": round(evidence.people, 4),
                "temporal": round(evidence.temporal, 4),
                "compatible_people": evidence.compatible_people,
            }
            previous = best_by_pair.get(group_pair)
            if previous is None or candidate["total"] > previous["total"]:
                best_by_pair[group_pair] = candidate

    ranked = sorted(best_by_pair.values(), key=lambda item: item["total"], reverse=True)
    review_candidates = [
        item for item in ranked
        if (
            item["compatible_people"]
            and item["scene"] >= args.review_scene
            and item["total"] >= args.review_total
        )
    ]
    strict_leaks = [
        item for item in review_candidates
        if item["scene"] >= args.strict_scene and item["total"] >= args.strict_total
    ]
    payload = {
        "project_id": result["project_id"],
        "group_count": len(result["groups"]),
        "photo_scope": "all" if args.all_photos else f"stars_at_least_{args.minimum_stars}",
        "photo_count": len(photos),
        "embedding": engine.status(),
        "thresholds": {
            "review_total": args.review_total,
            "review_scene": args.review_scene,
            "strict_total": args.strict_total,
            "strict_scene": args.strict_scene,
        },
        "metric_scope": "production_comparator_candidates_not_independent_ground_truth",
        "provenance": evaluation_provenance(result, args.rating_model_file),
        "strict_leak_count": len(strict_leaks),
        "strict_leaks": strict_leaks[: args.limit],
        "review_candidate_count": len(review_candidates),
        "review_candidates": review_candidates[: args.limit],
        "top_cross_group_pairs": ranked[: args.limit],
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
