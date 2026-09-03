"""使用同名人工已修照片训练本机摄影师偏好模型。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from photocull.feature_cache import FeatureCache  # noqa: E402
from photocull.grouping import group_similar_photos  # noqa: E402
from photocull.identity import IdentityClusterer  # noqa: E402
from photocull.imaging import discover_images  # noqa: E402
from photocull.preference import (  # noqa: E402
    PREFERENCE_FEATURES,
    PREFERENCE_MODEL_PATH,
    best_preference_threshold,
    fit_pairwise_preference_model,
    fit_preference_model,
)
from photocull.quality import rescore_quality  # noqa: E402
from photocull.scoring import prepare_group_ranking_features  # noqa: E402


REFERENCE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def _classification_metrics(probabilities: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, Any]:
    predicted = probabilities >= threshold
    positive = labels == 1
    true_positive = int(np.sum(predicted & positive))
    false_positive = int(np.sum(predicted & ~positive))
    false_negative = int(np.sum(~predicted & positive))
    true_negative = int(np.sum(~predicted & ~positive))
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {
        "threshold": round(float(threshold), 4),
        "predicted_positive": int(np.sum(predicted)),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _roc_auc(probabilities: np.ndarray, labels: np.ndarray) -> float:
    positive = probabilities[labels == 1]
    negative = probabilities[labels == 0]
    if not positive.size or not negative.size:
        return 0.0
    comparisons = positive[:, None] - negative[None, :]
    return float(np.mean(comparisons > 0) + 0.5 * np.mean(comparisons == 0))


def _load_training_data(
    source_dir: Path,
    reference_dir: Path,
    recursive: bool,
) -> tuple[list[Any], np.ndarray, dict[str, Any]]:
    source_dir = source_dir.expanduser().resolve()
    reference_dir = reference_dir.expanduser().resolve()
    paths = discover_images(source_dir, recursive)
    reference_stems = {
        path.stem.casefold()
        for path in reference_dir.rglob("*")
        if path.is_file() and path.suffix.casefold() in REFERENCE_EXTENSIONS
    }
    cache = FeatureCache()
    signature = cache.dominant_pipeline_signature(source_dir)
    if not signature:
        raise RuntimeError("素材目录尚无可用特征缓存，请先运行一次筛图")

    photos = []
    cached_stems: set[str] = set()
    for path in paths:
        photo = cache.load(path, source_dir, signature)
        if photo is None:
            continue
        photo.metrics, photo.score, photo.issues = rescore_quality(photo.metrics, photo.faces)
        photos.append(photo)
        cached_stems.add(path.stem.casefold())
    photos.sort(key=lambda photo: (photo.file_sequence, photo.filename.casefold()))
    IdentityClusterer(0.42).assign(photos)
    groups = group_similar_photos(photos, "balanced")
    prepare_group_ranking_features(groups)
    labels = np.asarray([photo.path.stem.casefold() in reference_stems for photo in photos], dtype=np.int8)
    metadata = {
        "source_dir": str(source_dir),
        "reference_dir": str(reference_dir),
        "source_photos": len(paths),
        "cached_photos": len(photos),
        "cache_coverage": round(len(photos) / max(1, len(paths)), 4),
        "reference_photos": len(reference_stems),
        "matched_reference": len(reference_stems & cached_stems),
        "groups": len(groups),
        "largest_group": max((len(group.photos) for group in groups), default=0),
        "unmatched_reference": sorted(reference_stems - {path.stem.casefold() for path in paths}),
        "pipeline_signature": signature,
        "cache_database": str(cache.path),
    }
    return photos, labels, metadata


def _cross_validate(
    photos: list[Any],
    labels: np.ndarray,
    folds: int,
    objective: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    indices = np.arange(len(photos))
    group_ids = list(dict.fromkeys(photo.group_id for photo in photos))
    group_sizes = {group_id: sum(photo.group_id == group_id for photo in photos) for group_id in group_ids}
    fold_count = min(folds, len(group_ids))
    group_blocks: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    assigned_size = 0
    for group_id in group_ids:
        remaining_folds = fold_count - len(group_blocks)
        target_size = (len(photos) - assigned_size) / max(1, remaining_folds)
        size = group_sizes[group_id]
        if current and current_size + size > target_size and len(group_blocks) < fold_count - 1:
            group_blocks.append(current)
            assigned_size += current_size
            current = []
            current_size = 0
        current.append(group_id)
        current_size += size
    if current:
        group_blocks.append(current)
    probabilities = np.full(len(photos), np.nan, dtype=np.float64)
    fold_reports: list[dict[str, Any]] = []
    for fold_number, group_block in enumerate(group_blocks, start=1):
        validation_groups = set(group_block)
        validation = np.asarray(
            [index for index, photo in enumerate(photos) if photo.group_id in validation_groups],
            dtype=np.int64,
        )
        training = np.setdiff1d(indices, validation, assume_unique=True)
        train_labels = labels[training]
        if np.sum(train_labels == 1) < 5 or np.sum(train_labels == 0) < 5:
            continue
        training_photos = [photos[index] for index in training]
        model = (
            fit_pairwise_preference_model(training_photos, train_labels)
            if objective == "pairwise"
            else fit_preference_model(training_photos, train_labels)
        )
        fold_probabilities = model.predict_probabilities(photos[index] for index in validation)
        probabilities[validation] = fold_probabilities
        fold_reports.append(
            {
                "fold": fold_number,
                "objective": objective,
                "train": len(training),
                "validation": len(validation),
                "validation_groups": len(validation_groups),
                "validation_positive": int(np.sum(labels[validation] == 1)),
                "validation_sequence": [
                    int(photos[int(validation[0])].file_sequence),
                    int(photos[int(validation[-1])].file_sequence),
                ],
            }
        )
    return probabilities, fold_reports


def _group_ranking_metrics(
    photos: list[Any],
    labels: np.ndarray,
    preference_values: np.ndarray,
) -> dict[str, Any]:
    groups: dict[str, list[int]] = {}
    for index, photo in enumerate(photos):
        groups.setdefault(photo.group_id, []).append(index)
    eligible = [
        members
        for members in groups.values()
        if len(members) > 1 and np.any(labels[members] == 1) and np.any(labels[members] == 0)
    ]
    preference_hits = 0
    generic_hits = 0
    preference_pairs: list[float] = []
    generic_pairs: list[float] = []
    for members in eligible:
        preference = preference_values[members]
        generic = np.asarray([photos[index].metrics.get("generic_group_score", 0.0) for index in members])
        group_labels = labels[members]
        preference_hits += int(group_labels[int(np.argmax(preference))] == 1)
        generic_hits += int(group_labels[int(np.argmax(generic))] == 1)
        positive = np.flatnonzero(group_labels == 1)
        negative = np.flatnonzero(group_labels == 0)
        for left in positive:
            for right in negative:
                preference_pairs.append(1.0 if preference[left] > preference[right] else 0.5 if preference[left] == preference[right] else 0.0)
                generic_pairs.append(1.0 if generic[left] > generic[right] else 0.5 if generic[left] == generic[right] else 0.0)
    count = len(eligible)
    return {
        "eligible_groups": count,
        "preference_top1_hit_rate": round(preference_hits / max(1, count), 4),
        "generic_top1_hit_rate": round(generic_hits / max(1, count), 4),
        "preference_pairwise_accuracy": round(float(np.mean(preference_pairs)) if preference_pairs else 0.0, 4),
        "generic_pairwise_accuracy": round(float(np.mean(generic_pairs)) if generic_pairs else 0.0, 4),
        "compared_pairs": len(preference_pairs),
    }


def _select_blend_weight(
    photos: list[Any],
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    generic = np.asarray([photo.metrics.get("generic_group_score", 0.0) for photo in photos], dtype=np.float64)
    curve: list[dict[str, Any]] = []
    best_weight = 0.0
    best_metrics = _group_ranking_metrics(photos, labels, generic)
    best_utility = best_metrics["preference_top1_hit_rate"] + best_metrics["preference_pairwise_accuracy"]
    for weight in np.linspace(0.05, 0.60, 12):
        blended = (1.0 - weight) * generic + weight * probabilities * 100.0
        metrics = _group_ranking_metrics(photos, labels, blended)
        utility = metrics["preference_top1_hit_rate"] + metrics["preference_pairwise_accuracy"]
        curve.append({"weight": round(float(weight), 4), **metrics})
        if utility > best_utility + 1e-6:
            best_weight = float(weight)
            best_metrics = metrics
            best_utility = utility
    return best_weight, best_metrics, curve


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 PhotoCull 本机摄影师偏好模型")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=PREFERENCE_MODEL_PATH)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--replace-unvalidated", action="store_true")
    args = parser.parse_args()

    photos, labels, data = _load_training_data(args.source_dir, args.reference_dir, args.recursive)
    if len(photos) < 20:
        raise RuntimeError("有效缓存样本不足 20 张")
    if not args.allow_partial and data["cache_coverage"] < 0.98:
        raise RuntimeError(
            f"当前特征缓存覆盖率仅 {data['cache_coverage']:.2%}；请等待全量筛图完成，或显式使用 --allow-partial"
        )

    pointwise_probabilities, pointwise_folds = _cross_validate(photos, labels, max(2, args.folds), "pointwise")
    pairwise_probabilities, pairwise_folds = _cross_validate(photos, labels, max(2, args.folds), "pairwise")
    valid = np.isfinite(pointwise_probabilities) & np.isfinite(pairwise_probabilities)
    if int(np.sum(valid)) < max(20, round(len(photos) * 0.8)):
        raise RuntimeError("交叉验证有效样本不足，无法可靠校准偏好阈值")
    oof_threshold, _ = best_preference_threshold(pointwise_probabilities[valid], labels[valid])
    oof_metrics = _classification_metrics(pointwise_probabilities[valid], labels[valid], oof_threshold)
    oof_metrics["roc_auc"] = round(_roc_auc(pointwise_probabilities[valid], labels[valid]), 4)
    oof_metrics["evaluated"] = int(np.sum(valid))
    positive_rate = float(np.mean(labels[valid]))
    baseline_f1 = 2.0 * positive_rate / (1.0 + positive_rate)
    oof_metrics["all_positive_baseline_f1"] = round(baseline_f1, 4)
    oof_metrics["f1_uplift"] = round(float(oof_metrics["f1"]) - baseline_f1, 4)
    pointwise_group_ranking = _group_ranking_metrics(photos, labels, pointwise_probabilities)
    pairwise_group_ranking = _group_ranking_metrics(photos, labels, pairwise_probabilities)
    pointwise_rank_score = (
        pointwise_group_ranking["preference_top1_hit_rate"]
        + pointwise_group_ranking["preference_pairwise_accuracy"]
    )
    pairwise_rank_score = (
        pairwise_group_ranking["preference_top1_hit_rate"]
        + pairwise_group_ranking["preference_pairwise_accuracy"]
    )
    objective = "pairwise" if pairwise_rank_score > pointwise_rank_score + 0.005 else "pointwise"
    group_ranking = pairwise_group_ranking if objective == "pairwise" else pointwise_group_ranking
    selected_probabilities = pairwise_probabilities if objective == "pairwise" else pointwise_probabilities
    blend_weight, blend_validation, blend_curve = _select_blend_weight(photos, labels, selected_probabilities)
    top1_uplift = blend_validation["preference_top1_hit_rate"] - blend_validation["generic_top1_hit_rate"]
    pairwise_uplift = blend_validation["preference_pairwise_accuracy"] - blend_validation["generic_pairwise_accuracy"]
    ranking_validated = (
        blend_validation["eligible_groups"] >= 20
        and blend_validation["preference_pairwise_accuracy"] >= 0.54
        and (top1_uplift >= 0.02 or pairwise_uplift >= 0.025)
        and blend_weight > 0.0
    )
    ranking_strength = (
        float(np.clip(max(top1_uplift, pairwise_uplift) / 0.12, 0.2, 1.0)) if ranking_validated else 0.0
    )
    if not ranking_validated:
        blend_weight = 0.0
    selection_filter_enabled = bool(
        objective == "pointwise" and oof_metrics["roc_auc"] >= 0.62 and oof_metrics["f1_uplift"] >= 0.04
    )

    model = fit_pairwise_preference_model(photos, labels) if objective == "pairwise" else fit_preference_model(photos, labels)
    if objective == "pointwise":
        model.threshold = oof_threshold
    training_probabilities = model.predict_probabilities(photos)
    model.metadata.update(
        {
            "source_dir": data["source_dir"],
            "reference_dir": data["reference_dir"],
            "pipeline_signature": data["pipeline_signature"],
            "cross_validation": oof_metrics,
            "group_ranking_validation": group_ranking,
            "blend_validation": {"selected_weight": round(blend_weight, 4), "selected": blend_validation, "curve": blend_curve},
            "objective_comparison": {
                "selected": objective,
                "pointwise": pointwise_group_ranking,
                "pairwise": pairwise_group_ranking,
            },
            "ranking_strength": round(ranking_strength, 4),
            "blend_weight": round(blend_weight, 4),
            "selection_filter_enabled": selection_filter_enabled,
        }
    )
    model_validated = ranking_validated or selection_filter_enabled
    preserve_existing = bool(
        not args.no_save
        and args.model_path.is_file()
        and not model_validated
        and not args.replace_unvalidated
    )
    saved_path = None if args.no_save or preserve_existing else model.save(args.model_path)
    coefficients = sorted(
        (
            {"feature": feature, "standardized_weight": round(float(weight), 6)}
            for feature, weight in zip(PREFERENCE_FEATURES, model.weights, strict=True)
        ),
        key=lambda item: abs(item["standardized_weight"]),
        reverse=True,
    )
    report = {
        "data": data,
        "model": model.status(),
        "cross_validation": oof_metrics,
        "group_ranking_validation": group_ranking,
        "blend_validation": {"selected_weight": round(blend_weight, 4), "selected": blend_validation, "curve": blend_curve},
        "objective_comparison": {
            "selected": objective,
            "pointwise": pointwise_group_ranking,
            "pairwise": pairwise_group_ranking,
        },
        "training_set": _classification_metrics(training_probabilities, labels, model.threshold),
        "folds": {"pointwise": pointwise_folds, "pairwise": pairwise_folds},
        "coefficients": coefficients,
        "saved_model": str(saved_path) if saved_path else None,
        "preserved_existing_model": preserve_existing,
        "notes": [
            "交叉验证按文件序号分块，避免把相邻连拍同时放入训练集和验证集。",
            "最终全量训练指标只用于拟合检查；泛化能力以交叉验证指标为准。",
            "偏好分只参与硬质量约束之后的软排序。",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
