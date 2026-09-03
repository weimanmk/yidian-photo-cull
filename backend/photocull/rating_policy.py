from __future__ import annotations

from collections import Counter
import math
from typing import Any

import numpy as np

from .coverage import SEVERE_ISSUES, select_person_stage_coverage
from .internal_models import PhotoGroupInternal, PhotoObservation
from .near_duplicates import DuplicateLayers, build_duplicate_layers
from .rating_model import FrozenRatingModel, RatingFeatureProvider, RatingModelError
from .rating_types import RatingOrigin, RatingReason, RatingReport, RatingTier, apply_rating


TARGET_REDUCTION = 0.35
DELIVERY_TARGET_REDUCTION = 0.25
LEARNED_ALPHA = 0.75
GROUP_DEMOTE = 0.55
DUPLICATE_DEMOTE = 0.10


def _is_decodable(photo: PhotoObservation) -> bool:
    return bool(
        photo.width > 0
        and photo.height > 0
        and not any(issue.startswith("文件读取失败") for issue in photo.issues)
    )


def _has_decisive_issue(photo: PhotoObservation) -> bool:
    return any(issue in SEVERE_ISSUES or issue.startswith("文件读取失败") for issue in photo.issues)


def _rank_value(photo: PhotoObservation, fallback: int) -> int:
    return photo.rank_in_group if photo.rank_in_group > 0 else fallback


def _rank_percentiles(groups: list[PhotoGroupInternal]) -> dict[str, float]:
    percentiles: dict[str, float] = {}
    for group in groups:
        members = [photo for photo in group.photos if _is_decodable(photo)]
        ordered = sorted(
            enumerate(members, start=1),
            key=lambda item: (
                _rank_value(item[1], item[0]),
                -float(item[1].metrics.get("generic_group_score", item[1].score)),
                item[1].filename.casefold(),
            ),
        )
        count = len(ordered)
        for position, (_fallback, photo) in enumerate(ordered):
            percentiles[photo.id] = 1.0 if count <= 1 else 1.0 - position / (count - 1)
    return percentiles


def _stable_scores(
    photos: list[PhotoObservation],
    rank_percentiles: dict[str, float],
) -> dict[str, float]:
    return {
        photo.id: (
            0.58
            * float(np.clip(photo.metrics.get("generic_group_score", photo.score) / 100.0, 0.0, 1.0))
            + 0.42 * rank_percentiles.get(photo.id, 0.5)
        )
        for photo in photos
    }


def _learned_scores(
    photos: list[PhotoObservation],
    groups: list[PhotoGroupInternal],
    *,
    model: Any | None,
    feature_provider: RatingFeatureProvider | None,
) -> tuple[dict[str, float], dict[str, float], str, str]:
    rank_percentiles = _rank_percentiles(groups)
    stable = _stable_scores(photos, rank_percentiles)
    try:
        active_model = model or FrozenRatingModel.load_default()
        if feature_provider is None:
            predicted = active_model.predict(photos)
        else:
            predicted = active_model.predict(photos, feature_provider=feature_provider)
        if set(predicted) != {photo.id for photo in photos}:
            raise RatingModelError("评分模型没有返回完整照片集合")
        learned = {photo_id: float(value) for photo_id, value in predicted.items()}
        if not all(math.isfinite(value) for value in learned.values()):
            raise RatingModelError("评分模型返回非有限数值")
        profile = str(getattr(active_model, "last_profile", "") or ("custom" if model is not None else "base"))
        fallback_reason = str(getattr(active_model, "last_fallback_reason", ""))
        return learned, stable, profile, fallback_reason
    except RatingModelError as exc:
        return dict(stable), stable, "stable_fallback", str(exc)


def _seed_photo(
    group: PhotoGroupInternal,
    learned: dict[str, float],
    eligible_ids: set[str],
) -> PhotoObservation | None:
    members = [photo for photo in group.photos if photo.id in eligible_ids]
    if not members:
        return None
    return min(
        enumerate(members, start=1),
        key=lambda item: (
            _rank_value(item[1], item[0]),
            -learned.get(item[1].id, 0.0),
            item[1].filename.casefold(),
        ),
    )[1]


def _select_primary(
    groups: list[PhotoGroupInternal],
    photos: list[PhotoObservation],
    population_count: int,
    layers: DuplicateLayers,
    learned: dict[str, float],
    stable: dict[str, float],
) -> tuple[set[str], int, int]:
    selected: set[str] = set()
    selected_clusters: set[str] = set()
    group_selected = {group.id: 0 for group in groups}
    beat_selected: dict[str, int] = {}
    eligible_ids = {photo.id for photo in photos}

    for group in groups:
        seed = _seed_photo(group, learned, eligible_ids)
        if seed is None:
            continue
        selected.add(seed.id)
        selected_clusters.add(layers.strict_cluster_by_photo[seed.id])
        group_selected[group.id] += 1
        beat_id = layers.beat_by_photo[seed.id]
        beat_selected[beat_id] = beat_selected.get(beat_id, 0) + 1

    target_count = max(
        len(selected),
        min(population_count, round(population_count * (1.0 - TARGET_REDUCTION))),
    )
    by_id = {photo.id: photo for photo in photos}
    remaining = set(by_id) - selected
    while len(selected) < target_count:
        eligible = [
            photo_id
            for photo_id in remaining
            if layers.strict_cluster_by_photo[photo_id] not in selected_clusters
        ]
        if not eligible:
            break

        def utility(photo_id: str) -> tuple[float, int, str]:
            photo = by_id[photo_id]
            beat_id = layers.beat_by_photo[photo_id]
            base = LEARNED_ALPHA * learned[photo_id] + (1.0 - LEARNED_ALPHA) * stable[photo_id]
            value = (
                base
                * DUPLICATE_DEMOTE ** beat_selected.get(beat_id, 0)
                * GROUP_DEMOTE ** max(0, group_selected[photo.group_id] - 1)
            )
            return value, -_rank_value(photo, 10**9), photo.filename.casefold()

        winner_id = max(eligible, key=utility)
        remaining.remove(winner_id)
        selected.add(winner_id)
        cluster_id = layers.strict_cluster_by_photo[winner_id]
        selected_clusters.add(cluster_id)
        winner = by_id[winner_id]
        group_selected[winner.group_id] += 1
        beat_id = layers.beat_by_photo[winner_id]
        beat_selected[beat_id] = beat_selected.get(beat_id, 0) + 1

    return selected, target_count, max(0, target_count - len(selected))


def _person_stage_keys_by_photo(
    photos: list[PhotoObservation],
    eligible_people: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    people = set(eligible_people)
    return {
        photo.id: tuple(
            sorted(
                f"{photo.stage_id}:{person_id}"
                for person_id in photo.significant_person_ids & people
                if photo.stage_id
            )
        )
        for photo in photos
    }


def _select_delivery_reserve(
    photos: list[PhotoObservation],
    layers: DuplicateLayers,
    learned: dict[str, float],
    stable: dict[str, float],
    *,
    eligible_people: tuple[str, ...],
    target_count: int,
) -> dict[str, tuple[str, ...]]:
    """选择人物×环节安全余量；不足时宁可报告缺口，也不填充严格重复或硬废片。"""
    keys_by_photo = _person_stage_keys_by_photo(photos, eligible_people)
    delivered = {photo.id for photo in photos if photo.stars >= 2}
    represented_clusters = {
        layers.strict_cluster_by_photo[photo_id]
        for photo_id in delivered
    }
    group_counts = Counter(photo.group_id for photo in photos if photo.id in delivered)
    beat_counts = Counter(layers.beat_by_photo[photo_id] for photo_id in delivered)
    key_counts: Counter[str] = Counter(
        key
        for photo in photos
        if photo.id in delivered
        for key in keys_by_photo[photo.id]
    )
    reserve: dict[str, tuple[str, ...]] = {}

    while len(delivered) < target_count:
        options = sorted(
            (
                photo
                for photo in photos
                if photo.id not in delivered
                and not photo.rating_locked
                and photo.stars < 2
                and keys_by_photo[photo.id]
                and layers.strict_cluster_by_photo[photo.id] not in represented_clusters
            ),
            key=lambda photo: photo.filename.casefold(),
        )
        if not options:
            break

        def utility(photo: PhotoObservation) -> tuple[float, ...]:
            keys = keys_by_photo[photo.id]
            counts = [key_counts[key] for key in keys]
            minimum = min(counts)
            quality = (
                LEARNED_ALPHA * learned.get(photo.id, stable.get(photo.id, 0.0))
                + (1.0 - LEARNED_ALPHA) * stable.get(photo.id, 0.0)
            )
            severe_issues = sum(issue in SEVERE_ISSUES for issue in photo.issues)
            return (
                -float(minimum),
                float(sum(count == minimum for count in counts)),
                sum(1.0 / (count + 1.0) for count in counts),
                -float(severe_issues),
                -float(beat_counts[layers.beat_by_photo[photo.id]]),
                -float(group_counts[photo.group_id]),
                quality,
                stable.get(photo.id, 0.0),
                -float(_rank_value(photo, 10**9)),
            )

        chosen = max(options, key=utility)
        chosen_keys = keys_by_photo[chosen.id]
        reserve[chosen.id] = chosen_keys
        delivered.add(chosen.id)
        represented_clusters.add(layers.strict_cluster_by_photo[chosen.id])
        group_counts[chosen.group_id] += 1
        beat_counts[layers.beat_by_photo[chosen.id]] += 1
        key_counts.update(chosen_keys)

    return reserve


def _duplicate_leaks(photo_ids: set[str], layers: DuplicateLayers) -> int:
    counts: dict[str, int] = {}
    for photo_id in photo_ids:
        cluster_id = layers.strict_cluster_by_photo[photo_id]
        counts[cluster_id] = counts.get(cluster_id, 0) + 1
    return sum(max(0, count - 1) for count in counts.values())


def assign_semantic_ratings(
    groups: list[PhotoGroupInternal],
    *,
    window_minutes: int,
    model: Any | None = None,
    feature_provider: RatingFeatureProvider | None = None,
) -> RatingReport:
    all_photos = [photo for group in groups for photo in group.photos]
    valid_photos = [photo for photo in all_photos if _is_decodable(photo)]
    by_id = {photo.id: photo for photo in all_photos}
    layers = build_duplicate_layers(groups)
    for photo in all_photos:
        photo.strict_duplicate_cluster_id = layers.strict_cluster_by_photo[photo.id]
        photo.beat_id = layers.beat_by_photo[photo.id]
        reason = (
            RatingReason.TECHNICAL_REJECT
            if not _is_decodable(photo) or _has_decisive_issue(photo)
            else RatingReason.REDUNDANT_REJECT
        )
        apply_rating(
            photo,
            tier=RatingTier.WASTE,
            origin=RatingOrigin.AI,
            reason=reason,
        )

    learned, stable, rating_model_profile, rating_model_fallback_reason = _learned_scores(
        valid_photos,
        groups,
        model=model,
        feature_provider=feature_provider,
    )
    primary_ids, target_count, initial_shortfall = _select_primary(
        groups,
        valid_photos,
        len(valid_photos),
        layers,
        learned,
        stable,
    )
    for photo_id in primary_ids:
        apply_rating(
            by_id[photo_id],
            tier=RatingTier.PRIMARY,
            origin=RatingOrigin.AI,
            reason=RatingReason.PRIMARY_RANK,
        )
    actual_primary_ids = {photo.id for photo in all_photos if photo.stars == 3}

    coverage = select_person_stage_coverage(
        groups,
        primary_photo_ids=actual_primary_ids,
        window_minutes=window_minutes,
    )
    for photo_id in coverage.selected_photo_ids:
        photo = by_id[photo_id]
        apply_rating(
            photo,
            tier=RatingTier.COVERAGE,
            origin=RatingOrigin.COVERAGE,
            reason=RatingReason.PERSON_STAGE_GAP,
            needs_review=bool(photo.issues) or photo.score < 55.0,
            coverage_keys=coverage.keys_by_photo[photo_id],
        )

    delivery_target_count = min(
        len(valid_photos),
        round(len(valid_photos) * (1.0 - DELIVERY_TARGET_REDUCTION)),
    )
    reserve_keys_by_photo = _select_delivery_reserve(
        valid_photos,
        layers,
        learned,
        stable,
        eligible_people=coverage.eligible_people,
        target_count=delivery_target_count,
    )
    for photo_id, coverage_keys in reserve_keys_by_photo.items():
        photo = by_id[photo_id]
        apply_rating(
            photo,
            tier=RatingTier.COVERAGE,
            origin=RatingOrigin.COVERAGE,
            reason=RatingReason.PERSON_STAGE_RESERVE,
            needs_review=bool(photo.issues) or photo.score < 55.0,
            coverage_keys=coverage_keys,
        )

    represented_clusters = {
        photo.strict_duplicate_cluster_id for photo in all_photos if photo.stars >= 2
    }
    for group in groups:
        candidates = [
            photo
            for photo in group.photos
            if photo.stars < 2
            and _is_decodable(photo)
            and not _has_decisive_issue(photo)
            and photo.score >= 55.0
            and photo.strict_duplicate_cluster_id not in represented_clusters
            and not photo.rating_locked
        ]
        if not candidates:
            continue
        winner = max(
            candidates,
            key=lambda photo: (
                learned.get(photo.id, stable.get(photo.id, 0.0)),
                stable.get(photo.id, 0.0),
                -_rank_value(photo, 10**9),
                photo.filename.casefold(),
            ),
        )
        if apply_rating(
            winner,
            tier=RatingTier.VALUABLE,
            origin=RatingOrigin.AI,
            reason=RatingReason.UNIQUE_MOMENT,
        ):
            represented_clusters.add(winner.strict_duplicate_cluster_id)

    for photo in all_photos:
        photo.coverage_protected = photo.stars == 2 and photo.rating_origin == RatingOrigin.COVERAGE.value
        photo.coverage_person_ids = sorted(
            {key.split(":", 1)[1] for key in photo.coverage_keys if ":" in key}
        )
        photo.is_best_pick = photo.stars >= 2

    actual_primary_ids = {photo.id for photo in all_photos if photo.stars == 3}
    unresolved_keys = set(coverage.unresolved_keys)
    for photo_id, keys in coverage.keys_by_photo.items():
        if by_id[photo_id].stars < 2:
            unresolved_keys.update(keys)
    counts = {stars: sum(photo.stars == stars for photo in all_photos) for stars in range(4)}
    actual_reduction = (
        1.0 - len(actual_primary_ids) / len(valid_photos)
        if valid_photos
        else 1.0
    )
    actual_delivery_ids = {photo.id for photo in all_photos if photo.stars >= 2}
    delivery_actual_reduction = (
        1.0 - len(actual_delivery_ids) / len(valid_photos)
        if valid_photos
        else 1.0
    )
    return RatingReport(
        target_reduction=TARGET_REDUCTION,
        actual_reduction=actual_reduction,
        delivery_target_reduction=DELIVERY_TARGET_REDUCTION,
        delivery_actual_reduction=delivery_actual_reduction,
        primary_count=counts[3],
        coverage_count=counts[2],
        coverage_reserve_count=sum(
            photo.rating_reason == RatingReason.PERSON_STAGE_RESERVE.value
            for photo in all_photos
        ),
        valuable_count=counts[1],
        waste_count=counts[0],
        primary_budget_shortfall=max(initial_shortfall, target_count - len(actual_primary_ids)),
        delivery_budget_shortfall=max(0, delivery_target_count - len(actual_delivery_ids)),
        primary_duplicate_leaks=_duplicate_leaks(actual_primary_ids, layers),
        required_coverage_keys=len(coverage.required_keys),
        unresolved_coverage_keys=len(unresolved_keys),
        rating_model_profile=rating_model_profile,
        rating_model_fallback_reason=rating_model_fallback_reason,
    )
