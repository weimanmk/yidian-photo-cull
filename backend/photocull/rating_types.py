from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .internal_models import PhotoObservation


class RatingTier(StrEnum):
    WASTE = "waste"
    VALUABLE = "valuable"
    COVERAGE = "coverage"
    PRIMARY = "primary"


class RatingOrigin(StrEnum):
    AI = "ai"
    COVERAGE = "coverage"
    MANUAL = "manual"
    LEGACY = "legacy"


class RatingReason(StrEnum):
    PRIMARY_RANK = "primary_rank"
    PERSON_STAGE_GAP = "person_stage_gap"
    PERSON_STAGE_RESERVE = "person_stage_reserve"
    UNIQUE_MOMENT = "unique_moment"
    TECHNICAL_REJECT = "technical_reject"
    REDUNDANT_REJECT = "redundant_reject"
    MANUAL_OVERRIDE = "manual_override"


STAR_BY_TIER = {
    RatingTier.WASTE: 0,
    RatingTier.VALUABLE: 1,
    RatingTier.COVERAGE: 2,
    RatingTier.PRIMARY: 3,
}
TIER_BY_STAR = {stars: tier for tier, stars in STAR_BY_TIER.items()}


@dataclass(frozen=True, slots=True)
class RatingReport:
    target_reduction: float
    actual_reduction: float
    delivery_target_reduction: float
    delivery_actual_reduction: float
    primary_count: int
    coverage_count: int
    coverage_reserve_count: int
    valuable_count: int
    waste_count: int
    primary_budget_shortfall: int
    delivery_budget_shortfall: int
    primary_duplicate_leaks: int
    required_coverage_keys: int
    unresolved_coverage_keys: int
    rating_model_profile: str
    rating_model_fallback_reason: str

    def public_dict(self) -> dict[str, Any]:
        return {
            "target_reduction": self.target_reduction,
            "actual_reduction": self.actual_reduction,
            "delivery_target_reduction": self.delivery_target_reduction,
            "delivery_actual_reduction": self.delivery_actual_reduction,
            "primary_count": self.primary_count,
            "coverage_count": self.coverage_count,
            "coverage_reserve_count": self.coverage_reserve_count,
            "valuable_count": self.valuable_count,
            "waste_count": self.waste_count,
            "primary_budget_shortfall": self.primary_budget_shortfall,
            "delivery_budget_shortfall": self.delivery_budget_shortfall,
            "primary_duplicate_leaks": self.primary_duplicate_leaks,
            "required_coverage_keys": self.required_coverage_keys,
            "unresolved_coverage_keys": self.unresolved_coverage_keys,
            "rating_model_profile": self.rating_model_profile,
            "rating_model_fallback_reason": self.rating_model_fallback_reason,
        }


def apply_rating(
    photo: PhotoObservation,
    *,
    tier: RatingTier,
    origin: RatingOrigin,
    reason: RatingReason,
    locked: bool = False,
    needs_review: bool = False,
    coverage_keys: tuple[str, ...] = (),
) -> bool:
    if photo.rating_locked and origin is not RatingOrigin.MANUAL:
        return False

    photo.stars = STAR_BY_TIER[tier]
    photo.rating_tier = tier.value
    photo.rating_origin = origin.value
    photo.rating_reason = reason.value
    photo.rating_locked = locked
    photo.needs_review = needs_review
    photo.coverage_keys = list(dict.fromkeys(coverage_keys))
    return True
