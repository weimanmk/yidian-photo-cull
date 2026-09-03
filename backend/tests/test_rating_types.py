from datetime import datetime
from pathlib import Path

import numpy as np

from photocull.internal_models import PhotoObservation, VisualDescriptor
from photocull.rating_types import RatingOrigin, RatingReason, RatingTier, apply_rating


def make_photo() -> PhotoObservation:
    return PhotoObservation(
        id="photo-a",
        path=Path("photo-a.jpg"),
        source_root=Path("."),
        filename="photo-a.jpg",
        relative_path="photo-a.jpg",
        width=1600,
        height=1000,
        capture_time=datetime(2026, 8, 30, 9, 0),
        file_sequence=1,
        descriptor=VisualDescriptor(
            phash=1,
            layout=np.ones(4, dtype=np.float32),
            color=np.ones(4, dtype=np.float32),
            edge=np.ones(4, dtype=np.float32),
        ),
    )


def test_each_semantic_tier_has_exactly_one_star_value() -> None:
    photo = make_photo()
    expected = {
        RatingTier.WASTE: 0,
        RatingTier.VALUABLE: 1,
        RatingTier.COVERAGE: 2,
        RatingTier.PRIMARY: 3,
    }

    for tier, stars in expected.items():
        reason = RatingReason.PRIMARY_RANK if stars == 3 else RatingReason.REDUNDANT_REJECT
        changed = apply_rating(photo, tier=tier, origin=RatingOrigin.AI, reason=reason)

        assert changed is True
        assert photo.stars == stars
        assert photo.rating_tier == tier.value
        assert photo.public_dict()["rating_tier"] == tier.value


def test_manual_rating_is_locked_and_survives_ai_reassignment() -> None:
    photo = make_photo()
    apply_rating(
        photo,
        tier=RatingTier.COVERAGE,
        origin=RatingOrigin.MANUAL,
        reason=RatingReason.MANUAL_OVERRIDE,
        locked=True,
    )

    changed = apply_rating(
        photo,
        tier=RatingTier.WASTE,
        origin=RatingOrigin.AI,
        reason=RatingReason.TECHNICAL_REJECT,
    )

    assert changed is False
    assert photo.stars == 2
    assert photo.rating_tier == "coverage"
    assert photo.rating_origin == "manual"
    assert photo.rating_locked is True
