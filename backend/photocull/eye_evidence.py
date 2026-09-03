from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from typing import Any

from .internal_models import FaceObservation


NEUTRAL_EYE_SCORE = 72.0
DEFAULT_EYE_EVIDENCE_PROFILE_NAME = "default"


@dataclass(frozen=True, slots=True)
class EyeEvidenceProfile:
    name: str
    min_area_ratio: float
    min_confidence: float
    max_abs_yaw: float = 35.0
    max_abs_pitch: float = 15.0
    max_abs_roll: float = 35.0
    min_eye_sharpness: float = 35.0
    max_occlusion: float = 0.55
    max_closed_probability: float = 0.18

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EyeEvidence:
    open_score: float
    reliability: float
    decisive_closed: bool
    uncertain: bool
    reasons: tuple[str, ...]


EYE_EVIDENCE_PROFILES = {
    "default": EyeEvidenceProfile("default", 0.0030, 0.82),
    "wide-hard": EyeEvidenceProfile("wide-hard", 0.0025, 0.80),
    "conservative-hard": EyeEvidenceProfile("conservative-hard", 0.0040, 0.86),
}


def active_eye_evidence_profile() -> EyeEvidenceProfile:
    fallback = EYE_EVIDENCE_PROFILES.get(
        DEFAULT_EYE_EVIDENCE_PROFILE_NAME,
        EYE_EVIDENCE_PROFILES["default"],
    )
    configured = os.getenv("PHOTOCULL_EYE_EVIDENCE_PROFILE", "").strip().casefold()
    return EYE_EVIDENCE_PROFILES.get(configured, fallback)


def eye_evidence_status() -> dict[str, Any]:
    return {
        **active_eye_evidence_profile().public_dict(),
        "ranking_applied": False,
        "validation_status": "development-gates-failed",
    }


def _finite_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def _ascending(value: float, low: float, high: float) -> float:
    return _clamp((value - low) / max(high - low, 1e-9))


def _descending(value: float, full: float, zero: float) -> float:
    return _clamp((zero - value) / max(zero - full, 1e-9))


def evaluate_eye_evidence(
    face: FaceObservation,
    profile: EyeEvidenceProfile | None = None,
) -> EyeEvidence:
    selected = profile or active_eye_evidence_profile()
    state = str(face.eye_state or "Unknown").strip().title()
    if state not in {"Open", "Partial", "Closed"}:
        return EyeEvidence(
            open_score=NEUTRAL_EYE_SCORE,
            reliability=0.0,
            decisive_closed=False,
            uncertain=False,
            reasons=("unknown_eye_state",),
        )

    raw_values = {
        "area_ratio": _finite_number(face.area_ratio),
        "confidence": _finite_number(face.confidence),
        "yaw": _finite_number(face.yaw),
        "pitch": _finite_number(face.pitch),
        "roll": _finite_number(face.roll),
        "eye_sharpness": _finite_number(face.eye_sharpness),
        "occlusion_risk": _finite_number(face.occlusion_risk),
        "open_probability": _finite_number(face.open_probability),
    }
    reasons: list[str] = []
    for field, value in raw_values.items():
        if value is None:
            reasons.append("missing_open_probability" if field == "open_probability" else f"invalid_{field}")

    valid_ranges: dict[str, tuple[float, float | None]] = {
        "area_ratio": (0.0, 1.0),
        "confidence": (0.0, 1.0),
        "yaw": (-90.0, 90.0),
        "pitch": (-90.0, 90.0),
        "roll": (-90.0, 90.0),
        "eye_sharpness": (0.0, None),
        "occlusion_risk": (0.0, 1.0),
        "open_probability": (0.0, 1.0),
    }
    for field, (minimum, maximum) in valid_ranges.items():
        value = raw_values[field]
        if value is not None and (value < minimum or (maximum is not None and value > maximum)):
            raw_values[field] = None
            reasons.append(f"invalid_{field}")

    probability = raw_values["open_probability"]

    area_ratio = raw_values["area_ratio"]
    confidence = raw_values["confidence"]
    yaw = raw_values["yaw"]
    pitch = raw_values["pitch"]
    roll = raw_values["roll"]
    eye_sharpness = raw_values["eye_sharpness"]
    occlusion_risk = raw_values["occlusion_risk"]

    if area_ratio is not None and area_ratio < selected.min_area_ratio:
        reasons.append("face_too_small")
    if confidence is not None and confidence < selected.min_confidence:
        reasons.append("low_detection_confidence")
    if yaw is not None and abs(yaw) > selected.max_abs_yaw:
        reasons.append("side_pose")
    if pitch is not None and abs(pitch) > selected.max_abs_pitch:
        reasons.append("pitch_out_of_range")
    if roll is not None and abs(roll) > selected.max_abs_roll:
        reasons.append("roll_out_of_range")
    if eye_sharpness is not None and eye_sharpness < selected.min_eye_sharpness:
        reasons.append("eyes_not_sharp")
    if occlusion_risk is not None and occlusion_risk >= selected.max_occlusion:
        reasons.append("occluded")

    complete = all(value is not None for value in raw_values.values()) and probability is not None
    if complete:
        assert area_ratio is not None
        assert confidence is not None
        assert yaw is not None
        assert pitch is not None
        assert roll is not None
        assert eye_sharpness is not None
        assert occlusion_risk is not None
        components = (
            _ascending(area_ratio, 0.0012, selected.min_area_ratio),
            _ascending(confidence, 0.70, selected.min_confidence),
            _descending(abs(yaw), 20.0, selected.max_abs_yaw),
            _descending(abs(pitch), 10.0, selected.max_abs_pitch),
            _descending(abs(roll), 20.0, selected.max_abs_roll),
            _ascending(eye_sharpness, 20.0, selected.min_eye_sharpness),
            _descending(occlusion_risk, 0.25, selected.max_occlusion),
            _clamp(abs(probability - 0.5) / 0.5),
        )
        reliability = min(components)
    else:
        reliability = 0.0

    looking_down = pitch is not None and yaw is not None and pitch >= 15.0 and abs(yaw) <= 45.0
    if state == "Open":
        state_score = 100.0
    elif state == "Partial":
        state_score = 70.0 if looking_down else 52.0
    else:
        state_score = 58.0 if looking_down else 5.0
    if probability is not None and not looking_down:
        state_score = 0.68 * state_score + 32.0 * probability
    open_score = _clamp(
        reliability * state_score + (1.0 - reliability) * NEUTRAL_EYE_SCORE,
        0.0,
        100.0,
    )

    decisive_closed = bool(
        state == "Closed"
        and complete
        and probability is not None
        and probability <= selected.max_closed_probability
        and area_ratio is not None
        and area_ratio >= selected.min_area_ratio
        and confidence is not None
        and confidence >= selected.min_confidence
        and yaw is not None
        and abs(yaw) <= selected.max_abs_yaw
        and pitch is not None
        and abs(pitch) <= selected.max_abs_pitch
        and roll is not None
        and abs(roll) <= selected.max_abs_roll
        and eye_sharpness is not None
        and eye_sharpness >= selected.min_eye_sharpness
        and occlusion_risk is not None
        and occlusion_risk < selected.max_occlusion
    )
    uncertain = bool(
        state == "Partial"
        or (state == "Closed" and not decisive_closed)
        or (state == "Open" and reliability < 0.5)
    )
    return EyeEvidence(
        open_score=open_score,
        reliability=reliability,
        decisive_closed=decisive_closed,
        uncertain=uncertain,
        reasons=tuple(dict.fromkeys(reasons)),
    )
