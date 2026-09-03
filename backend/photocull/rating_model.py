from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable, Literal, Mapping, Protocol

import numpy as np

from .face_quality import select_quality_faces
from .internal_models import PhotoObservation


BASE_FEATURE_NAMES = (
    "sharpness_score",
    "motion_blur_score",
    "exposure_score",
    "contrast_score",
    "noise_score",
    "composition_score",
    "face_quality_score",
    "min_face_score",
    "face_sharpness_score",
    "eye_sharpness_score",
    "eye_score",
    "expression_score",
    "technical_score",
    "generic_group_score",
    "group_relative_score",
    "group_quality_percentile",
    "group_size",
    "relative_face_quality",
    "relative_eye",
    "relative_motion",
    "relative_exposure",
    "relative_composition",
    "bad_face_count",
    "face_count",
    "largest_face_ratio",
    "mean_occlusion_risk",
    "mean_abs_yaw",
    "mean_smile_score",
    "portrait_present",
    "subject_sharpness_score",
    "depth_focus_score",
    "depth_background_blur_score",
    "depth_separation_score",
    "depth_occlusion_risk",
    "main_subject_score",
    "person_count",
    "dof_portrait_quality",
)
IQA_FEATURE_NAMES = BASE_FEATURE_NAMES + (
    "iqa_musiq",
    "iqa_qualiclip",
    "iqa_relative_musiq",
    "iqa_relative_qualiclip",
    "iqa_consensus",
    "iqa_disagreement",
)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "assets" / "rating_model_v1.json"


class RatingModelError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RatingProfile:
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float

    @classmethod
    def from_payload(cls, name: str, payload: Mapping[str, object]) -> "RatingProfile":
        try:
            profile = cls(
                feature_names=tuple(str(value) for value in payload["feature_names"]),
                mean=np.asarray(payload["mean"], dtype=np.float64),
                scale=np.asarray(payload["scale"], dtype=np.float64),
                weights=np.asarray(payload["weights"], dtype=np.float64),
                bias=float(payload["bias"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RatingModelError(f"评分模型档案 {name} 无法解析") from exc

        expected = len(profile.feature_names)
        if profile.mean.shape != (expected,) or profile.scale.shape != (expected,) or profile.weights.shape != (expected,):
            raise RatingModelError(f"评分模型档案 {name} 特征维度不一致")
        if (
            not np.isfinite(profile.mean).all()
            or not np.isfinite(profile.scale).all()
            or not np.isfinite(profile.weights).all()
            or not math.isfinite(profile.bias)
            or np.any(profile.scale <= 0.0)
        ):
            raise RatingModelError(f"评分模型档案 {name} 包含非法数值")
        return profile


class RatingFeatureProvider(Protocol):
    def matrix(
        self,
        photos: list[PhotoObservation],
        profiles: Mapping[str, RatingProfile],
    ) -> tuple[np.ndarray, Literal["base", "iqa"]]: ...


def _photo_values(photo: PhotoObservation) -> dict[str, float]:
    values = {name: float(value) for name, value in photo.metrics.items()}
    quality_faces = select_quality_faces(photo.faces)
    values.update(
        {
            "face_count": float(len(photo.faces)),
            "largest_face_ratio": max((face.area_ratio for face in quality_faces), default=0.0) * 1000.0,
            "mean_occlusion_risk": (
                float(np.mean([face.occlusion_risk for face in quality_faces])) * 100.0
                if quality_faces
                else 0.0
            ),
            "mean_abs_yaw": (
                float(np.mean([abs(face.yaw) for face in quality_faces])) if quality_faces else 0.0
            ),
            "mean_smile_score": (
                float(np.mean([face.smile_score for face in quality_faces])) if quality_faces else 0.0
            ),
            "portrait_present": float(bool(quality_faces)),
            "person_count": float(len(set(photo.person_ids))),
        }
    )
    subject_sharpness = float(
        values.get(
            "subject_sharpness_score",
            values.get("motion_blur_score", values.get("sharpness_score", 0.0)),
        )
    )
    background_blur = float(values.get("depth_background_blur_score", 0.0))
    face_quality = float(values.get("face_quality_score", 0.0))
    values["dof_portrait_quality"] = (
        max(subject_sharpness, min(face_quality, 100.0))
        if quality_faces and face_quality >= 70.0 and background_blur >= 60.0
        else subject_sharpness
    )
    return values


def _matrix(values_by_photo: list[dict[str, float]], feature_names: tuple[str, ...]) -> np.ndarray:
    if not values_by_photo:
        return np.empty((0, len(feature_names)), dtype=np.float64)
    return np.asarray(
        [[float(values.get(name, 0.0)) for name in feature_names] for values in values_by_photo],
        dtype=np.float64,
    )


class BuiltInRatingFeatures:
    def matrix(
        self,
        photos: list[PhotoObservation],
        profiles: Mapping[str, RatingProfile],
    ) -> tuple[np.ndarray, Literal["base"]]:
        profile = profiles.get("base")
        if profile is None:
            raise RatingModelError("评分模型缺少 base 档案")
        return _matrix([_photo_values(photo) for photo in photos], profile.feature_names), "base"


def _percentiles(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if len(array) <= 1 or float(np.ptp(array)) < 1e-9:
        return np.full(len(array), 0.5, dtype=np.float64)
    order = np.argsort(array, kind="stable")
    ranks = np.empty(len(array), dtype=np.float64)
    ranks[order] = np.arange(len(array), dtype=np.float64)
    return ranks / (len(array) - 1)


class IqaRatingFeatures:
    def __init__(
        self,
        fallback: BuiltInRatingFeatures | None = None,
        *,
        scores_by_photo: Mapping[str, Mapping[str, float]] | None = None,
    ) -> None:
        self._fallback = fallback or BuiltInRatingFeatures()
        self._scores_by_photo = scores_by_photo
        self.last_fallback_reason = ""

    def matrix(
        self,
        photos: list[PhotoObservation],
        profiles: Mapping[str, RatingProfile],
    ) -> tuple[np.ndarray, Literal["base", "iqa"]]:
        score_pairs = [self._score_pair(photo) for photo in photos]
        if not photos or any(pair is None for pair in score_pairs):
            self.last_fallback_reason = "optional_iqa_scores_incomplete"
            return self._fallback.matrix(photos, profiles)
        profile = profiles.get("iqa")
        if profile is None:
            self.last_fallback_reason = "iqa_profile_missing"
            return self._fallback.matrix(photos, profiles)
        self.last_fallback_reason = ""

        values_by_photo = [_photo_values(photo) for photo in photos]
        groups: dict[str, list[int]] = {}
        for index, (photo, score_pair) in enumerate(zip(photos, score_pairs, strict=True)):
            assert score_pair is not None
            groups.setdefault(photo.group_id, []).append(index)
            values_by_photo[index]["iqa_musiq"] = score_pair[0]
            values_by_photo[index]["iqa_qualiclip"] = score_pair[1] * 100.0

        for indices in groups.values():
            relative_musiq = _percentiles([values_by_photo[index]["iqa_musiq"] for index in indices]) * 100.0
            relative_qualiclip = (
                _percentiles([values_by_photo[index]["iqa_qualiclip"] for index in indices]) * 100.0
            )
            for index, musiq, qualiclip in zip(indices, relative_musiq, relative_qualiclip, strict=True):
                values_by_photo[index]["iqa_relative_musiq"] = float(musiq)
                values_by_photo[index]["iqa_relative_qualiclip"] = float(qualiclip)
                values_by_photo[index]["iqa_consensus"] = float((musiq + qualiclip) / 2.0)
                values_by_photo[index]["iqa_disagreement"] = float(abs(musiq - qualiclip))
        return _matrix(values_by_photo, profile.feature_names), "iqa"

    def _score_pair(self, photo: PhotoObservation) -> tuple[float, float] | None:
        try:
            if self._scores_by_photo is None:
                musiq = float(photo.metrics["musiq_score"])
                qualiclip = float(photo.metrics["qualiclip_score"])
            else:
                external = self._scores_by_photo[photo.id]
                musiq = float(external["musiq_score"] if "musiq_score" in external else external["musiq"])
                qualiclip = float(
                    external["qualiclip_score"]
                    if "qualiclip_score" in external
                    else external["qualiclip+"]
                )
            if math.isfinite(musiq) and math.isfinite(qualiclip):
                return musiq, qualiclip
            return None
        except (KeyError, TypeError, ValueError):
            return None


class FrozenRatingModel:
    def __init__(
        self,
        *,
        version: str,
        profiles: dict[str, RatingProfile],
        training_hashes: dict[str, str],
        selection_parameters: dict[str, float | int],
    ) -> None:
        self.version = version
        self.profiles = profiles
        self.training_hashes = training_hashes
        self.selection_parameters = selection_parameters
        self.last_profile: str | None = None
        self.last_fallback_reason: str = ""

    @classmethod
    def load_default(cls) -> "FrozenRatingModel":
        return cls.load(DEFAULT_MODEL_PATH)

    @classmethod
    def load(cls, path: Path) -> "FrozenRatingModel":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            version = str(payload["version"])
            raw_profiles = dict(payload["profiles"])
            training_hashes = {
                str(name): str(value).upper() for name, value in dict(payload["training_hashes"]).items()
            }
            selection_parameters = {
                str(name): value for name, value in dict(payload["selection_parameters"]).items()
            }
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RatingModelError(f"无法读取评分模型资产: {path}") from exc
        if version != "rating-pointwise-v1":
            raise RatingModelError(f"不支持的评分模型版本: {version}")
        if set(raw_profiles) != {"base", "iqa"}:
            raise RatingModelError("评分模型必须同时包含 base 与 iqa 档案")
        profiles = {
            name: RatingProfile.from_payload(name, dict(raw_payload))
            for name, raw_payload in raw_profiles.items()
        }
        if profiles["base"].feature_names != BASE_FEATURE_NAMES:
            raise RatingModelError("base 档案特征契约不匹配")
        if profiles["iqa"].feature_names != IQA_FEATURE_NAMES:
            raise RatingModelError("iqa 档案特征契约不匹配")
        return cls(
            version=version,
            profiles=profiles,
            training_hashes=training_hashes,
            selection_parameters=selection_parameters,
        )

    def predict(
        self,
        photos: Iterable[PhotoObservation],
        *,
        feature_provider: RatingFeatureProvider | None = None,
    ) -> dict[str, float]:
        members = list(photos)
        provider = feature_provider or IqaRatingFeatures()
        rows, profile_name = provider.matrix(members, self.profiles)
        self.last_fallback_reason = str(getattr(provider, "last_fallback_reason", ""))
        profile = self.profiles.get(profile_name)
        if profile is None:
            raise RatingModelError(f"未知评分模型档案: {profile_name}")
        matrix = np.asarray(rows, dtype=np.float64)
        expected_shape = (len(members), len(profile.feature_names))
        if matrix.shape != expected_shape:
            raise RatingModelError(
                f"评分特征维度不匹配: expected={expected_shape}, actual={matrix.shape}"
            )
        if not np.isfinite(matrix).all():
            raise RatingModelError("评分特征包含非有限数值")

        standardized = (matrix - profile.mean) / profile.scale
        logits = np.clip(standardized @ profile.weights + profile.bias, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        if not np.isfinite(probabilities).all():
            raise RatingModelError("评分模型输出包含非有限数值")
        self.last_profile = profile_name
        return {
            photo.id: float(score)
            for photo, score in zip(members, probabilities, strict=True)
        }
