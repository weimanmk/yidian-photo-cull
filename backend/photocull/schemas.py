from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ScanRequest(BaseModel):
    folder: str = Field(min_length=1)
    grouping_preset: Literal["cautious", "balanced", "aggressive"] = "balanced"
    keep_per_group: int = Field(default=1, ge=1, le=5)
    coverage_enabled: bool = True
    coverage_window_minutes: int = Field(default=15, ge=5, le=60)
    recursive: bool = True


class SettingsPatch(BaseModel):
    grouping_preset: Literal["cautious", "balanced", "aggressive"] | None = None
    keep_per_group: int | None = Field(default=None, ge=1, le=5)
    coverage_enabled: bool | None = None
    coverage_window_minutes: int | None = Field(default=None, ge=5, le=60)
    face_identity_threshold: float | None = Field(default=None, ge=0.32, le=0.62)
    use_gpu: bool | None = None
    recursive: bool | None = None
    jpeg_preview_quality: int | None = Field(default=None, ge=60, le=95)
    vlm_enabled: bool | None = None
    vlm_server_url: str | None = Field(default=None, min_length=1, max_length=200)
    vlm_executable_path: str | None = Field(default=None, max_length=1000)
    vlm_model_path: str | None = Field(default=None, max_length=1000)
    vlm_mmproj_path: str | None = Field(default=None, max_length=1000)
    vlm_model_id: str | None = Field(default=None, min_length=1, max_length=160)
    vlm_quantization: str | None = Field(default=None, min_length=1, max_length=40)
    vlm_context_size: int | None = Field(default=None, ge=4096, le=32768)
    vlm_gpu_layers: int | None = Field(default=None, ge=0, le=256)
    vlm_max_groups: int | None = Field(default=None, ge=1, le=500)
    vlm_max_candidates: int | None = Field(default=None, ge=2, le=8)
    vlm_ambiguity_margin: float | None = Field(default=None, ge=0.5, le=30.0)
    vlm_min_confidence: float | None = Field(default=None, ge=0.5, le=0.95)
    vlm_timeout_seconds: int | None = Field(default=None, ge=30, le=900)


class PhotoLabelRequest(BaseModel):
    category: Literal["selected", "duplicate", "blurred", "closed_eyes", "exposure", "rejected"]
    stars: int | None = Field(default=None, ge=0, le=5)


class PhotoRatingRequest(BaseModel):
    stars: int = Field(ge=0, le=3)
    locked: bool = True


class ExportRequest(BaseModel):
    destination: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    selected_only: bool = True
    copy_originals: bool = True
    write_xmp: bool = True


class ExportPreflightRequest(BaseModel):
    destination: str = Field(min_length=1, max_length=2000)
    project_id: str = Field(min_length=1, max_length=160)
    minimum_stars: Literal[1, 2, 3] = 2


class ExportExecuteRequest(BaseModel):
    plan_hash: str = Field(min_length=64, max_length=64)
    confirmed: Literal[True]


class LightroomPreflightRequest(BaseModel):
    project_id: str = Field(min_length=1, max_length=160)
