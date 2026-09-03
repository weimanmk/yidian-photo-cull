from __future__ import annotations

import time
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from threading import Event, RLock, Thread
from typing import Any

import numpy as np

from .body_engine import BodyEngine
from .config import EngineSettings, settings_store
from .depth_engine import DepthEngine
from .eye_evidence import eye_evidence_status
from .face_engine import FaceEngine
from .face_quality import refine_face_quality
from .feature_cache import FeatureCache
from .grouping import group_similar_photos
from .identity import IdentityClusterer
from .imaging import (
    build_descriptor,
    cached_images_exist,
    capture_time,
    discover_images,
    ensure_cached_images,
    file_sequence,
    load_image,
    photo_id,
    resize_for_analysis,
)
from .internal_models import PhotoGroupInternal, PhotoObservation, VisualDescriptor
from .project_store import ProjectStore
from .preference import PREFERENCE_MODEL_VERSION, PreferenceModel
from .pose_engine import PoseEngine
from .quality import analyze_quality, rescore_quality
from .rating_policy import assign_semantic_ratings
from .rating_types import RatingOrigin, RatingReason, TIER_BY_STAR
from .scene_engine import SceneEmbeddingEngine
from .scoring import prepare_group_ranking_features, rank_groups
from .vlm import LlamaServerManager, review_groups_with_vlm


FEATURE_PIPELINE_VERSION = "0.9.0"
ENGINE_VERSION = "0.9.0"


def ensure_cache_hit_images(
    path: Path,
    identifier: str,
    jpeg_quality: int,
    *,
    enabled: bool,
) -> None:
    if not enabled or cached_images_exist(identifier):
        return
    image = load_image(path)
    ensure_cached_images(image, identifier, jpeg_quality)


class ScanConflictError(RuntimeError):
    pass


class ScannerService:
    def __init__(
        self,
        projects: ProjectStore | None = None,
        feature_cache: FeatureCache | None = None,
        vlm_runtime: LlamaServerManager | None = None,
        rating_feature_provider: Any | None = None,
    ) -> None:
        self.projects = projects or ProjectStore()
        cache_path = None if projects is None else self.projects.root.parent / "cache.db"
        self.preference_model_path = None if projects is None else self.projects.root.parent / "preference-model.json"
        self.feature_cache = feature_cache or FeatureCache(cache_path)
        self.vlm_runtime = vlm_runtime or LlamaServerManager()
        self.rating_feature_provider = rating_feature_provider
        self._lock = RLock()
        self._cancel = Event()
        self._thread: Thread | None = None
        self._status = self._idle_status()
        self._results: dict[str, Any] | None = None
        self._files: dict[str, str] = {}

    @staticmethod
    def _idle_status() -> dict[str, Any]:
        return {
            "status": "idle",
            "phase": "等待",
            "message": "等待开始",
            "processed": 0,
            "total": 0,
            "progress": 0.0,
            "current_file": "",
            "elapsed_seconds": 0.0,
            "eta_seconds": None,
            "error": None,
            "project_id": None,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            status = dict(self._status)
        if status["status"] not in {"idle", "completed", "cancelled", "failed"}:
            elapsed = max(0.0, time.monotonic() - status.pop("_started_at", time.monotonic()))
            status["elapsed_seconds"] = round(elapsed, 1)
            if status["processed"] and status["total"]:
                remaining = max(0, status["total"] - status["processed"])
                status["eta_seconds"] = round(elapsed / status["processed"] * remaining)
        else:
            status.pop("_started_at", None)
        return status

    def start(
        self,
        folder: str,
        grouping_preset: str,
        keep_per_group: int,
        recursive: bool,
        *,
        coverage_enabled: bool | None = None,
        coverage_window_minutes: int | None = None,
        cache_hit_previews: bool = True,
    ) -> dict[str, Any]:
        root = Path(folder).expanduser().resolve()
        if not root.is_dir():
            raise ValueError("所选照片文件夹不存在")
        with self._lock:
            if self._thread and self._thread.is_alive():
                raise ScanConflictError("已有筛选任务正在运行")
            self._cancel.clear()
            self._results = None
            self._files = {}
            self._status = {
                **self._idle_status(),
                "status": "discovering",
                "phase": "扫描目录",
                "message": "正在查找支持的照片",
                "_started_at": time.monotonic(),
            }
            task_settings = settings_store.get()
            task_settings.grouping_preset = grouping_preset
            task_settings.keep_per_group = keep_per_group
            task_settings.recursive = recursive
            if coverage_enabled is not None:
                task_settings.coverage_enabled = bool(coverage_enabled)
            if coverage_window_minutes is not None:
                task_settings.coverage_window_minutes = min(60, max(5, int(coverage_window_minutes)))
            task_settings.coverage_enabled = True
            self._thread = Thread(
                target=self._run,
                args=(root, task_settings, cache_hit_previews),
                name="photocull-scan",
                daemon=True,
            )
            self._thread.start()
        return self.status()

    def cancel(self) -> dict[str, Any]:
        self._cancel.set()
        self.vlm_runtime.stop()
        with self._lock:
            if self._status["status"] not in {"idle", "completed", "failed"}:
                self._status.update(status="cancelled", phase="已取消", message="筛选任务已取消")
        return self.status()

    def results(self) -> dict[str, Any] | None:
        with self._lock:
            return self._results

    def files(self) -> dict[str, str]:
        with self._lock:
            return dict(self._files)

    def resolve_file(self, identifier: str) -> Path | None:
        with self._lock:
            path = self._files.get(identifier)
        candidate = Path(path) if path else None
        return candidate if candidate and candidate.is_file() else None

    def load_project(self, project_id: str) -> dict[str, Any]:
        results, files = self.projects.load(project_id)
        with self._lock:
            self._results = results
            self._files = files
            self._status = {
                **self._idle_status(),
                "status": "completed",
                "phase": "已完成",
                "message": "已载入本机项目",
                "processed": results.get("summary", {}).get("total", 0),
                "total": results.get("summary", {}).get("total", 0),
                "progress": 100.0,
                "project_id": project_id,
            }
        return results

    def label_photo(self, photo_identifier: str, category: str, stars: int | None) -> bool:
        with self._lock:
            if not self._results:
                return False
            target = next((photo for photo in self._results.get("photos", []) if photo.get("id") == photo_identifier), None)
            if target is None:
                return False
            previous_category = target.get("category")
            target["category"] = category
            target["is_best_pick"] = category == "selected"
            if target.get("coverage_protected"):
                target["coverage_protected"] = False
                target["coverage_person_ids"] = []
                target["selection_reasons"] = [
                    reason
                    for reason in target.get("selection_reasons", [])
                    if not str(reason).startswith("覆盖保底：")
                    and not str(reason).startswith("保留不代表画质合格")
                ]
            if stars is not None:
                target["stars"] = stars
            group = next(
                (group for group in self._results.get("groups", []) if group.get("id") == target.get("group_id")),
                None,
            )
            if group is not None:
                photo_by_id = {photo.get("id"): photo for photo in self._results.get("photos", [])}
                group["best_photo_ids"] = [
                    identifier
                    for identifier in group.get("photo_ids", [])
                    if photo_by_id.get(identifier, {}).get("is_best_pick")
                ]
                group["coverage_protected"] = any(
                    photo_by_id.get(identifier, {}).get("coverage_protected")
                    for identifier in group.get("photo_ids", [])
                )
            self._refresh_summary(self._results)
            project_id = self._results["project_id"]
            self.projects.save(project_id, self._results, self._files)
            self.feature_cache.record_user_action(
                project_id,
                photo_identifier,
                target.get("group_id"),
                previous_category,
                category,
                stars,
            )
        return True

    def rate_photo(self, photo_identifier: str, stars: int, *, locked: bool = True) -> bool:
        if stars not in TIER_BY_STAR:
            raise ValueError("人工星级必须是 0 到 3")
        with self._lock:
            if not self._results:
                return False
            target = next(
                (photo for photo in self._results.get("photos", []) if photo.get("id") == photo_identifier),
                None,
            )
            if target is None:
                return False
            previous_tier = str(target.get("rating_tier", target.get("category", "")))
            tier = TIER_BY_STAR[stars]
            target["stars"] = stars
            target["rating_tier"] = tier.value
            target["rating_origin"] = RatingOrigin.MANUAL.value
            target["rating_reason"] = RatingReason.MANUAL_OVERRIDE.value
            target["rating_locked"] = bool(locked)
            target["needs_review"] = False
            target["is_best_pick"] = stars >= 2
            target["coverage_protected"] = False
            target["coverage_person_ids"] = []

            photo_by_id = {photo.get("id"): photo for photo in self._results.get("photos", [])}
            group = next(
                (group for group in self._results.get("groups", []) if group.get("id") == target.get("group_id")),
                None,
            )
            if group is not None:
                group["best_photo_ids"] = [
                    identifier
                    for identifier in group.get("photo_ids", [])
                    if int(photo_by_id.get(identifier, {}).get("stars", 0)) >= 2
                ]
                group["coverage_protected"] = any(
                    bool(photo_by_id.get(identifier, {}).get("coverage_protected"))
                    for identifier in group.get("photo_ids", [])
                )
            self._refresh_summary(self._results)
            project_id = self._results["project_id"]
            self.projects.save(project_id, self._results, self._files)
            self.feature_cache.record_user_action(
                project_id,
                photo_identifier,
                target.get("group_id"),
                previous_tier,
                tier.value,
                stars,
            )
        return True

    def _update(self, **changes: Any) -> None:
        with self._lock:
            self._status.update(changes)

    def _run(self, root: Path, settings: EngineSettings, cache_hit_previews: bool = True) -> None:
        started_at = time.monotonic()
        try:
            paths = discover_images(root, settings.recursive)
            if not paths:
                raise ValueError("所选文件夹中没有支持的照片")
            self._update(
                status="analyzing",
                phase="视觉分析",
                message="正在提取场景、人脸、3D 姿态、景深与画质特征",
                total=len(paths),
            )
            face_engine = FaceEngine(use_gpu=settings.use_gpu)
            body_engine = BodyEngine(use_gpu=settings.use_gpu)
            pose_engine = PoseEngine()
            depth_engine = DepthEngine(use_gpu=settings.use_gpu)
            scene_engine = SceneEmbeddingEngine(use_gpu=settings.use_gpu)
            preference_model = PreferenceModel.load(self.preference_model_path)
            preference_applied = bool(preference_model and preference_model.applies_to_source(root))
            pipeline_signature = (
                f"{FEATURE_PIPELINE_VERSION}|image={scene_engine.signature()}|face={face_engine.signature()}"
                f"|body={body_engine.signature()}|pose={pose_engine.signature()}|depth={depth_engine.signature()}"
            )
            self.feature_cache.record_model("image_embedding", scene_engine.signature(), scene_engine.status())
            self.feature_cache.record_model("face_pipeline", face_engine.signature(), face_engine.status())
            self.feature_cache.record_model("body_pipeline", body_engine.signature(), body_engine.status())
            self.feature_cache.record_model("pose_pipeline", pose_engine.signature(), pose_engine.status())
            self.feature_cache.record_model("depth_pipeline", depth_engine.signature(), depth_engine.status())
            if preference_model is not None:
                self.feature_cache.record_model(
                    "photographer_preference",
                    PREFERENCE_MODEL_VERSION,
                    preference_model.status(),
                )
            photos: list[PhotoObservation] = []
            cache_hits = 0
            cache_misses = 0

            for index, path in enumerate(paths, start=1):
                if self._cancel.is_set():
                    self._update(status="cancelled", phase="已取消", message="筛选任务已取消")
                    return
                identifier = photo_id(path)
                try:
                    observation = self.feature_cache.load(path, root, pipeline_signature)
                    if observation is not None:
                        cache_hits += 1
                        observation.metrics, observation.score, observation.issues = rescore_quality(
                            observation.metrics,
                            observation.faces,
                        )
                        ensure_cache_hit_images(
                            path,
                            identifier,
                            settings.jpeg_preview_quality,
                            enabled=cache_hit_previews,
                        )
                    else:
                        cache_misses += 1
                        image = load_image(path)
                        width, height = image.size
                        taken_at = capture_time(image, path)
                        ensure_cached_images(image, identifier, settings.jpeg_preview_quality)
                        rgb = resize_for_analysis(image)
                        semantic = scene_engine.embed(rgb)
                        descriptor = build_descriptor(rgb, semantic)
                        faces = face_engine.analyze(rgb, identifier, image)
                        bodies = body_engine.analyze(rgb)
                        poses = pose_engine.analyze(rgb, image)
                        depth = depth_engine.analyze(rgb, faces, bodies, poses)
                        refine_face_quality(image, faces)
                        metrics, score, issues = analyze_quality(rgb, faces, depth)
                        observation = PhotoObservation(
                            id=identifier,
                            path=path,
                            source_root=root,
                            filename=path.name,
                            relative_path=str(path.relative_to(root)),
                            width=width,
                            height=height,
                            capture_time=taken_at,
                            file_sequence=file_sequence(path),
                            descriptor=descriptor,
                            faces=faces,
                            bodies=bodies,
                            poses=poses,
                            depth=depth,
                            metrics=metrics,
                            score=score,
                            issues=issues,
                        )
                        self.feature_cache.save(observation, pipeline_signature)
                except Exception as exc:
                    zeros = np.zeros(64, dtype=np.float16)
                    observation = PhotoObservation(
                        id=identifier,
                        path=path,
                        source_root=root,
                        filename=path.name,
                        relative_path=str(path.relative_to(root)),
                        width=0,
                        height=0,
                        capture_time=None,
                        file_sequence=file_sequence(path),
                        descriptor=VisualDescriptor(phash=0, layout=np.zeros(576, dtype=np.float16), color=np.zeros(96, dtype=np.float16), edge=zeros),
                        score=0.0,
                        issues=[f"文件读取失败：{exc}"],
                        category="rejected",
                    )
                photos.append(observation)
                self._files[identifier] = str(path)
                self._update(
                    processed=index,
                    current_file=path.name,
                    progress=round(index / len(paths) * 74.0, 2),
                    cache_hits=cache_hits,
                    cache_misses=cache_misses,
                )

            good_photos = [photo for photo in photos if photo.width > 0]
            self._update(status="identifying", phase="人物识别", message="正在聚合同一人物的人脸向量", progress=76.0)
            clusters = IdentityClusterer(settings.face_identity_threshold).assign(good_photos)

            self._update(
                status="grouping",
                phase="相似分组",
                message="正在融合 AI 场景、人物身份、3D 动作、景深与连拍时序",
                progress=83.0,
            )
            groups = group_similar_photos(good_photos, settings.grouping_preset)
            prepare_group_ranking_features(groups)
            if preference_applied and preference_model is not None:
                self._update(
                    status="personalizing",
                    phase="偏好校准",
                    message="正在应用本机组内相对选片偏好模型",
                    progress=88.0,
                )
                preference_model.apply(good_photos)
            for failed in (photo for photo in photos if photo.width == 0):
                failed.group_id = f"group-error-{failed.id}"

            self._update(
                status="ranking",
                phase="组内优选",
                message="正在比较睁眼、主体合焦、背景分离与整体画质",
                progress=92.0,
            )
            rank_groups(groups, settings.keep_per_group)
            self._update(
                status="vlm_reviewing",
                phase="大模型复核",
                message="正在复核表情、动作、互动与构图难分胜负的候选组",
                progress=94.0,
            )

            def vlm_progress(completed: int, total: int) -> None:
                self._update(
                    message=f"正在复核候选组 {completed}/{total}",
                    progress=round(94.0 + 5.0 * completed / max(1, total), 2),
                )

            vlm_report = review_groups_with_vlm(
                groups,
                settings,
                self.feature_cache,
                self.vlm_runtime,
                cancel_check=self._cancel.is_set,
                progress_callback=vlm_progress,
            )
            if self._cancel.is_set():
                self._update(status="cancelled", phase="已取消", message="筛选任务已取消")
                return

            self._update(
                status="ranking",
                phase="覆盖校验",
                message="正在生成 3 星精选并补齐每个环节的已识别人物",
                progress=99.0,
            )
            failed_groups = [
                PhotoGroupInternal(
                    id=photo.group_id,
                    photos=[photo],
                    confidence=0.0,
                    reason="文件读取失败",
                )
                for photo in photos
                if photo.width == 0
            ]
            rating_groups = [*groups, *failed_groups]
            rating_report = assign_semantic_ratings(
                rating_groups,
                window_minutes=settings.coverage_window_minutes,
                feature_provider=self.rating_feature_provider,
            )
            coverage_report = self._coverage_payload(
                rating_groups,
                rating_report.public_dict(),
                settings.coverage_window_minutes,
            )

            for failed in (photo for photo in photos if photo.width == 0):
                failed.rank_in_group = 1
                failed.selection_reasons = ["文件无法解码，未参与相似度比较"]

            created_at = datetime.now().astimezone().isoformat(timespec="seconds")
            project_id = uuid.uuid4().hex[:16]
            public_groups = [group.public_dict(settings.keep_per_group) for group in rating_groups]
            results = {
                "schema_version": 2,
                "rating_migration_status": "native",
                "lightroom_ready": True,
                "project_id": project_id,
                "project_name": f"{root.name} · {datetime.now():%m月%d日 %H:%M}",
                "source_name": root.name,
                "created_at": created_at,
                "engine": {
                    "version": ENGINE_VERSION,
                    "grouping_preset": settings.grouping_preset,
                    "eye_evidence": eye_evidence_status(),
                    "face_ai": face_engine.status(),
                    "body_ai": body_engine.status(),
                    "pose_ai": pose_engine.status(),
                    "depth_ai": depth_engine.status(),
                    "scene_ai": scene_engine.status(),
                    "preference_ai": self._preference_status(preference_model, preference_applied),
                    "vlm_ai": vlm_report,
                    "coverage_guard": rating_report.public_dict(),
                    "feature_cache": {
                        "database": str(self.feature_cache.path),
                        "pipeline_signature": pipeline_signature,
                        "hits": cache_hits,
                        "misses": cache_misses,
                        "error": self.feature_cache.error or None,
                    },
                },
                "photos": [photo.public_dict() for photo in photos],
                "groups": public_groups,
                "coverage": coverage_report,
                "rating_policy": rating_report.public_dict(),
                "summary": {},
            }
            self._refresh_summary(results, elapsed=time.monotonic() - started_at, people=sum(len(cluster.photo_ids) >= 2 for cluster in clusters))
            self.projects.save(project_id, results, self._files)
            with self._lock:
                self._results = results
            self._update(
                status="completed",
                phase="已完成",
                message=(
                    f"筛选结果已保存，其中 {coverage_report['protected_photos']} 张为人物覆盖保底"
                    if coverage_report["protected_photos"]
                    else "筛选结果已保存在本机"
                ),
                processed=len(paths),
                total=len(paths),
                progress=100.0,
                current_file="",
                elapsed_seconds=round(time.monotonic() - started_at, 1),
                eta_seconds=0,
                project_id=project_id,
            )
        except Exception as exc:
            self.vlm_runtime.stop()
            self._update(status="failed", phase="任务失败", message="本地筛选未完成", error=str(exc), progress=0.0)

    @staticmethod
    def _preference_status(model: PreferenceModel | None, applied: bool) -> dict[str, Any]:
        if model is None:
            return {
                "available": False,
                "applied": False,
                "version": PREFERENCE_MODEL_VERSION,
                "reason": "尚未使用人工选片样本训练",
            }
        status = model.status()
        status["applied"] = applied
        if not applied:
            status["reason"] = "训练来源与当前素材目录不同，已自动使用通用排序"
        return status

    @staticmethod
    def _coverage_payload(
        groups: list[PhotoGroupInternal],
        rating_policy: dict[str, Any],
        window_minutes: int,
    ) -> dict[str, Any]:
        stage_photos: dict[str, list[PhotoObservation]] = {}
        for group in groups:
            for photo in group.photos:
                if photo.stage_id:
                    stage_photos.setdefault(photo.stage_id, []).append(photo)
        stages = []
        for stage_id, members in sorted(stage_photos.items()):
            times = sorted(photo.capture_time for photo in members if photo.capture_time)
            people = {person for photo in members for person in photo.significant_person_ids}
            stages.append(
                {
                    "id": stage_id,
                    "label": members[0].stage_label,
                    "photo_count": len(members),
                    "person_count": len(people),
                    "start_time": times[0].isoformat() if times else None,
                    "end_time": times[-1].isoformat() if times else None,
                }
            )
        all_photos = [photo for group in groups for photo in group.photos]
        protected_cells = sum(
            len(photo.coverage_keys)
            for photo in all_photos
            if photo.stars == 2 and photo.rating_origin == RatingOrigin.COVERAGE.value
        )
        required = int(rating_policy.get("required_coverage_keys", 0))
        unresolved = int(rating_policy.get("unresolved_coverage_keys", 0))
        return {
            "enabled": True,
            "stage_source": "folder" if any(stage["label"].startswith("环节 ") for stage in stages) else "time",
            "window_minutes": int(window_minutes),
            "stages": stages,
            "eligible_people": len(
                {person for photo in all_photos for person in photo.significant_person_ids}
            ),
            "required_cells": required,
            "already_covered_cells": max(0, required - protected_cells - unresolved),
            "protected_photos": sum(
                photo.stars == 2 and photo.rating_origin == RatingOrigin.COVERAGE.value
                for photo in all_photos
            ),
            "protected_cells": protected_cells,
            "unresolved_cells": unresolved,
        }

    @staticmethod
    def _refresh_summary(results: dict[str, Any], elapsed: float | None = None, people: int | None = None) -> None:
        photos = results.get("photos", [])
        existing = results.get("summary", {})
        coverage = results.get("coverage", {})
        rating_policy = results.get("rating_policy", {})
        protected = sum(bool(photo.get("coverage_protected")) for photo in photos)
        star_counts = {
            stars: sum(int(photo.get("stars", 0)) == stars for photo in photos)
            for stars in range(4)
        }
        primary_clusters: dict[str, int] = {}
        for photo in photos:
            if int(photo.get("stars", 0)) != 3:
                continue
            cluster_id = str(photo.get("strict_duplicate_cluster_id", ""))
            if cluster_id:
                primary_clusters[cluster_id] = primary_clusters.get(cluster_id, 0) + 1
        primary_duplicate_leaks = sum(max(0, count - 1) for count in primary_clusters.values())
        if isinstance(rating_policy, dict):
            rating_policy.update(
                {
                    "primary_count": star_counts[3],
                    "coverage_count": star_counts[2],
                    "valuable_count": star_counts[1],
                    "waste_count": star_counts[0],
                    "primary_duplicate_leaks": primary_duplicate_leaks,
                }
            )
        results["summary"] = {
            "total": len(photos),
            "selected": sum(int(photo.get("stars", 0)) >= 2 for photo in photos),
            "duplicates": sum(photo.get("category") == "duplicate" for photo in photos),
            "issues": sum(
                bool(
                    photo.get("category") not in {"selected", "duplicate"}
                    or (bool(photo.get("coverage_protected")) and bool(photo.get("issues")))
                )
                for photo in photos
            ),
            "groups": len(results.get("groups", [])),
            "people": int(people if people is not None else existing.get("people", 0)),
            "coverage_protected": protected,
            "coverage_stages": len(coverage.get("stages", [])),
            "coverage_required_cells": int(
                rating_policy.get("required_coverage_keys", coverage.get("required_cells", 0))
            ),
            "coverage_unresolved_cells": int(
                rating_policy.get("unresolved_coverage_keys", coverage.get("unresolved_cells", 0))
            ),
            "primary_duplicate_leaks": primary_duplicate_leaks,
            "primary": star_counts[3],
            "coverage": star_counts[2],
            "valuable": star_counts[1],
            "waste": star_counts[0],
            "stars_0": star_counts[0],
            "stars_1": star_counts[1],
            "stars_2": star_counts[2],
            "stars_3": star_counts[3],
            "elapsed_seconds": round(float(elapsed if elapsed is not None else existing.get("elapsed_seconds", 0.0)), 1),
        }
