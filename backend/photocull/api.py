from __future__ import annotations

import os
import signal
from dataclasses import asdict
from pathlib import Path
from threading import Timer

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from . import __version__
from .body_engine import BodyEngine
from .config import HOST, PORT, PREVIEW_DIR, THUMBNAIL_DIR, settings_store
from .depth_engine import DepthEngine
from .exporter import (
    ExportOperationStore,
    ExportPlanChangedError,
    ExportVerificationError,
    build_export_plan,
)
from .face_engine import FaceEngine
from .imaging import ensure_cached_images, load_image
from .lightroom_bridge import LightroomBridge
from .lightroom_service import (
    LightroomOperationNotFound,
    LightroomPlanError,
    LightroomService,
    LightroomStateError,
)
from .project_store import ProjectStore
from .preference import PREFERENCE_MODEL_VERSION, PreferenceModel
from .pose_engine import PoseEngine
from .scanner import ScanConflictError, ScannerService
from .schemas import (
    ExportExecuteRequest,
    ExportPreflightRequest,
    ExportRequest,
    LightroomPreflightRequest,
    PhotoLabelRequest,
    PhotoRatingRequest,
    ScanRequest,
    SettingsPatch,
)
from .scene_engine import SceneEmbeddingEngine
from .vlm import LlamaServerManager


projects = ProjectStore()
export_operations = ExportOperationStore(projects.root.parent / "export-operations")
lightroom_service: LightroomService | None = None
vlm_runtime = LlamaServerManager()
scanner = ScannerService(projects, vlm_runtime=vlm_runtime)
health_pose_engine = PoseEngine()
health_depth_engine = DepthEngine(use_gpu=settings_store.get().use_gpu)
app = FastAPI(title="PhotoCull Local Engine", version=__version__, docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173", "null"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH"],
    allow_headers=["Content-Type", "X-PhotoCull-Token"],
)


def _lightroom_service() -> LightroomService:
    global lightroom_service
    if lightroom_service is None:
        lightroom_service = LightroomService(projects, LightroomBridge())
    return lightroom_service


@app.middleware("http")
async def local_token_guard(request: Request, call_next):
    expected = os.getenv("PHOTOCULL_API_TOKEN", "")
    if expected and request.url.path.startswith("/api/"):
        supplied = request.headers.get("X-PhotoCull-Token") or request.query_params.get("token")
        if supplied != expected:
            return JSONResponse(status_code=403, content={"detail": "本地 API 令牌无效"})
    return await call_next(request)


@app.get("/api/health")
def health():
    settings = settings_store.get()
    preference_model = PreferenceModel.load()
    return {
        "status": "ok",
        "version": __version__,
        "offline": True,
        "face_ai": FaceEngine(use_gpu=settings.use_gpu).status(),
        "body_ai": BodyEngine(use_gpu=settings.use_gpu).status(),
        "pose_ai": health_pose_engine.status(),
        "depth_ai": health_depth_engine.status(),
        "scene_ai": SceneEmbeddingEngine(use_gpu=settings.use_gpu).status(),
        "preference_ai": (
            preference_model.status()
            if preference_model is not None
            else {"available": False, "version": PREFERENCE_MODEL_VERSION, "ranking_strength": 0.0}
        ),
        "vlm_ai": vlm_runtime.status(settings),
    }


@app.get("/api/vlm/status")
def vlm_status():
    return vlm_runtime.status(settings_store.get())


@app.get("/api/settings")
def get_settings():
    return asdict(settings_store.get())


@app.patch("/api/settings")
def update_settings(changes: SettingsPatch):
    return asdict(settings_store.update(changes.model_dump(exclude_none=True)))


@app.post("/api/scan/start")
def start_scan(payload: ScanRequest):
    try:
        return scanner.start(
            payload.folder,
            payload.grouping_preset,
            payload.keep_per_group,
            payload.recursive,
            coverage_enabled=payload.coverage_enabled,
            coverage_window_minutes=payload.coverage_window_minutes,
        )
    except ScanConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/scan/status")
def scan_status():
    return scanner.status()


@app.post("/api/scan/cancel")
def cancel_scan():
    return scanner.cancel()


@app.get("/api/scan/results")
def scan_results():
    results = scanner.results()
    if results is None:
        raise HTTPException(status_code=404, detail="还没有可用的筛选结果")
    return results


@app.get("/api/projects")
def list_projects():
    return projects.list()


@app.get("/api/projects/{project_id}")
def load_project(project_id: str):
    try:
        return scanner.load_project(project_id)
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=404, detail="项目不存在或已损坏") from exc


@app.patch("/api/photos/{photo_id}/label")
def label_photo(photo_id: str, payload: PhotoLabelRequest):
    if not scanner.label_photo(photo_id, payload.category, payload.stars):
        raise HTTPException(status_code=404, detail="照片不存在")
    return {"ok": True}


@app.patch("/api/photos/{photo_id}/rating")
def rate_photo(photo_id: str, payload: PhotoRatingRequest):
    if not scanner.rate_photo(photo_id, payload.stars, locked=payload.locked):
        raise HTTPException(status_code=404, detail="照片不存在")
    return {"ok": True}


def _cache_file(identifier: str, preview: bool) -> Path:
    directory = PREVIEW_DIR if preview else THUMBNAIL_DIR
    target = directory / f"{identifier}.jpg"
    if target.is_file():
        return target
    source = scanner.resolve_file(identifier)
    if source is None:
        raise HTTPException(status_code=404, detail="照片不存在")
    try:
        image = load_image(source)
        ensure_cached_images(image, identifier, settings_store.get().jpeg_preview_quality)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"无法生成预览：{exc}") from exc
    if not target.is_file():
        raise HTTPException(status_code=404, detail="预览不存在")
    return target


@app.get("/api/thumbnails/{photo_id}")
def thumbnail(photo_id: str):
    return FileResponse(_cache_file(photo_id, preview=False), media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})


@app.get("/api/images/{photo_id}")
def preview(photo_id: str):
    return FileResponse(_cache_file(photo_id, preview=True), media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})


@app.get("/api/files/{photo_id}")
def file_location(photo_id: str):
    path = scanner.resolve_file(photo_id)
    if path is None:
        raise HTTPException(status_code=404, detail="照片不存在")
    return {"path": str(path)}


@app.get("/api/lightroom/status")
def lightroom_status():
    return _lightroom_service().status()


@app.post("/api/lightroom/preflights")
def create_lightroom_preflight(payload: LightroomPreflightRequest):
    try:
        return _lightroom_service().create_preflight(payload.project_id).public_dict()
    except LightroomPlanError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LightroomStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/lightroom/operations/{operation_id}")
def get_lightroom_operation(operation_id: str):
    try:
        return _lightroom_service().refresh(operation_id).public_dict()
    except LightroomOperationNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except LightroomStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/lightroom/operations/{operation_id}/execute")
def execute_lightroom_operation(operation_id: str):
    try:
        return _lightroom_service().confirm_execute(operation_id).public_dict()
    except LightroomOperationNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except LightroomStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/lightroom/operations/{operation_id}/retry")
def retry_lightroom_operation(operation_id: str):
    try:
        return _lightroom_service().retry_pending_rating(operation_id).public_dict()
    except LightroomOperationNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except LightroomStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/lightroom/operations/{operation_id}/rollback")
def rollback_lightroom_operation(operation_id: str):
    try:
        return _lightroom_service().request_rollback(operation_id).public_dict()
    except LightroomOperationNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except LightroomStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def _load_export_project(project_id: str) -> tuple[dict, dict[str, str]]:
    results = scanner.results()
    if results is None or results.get("project_id") != project_id:
        try:
            results = scanner.load_project(project_id)
        except (OSError, ValueError, KeyError) as exc:
            raise HTTPException(status_code=404, detail="待导出的项目不存在") from exc
    return results, scanner.files()


@app.post("/api/exports/preflights")
def preflight_export(payload: ExportPreflightRequest):
    results, files = _load_export_project(payload.project_id)
    try:
        plan = build_export_plan(
            results,
            files,
            Path(payload.destination),
            payload.minimum_stars,
        )
        export_operations.save_plan(plan)
        return plan.to_dict()
    except (OSError, ValueError, ExportPlanChangedError) as exc:
        raise HTTPException(status_code=400, detail=f"导出预检失败：{exc}") from exc


@app.post("/api/exports/{operation_id}/execute")
def execute_export(operation_id: str, payload: ExportExecuteRequest):
    try:
        return export_operations.execute(operation_id, payload.plan_hash).to_dict()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ExportPlanChangedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ExportVerificationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"导出失败：{exc}") from exc


@app.post("/api/export")
def legacy_export(_: ExportRequest):
    raise HTTPException(status_code=410, detail="旧版导出已停用，请先预检并确认安全导出计划")


@app.post("/api/shutdown")
def shutdown():
    scanner.cancel()
    vlm_runtime.stop()
    Timer(0.2, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
    return {"ok": True}


def run() -> None:
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT, log_level="info", access_log=False)
