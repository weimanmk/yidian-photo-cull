from pathlib import Path

from importlib.util import find_spec

from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs, collect_submodules, copy_metadata


backend_root = Path(SPECPATH)
datas = []
binaries = []
hiddenimports = collect_submodules("uvicorn")

datas.append((
    str(backend_root / "photocull" / "assets" / "rating_model_v1.json"),
    "photocull/assets",
))

for package_name in ("onnxruntime", "ai_edge_litert", "pillow_heif", "rawpy"):
    package_datas, package_binaries, package_hiddenimports = collect_all(package_name)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports

for package_name in (
    "nvidia.cublas",
    "nvidia.cuda_nvrtc",
    "nvidia.cuda_runtime",
    "nvidia.cudnn",
    "nvidia.cufft",
    "nvidia.curand",
    "nvidia.nvjitlink",
):
    try:
        if find_spec(package_name) is not None:
            binaries += collect_dynamic_libs(package_name)
    except (ImportError, ModuleNotFoundError):
        pass

for distribution_name in (
    "fastapi",
    "uvicorn",
    "pillow-heif",
    "rawpy",
    "onnxruntime-directml",
    "onnxruntime-gpu",
    "nvidia-cublas-cu12",
    "nvidia-cuda-nvrtc-cu12",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cudnn-cu12",
    "nvidia-cufft-cu12",
    "nvidia-curand-cu12",
    "nvidia-nvjitlink-cu12",
    "ai-edge-litert",
):
    try:
        datas += copy_metadata(distribution_name)
    except Exception:
        pass

analysis = Analysis(
    [str(backend_root / "photocull_backend.py")],
    pathex=[str(backend_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest", "httpx"],
    noarchive=False,
    optimize=1,
)

python_archive = PYZ(analysis.pure)

executable = EXE(
    python_archive,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="photocull-backend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
)

bundle = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="photocull-backend",
)
