from __future__ import annotations

import os
from pathlib import Path

try:
    import onnxruntime as ort
except ImportError:
    ort = None


_DLL_DIRECTORY_HANDLES: list[object] = []
_DLL_DIRECTORY_PATHS: set[Path] = set()


class InferenceRuntime:
    """统一管理 ONNX Execution Provider，业务模型不绑定具体 GPU 后端。"""

    def __init__(self, use_gpu: bool = True) -> None:
        self.use_gpu = use_gpu
        self.cuda_preload_error: str | None = None
        if use_gpu:
            self._preload_cuda_dlls()

    def _preload_cuda_dlls(self) -> None:
        if ort is None or not hasattr(ort, "preload_dlls"):
            return
        try:
            self._register_nvidia_dll_directories()
            # 空字符串要求 ORT 从随应用打包的 NVIDIA site-packages 中加载 DLL。
            ort.preload_dlls(directory="")
        except Exception as error:  # pragma: no cover - 取决于宿主 CUDA 环境
            self.cuda_preload_error = str(error)

    @staticmethod
    def _register_nvidia_dll_directories() -> None:
        if ort is None or not hasattr(os, "add_dll_directory"):
            return
        ort_file = getattr(ort, "__file__", None)
        if not ort_file:
            return
        site_packages = Path(ort_file).resolve().parent.parent
        discovered: list[Path] = []
        for component in ("cublas", "cuda_nvrtc", "cuda_runtime", "cudnn", "cufft", "curand", "nvjitlink"):
            dll_directory = site_packages / "nvidia" / component / "bin"
            if not dll_directory.is_dir():
                continue
            discovered.append(dll_directory)
            if dll_directory not in _DLL_DIRECTORY_PATHS:
                _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(dll_directory)))
                _DLL_DIRECTORY_PATHS.add(dll_directory)

        # cuDNN 9 会在首次卷积时再次按文件名加载引擎子库；该内部加载不总是
        # 遵循 AddDllDirectory，因此还需把目录加入当前后端进程的 PATH。
        current_entries = [entry for entry in os.environ.get("PATH", "").split(os.pathsep) if entry]
        known_entries = {entry.casefold() for entry in current_entries}
        missing_entries = [str(path) for path in discovered if str(path).casefold() not in known_entries]
        if missing_entries:
            os.environ["PATH"] = os.pathsep.join([*missing_entries, *current_entries])

    @property
    def available(self) -> bool:
        return ort is not None

    @property
    def providers(self) -> list[str]:
        if ort is None:
            return []
        available = ort.get_available_providers()
        if self.use_gpu and "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if self.use_gpu and "DmlExecutionProvider" in available:
            return ["DmlExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    @property
    def primary_provider(self) -> str:
        return self.providers[0] if self.providers else "unavailable"

    def actual_providers(self, *sessions: object | None) -> list[str]:
        actual: list[str] = []
        for session in sessions:
            if session is None or not hasattr(session, "get_providers"):
                continue
            for provider in session.get_providers():
                if provider not in actual:
                    actual.append(provider)
        return actual

    def create_session(self, path: Path):
        if ort is None:
            raise RuntimeError("ONNX Runtime 未安装")
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        if "DmlExecutionProvider" in self.providers:
            options.enable_mem_pattern = False
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        return ort.InferenceSession(str(path), sess_options=options, providers=self.providers)
