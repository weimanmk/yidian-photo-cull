from __future__ import annotations

from types import SimpleNamespace

from photocull import runtime as runtime_module


def fake_ort(providers: list[str], *, preload_error: Exception | None = None) -> SimpleNamespace:
    def preload_dlls(*, directory: str) -> None:
        assert directory == ""
        if preload_error is not None:
            raise preload_error

    return SimpleNamespace(
        get_available_providers=lambda: providers,
        preload_dlls=preload_dlls,
    )


def test_cuda_is_preferred_over_directml(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_module,
        "ort",
        fake_ort(["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]),
    )

    runtime = runtime_module.InferenceRuntime(use_gpu=True)

    assert runtime.providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert runtime.primary_provider == "CUDAExecutionProvider"
    assert runtime.cuda_preload_error is None


def test_directml_remains_the_secondary_gpu_backend(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_module,
        "ort",
        fake_ort(["DmlExecutionProvider", "CPUExecutionProvider"]),
    )

    runtime = runtime_module.InferenceRuntime(use_gpu=True)

    assert runtime.providers == ["DmlExecutionProvider", "CPUExecutionProvider"]


def test_cpu_mode_does_not_preload_or_select_gpu(monkeypatch) -> None:
    calls: list[str] = []
    fake = fake_ort(["CUDAExecutionProvider", "CPUExecutionProvider"])
    fake.preload_dlls = lambda *, directory: calls.append(directory)
    monkeypatch.setattr(runtime_module, "ort", fake)

    runtime = runtime_module.InferenceRuntime(use_gpu=False)

    assert runtime.providers == ["CPUExecutionProvider"]
    assert calls == []


def test_cuda_preload_failure_is_diagnostic_not_fatal(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_module,
        "ort",
        fake_ort(["CUDAExecutionProvider", "CPUExecutionProvider"], preload_error=RuntimeError("missing DLL")),
    )

    runtime = runtime_module.InferenceRuntime(use_gpu=True)

    assert runtime.providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert runtime.cuda_preload_error == "missing DLL"


def test_actual_session_providers_override_configured_order(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_module,
        "ort",
        fake_ort(["CUDAExecutionProvider", "CPUExecutionProvider"]),
    )
    runtime = runtime_module.InferenceRuntime(use_gpu=True)
    cpu_session = SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])

    assert runtime.actual_providers(cpu_session) == ["CPUExecutionProvider"]
