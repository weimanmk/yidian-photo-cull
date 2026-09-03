from __future__ import annotations

import ctypes
import os
from pathlib import Path

import pytest


LIGHTROOM_CLASSIC_DIR = Path(
    os.environ.get(
        "LIGHTROOM_CLASSIC_DIR",
        r"C:\Program Files\Adobe\Adobe Lightroom Classic",
    )
)


def run_lightroom_lua(source: str) -> None:
    """使用 Lightroom Classic 自带的 Lua 5.1 运行测试脚本。"""

    runtime_path = LIGHTROOM_CLASSIC_DIR / "AgKernel.dll"
    if not runtime_path.is_file():
        pytest.skip("本机未安装 Lightroom Classic Lua 运行时")

    dll_directory = os.add_dll_directory(str(LIGHTROOM_CLASSIC_DIR))
    runtime = ctypes.CDLL(str(runtime_path))
    runtime.luaL_newstate.restype = ctypes.c_void_p
    runtime.luaL_openlibs.argtypes = [ctypes.c_void_p]
    runtime.luaL_loadstring.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    runtime.luaL_loadstring.restype = ctypes.c_int
    runtime.lua_pcall.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    runtime.lua_pcall.restype = ctypes.c_int
    runtime.lua_tolstring.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    runtime.lua_tolstring.restype = ctypes.c_char_p
    runtime.lua_close.argtypes = [ctypes.c_void_p]

    state = runtime.luaL_newstate()
    if not state:
        dll_directory.close()
        raise RuntimeError("无法创建 Lightroom Lua 状态")

    try:
        runtime.luaL_openlibs(state)
        encoded = source.encode("utf-8")
        load_status = runtime.luaL_loadstring(state, encoded)
        if load_status != 0:
            raise RuntimeError(_lua_error(runtime, state, "Lua 脚本加载失败"))
        run_status = runtime.lua_pcall(state, 0, 0, 0)
        if run_status != 0:
            raise RuntimeError(_lua_error(runtime, state, "Lua 脚本执行失败"))
    finally:
        runtime.lua_close(state)
        dll_directory.close()


def _lua_error(runtime: ctypes.CDLL, state: int, fallback: str) -> str:
    size = ctypes.c_size_t()
    raw = runtime.lua_tolstring(state, -1, ctypes.byref(size))
    if not raw:
        return fallback
    return ctypes.string_at(raw, size.value).decode("utf-8", errors="replace")
