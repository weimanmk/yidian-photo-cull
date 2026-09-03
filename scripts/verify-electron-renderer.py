from __future__ import annotations

import argparse
import asyncio
import base64
import json
from pathlib import Path
from urllib.request import urlopen

import websockets


async def cdp_call(socket, request_id: int, method: str, params: dict | None = None) -> dict:
    await socket.send(json.dumps({"id": request_id, "method": method, "params": params or {}}))
    while True:
        payload = json.loads(await socket.recv())
        if payload.get("id") != request_id:
            continue
        if "error" in payload:
            raise RuntimeError(f"CDP {method} failed: {payload['error']}")
        return payload["result"]


async def inspect_renderer(port: int, output_dir: Path) -> dict:
    with urlopen(f"http://127.0.0.1:{port}/json/list", timeout=5) as response:
        targets = json.load(response)
    pages = [target for target in targets if target.get("type") == "page"]
    if len(pages) != 1:
        raise RuntimeError(f"Expected one Electron renderer page, found {len(pages)}")

    expression = """(() => ({
      title: document.title,
      readyState: document.readyState,
      rootChildren: document.querySelector('#root')?.childElementCount ?? -1,
      bodyText: document.body.innerText.slice(0, 2000),
      scriptSources: [...document.scripts].map((element) => element.src),
      stylesheetSources: [...document.querySelectorAll('link[rel="stylesheet"]')].map((element) => element.href),
      loadedStylesheets: document.styleSheets.length,
      hasDesktopBridge: Boolean(window.desktop)
    }))()"""

    async with websockets.connect(pages[0]["webSocketDebuggerUrl"], max_size=16 * 1024 * 1024) as socket:
        evaluation = await cdp_call(
            socket,
            1,
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True},
        )
        if "exceptionDetails" in evaluation:
            raise RuntimeError(f"Renderer evaluation failed: {evaluation['exceptionDetails']}")
        screenshot = await cdp_call(
            socket,
            2,
            "Page.captureScreenshot",
            {"format": "png", "captureBeyondViewport": False},
        )

    renderer = evaluation["result"]["value"]
    result = {
        "target": {"title": pages[0].get("title"), "url": pages[0].get("url")},
        "renderer": renderer,
    }
    if renderer["readyState"] not in {"interactive", "complete"}:
        raise RuntimeError(f"Renderer document is not ready: {renderer['readyState']}")
    if renderer["rootChildren"] < 1 or not renderer["bodyText"].strip():
        raise RuntimeError(f"Renderer root is empty: {json.dumps(result, ensure_ascii=False)}")
    if renderer["loadedStylesheets"] < 1 or not renderer["hasDesktopBridge"]:
        raise RuntimeError(f"Renderer assets or preload bridge are unavailable: {json.dumps(result, ensure_ascii=False)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "renderer-state.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "renderer.png").write_bytes(base64.b64decode(screenshot["data"]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a packaged Electron renderer over its local CDP port.")
    parser.add_argument("--port", type=int, default=9223)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(inspect_renderer(args.port, args.output_dir.resolve()))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
