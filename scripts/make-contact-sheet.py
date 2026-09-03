"""从正在运行的本地引擎生成回归集联系表，便于人工检查组边界。"""

from __future__ import annotations

import argparse
import io
import json
import math
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


GROUP_COLORS = (
    "#8b5cf6", "#22d3ee", "#34d399", "#f59e0b", "#fb7185",
    "#60a5fa", "#a3e635", "#f472b6", "#f97316", "#2dd4bf",
)


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def fetch_image(url: str) -> Image.Image:
    with urllib.request.urlopen(url, timeout=30) as response:
        return Image.open(io.BytesIO(response.read())).convert("RGB")


def render_sheet(
    photos: list[dict],
    api_base: str,
    destination: Path,
    columns: int,
    cell_width: int,
    image_height: int,
) -> None:
    label_height = 42
    cell_height = image_height + label_height
    rows = max(1, math.ceil(len(photos) / columns))
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "#11131a")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=14)

    group_order = {group_id: index for index, group_id in enumerate(dict.fromkeys(p["group_id"] for p in photos))}
    for index, photo in enumerate(photos):
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        source = fetch_image(f"{api_base}{photo['thumbnail_url']}")
        fitted = ImageOps.fit(source, (cell_width - 10, image_height - 10), Image.Resampling.LANCZOS)
        color = GROUP_COLORS[group_order[photo["group_id"]] % len(GROUP_COLORS)]
        sheet.paste(fitted, (x + 5, y + 5))
        draw.rectangle((x + 3, y + 3, x + cell_width - 3, y + image_height - 3), outline=color, width=3)
        marker = "BEST" if photo["is_best_pick"] else f"R{photo['rank_in_group']}"
        draw.text((x + 7, y + image_height + 4), f"{photo['filename']}  {marker}", fill="#f8fafc", font=font)
        draw.text((x + 7, y + image_height + 22), photo["group_id"], fill=color, font=font)

    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, "JPEG", quality=91, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 PhotoCull 回归集联系表")
    parser.add_argument("--api-base", default="http://127.0.0.1:8767")
    parser.add_argument("--output-dir", type=Path, default=Path("output/playwright/real-set"))
    args = parser.parse_args()

    result = fetch_json(f"{args.api_base}/api/scan/results")
    photos = sorted(result["photos"], key=lambda item: item["filename"].casefold())
    selected = [photo for photo in photos if photo["is_best_pick"]]
    render_sheet(photos, args.api_base, args.output_dir / "all-groups.jpg", 8, 240, 160)
    render_sheet(selected, args.api_base, args.output_dir / "selected-only.jpg", 5, 300, 200)
    print(json.dumps({
        "project_id": result["project_id"],
        "total": len(photos),
        "groups": len(result["groups"]),
        "selected": len(selected),
        "output_dir": str(args.output_dir.resolve()),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
