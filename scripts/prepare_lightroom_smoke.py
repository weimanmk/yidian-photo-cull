from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE_ROOT = PROJECT_ROOT / "output" / "lightroom-smoke"
SOURCE_ROOT = SMOKE_ROOT / "source"
SAMPLES = (
    ("01-new-primary.jpg", (37, 99, 235), 3),
    ("02-update-coverage.jpg", (15, 118, 110), 2),
    ("03-unchanged-valuable.jpg", (202, 138, 4), 1),
    ("04-protected-five-star.jpg", (190, 24, 93), 3),
    ("05-new-waste.jpg", (71, 85, 105), 0),
)


def main() -> None:
    if SMOKE_ROOT.exists():
        raise RuntimeError(f"烟测目录已存在，拒绝覆盖：{SMOKE_ROOT}")
    SOURCE_ROOT.mkdir(parents=True)
    records: list[dict[str, object]] = []
    for index, (filename, color, target_rating) in enumerate(SAMPLES, start=1):
        target = SOURCE_ROOT / filename
        image = Image.new("RGB", (1600, 1067), color)
        draw = ImageDraw.Draw(image)
        draw.rectangle((80, 80, 1520, 987), outline="white", width=16)
        draw.text((130, 130), f"YIDIAN LIGHTROOM SMOKE {index}", fill="white", stroke_width=2, stroke_fill="black")
        draw.text((130, 190), f"TARGET RATING {target_rating}", fill="white", stroke_width=2, stroke_fill="black")
        temporary = target.with_suffix(".jpg.tmp")
        with temporary.open("xb") as stream:
            image.save(stream, format="JPEG", quality=94, subsampling=0)
            stream.flush()
        temporary.replace(target)
        stat = target.stat()
        records.append(
            {
                "id": f"smoke-{index}",
                "filename": filename,
                "target_rating": target_rating,
                "file_size": stat.st_size,
                "modified_ns": stat.st_mtime_ns,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        )
    manifest = SMOKE_ROOT / "samples.json"
    with manifest.open("x", encoding="utf-8") as stream:
        json.dump({"schema_version": 1, "samples": records}, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"smoke_root": str(SMOKE_ROOT), "sample_count": len(records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
