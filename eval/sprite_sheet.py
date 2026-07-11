import json
from pathlib import Path
from typing import List, Tuple

from PIL import Image


def pack_sprite_sheet(
    sprites: List[Tuple[str, Image.Image]],
    max_width: int = 1024,
    max_height: int = 1024,
    padding: int = 1,
) -> Tuple[Image.Image, dict]:
    if not sprites:
        return Image.new("RGBA", (0, 0)), {"frames": {}, "meta": {}}

    placements = []
    x, y = 0, 0
    row_h = 0

    for name, img in sprites:
        w, h = img.size
        if x > 0 and x + w + padding > max_width:
            x = 0
            y += row_h + padding
            row_h = 0
        if y + h > max_height:
            break
        placements.append((name, img, (x, y)))
        x += w + padding
        row_h = max(row_h, h)

    sheet_w = max_width
    sheet_h = y + row_h

    sheet = Image.new("RGBA", (sheet_w, sheet_h), (0, 0, 0, 0))
    frames = {}

    for name, img, (fx, fy) in placements:
        sheet.paste(img, (fx, fy))
        frames[name] = {
            "frame": {"x": fx, "y": fy, "w": img.width, "h": img.height},
            "spriteSourceSize": {"x": 0, "y": 0, "w": img.width, "h": img.height},
            "sourceSize": {"w": img.width, "h": img.height},
        }

    meta = {
        "app": "sprite-generator",
        "version": "1.0",
        "image": "sprite_sheet.png",
        "size": {"w": sheet_w, "h": sheet_h},
        "scale": "1",
    }

    missing_count = len(sprites) - len(placements)
    if missing_count > 0:
        meta["truncated"] = missing_count

    return sheet, {"frames": frames, "meta": meta}


def save_sprite_sheet(
    sprites: List[Tuple[str, Image.Image]],
    output_stem: str = "sprite_sheet",
    output_dir: str = ".",
    max_width: int = 1024,
    max_height: int = 1024,
    padding: int = 1,
) -> Tuple[Path, Path]:
    sheet, data = pack_sprite_sheet(sprites, max_width, max_height, padding)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    png_path = out_dir / f"{output_stem}.png"
    json_path = out_dir / f"{output_stem}.json"

    sheet.save(png_path)
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    return png_path, json_path
