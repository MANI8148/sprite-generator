from PIL import Image
import json
import math
import os
from typing import List, Tuple, Optional


def sprite_sheet(
    images: List[Image.Image],
    cols: Optional[int] = None,
    rows: Optional[int] = None,
    padding: int = 2,
    sheet_size: Optional[Tuple[int, int]] = None,
) -> Tuple[Image.Image, dict]:
    n = len(images)
    if n == 0:
        raise ValueError("No images to pack")
    if cols is None and rows is None:
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
    elif cols is None:
        cols = math.ceil(n / rows)
    elif rows is None:
        rows = math.ceil(n / cols)
    cell_w = max(img.size[0] for img in images)
    cell_h = max(img.size[1] for img in images)
    sheet_w = cols * (cell_w + padding) + padding
    sheet_h = rows * (cell_h + padding) + padding
    sheet = Image.new("RGBA", (sheet_w, sheet_h), (0, 0, 0, 0))
    frames = []
    for idx, img in enumerate(images):
        col = idx % cols
        row = idx // cols
        x = padding + col * (cell_w + padding)
        y = padding + row * (cell_h + padding)
        centered = Image.new("RGBA", (cell_w, cell_h), (0, 0, 0, 0))
        ox = (cell_w - img.size[0]) // 2
        oy = (cell_h - img.size[1]) // 2
        centered.paste(img, (ox, oy), img)
        sheet.paste(centered, (x, y), centered)
        frames.append({
            "index": idx,
            "x": x, "y": y,
            "w": cell_w, "h": cell_h,
            "source_w": img.size[0],
            "source_h": img.size[1],
            "offset_x": ox,
            "offset_y": oy,
        })
    metadata = {
        "type": "sprite_sheet",
        "size": {"w": sheet_w, "h": sheet_h},
        "cols": cols,
        "rows": rows,
        "cell_size": {"w": cell_w, "h": cell_h},
        "padding": padding,
        "frames": frames,
    }
    return sheet, metadata


def tileset(
    images: List[Image.Image],
    tile_size: Optional[Tuple[int, int]] = None,
    padding: int = 1,
) -> Tuple[Image.Image, dict]:
    if tile_size is None:
        tile_size = images[0].size if images else (16, 16)
    return sprite_sheet(images, padding=padding)


def animation_strip(
    frames: List[Image.Image],
    direction: str = "horizontal",
    padding: int = 2,
) -> Tuple[Image.Image, dict]:
    if not frames:
        raise ValueError("No frames to pack")
    frame_w = max(f.size[0] for f in frames)
    frame_h = max(f.size[1] for f in frames)
    if direction == "horizontal":
        total_w = len(frames) * (frame_w + padding) + padding
        total_h = frame_h + padding * 2
    else:
        total_w = frame_w + padding * 2
        total_h = len(frames) * (frame_h + padding) + padding
    strip = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 0))
    for idx, img in enumerate(frames):
        centered = Image.new("RGBA", (frame_w, frame_h), (0, 0, 0, 0))
        ox = (frame_w - img.size[0]) // 2
        oy = (frame_h - img.size[1]) // 2
        centered.paste(img, (ox, oy), img)
        if direction == "horizontal":
            x = padding + idx * (frame_w + padding)
            y = padding
        else:
            x = padding
            y = padding + idx * (frame_h + padding)
        strip.paste(centered, (x, y), centered)
    metadata = {
        "type": "animation_strip",
        "direction": direction,
        "frame_count": len(frames),
        "frame_size": {"w": frame_w, "h": frame_h},
        "total_size": {"w": total_w, "h": total_h},
        "padding": padding,
    }
    return strip, metadata


def individual_pngs(
    images: List[Image.Image],
    names: List[str],
    output_dir: str,
) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for img, name in zip(images, names):
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        path = os.path.join(output_dir, f"{safe}.png")
        img.save(path)
        paths.append(path)
    return paths
