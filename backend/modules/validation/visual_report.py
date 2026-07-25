from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Optional, Tuple
import io
import textwrap

from .metrics import assess_all

_TIER_COLORS = {
    "clean": (76, 175, 80),
    "acceptable": (255, 193, 7),
    "noisy": (255, 87, 34),
    "blurry": (255, 87, 34),
    "broken_outline": (244, 67, 54),
    "extreme_aspect": (244, 67, 54),
    "empty": (158, 158, 158),
}

_TIER_ORDER = ["clean", "acceptable", "noisy", "blurry", "broken_outline", "extreme_aspect", "empty"]

_CELL_SIZE = 160
_LABEL_HEIGHT = 60
_SPRITE_AREA = _CELL_SIZE - _LABEL_HEIGHT


def _quality_color(tier: str) -> Tuple[int, int, int]:
    return _TIER_COLORS.get(tier, (0, 0, 0))


def _format_metrics(metrics: Dict) -> List[str]:
    lines = []
    if metrics.get("quality_tier"):
        lines.append(f"Tier: {metrics['quality_tier']}")
    if "palette_size" in metrics:
        lines.append(f"Palette: {metrics['palette_size']} colors")
    if "sharpness" in metrics:
        lines.append(f"Sharpness: {metrics['sharpness']}")
    if "transparency_ratio" in metrics:
        lines.append(f"Alpha: {metrics['transparency_ratio']:.0%}")
    if "outline_continuity" in metrics:
        lines.append(f"Outline: {metrics['outline_continuity']:.0%}")
    if "aspect_ratio" in metrics:
        lines.append(f"Aspect: {metrics['aspect_ratio']}")
    return lines


def _draw_label(draw: ImageDraw, x: int, y: int, w: int, h: int, metrics: Dict):
    tier = metrics.get("quality_tier", "unknown")
    color = _quality_color(tier)
    draw.rectangle([x, y, x + w, y + h], fill=(250, 250, 250))
    draw.rectangle([x, y, x + w, y + h], outline=color, width=2)

    lines = _format_metrics(metrics)
    font = None
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except (IOError, OSError):
        font = ImageFont.load_default()

    line_y = y + 4
    for line in lines:
        draw.text((x + 4, line_y), line, fill=(30, 30, 30), font=font)
        line_y += 12


def _fit_sprite(sprite: Image.Image, size: int) -> Image.Image:
    s = sprite.copy()
    s.thumbnail((size, size), Image.NEAREST)
    padded = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ox = (size - s.width) // 2
    oy = (size - s.height) // 2
    padded.paste(s, (ox, oy), s)
    return padded


def render_validation_grid(
    sprites: List[Image.Image],
    metrics_list: Optional[List[Dict]] = None,
    title: str = "Sprite Validation Report",
    cells_per_row: int = 4,
) -> Image.Image:
    if metrics_list is None:
        metrics_list = [assess_all(img, batch=sprites) for img in sprites]

    n = len(sprites)
    if n == 0:
        raise ValueError("At least one sprite is required")

    cols = min(cells_per_row, n)
    rows = (n + cols - 1) // cols

    cell_w = _CELL_SIZE
    cell_h = _CELL_SIZE

    header_h = 30 if title else 0
    canvas_w = cols * cell_w
    canvas_h = header_h + rows * cell_h

    canvas = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    if title:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except (IOError, OSError):
            font = ImageFont.load_default()
        draw.text((8, 4), title, fill=(0, 0, 0), font=font)

    for i, sprite in enumerate(sprites):
        col = i % cols
        row = i // cols
        x = col * cell_w
        y = header_h + row * cell_h

        sprite_region = _fit_sprite(sprite, _SPRITE_AREA)
        sprite_x = x + (cell_w - _SPRITE_AREA) // 2
        sprite_y = y
        canvas.paste(sprite_region, (sprite_x, sprite_y), sprite_region)

        metrics = metrics_list[i] if i < len(metrics_list) else {}
        label_y = y + _SPRITE_AREA
        _draw_label(draw, x, label_y, cell_w, _LABEL_HEIGHT, metrics)

    return canvas


def generate_validation_report(
    sprites: List[Image.Image],
    metrics_list: Optional[List[Dict]] = None,
    title: str = "Sprite Validation Report",
    output_path: Optional[str] = None,
) -> bytes:
    grid = render_validation_grid(sprites, metrics_list, title=title)

    buf = io.BytesIO()
    grid.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    if output_path:
        grid.save(output_path)

    return png_bytes


def summarize_metrics(metrics_list: List[Dict]) -> Dict:
    if not metrics_list:
        return {}

    tiers = [m.get("quality_tier", "unknown") for m in metrics_list]
    tier_counts = {t: tiers.count(t) for t in _TIER_ORDER if t in tiers}
    unknown = sum(1 for t in tiers if t not in _TIER_ORDER)
    if unknown:
        tier_counts["unknown"] = unknown

    palette_sizes = [m.get("palette_size", 0) for m in metrics_list]
    sharpness = [m.get("sharpness", 0) for m in metrics_list]
    outlines = [m.get("outline_continuity", 0) for m in metrics_list]

    summary = {
        "total": len(metrics_list),
        "tier_counts": tier_counts,
        "clean_percent": round(tier_counts.get("clean", 0) / len(metrics_list) * 100, 1),
        "avg_palette_size": round(sum(palette_sizes) / len(palette_sizes), 1),
        "avg_sharpness": round(sum(sharpness) / len(sharpness), 1),
        "avg_outline_continuity": round(sum(outlines) / len(outlines), 3),
    }
    return summary
