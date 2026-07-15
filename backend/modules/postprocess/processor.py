from PIL import Image, ImageFilter, ImageOps
import numpy as np
from io import BytesIO
from typing import Optional, Tuple

from .palettes import get_palette


def to_rgba(image: Image.Image) -> Image.Image:
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    return image


def remove_background(
    image: Image.Image,
    model: str = "u2net",
    alpha_threshold: int = 128,
) -> Image.Image:
    try:
        from rembg import remove as rembg_remove

        rgba = to_rgba(image)
        result = rembg_remove(rgba, model_name=model)
        arr = np.array(result)
        arr[..., 3] = (arr[..., 3] > alpha_threshold).astype(np.uint8) * 255
        return Image.fromarray(arr)
    except ImportError:
        rgba = to_rgba(image)
        arr = np.array(rgba)
        bg_color = arr[0, 0, :3].copy()
        mask = np.all(arr[:, :, :3] == bg_color, axis=2)
        arr[..., 3] = np.where(mask, 0, 255)
        return Image.fromarray(arr)


def reduce_palette(
    image: Image.Image, max_colors: int = 32, dither: bool = False
) -> Image.Image:
    rgba = to_rgba(image)
    if max_colors >= 256:
        return rgba
    # Use PIL's built-in quantize for pixel art (fast, good results)
    rgb = rgba.convert("RGB")
    quantized = rgb.quantize(colors=min(max_colors, 256), method=Image.Quantize.MEDIANCUT)
    quant_rgba = quantized.convert("RGBA")
    arr_in = np.array(rgba)
    arr_out = np.array(quant_rgba)
    arr_out[:, :, 3] = arr_in[:, :, 3]
    return Image.fromarray(arr_out)


def pixel_cleanup(
    image: Image.Image, min_region_size: int = 3
) -> Image.Image:
    rgba = to_rgba(image)
    arr = np.array(rgba)
    alpha = arr[:, :, 3]
    from scipy.ndimage import label as ndi_label

    labeled, num_features = ndi_label(alpha > 0)
    for i in range(1, num_features + 1):
        if (labeled == i).sum() < min_region_size:
            alpha[labeled == i] = 0
    arr[:, :, 3] = alpha
    return Image.fromarray(arr)


def auto_center(
    image: Image.Image,
    canvas_size: Optional[Tuple[int, int]] = None,
    padding: int = 4,
) -> Image.Image:
    rgba = to_rgba(image)
    arr = np.array(rgba)
    alpha = arr[:, :, 3]
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)
    if not rows.any() or not cols.any():
        return rgba
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    cropped = rgba.crop((x_min, y_min, x_max + 1, y_max + 1))
    if canvas_size is None:
        max_side = max(cropped.size) + padding * 2
        canvas_size = (max_side, max_side)
    canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    x_offset = (canvas_size[0] - cropped.size[0]) // 2
    y_offset = (canvas_size[1] - cropped.size[1]) // 2
    canvas.paste(cropped, (x_offset, y_offset), cropped)
    return canvas


def auto_pad(
    image: Image.Image,
    padding: int = 8,
    target_size: Optional[Tuple[int, int]] = None,
) -> Image.Image:
    rgba = to_rgba(image)
    if target_size:
        canvas = Image.new("RGBA", target_size, (0, 0, 0, 0))
        x_offset = (target_size[0] - rgba.size[0]) // 2
        y_offset = (target_size[1] - rgba.size[1]) // 2
        canvas.paste(rgba, (x_offset, y_offset), rgba)
        return canvas
    new_size = (rgba.size[0] + padding * 2, rgba.size[1] + padding * 2)
    canvas = Image.new("RGBA", new_size, (0, 0, 0, 0))
    canvas.paste(rgba, (padding, padding), rgba)
    return canvas


def normalize(
    image: Image.Image,
    target_size: Tuple[int, int] = (512, 512),
    padding: int = 8,
) -> Image.Image:
    centered = auto_center(image, padding=padding)
    max_dim = max(centered.size)
    if max_dim > max(target_size):
        ratio = max(target_size) / max_dim
        new_w = max(1, int(centered.size[0] * ratio))
        new_h = max(1, int(centered.size[1] * ratio))
        centered = centered.resize((new_w, new_h), Image.NEAREST)
    return auto_pad(centered, target_size=target_size)


def upscale(
    image: Image.Image, factor: int = 4, method: str = "nearest"
) -> Image.Image:
    if factor <= 1:
        return image
    resample = Image.NEAREST if method == "nearest" else Image.LANCZOS
    w, h = image.size
    return image.resize((w * factor, h * factor), resample)


def outline_cleanup(
    image: Image.Image,
    outline_color: Optional[Tuple[int, int, int, int]] = None,
    threshold: int = 30,
) -> Image.Image:
    rgba = to_rgba(image)
    arr = np.array(rgba)
    alpha = arr[:, :, 3]
    from scipy.ndimage import binary_dilation, binary_erosion

    mask = alpha > threshold
    eroded = binary_erosion(mask, iterations=1)
    outline = mask & ~eroded
    if outline_color is None:
        outline_color = (0, 0, 0, 255)
    arr[outline] = outline_color
    interior = eroded & (alpha > 0)
    interior_alpha = alpha.copy()
    interior_alpha[~interior] = 0
    arr[:, :, 3] = interior_alpha
    return Image.fromarray(arr)


def palette_lock(
    image: Image.Image,
    palette_name: str = "retro_16",
) -> Image.Image:
    rgba = to_rgba(image)
    palette = get_palette(palette_name)
    palette_arr = np.array(palette, dtype=np.int32)
    arr = np.array(rgba)
    alpha = arr[:, :, 3]
    opaque = alpha > 128
    if not opaque.any():
        return rgba
    pixels = arr[opaque, :3].astype(np.int32)
    diff = pixels[:, np.newaxis, :] - palette_arr[np.newaxis, :, :]
    dist = np.sum(diff ** 2, axis=2)
    nearest = np.argmin(dist, axis=1)
    arr[opaque, :3] = palette_arr[nearest]
    return Image.fromarray(arr)
