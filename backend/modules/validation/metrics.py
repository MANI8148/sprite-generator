import hashlib
import numpy as np
from PIL import Image
from typing import Dict, Tuple, Optional


def palette_size(image: Image.Image) -> int:
    rgba = image.convert("RGBA")
    arr = np.array(rgba)
    alpha = arr[:, :, 3]
    opaque = alpha > 128
    if opaque.sum() == 0:
        return 0
    pixels = arr[opaque][:, :3]
    return len(set(tuple(p) for p in pixels))


def sprite_centering(image: Image.Image) -> Tuple[float, float]:
    rgba = image.convert("RGBA")
    arr = np.array(rgba)
    alpha = arr[:, :, 3] > 128
    rows = np.any(alpha, axis=1)
    cols = np.any(alpha, axis=0)
    if not rows.any():
        return (0.5, 0.5)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    center_x = (x_min + x_max) / 2 / image.size[0]
    center_y = (y_min + y_max) / 2 / image.size[1]
    return (float(center_x), float(center_y))


def bounding_box(image: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    rgba = image.convert("RGBA")
    arr = np.array(rgba)
    alpha = arr[:, :, 3] > 128
    rows = np.any(alpha, axis=1)
    cols = np.any(alpha, axis=0)
    if not rows.any():
        return None
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return (int(x_min), int(y_min), int(x_max), int(y_max))


def transparency_coverage(image: Image.Image) -> float:
    rgba = image.convert("RGBA")
    arr = np.array(rgba)
    total = arr.shape[0] * arr.shape[1]
    transparent = np.sum(arr[:, :, 3] < 128)
    return float(transparent) / total


def outline_continuity(image: Image.Image) -> float:
    rgba = image.convert("RGBA")
    arr = np.array(rgba)
    alpha = arr[:, :, 3] > 128
    h, w = alpha.shape
    from scipy.ndimage import binary_dilation
    struct = np.ones((3, 3), dtype=bool)
    dilated = binary_dilation(alpha, iterations=1, structure=struct)
    edge = dilated & ~alpha
    # Count connected components in the edge
    from scipy.ndimage import label
    labeled, num = label(edge)
    if num <= 1:
        return 1.0
    sizes = [np.sum(labeled == i) for i in range(1, num + 1)]
    main = max(sizes)
    return float(main) / sum(sizes) if sum(sizes) > 0 else 1.0


def pixel_sharpness(image: Image.Image) -> float:
    gray = image.convert("L")
    arr = np.array(gray, dtype=np.float32)
    from scipy.ndimage import convolve
    lap = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    lap_out = convolve(arr, lap, mode="reflect")
    return float(np.var(lap_out))


def _image_hash(img: Image.Image) -> str:
    small = img.resize((16, 16), Image.NEAREST).convert("L")
    arr = np.array(small, dtype=np.uint8)
    if arr.max() == arr.min():
        return hashlib.md5(arr.tobytes()).hexdigest()
    avg = arr.mean()
    bits = "".join(["1" if p > avg else "0" for p in arr.flatten()])
    return bits


def duplicate_detection(images: list) -> int:
    if len(images) < 2:
        return 0
    hashes = [_image_hash(img) for img in images]
    dupes = 0
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            diff = sum(a != b for a, b in zip(hashes[i], hashes[j]))
            if diff < 4:
                dupes += 1
    return dupes


def palette_consistency(image: Image.Image, palette_name: str = "retro_16") -> float:
    from ..postprocess.palettes import get_palette
    target = get_palette(palette_name)
    palette_arr = np.array(target, dtype=np.int32)
    rgba = image.convert("RGBA")
    arr = np.array(rgba)
    alpha = arr[:, :, 3]
    opaque = alpha > 128
    if not opaque.any():
        return 1.0
    pixels = arr[opaque, :3].astype(np.int32)
    min_dist = np.full(pixels.shape[0], np.inf, dtype=np.float64)
    for c in palette_arr:
        d = np.sum((pixels.astype(np.float64) - c.astype(np.float64)) ** 2, axis=1)
        min_dist = np.minimum(min_dist, d)
    max_dist = 3 * (255 ** 2)
    scores = 1.0 - np.sqrt(min_dist) / np.sqrt(max_dist)
    return float(np.mean(scores))


def sprite_aspect_ratio(image: Image.Image) -> float:
    bbox = bounding_box(image)
    if bbox is None:
        return 1.0
    x, y, x2, y2 = bbox
    w = x2 - x + 1
    h = y2 - y + 1
    if h == 0:
        return 1.0
    ratio = w / h
    if ratio < 1.0:
        ratio = 1.0 / ratio
    return round(ratio, 2)


def assess_all(image: Image.Image, batch: list = None) -> Dict:
    result = {}
    result["palette_size"] = palette_size(image)
    cx, cy = sprite_centering(image)
    result["center_x"] = round(cx, 3)
    result["center_y"] = round(cy, 3)
    bbox = bounding_box(image)
    if bbox:
        result["bbox"] = {
            "x": bbox[0], "y": bbox[1],
            "w": bbox[2] - bbox[0], "h": bbox[3] - bbox[1],
        }
        result["bbox_area"] = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    result["transparency_ratio"] = round(transparency_coverage(image), 3)
    result["outline_continuity"] = round(outline_continuity(image), 3)
    result["sharpness"] = round(pixel_sharpness(image), 1)
    result["palette_consistency"] = round(palette_consistency(image), 3)
    result["aspect_ratio"] = sprite_aspect_ratio(image)
    quality = "clean"
    if result["palette_size"] == 0 or result["transparency_ratio"] >= 0.99:
        quality = "empty"
    elif result["aspect_ratio"] > 4.0:
        quality = "extreme_aspect"
    elif result["palette_size"] > 128:
        quality = "noisy"
    elif result["outline_continuity"] < 0.8:
        quality = "broken_outline"
    elif result["sharpness"] < 50:
        quality = "blurry"
    elif result["palette_size"] > 64:
        quality = "acceptable"
    result["quality_tier"] = quality
    if batch:
        result["duplicates"] = duplicate_detection(batch)
    return result
