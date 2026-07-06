"""
Clean, normalize, and quantize sprites to a fixed palette.
Pipeline:
  1. Find all PNGs in raw directory
  2. Resize/crop to fixed canvas (32x32 default)
  3. Detect and remove background, center sprite
  4. Dedup near-duplicates via perceptual hash
  5. Quantize to global palette (24 colors default)
  6. Save as indexed PNGs + metadata CSV
"""
import os
import sys
import json
import argparse
import hashlib
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm
import imagehash


def remove_background(img: Image.Image, bg_color=None) -> Image.Image:
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    if bg_color is None:
        pixels = np.array(img)
        # sample corners to guess bg color
        corners = [
            pixels[0, 0],
            pixels[0, -1],
            pixels[-1, 0],
            pixels[-1, -1],
        ]
        bg_color = tuple(int(x) for x in np.median(corners, axis=0)[:3])
        bg_color = bg_color + (255,)

    # Make pixels matching bg_color transparent
    data = np.array(img)
    mask = np.all(data[:, :, :3] == bg_color[:3], axis=2)
    data[mask] = [0, 0, 0, 0]
    return Image.fromarray(data)


def find_content_bbox(img: Image.Image) -> tuple:
    arr = np.array(img)
    if arr.shape[2] == 4:
        alpha = arr[:, :, 3]
    else:
        alpha = 255 - np.all(arr == 0, axis=2).astype(np.uint8) * 255

    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)

    if not np.any(rows) or not np.any(cols):
        return (0, 0, img.width, img.height)

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return (int(x_min), int(y_min), int(x_max + 1), int(y_max + 1))


def center_on_canvas(img: Image.Image, canvas_size: int = 32) -> Image.Image:
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    bbox = find_content_bbox(img)
    sprite = img.crop(bbox)

    # Compute scale to fit within canvas with padding
    max_dim = max(sprite.width, sprite.height)
    scale = (canvas_size * 0.8) / max_dim
    new_w = max(1, int(sprite.width * scale))
    new_h = max(1, int(sprite.height * scale))
    sprite = sprite.resize((new_w, new_h), Image.NEAREST)

    canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    x_offset = (canvas_size - new_w) // 2
    y_offset = canvas_size - new_h  # bottom-center anchor for characters

    canvas.paste(sprite, (x_offset, y_offset), sprite)
    return canvas


def build_global_palette(images: list, n_colors: int = 24) -> list:
    all_pixels = []
    for img in images:
        arr = np.array(img.convert("RGBA"))
        mask = arr[:, :, 3] > 0
        pixels = arr[:, :, :3][mask]
        if len(pixels) > 0:
            # Subsample for speed
            if len(pixels) > 50000:
                idx = np.random.choice(len(pixels), 50000, replace=False)
                pixels = pixels[idx]
            all_pixels.append(pixels)

    if not all_pixels:
        return [(0, 0, 0)]

    all_pixels = np.concatenate(all_pixels, axis=0)

    # Use k-means++ approximation via random sampling
    if len(all_pixels) > n_colors * 100:
        idx = np.random.choice(len(all_pixels), n_colors * 100, replace=False)
        samples = all_pixels[idx]
    else:
        samples = all_pixels

    # Simple k-means
    centroids = samples[np.random.choice(len(samples), n_colors, replace=False)].astype(float)

    for _ in range(20):
        distances = np.linalg.norm(samples[:, np.newaxis, :].astype(float) - centroids[np.newaxis, :, :], axis=2)
        labels = np.argmin(distances, axis=1)
        for k in range(n_colors):
            mask_k = labels == k
            if np.any(mask_k):
                centroids[k] = samples[mask_k].mean(axis=0)

    palette = [(int(c[0]), int(c[1]), int(c[2])) for c in centroids]
    return palette


def quantize_to_palette(img: Image.Image, palette: list) -> Image.Image:
    arr = np.array(img.convert("RGBA"))
    h, w = arr.shape[:2]
    pixels = arr[:, :, :3].reshape(-1, 3)
    alpha = arr[:, :, 3]

    # Find nearest palette color for each pixel
    palette_arr = np.array(palette, dtype=float)
    pixel_float = pixels.astype(float)
    distances = np.linalg.norm(
        pixel_float[:, np.newaxis, :] - palette_arr[np.newaxis, :, :],
        axis=2
    )
    nearest = np.argmin(distances, axis=1)

    out = np.zeros((h, w, 4), dtype=np.uint8)
    out[:, :, :3] = np.array(palette)[nearest].reshape(h, w, 3)
    out[:, :, 3] = alpha
    return Image.fromarray(out)


def main():
    parser = argparse.ArgumentParser(description="Clean and normalize sprite dataset")
    parser.add_argument("--input", "-i", default="data/raw", help="Input raw directory")
    parser.add_argument("--output", "-o", default="data/processed", help="Output directory")
    parser.add_argument("--canvas-size", type=int, default=32, help="Target canvas size")
    parser.add_argument("--palette-size", type=int, default=24, help="Number of palette colors")
    parser.add_argument("--dedup-threshold", type=int, default=5,
                        help="Perceptual hash distance for dedup")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all images
    image_paths = list(input_dir.rglob("*.png")) + list(input_dir.rglob("*.jpg"))
    print(f"Found {len(image_paths)} images")

    if len(image_paths) == 0:
        print("No images found. Run scrape_sources.py first.")
        return 1

    # Clean each image
    cleaned = []
    valid_paths = []

    for path in tqdm(image_paths, desc="Cleaning images"):
        try:
            img = Image.open(path).convert("RGBA")
            img = remove_background(img)
            img = center_on_canvas(img, args.canvas_size)
            cleaned.append(img)
            valid_paths.append(path)
        except Exception:
            continue

    print(f"Cleaned {len(cleaned)} images")

    # Dedup with perceptual hash
    hashes = []
    dedup_indices = []
    for i, img in enumerate(tqdm(cleaned, desc="Deduping")):
        gray = img.convert("L")
        phash = imagehash.phash(gray)

        is_dup = False
        for existing_idx, existing_hash in hashes:
            if phash - existing_hash < args.dedup_threshold:
                is_dup = True
                break

        if not is_dup:
            hashes.append((i, phash))
            dedup_indices.append(i)

    deduped = [cleaned[i] for i in dedup_indices]
    print(f"After dedup: {len(deduped)} unique sprites")

    # Build global palette
    print("Building global palette...")
    palette = build_global_palette(deduped, args.palette_size)
    print(f"Palette: {len(palette)} colors")

    # Save palette
    palette_path = output_dir / "palette.json"
    with open(palette_path, "w") as f:
        json.dump(palette, f)

    # Quantize and save
    metadata = []
    for i, img in enumerate(tqdm(deduped, desc="Quantizing and saving")):
        quantized = quantize_to_palette(img, palette)
        filename = f"sprite_{i:06d}.png"
        quantized.save(output_dir / filename)
        metadata.append({
            "id": i,
            "filename": filename,
            "source": str(valid_paths[dedup_indices[i]]),
        })

    # Save metadata
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved {len(metadata)} processed sprites to {output_dir}")
    print(f"Palette: {palette_path}")
    print(f"Metadata: {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
