"""
Augment sprite dataset with safe augmentations.
For each sprite, creates N augmented copies with adjusted labels.
Augmentations:
  - Color Jitter (brightness, contrast, saturation)
  - Random Translation (1-2px shift, transparent padding)
  - HorizontalFlip (swaps left<->right labels when applied)
"""
import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


DIRECTION_SWAP = {
    "left": "right",
    "right": "left",
    "front_left": "front_right",
    "front_right": "front_left",
    "back_left": "back_right",
    "back_right": "back_left",
}


def color_jitter(img: Image.Image, brightness=0.1, contrast=0.1, saturation=0.1) -> Image.Image:
    import torchvision.transforms.functional as TF
    import torch
    if np.random.rand() > 0.5:
        img = TF.adjust_brightness(img, 1.0 + np.random.uniform(-brightness, brightness))
    if np.random.rand() > 0.5:
        img = TF.adjust_contrast(img, 1.0 + np.random.uniform(-contrast, contrast))
    if np.random.rand() > 0.5:
        img = TF.adjust_saturation(img, 1.0 + np.random.uniform(-saturation, saturation))
    return img


def random_translate(img: Image.Image, max_shift=2) -> Image.Image:
    dx = np.random.randint(-max_shift, max_shift + 1)
    dy = np.random.randint(-max_shift, max_shift + 1)
    result = Image.new("RGBA", img.size, (0, 0, 0, 0))
    result.paste(img, (dx, dy))
    return result


def horizontal_flip(img: Image.Image) -> Image.Image:
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def load_metadata(path: Path) -> list:
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Augment sprite dataset")
    parser.add_argument("--input", "-i", default="data/processed",
                        help="Processed dataset directory with metadata_labeled.json")
    parser.add_argument("--output", "-o", default="data/augmented",
                        help="Output directory for augmented dataset")
    parser.add_argument("--copies", type=int, default=4,
                        help="Number of augmented copies per original sprite")
    parser.add_argument("--include-flip", action="store_true", default=True,
                        help="Include horizontal flips (label-aware)")
    parser.add_argument("--include-jitter", action="store_true", default=True,
                        help="Include color jitter")
    parser.add_argument("--include-translate", action="store_true", default=True,
                        help="Include random translation")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_path = input_dir / "metadata_labeled.json"
    if not meta_path.exists():
        meta_path = input_dir / "metadata.json"

    metadata = load_metadata(meta_path)
    print(f"Loaded {len(metadata)} original entries from {meta_path}")

    aug_metadata = []
    next_id = len(metadata)

    pbar = tqdm(metadata, desc="Augmenting")
    for entry in pbar:
        src_path = input_dir / entry["filename"]
        if not src_path.exists():
            continue

        img = Image.open(src_path).convert("RGBA")

        # Create augmented copies
        for copy_idx in range(args.copies):
            aug_img = img.copy()
            aug_entry = dict(entry)
            original_filename = Path(entry["filename"])
            stem = original_filename.stem
            ext = original_filename.suffix

            applied = []
            direction = entry.get("direction", "front")

            if args.include_translate and np.random.rand() > 0.3:
                aug_img = random_translate(aug_img)
                applied.append("translate")

            if args.include_jitter and np.random.rand() > 0.3:
                aug_img = color_jitter(aug_img)
                applied.append("color_jitter")

            if args.include_flip and np.random.rand() > 0.3:
                aug_img = horizontal_flip(aug_img)
                new_direction = DIRECTION_SWAP.get(direction, direction)
                aug_entry["direction"] = new_direction
                aug_entry["flipped"] = True
                applied.append("flip")
            else:
                aug_entry["flipped"] = False

            aug_filename = f"{stem}_aug{next_id:06d}.png"
            aug_img.save(output_dir / aug_filename)

            aug_entry["id"] = next_id
            aug_entry["filename"] = aug_filename
            aug_entry["augmented"] = True
            aug_entry["augmentations"] = "_".join(applied) if applied else "none"
            aug_entry["source_id"] = entry["id"]
            aug_metadata.append(aug_entry)
            next_id += 1

        # Copy original to output as well
        orig_filename = f"{stem}_orig.png"
        img.save(output_dir / orig_filename)
        entry["filename"] = orig_filename
        entry["augmented"] = False
        entry["flipped"] = False
        entry["augmentations"] = "none"
        entry["source_id"] = entry["id"]
        aug_metadata.append(entry)

    print(f"Total entries after augmentation: {len(aug_metadata)}")

    meta_out = output_dir / "metadata_labeled.json"
    with open(meta_out, "w") as f:
        json.dump(aug_metadata, f, indent=2)
    print(f"Saved metadata to {meta_out}")

    # Copy palette
    palette_src = input_dir / "palette.json"
    if palette_src.exists():
        import shutil
        shutil.copy(palette_src, output_dir / "palette.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
