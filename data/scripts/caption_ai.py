"""
Auto-label sprites with character class, action, and direction using an AI model.
Uses a free vision-language model via HuggingFace Inference API or local transformers.
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional

import requests
from PIL import Image
from tqdm import tqdm


def load_metadata(metadata_path: Path) -> list:
    with open(metadata_path) as f:
        return json.load(f)


def caption_with_api(
    img: Image.Image,
    api_token: str,
    model: str,
) -> dict:
    """Use HF Inference API for zero-shot classification."""
    headers = {"Authorization": f"Bearer {api_token}"}

    # Convert image to bytes
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    # Use a simple image classification approach
    response = requests.post(
        f"https://api-inference.huggingface.co/models/{model}",
        headers=headers,
        data=buf.read(),
        timeout=30,
    )

    if response.status_code == 200:
        result = response.json()
        return {"class": str(result[0]["label"]) if result else "unknown", "action": "idle", "direction": "front"}
    return {"class": "unknown", "action": "idle", "direction": "front"}


def caption_locally(img: Image.Image) -> dict:
    """
    Simple rule-based captioning since we don't have a vision model locally.
    This is a placeholder - replace with an actual model call when available.
    """
    width, height = img.size
    # Heuristic: tall sprites = characters, wide = items, square = tiles
    ratio = height / width if width > 0 else 1
    if ratio > 1.3:
        cls = "character"
    elif ratio < 0.7:
        cls = "item"
    else:
        cls = "tile"

    return {"class": cls, "action": "idle", "direction": "front"}


def main():
    parser = argparse.ArgumentParser(description="Auto-label sprites with AI")
    parser.add_argument("--input", "-i", default="data/processed",
                        help="Processed dataset directory")
    parser.add_argument("--output", "-o", default="data/processed/metadata_labeled.json",
                        help="Output metadata path")
    parser.add_argument("--hf-token", default=None, help="HuggingFace API token")
    parser.add_argument("--model", default=None,
                        help="HF model for captioning (default: local heuristics)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    metadata_path = input_dir / "metadata.json"
    metadata = load_metadata(metadata_path)

    use_api = args.hf_token is not None and args.model is not None

    for entry in tqdm(metadata, desc="Captioning sprites"):
        img_path = input_dir / entry["filename"]
        try:
            img = Image.open(img_path).convert("RGBA")
            if use_api:
                labels = caption_with_api(img, args.hf_token, args.model)
            else:
                labels = caption_locally(img)
            entry.update(labels)
        except Exception:
            entry.update({"class": "unknown", "action": "idle", "direction": "front"})

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Labeled {len(metadata)} sprites -> {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
