"""
Auto-label sprites with character class, action, and direction using an AI model.
Uses a free vision-language model via HuggingFace Inference API or local transformers.
"""
import os
import sys
import json
import argparse
import io
import base64
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm


def load_metadata(metadata_path: Path) -> list:
    with open(metadata_path) as f:
        return json.load(f)


def _analyze_direction(img: Image.Image) -> str:
    """Determine facing direction via horizontal center-of-mass of alpha channel."""
    arr = np.array(img)
    alpha = arr[:, :, 3].astype(np.float64)
    if alpha.sum() == 0:
        return "front"

    h, w = alpha.shape
    x_coords = np.arange(w, dtype=np.float64)
    x_weights = alpha.sum(axis=0)
    if x_weights.sum() == 0:
        return "front"
    cog_x = np.average(x_coords, weights=x_weights)
    rel_x = cog_x / w

    if rel_x < 0.4:
        return "right"
    elif rel_x > 0.6:
        return "left"
    return "front"


def _analyze_action(img: Image.Image) -> str:
    """Determine action via vertical variance comparison in alpha channel."""
    arr = np.array(img)
    alpha = arr[:, :, 3].astype(np.float64)
    if alpha.sum() == 0:
        return "idle"

    h, w = alpha.shape
    mid = h // 2
    if mid == 0 or h == mid:
        return "idle"

    upper_var = alpha[:mid, :].var()
    lower_var = alpha[mid:, :].var()

    if lower_var > upper_var * 1.5 and upper_var > 0:
        return "walking"
    return "idle"


def _classify_by_ratio(img: Image.Image) -> str:
    """Classify sprite type by aspect ratio."""
    width, height = img.size
    ratio = height / width if width > 0 else 1
    if ratio > 1.3:
        return "character"
    elif ratio < 0.7:
        return "item"
    return "tile"


def _query_hf_vqa(
    img: Image.Image,
    api_token: str,
    model: str,
    question: str,
) -> Optional[str]:
    """Query a HF Inference API VQA model and return the answer text."""
    headers = {"Authorization": f"Bearer {api_token}"}
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    image_b64 = base64.b64encode(buf.getvalue()).decode()

    payload = {
        "inputs": {
            "image": image_b64,
            "question": question,
        }
    }
    resp = requests.post(
        f"https://api-inference.huggingface.co/models/{model}",
        headers=headers,
        json=payload,
        timeout=30,
    )
    if resp.status_code == 200:
        data = resp.json()
        if isinstance(data, list) and data:
            if isinstance(data[0], dict):
                return str(data[0].get("answer", ""))
            return str(data[0])
    return None


def caption_with_api(
    img: Image.Image,
    api_token: str,
    model: str,
) -> dict:
    """Use HF Inference API for multi-attribute VQA labeling."""
    result = {}

    cls = _query_hf_vqa(img, api_token, model, "What type of game sprite is this?")
    result["class"] = cls if cls else _classify_by_ratio(img)

    action = _query_hf_vqa(img, api_token, model, "What action is this sprite performing?")
    result["action"] = action if action else _analyze_action(img)

    direction = _query_hf_vqa(img, api_token, model, "Which direction is this sprite facing?")
    result["direction"] = direction if direction else _analyze_direction(img)

    return result


def caption_locally(img: Image.Image) -> dict:
    """Rule-based captioning using pixel analysis heuristics."""
    return {
        "class": _classify_by_ratio(img),
        "action": _analyze_action(img),
        "direction": _analyze_direction(img),
    }


def labels_to_caption(labels: dict, prefix: str = "pixel art sprite") -> str:
    """Convert structured labels from caption_locally/caption_with_api into a
    natural language caption string suitable for SD/LoRA training."""
    cls = labels.get("class", "character")
    action = labels.get("action", "idle")
    direction = labels.get("direction", "front")
    parts = [prefix, "of a", cls]
    if action and action != "idle":
        parts.append(action)
    if direction:
        parts.append(f"facing {direction}")
    return " ".join(parts)


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
