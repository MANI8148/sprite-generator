"""
Upload cleaned sprite dataset to HuggingFace Datasets with versioning.
"""
import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from datasets import Dataset, Features, Image as HFImage, Value
from huggingface_hub import HfApi, login


def load_metadata(metadata_path: Path) -> list:
    with open(metadata_path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Push sprite dataset to HF Datasets")
    parser.add_argument("--input", "-i", default="data/processed",
                        help="Processed dataset directory")
    parser.add_argument("--label-file", default="metadata_labeled.json",
                        help="Labeled metadata filename")
    parser.add_argument("--repo", "-r", required=True,
                        help="HF Dataset repo name (e.g. username/sprites)")
    parser.add_argument("--token", required=True, help="HF write token")
    parser.add_argument("--private", action="store_true",
                        help="Create private dataset")
    parser.add_argument("--split", default="train",
                        help="Dataset split name")
    args = parser.parse_args()

    input_dir = Path(args.input)

    # Try labeled metadata first, fallback to unlabeled
    label_path = input_dir / args.label_file
    if label_path.exists():
        metadata = load_metadata(label_path)
    else:
        metadata = load_metadata(input_dir / "metadata.json")

    # Load palette
    palette_path = input_dir / "palette.json"
    palette = []
    if palette_path.exists():
        with open(palette_path) as f:
            palette = json.load(f)

    print(f"Loading {len(metadata)} sprites...")

    images = []
    classes = []
    actions = []
    directions = []

    for entry in metadata:
        img_path = input_dir / entry["filename"]
        if not img_path.exists():
            continue
        # Load as RGBA PIL
        img = Image.open(img_path).convert("RGBA")
        images.append(img)
        classes.append(entry.get("class", "unknown"))
        actions.append(entry.get("action", "idle"))
        directions.append(entry.get("direction", "front"))

    # Create HF Dataset
    data = {
        "image": images,
        "class": classes,
        "action": actions,
        "direction": directions,
    }

    features = Features({
        "image": HFImage(),
        "class": Value("string"),
        "action": Value("string"),
        "direction": Value("string"),
    })

    dataset = Dataset.from_dict(data, features=features)

    # Push to HF Hub
    login(token=args.token)
    api = HfApi()

    dataset.push_to_hub(
        args.repo,
        split=args.split,
        private=args.private,
        token=args.token,
    )

    print(f"Pushed {len(dataset)} sprites to {args.repo}")

    # Also push palette as a dataset card
    if palette:
        card_content = f"""---
license: cc0
dataset_info:
- config_name: default
  features:
  - name: image
    dtype: image
  - name: class
    dtype: string
  - name: action
    dtype: string
  - name: direction
    dtype: string
  splits:
  - name: train
    num_examples: {len(dataset)}
---

# Sprite Dataset

Global palette ({len(palette)} colors):
{palette}

Generated from Kenney.nl CC0 sprite packs.
"""
        api.upload_file(
            path_or_fileobj=card_content.encode(),
            path_in_repo="README.md",
            repo_id=args.repo,
            repo_type="dataset",
            token=args.token,
        )

    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
