"""
Evaluation metrics for sprite quality assessment.
Metrics: palette-adherence rate, grid-alignment, reconstruction loss.
"""
import sys
import json
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from datasets import load_dataset
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.vqvae.model import VQVAE
from models.transformer.model import SpriteTransformer


def palette_adherence_rate(img: Image.Image, palette: list) -> float:
    """What percentage of pixels match the fixed palette?"""
    arr = np.array(img.convert("RGBA"))
    h, w = arr.shape[:2]
    pixels = arr[:, :, :3].reshape(-1, 3)
    alpha = arr[:, :, 3].reshape(-1)

    palette_arr = np.array(palette)
    total_pixels = (alpha > 0).sum()
    if total_pixels == 0:
        return 1.0

    matches = 0
    for pixel in pixels[alpha > 0]:
        distances = np.linalg.norm(palette_arr.astype(float) - pixel.astype(float), axis=1)
        if distances.min() < 5:  # within 5 RGB levels
            matches += 1

    return matches / total_pixels


def grid_alignment_check(img: Image.Image, grid_size: int = 32) -> float:
    """Check that sprite is aligned to pixel grid (no sub-pixel blur)."""
    arr = np.array(img)
    if arr.shape[2] == 4:
        alpha = arr[:, :, 3]
    else:
        alpha = np.ones((arr.shape[0], arr.shape[1])) * 255

    # Check for smooth (non-binary) alpha edges
    edge_pixels = 0
    smooth_pixels = 0

    for y in range(1, arr.shape[0] - 1):
        for x in range(1, arr.shape[0] - 1):
            if alpha[y, x] > 0 and alpha[y, x] < 255:
                edge_pixels += 1
                if alpha[y, x] > 0:
                    smooth_pixels += 1

    if edge_pixels == 0:
        return 1.0  # perfectly hard edges

    return 1.0 - (smooth_pixels / edge_pixels)


def compute_reconstruction_loss(vqvae, dataloader, device) -> float:
    """Compute average reconstruction MSE over a dataset."""
    vqvae.eval()
    total_loss = 0
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            output = vqvae(batch)
            total_loss += output["recon_loss"].item() * batch.size(0)
            count += batch.size(0)

    return total_loss / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="HF Dataset path")
    parser.add_argument("--vqvae-checkpoint", required=True)
    parser.add_argument("--output", "-o", default="eval_results.json")
    parser.add_argument("--num-samples", type=int, default=100)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load palette
    palette_path = Path("data/processed/palette.json")
    palette = []
    if palette_path.exists():
        with open(palette_path) as f:
            palette = json.load(f)

    # Load a subset of the dataset
    dataset = load_dataset(args.dataset, split=f"train[:{args.num_samples}]")

    # Run palette adherence on dataset
    adherence_scores = []
    for item in dataset:
        img = item["image"].convert("RGBA")
        adherence_scores.append(palette_adherence_rate(img, palette))

    metrics = {
        "palette_adherence_mean": float(np.mean(adherence_scores)) if adherence_scores else 0,
        "palette_adherence_std": float(np.std(adherence_scores)) if adherence_scores else 0,
        "num_samples": args.num_samples,
        "palette_size": len(palette),
    }

    print(json.dumps(metrics, indent=2))

    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to {args.output}")


if __name__ == "__main__":
    main()
