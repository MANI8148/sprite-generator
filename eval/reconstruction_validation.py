"""
Visually validate VQ-VAE reconstructions against original images.
Produces side-by-side comparison grids and per-sample metrics (PSNR, MSE).
Phase 0 Item 2: Visually validate reconstructions and generated samples.
"""
import json
import math
import sys
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.vqvae.model import VQVAE

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


def compute_mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    diff = original.astype(np.float64) - reconstructed.astype(np.float64)
    return float(np.mean(diff ** 2))


def compute_psnr(original: np.ndarray, reconstructed: np.ndarray, max_val: int = 255) -> float:
    mse = compute_mse(original, reconstructed)
    if mse == 0:
        return float("inf")
    return float(20 * math.log10(max_val) - 10 * math.log10(mse))


def compute_metrics(original: Image.Image, reconstructed: Image.Image) -> dict:
    orig_arr = np.array(original.convert("RGBA"), dtype=np.uint8)
    recon_arr = np.array(reconstructed.convert("RGBA"), dtype=np.uint8)
    alpha_mask = orig_arr[:, :, 3] > 128
    if alpha_mask.sum() == 0:
        return {"mse": 0.0, "psnr": float("inf"), "pixel_count": 0}
    mse = compute_mse(orig_arr[alpha_mask], recon_arr[alpha_mask])
    psnr = float("inf") if mse == 0 else float(20 * math.log10(255) - 10 * math.log10(mse))
    return {"mse": round(mse, 4), "psnr": round(psnr, 2), "pixel_count": int(alpha_mask.sum())}


def reconstruct_image(vqvae: VQVAE, image: Image.Image, device: torch.device) -> Image.Image:
    img_resized = image.convert("RGBA").resize((32, 32))
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    vqvae.eval()
    with torch.no_grad():
        output = vqvae(tensor)
        recon = output["recon"]
    recon_arr = recon[0].permute(1, 2, 0).cpu().numpy()
    recon_arr = (recon_arr * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(recon_arr, "RGBA")


def create_comparison_grid(
    originals: List[Image.Image],
    reconstructions: List[Image.Image],
    metrics_list: List[dict],
    title: str = "Reconstruction Validation",
    cells_per_row: int = 4,
) -> Image.Image:
    n = len(originals)
    if n == 0:
        raise ValueError("At least one image pair is required")
    cols = min(cells_per_row, n)
    rows = (n + cols - 1) // cols
    cell_w = 128
    cell_h = 160
    header_h = 30 if title else 0
    canvas_w = cols * cell_w
    canvas_h = header_h + rows * cell_h
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except (IOError, OSError):
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()
    if title:
        draw.text((8, 4), title, fill=(0, 0, 0), font=title_font)
    for i in range(n):
        col = i % cols
        row = i // cols
        x = col * cell_w
        y = header_h + row * cell_h
        half_w = cell_w // 2
        orig_small = originals[i].copy()
        orig_small.thumbnail((half_w - 4, 64), Image.NEAREST)
        orig_x = x + 2
        orig_y = y + 2
        canvas.paste(orig_small, (orig_x, orig_y), orig_small)
        recon_small = reconstructions[i].copy()
        recon_small.thumbnail((half_w - 4, 64), Image.NEAREST)
        recon_x = x + half_w + 2
        recon_y = y + 2
        canvas.paste(recon_small, (recon_x, recon_y), recon_small)
        draw.line([(x + half_w, y), (x + half_w, y + 64)], fill=(200, 200, 200), width=1)
        metrics = metrics_list[i] if i < len(metrics_list) else {}
        label_y = y + 68
        psnr_val = metrics.get("psnr", "N/A")
        mse_val = metrics.get("mse", "N/A")
        psnr_str = "inf" if psnr_val == float("inf") else str(psnr_val)
        draw.text((x + 4, label_y), f"PSNR: {psnr_str}", fill=(30, 30, 30), font=font)
        draw.text((x + 4, label_y + 12), f"MSE: {mse_val}", fill=(30, 30, 30), font=font)
        pixel_count = metrics.get("pixel_count", 0)
        draw.text((x + 4, label_y + 24), f"Pixels: {pixel_count}", fill=(30, 30, 30), font=font)
        draw.text((x + 4, label_y + 36), f"#{i + 1}", fill=(128, 128, 128), font=font)
    return canvas


def validate_reconstructions(
    vqvae: VQVAE,
    images: List[Image.Image],
    device: torch.device,
    title: str = "Reconstruction Validation",
    output_path: Optional[str] = None,
) -> Tuple[Image.Image, List[dict]]:
    reconstructions = []
    metrics_list = []
    for img in images:
        recon = reconstruct_image(vqvae, img, device)
        reconstructions.append(recon)
        metrics_list.append(compute_metrics(img, recon))
    grid = create_comparison_grid(images, reconstructions, metrics_list, title=title)
    if output_path:
        grid.save(output_path)
    return grid, metrics_list


def main():
    parser = argparse.ArgumentParser(
        description="Visually validate VQ-VAE reconstructions"
    )
    parser.add_argument("--vqvae-checkpoint", required=True)
    parser.add_argument("--output", "-o", default="reconstruction_validation.png")
    parser.add_argument("--metrics-output", default="reconstruction_metrics.json")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--dataset", default=None, help="HF dataset path or image directory")
    parser.add_argument("--image-dir", default=None, help="Path to directory of images")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.vqvae_checkpoint, map_location=device)
    num_emb = checkpoint.get("config", {}).get("num_embeddings")
    if num_emb is None:
        num_emb = checkpoint["model_state"]["quantizer.embedding.weight"].size(0)
    vqvae = VQVAE(num_embeddings=num_emb).to(device)
    vqvae.load_state_dict(checkpoint["model_state"])
    vqvae.eval()

    images = []
    if args.image_dir:
        img_dir = Path(args.image_dir)
        exts = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}
        paths = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in exts])[:args.num_samples]
        for p in paths:
            images.append(Image.open(p).convert("RGBA"))
    elif args.dataset:
        if load_dataset is None:
            print("datasets library not available")
            return 1
        ds = load_dataset(args.dataset, split=f"train[:{args.num_samples}]")
        for item in ds:
            images.append(item["image"].convert("RGBA"))
    else:
        print("Either --dataset or --image-dir must be provided")
        return 1

    if not images:
        print("No images found")
        return 1

    grid, metrics_list = validate_reconstructions(vqvae, images, device, output_path=args.output)

    psnr_vals = [m["psnr"] for m in metrics_list if m["psnr"] != float("inf")]
    mse_vals = [m["mse"] for m in metrics_list]

    summary = {
        "num_samples": len(images),
        "avg_psnr": round(float(np.mean(psnr_vals)), 2) if psnr_vals else None,
        "avg_mse": round(float(np.mean(mse_vals)), 4) if mse_vals else None,
        "min_psnr": round(float(np.min(psnr_vals)), 2) if psnr_vals else None,
        "max_psnr": round(float(np.max(psnr_vals)), 2) if psnr_vals else None,
        "per_sample": metrics_list,
    }

    metrics_path = Path(args.metrics_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Reconstruction grid saved to {args.output}")
    print(f"Metrics saved to {metrics_path}")
    print(f"Average PSNR: {summary['avg_psnr']}, Average MSE: {summary['avg_mse']}")
    return 0


if __name__ == "__main__":
    main()
