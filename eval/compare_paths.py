"""
Compare Path A (VQ-VAE + Transformer) vs Path B (LoRA fine-tuning).
Generates side-by-side sprite grids and computes metrics for both.
"""
import sys
import json
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.vqvae.model import VQVAE
from models.transformer.model import SpriteTransformer
from models.transformer.train import CLASS_VOCAB, ACTION_VOCAB, DIRECTION_VOCAB
from models.lora.model import SpriteLoRAWrapper
from eval.metrics import palette_adherence_rate, grid_alignment_check


def generate_vqvae_samples(vqvae, transformer, device, conditions, temperature=1.0):
    vqvae.eval()
    transformer.eval()
    samples = []

    with torch.no_grad():
        for cls_name, act_name, dir_name in conditions:
            cls_id = torch.tensor([
                max(0, min(CLASS_VOCAB.index(cls_name) if cls_name in CLASS_VOCAB else 0, 63))
            ]).to(device)
            act_id = torch.tensor([
                max(0, min(ACTION_VOCAB.index(act_name) if act_name in ACTION_VOCAB else 0, 63))
            ]).to(device)
            dir_id = torch.tensor([
                max(0, min(DIRECTION_VOCAB.index(dir_name) if dir_name in DIRECTION_VOCAB else 0, 63))
            ]).to(device)

            indices = transformer.generate(
                cls_id, act_id, dir_id,
                max_tokens=64,
                temperature=temperature,
                top_k=40,
                top_p=0.9,
            )
            recon = vqvae.decode_from_indices(indices, (vqvae.latent_dim, 8, 8))
            img_arr = recon[0].permute(1, 2, 0).cpu().numpy()
            img_arr = (img_arr * 255).clip(0, 255).astype(np.uint8)
            samples.append(Image.fromarray(img_arr, "RGBA"))

    return samples


def generate_lora_samples(lora_model, device, num_samples):
    lora_model.eval()
    samples = []

    with torch.no_grad():
        generated = lora_model.generate(num_samples, device=device)
        for i in range(num_samples):
            img_arr = generated[i].permute(1, 2, 0).cpu().numpy()
            img_arr = (img_arr * 255).clip(0, 255).astype(np.uint8)
            samples.append(Image.fromarray(img_arr, "RGBA"))

    return samples


def compute_metrics(samples, palette):
    adherence_scores = [palette_adherence_rate(img, palette) for img in samples]
    alignment_scores = [grid_alignment_check(img) for img in samples]
    return {
        "palette_adherence_mean": float(np.mean(adherence_scores)),
        "palette_adherence_std": float(np.std(adherence_scores)),
        "grid_alignment_mean": float(np.mean(alignment_scores)),
        "grid_alignment_std": float(np.std(alignment_scores)),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare VQ-VAE+Transformer vs LoRA")
    parser.add_argument("--vqvae-checkpoint", required=True)
    parser.add_argument("--transformer-checkpoint", required=True)
    parser.add_argument("--lora-checkpoint", required=True)
    parser.add_argument("--output", "-o", default="comparison_results.json")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--visualization", default="comparison_grid.png")
    parser.add_argument("--palette", default=None, help="Path to palette JSON file")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    palette = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 255, 255), (0, 0, 0), (128, 128, 128), (255, 128, 0),
        (128, 0, 128), (0, 128, 128), (128, 128, 0), (64, 64, 64),
    ]
    palette_path = Path(args.palette) if args.palette else Path("data/processed/palette.json")
    if palette_path.exists():
        with open(palette_path) as f:
            raw = json.load(f)
            palette = [tuple(c) for c in raw] if isinstance(raw[0], list) else raw

    print("Loading VQ-VAE...")
    vqvae_ckpt = torch.load(args.vqvae_checkpoint, map_location=device)
    num_emb = vqvae_ckpt.get("config", {}).get("num_embeddings")
    if num_emb is None:
        num_emb = vqvae_ckpt["model_state"]["quantizer.embedding.weight"].size(0)
    vqvae = VQVAE(num_embeddings=num_emb).to(device)
    vqvae.load_state_dict(vqvae_ckpt["model_state"])
    vqvae.eval()

    print("Loading Transformer...")
    transformer = SpriteTransformer(
        vocab_size=vqvae.quantizer.num_embeddings,
        condition_vocab_size=64,
    ).to(device)
    t_ckpt = torch.load(args.transformer_checkpoint, map_location=device)
    transformer.load_state_dict(t_ckpt["model_state"])
    transformer.eval()

    print("Loading LoRA model...")
    lora_model = SpriteLoRAWrapper().to(device)
    lora_ckpt = torch.load(args.lora_checkpoint, map_location=device)
    lora_model.load_state_dict(lora_ckpt["model_state"], strict=False)
    lora_model.eval()

    classes = ["character", "enemy", "item", "weapon", "tile"]
    actions = ["idle", "walk", "attack", "jump"]
    directions = ["front", "back", "left", "right"]
    conditions = [
        (classes[i % len(classes)], actions[i % len(actions)], directions[i % len(directions)])
        for i in range(args.num_samples)
    ]

    print("Generating VQ-VAE + Transformer samples...")
    vqvae_samples = generate_vqvae_samples(vqvae, transformer, device, conditions, args.temperature)
    print("Generating LoRA samples...")
    lora_samples = generate_lora_samples(lora_model, device, args.num_samples)

    vqvae_metrics = compute_metrics(vqvae_samples, palette)
    lora_metrics = compute_metrics(lora_samples, palette)

    results = {
        "comparison": {
            "vqvae_transformer": vqvae_metrics,
            "lora": lora_metrics,
        },
        "difference": {
            "palette_adherence": vqvae_metrics["palette_adherence_mean"] - lora_metrics["palette_adherence_mean"],
            "grid_alignment": vqvae_metrics["grid_alignment_mean"] - lora_metrics["grid_alignment_mean"],
        },
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "conditions": [{"class": c, "action": a, "direction": d} for c, a, d in conditions],
    }

    print(json.dumps(results, indent=2))
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output}")

    sprite_size = 32
    cols = args.num_samples
    rows = 2
    canvas = Image.new("RGBA", (cols * sprite_size, rows * sprite_size), (0, 0, 0, 0))

    for i in range(args.num_samples):
        canvas.paste(vqvae_samples[i], (i * sprite_size, 0))
        canvas.paste(lora_samples[i], (i * sprite_size, sprite_size))

    if args.visualization:
        canvas.save(args.visualization)
        print(f"Visualization saved to {args.visualization}")

    return 0


if __name__ == "__main__":
    main()
