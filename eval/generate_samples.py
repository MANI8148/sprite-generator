"""
Generate a grid of sample sprites from the trained model for visual inspection.
"""
import sys
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.vqvae.model import VQVAE
from models.transformer.model import SpriteTransformer
from models.transformer.train import CLASS_VOCAB, ACTION_VOCAB, DIRECTION_VOCAB


def quantize_to_palette(img_array: np.ndarray, palette: list) -> np.ndarray:
    result = img_array.copy()
    pixels = result[:, :, :3].reshape(-1, 3)
    alpha = result[:, :, 3].reshape(-1)
    palette_arr = np.array(palette, dtype=float)

    for i in range(len(pixels)):
        if alpha[i] > 0:
            d = np.linalg.norm(palette_arr - pixels[i].astype(float), axis=1)
            pixels[i] = palette_arr[d.argmin()].astype(np.uint8)
    return result


def hard_alpha_edges(img_array: np.ndarray, threshold: int = 128) -> np.ndarray:
    result = img_array.copy()
    result[:, :, 3] = np.where(result[:, :, 3] > threshold, 255, 0)
    return result


def post_process_sprite(img_array: np.ndarray, palette: list = None) -> np.ndarray:
    if palette is not None:
        img_array = quantize_to_palette(img_array, palette)
    img_array = hard_alpha_edges(img_array)
    return img_array


def load_models(hf_repo: str, device: str):
    vqvae_ckpt = torch.load(hf_hub_download(hf_repo, "vqvae_latest.pt"), map_location=device)
    num_emb = vqvae_ckpt.get("config", {}).get("num_embeddings")
    if num_emb is None:
        num_emb = vqvae_ckpt["model_state"]["quantizer.embedding"].size(0)
    vqvae = VQVAE(num_embeddings=num_emb).to(device)
    sd = vqvae_ckpt["model_state"]
    sd = {k: v for k, v in sd.items() if not k.startswith("perceptual_loss.")}
    vqvae.load_state_dict(sd)
    vqvae.eval()

    t_ckpt = torch.load(hf_hub_download(hf_repo, "transformer_latest.pt"), map_location=device)
    cfg = t_ckpt.get("config", {})
    sd = t_ckpt["model_state"]
    d_model = cfg.get("d_model", sd["ln_f.weight"].size(0))
    n_layers = cfg.get("n_layers", sum(1 for k in sd if k.startswith("blocks.") and k.endswith(".ln1.weight")))
    n_heads = cfg.get("n_heads", 8)
    max_seq_len = sd["pos_embedding"].size(1)

    transformer = SpriteTransformer(
        vocab_size=sd["head.weight"].size(0),
        condition_vocab_size=64,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        max_seq_len=max_seq_len,
    ).to(device)
    transformer.load_state_dict(sd)
    transformer.eval()

    return vqvae, transformer


def generate_grid(vqvae, transformer, device, grid_size=(4, 4), output_path="samples.png", palette=None):
    rows, cols = grid_size
    total = rows * cols
    sprite_size = 32
    canvas = Image.new("RGBA", (cols * sprite_size, rows * sprite_size), (0, 0, 0, 0))

    classes = ["character", "item", "enemy", "weapon", "vehicle", "animal", "plant", "tile",
               "ui_element", "projectile", "furniture", "decoration", "food", "tool", "accessory", "building"]
    actions = ["idle", "walk", "run", "attack", "jump", "hurt", "death", "block",
               "shoot", "cast", "interact", "fly", "swim", "climb"]
    directions = ["front", "back", "left", "right"]
    temperatures = [0.6, 0.8, 1.0, 1.2]

    for i in range(total):
        cls = classes[i % len(classes)]
        act = actions[i % len(actions)]
        dire = directions[i % len(directions)]
        temp = temperatures[i % len(temperatures)]

        class_id = torch.tensor([max(0, min(CLASS_VOCAB.index(cls) if cls in CLASS_VOCAB else 0, 63))]).to(device)
        action_id = torch.tensor([max(0, min(ACTION_VOCAB.index(act) if act in ACTION_VOCAB else 0, 63))]).to(device)
        direction_id = torch.tensor([max(0, min(DIRECTION_VOCAB.index(dire) if dire in DIRECTION_VOCAB else 0, 63))]).to(device)

        with torch.no_grad():
            indices = transformer.generate(
                class_id, action_id, direction_id,
                max_tokens=64,
                temperature=temp,
                top_k=40,
                top_p=0.9,
            )

        recon = vqvae.decode_from_indices(indices, (vqvae.latent_dim, 8, 8))

        img_arr = recon[0].permute(1, 2, 0).cpu().detach().numpy()
        img_arr = (img_arr * 255).clip(0, 255).astype(np.uint8)
        img_arr = post_process_sprite(img_arr, palette)
        img = Image.fromarray(img_arr, "RGBA")

        row, col = divmod(i, cols)
        canvas.paste(img, (col * sprite_size, row * sprite_size))

    canvas.save(output_path)
    print(f"Sample grid saved to {output_path}")
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-repo", default="darklord8777/sprite-generator-model")
    parser.add_argument("--output", "-o", default="samples.png")
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--palette", default=None, help="Path to palette JSON file")
    args = parser.parse_args()

    palette = None
    if args.palette:
        palette_path = Path(args.palette)
        if palette_path.exists():
            import json
            with open(palette_path) as f:
                raw = json.load(f)
                palette = [tuple(c) if isinstance(c, list) else c for c in raw]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vqvae, transformer = load_models(args.hf_repo, device)
    generate_grid(vqvae, transformer, device, (args.rows, args.cols), args.output, palette)


if __name__ == "__main__":
    main()
