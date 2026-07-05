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


def load_models(hf_repo: str, device: str):
    vqvae = VQVAE().to(device)
    ckpt = torch.load(hf_hub_download(hf_repo, "vqvae_latest.pt"), map_location=device)
    vqvae.load_state_dict(ckpt["model_state"])
    vqvae.eval()

    transformer = SpriteTransformer(
        vocab_size=vqvae.quantizer.num_embeddings,
        condition_vocab_size=64,
    ).to(device)
    ckpt = torch.load(hf_hub_download(hf_repo, "transformer_latest.pt"), map_location=device)
    transformer.load_state_dict(ckpt["model_state"])
    transformer.eval()

    return vqvae, transformer


def generate_grid(vqvae, transformer, device, grid_size=(4, 4), output_path="samples.png"):
    rows, cols = grid_size
    total = rows * cols
    sprite_size = 32
    canvas = Image.new("RGBA", (cols * sprite_size, rows * sprite_size), (0, 0, 0, 0))

    classes = ["character", "item", "enemy", "weapon", "vehicle", "animal", "plant", "tile",
               "ui_element", "projectile", "furniture", "decoration", "food", "tool", "accessory", "building"]
    actions = ["idle", "walk", "run", "attack", "jump", "hurt", "death", "block",
               "shoot", "cast", "interact", "fly", "swim", "climb", "front", "left"]
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

        latent_shape = (vqvae.latent_dim, 8, 8)
        z = vqvae.quantizer.get_codebook_entry(indices.view(-1))
        z = z.view(-1, *latent_shape)
        recon = vqvae.decoder(z)

        img_arr = recon[0].permute(1, 2, 0).cpu().numpy()
        img_arr = (img_arr * 255).clip(0, 255).astype(np.uint8)
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
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vqvae, transformer = load_models(args.hf_repo, device)
    generate_grid(vqvae, transformer, device, (args.rows, args.cols), args.output)


if __name__ == "__main__":
    main()
