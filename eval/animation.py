"""
Generate sprite animation sequences with GIF export.
Chains VQ-VAE + Transformer generation across varying conditions
to produce animated sprite sheets.
"""
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.vqvae.model import VQVAE
from models.transformer.model import SpriteTransformer
from models.transformer.train import CLASS_VOCAB, ACTION_VOCAB, DIRECTION_VOCAB
from eval.generate_samples import post_process_sprite


DEFAULT_DIRECTIONS = ["front", "front_left", "left", "back_left",
                      "back", "back_right", "right", "front_right"]


def generate_animation_sequence(
    vqvae: VQVAE,
    transformer: SpriteTransformer,
    device: torch.device,
    class_name: str = "character",
    action: str = "walk",
    directions: Optional[List[str]] = None,
    num_repeats: int = 1,
    temperature: float = 1.0,
    top_k: int = 40,
    top_p: float = 0.9,
    palette: Optional[List[Tuple[int, int, int]]] = None,
) -> List[Image.Image]:
    if directions is None:
        directions = DEFAULT_DIRECTIONS

    vqvae.eval()
    transformer.eval()

    class_id = max(0, min(
        CLASS_VOCAB.index(class_name) if class_name in CLASS_VOCAB else 0, 63))
    action_id = max(0, min(
        ACTION_VOCAB.index(action) if action in ACTION_VOCAB else 0, 63))

    frames = []

    with torch.no_grad():
        for direction in directions:
            dir_id_val = max(0, min(
                DIRECTION_VOCAB.index(direction) if direction in DIRECTION_VOCAB else 0, 63))

            for _ in range(num_repeats):
                cls_t = torch.tensor([class_id]).to(device)
                act_t = torch.tensor([action_id]).to(device)
                dir_t = torch.tensor([dir_id_val]).to(device)

                indices = transformer.generate(
                    cls_t, act_t, dir_t,
                    max_tokens=64,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )

                recon = vqvae.decode_from_indices(indices, (vqvae.latent_dim, 8, 8))
                img_arr = recon[0].permute(1, 2, 0).cpu().numpy()
                img_arr = (img_arr * 255).clip(0, 255).astype(np.uint8)
                img_arr = post_process_sprite(img_arr, palette)
                frames.append(Image.fromarray(img_arr, "RGBA"))

    return frames


def create_animated_gif(
    frames: List[Image.Image],
    output_path: str,
    duration: int = 200,
    loop: int = 0,
) -> str:
    if not frames:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGBA", (1, 1)).save(str(out))
        return str(out)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    frames[0].save(
        str(out),
        save_all=True,
        append_images=frames[1:] if len(frames) > 1 else [],
        duration=duration,
        loop=loop,
        disposal=2,
    )

    return str(out)


def main():
    parser = argparse.ArgumentParser(
        description="Generate sprite animation sequence and export as GIF")
    parser.add_argument("--vqvae-checkpoint", required=True)
    parser.add_argument("--transformer-checkpoint", required=True)
    parser.add_argument("--output", "-o", default="animation.gif")
    parser.add_argument("--class-name", default="character")
    parser.add_argument("--action", default="walk")
    parser.add_argument("--directions", nargs="+", default=None)
    parser.add_argument("--num-repeats", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--duration", type=int, default=200)
    parser.add_argument("--palette", default=None, help="Path to palette JSON file")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    vqvae_ckpt = torch.load(args.vqvae_checkpoint, map_location=device)
    num_emb = vqvae_ckpt.get("config", {}).get("num_embeddings")
    if num_emb is None:
        num_emb = vqvae_ckpt["model_state"]["quantizer.embedding"].size(0)
    vqvae = VQVAE(num_embeddings=num_emb).to(device)
    vqvae.load_state_dict(vqvae_ckpt["model_state"])
    vqvae.eval()

    t_ckpt = torch.load(args.transformer_checkpoint, map_location=device)
    cfg = t_ckpt.get("config", {})
    sd = t_ckpt["model_state"]
    d_model = cfg.get("d_model", sd["ln_f.weight"].size(0))
    n_layers = cfg.get("n_layers",
                        sum(1 for k in sd if k.startswith("blocks.") and k.endswith(".ln1.weight")))
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

    palette = None
    if args.palette:
        palette_path = Path(args.palette)
        if palette_path.exists():
            import json
            with open(palette_path) as f:
                raw = json.load(f)
            palette = [tuple(c) if isinstance(c, list) else c for c in raw]

    frames = generate_animation_sequence(
        vqvae, transformer, device,
        class_name=args.class_name,
        action=args.action,
        directions=args.directions,
        num_repeats=args.num_repeats,
        temperature=args.temperature,
        palette=palette,
    )

    result_path = create_animated_gif(frames, args.output, duration=args.duration)
    print(f"Animation saved to {result_path} ({len(frames)} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
