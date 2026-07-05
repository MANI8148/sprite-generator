"""
Gradio demo for sprite generator.
Loads trained VQ-VAE + Transformer from HF Hub and generates sprites.
"""
import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import gradio as gr
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.vqvae.model import VQVAE
from models.transformer.model import SpriteTransformer
from models.transformer.train import CLASS_VOCAB, ACTION_VOCAB, DIRECTION_VOCAB


def load_models(hf_repo: str, hf_token: str = None, device: str = "cpu"):
    vqvae = VQVAE().to(device)
    vqvae_path = hf_hub_download(hf_repo, "vqvae_latest.pt", token=hf_token)
    vqvae.load_state_dict(torch.load(vqvae_path, map_location=device)["model_state"])
    vqvae.eval()

    transformer = SpriteTransformer(
        vocab_size=vqvae.quantizer.num_embeddings,
        condition_vocab_size=64,
    ).to(device)
    transformer_path = hf_hub_download(hf_repo, "transformer_latest.pt", token=hf_token)
    transformer.load_state_dict(torch.load(transformer_path, map_location=device)["model_state"])
    transformer.eval()

    return vqvae, transformer


def encode_condition(value: str, vocab: list) -> int:
    try:
        return vocab.index(value)
    except ValueError:
        return 0


def generate_sprite(
    vqvae, transformer,
    class_name: str, action: str, direction: str,
    temperature: float = 1.0,
    top_k: int = 40,
    top_p: float = 0.9,
    device: str = "cpu",
) -> Image.Image:
    class_id = torch.tensor([encode_condition(class_name, CLASS_VOCAB)]).to(device)
    action_id = torch.tensor([encode_condition(action, ACTION_VOCAB)]).to(device)
    direction_id = torch.tensor([encode_condition(direction, DIRECTION_VOCAB)]).to(device)

    # Latent grid size: 8x8 for 32x32 input with 4x downsampling
    num_tokens = 64

    with torch.no_grad():
        indices = transformer.generate(
            class_id, action_id, direction_id,
            max_tokens=num_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        # Decode through VQ-VAE
        latent_shape = (vqvae.latent_dim, 8, 8)
        z = vqvae.quantizer.get_codebook_entry(indices.view(-1))
        z = z.view(-1, *latent_shape)
        recon = vqvae.decoder(z)

    img = recon[0].permute(1, 2, 0).cpu().numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img, "RGBA")


def build_demo(vqvae, transformer, device):
    def generate(class_name, action, direction, temperature, top_k, top_p):
        img = generate_sprite(
            vqvae, transformer,
            class_name, action, direction,
            temperature, top_k, top_p, device,
        )
        # Scale up for display
        img = img.resize((128, 128), Image.NEAREST)
        return img

    iface = gr.Interface(
        fn=generate,
        inputs=[
            gr.Dropdown(choices=CLASS_VOCAB, label="Character Class", value="character"),
            gr.Dropdown(choices=ACTION_VOCAB, label="Action", value="idle"),
            gr.Dropdown(choices=DIRECTION_VOCAB, label="Direction", value="front"),
            gr.Slider(0.1, 2.0, value=1.0, label="Temperature"),
            gr.Slider(1, 100, value=40, label="Top-K"),
            gr.Slider(0.0, 1.0, value=0.9, label="Top-P"),
        ],
        outputs=gr.Image(type="pil", label="Generated Sprite"),
        title="Sprite Generator",
        description="Generate pixel-art sprites using VQ-VAE + Transformer.",
    )

    return iface


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-repo", default="mani8148/sprite-generator-model")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    print(f"Loading models from {args.hf_repo}...")
    vqvae, transformer = load_models(args.hf_repo, args.hf_token, args.device)
    print("Models loaded!")

    iface = build_demo(vqvae, transformer, args.device)
    iface.launch(share=args.share)


if __name__ == "__main__":
    main()
