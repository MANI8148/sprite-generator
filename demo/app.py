"""
Gradio demo for sprite generator.
Loads trained VQ-VAE + Transformer from HF Hub and generates sprites.
Designed for deployment on Hugging Face Spaces.
"""
import os
import sys
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


HF_REPO = os.environ.get("HF_REPO", "darklord8777/sprite-generator-model")
HF_TOKEN = os.environ.get("HF_TOKEN", None)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_models(hf_repo: str = HF_REPO, hf_token: str = None, device: str = DEVICE):
    vqvae_ckpt = torch.load(hf_hub_download(hf_repo, "vqvae_latest.pt", token=hf_token), map_location=device)
    num_emb = vqvae_ckpt.get("config", {}).get("num_embeddings")
    if num_emb is None:
        num_emb = vqvae_ckpt["model_state"]["quantizer.embedding.weight"].size(0)
    vqvae = VQVAE(num_embeddings=num_emb).to(device)
    vqvae.load_state_dict(vqvae_ckpt["model_state"])
    vqvae.eval()

    t_ckpt = torch.load(hf_hub_download(hf_repo, "transformer_latest.pt", token=hf_token), map_location=device)
    sd = t_ckpt["model_state"]
    cfg = t_ckpt.get("config", {})
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
    device: str = DEVICE,
) -> Image.Image:
    class_id = torch.tensor([encode_condition(class_name, CLASS_VOCAB)]).to(device)
    action_id = torch.tensor([encode_condition(action, ACTION_VOCAB)]).to(device)
    direction_id = torch.tensor([encode_condition(direction, DIRECTION_VOCAB)]).to(device)

    num_tokens = 64

    with torch.no_grad():
        indices = transformer.generate(
            class_id, action_id, direction_id,
            max_tokens=num_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        recon = vqvae.decode_from_indices(indices, (vqvae.latent_dim, 8, 8))

    img = recon[0].permute(1, 2, 0).cpu().numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img, "RGBA")


def generate_grid(
    vqvae, transformer, device, rows=4, cols=4,
    class_name="character", action="idle", direction="front",
    temperature=1.0, top_k=40, top_p=0.9,
) -> Image.Image:
    total = rows * cols
    sprite_size = 32
    canvas = Image.new("RGBA", (cols * sprite_size, rows * sprite_size), (0, 0, 0, 0))

    temps = [max(0.4, min(2.0, temperature * (0.6 + 0.4 * (i % cols) / max(cols - 1, 1)))) for i in range(total)]

    for i in range(total):
        cls = class_name
        act = action
        dire = direction

        class_id = torch.tensor([encode_condition(cls, CLASS_VOCAB)]).to(device)
        action_id = torch.tensor([encode_condition(act, ACTION_VOCAB)]).to(device)
        direction_id = torch.tensor([encode_condition(dire, DIRECTION_VOCAB)]).to(device)

        with torch.no_grad():
            indices = transformer.generate(
                class_id, action_id, direction_id,
                max_tokens=64,
                temperature=temps[i],
                top_k=top_k,
                top_p=top_p,
            )

        recon = vqvae.decode_from_indices(indices, (vqvae.latent_dim, 8, 8))
        img_arr = recon[0].permute(1, 2, 0).cpu().detach().numpy()
        img_arr = (img_arr * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(img_arr, "RGBA")

        row, col = divmod(i, cols)
        canvas.paste(img, (col * sprite_size, row * sprite_size))

    return canvas


def build_demo(vqvae, transformer, device=DEVICE):
    def generate_single(class_name, action, direction, temperature, top_k, top_p):
        img = generate_sprite(
            vqvae, transformer,
            class_name, action, direction,
            temperature, top_k, top_p, device,
        )
        img = img.resize((128, 128), Image.NEAREST)
        return img

    def generate_grid_tab(class_name, action, direction, temperature, top_k, top_p):
        img = generate_grid(
            vqvae, transformer, device,
            rows=4, cols=4,
            class_name=class_name, action=action, direction=direction,
            temperature=temperature, top_k=top_k, top_p=top_p,
        )
        img = img.resize((512, 512), Image.NEAREST)
        return img

    single_tab = gr.Interface(
        fn=generate_single,
        inputs=[
            gr.Dropdown(choices=CLASS_VOCAB, label="Character Class", value="character"),
            gr.Dropdown(choices=ACTION_VOCAB, label="Action", value="idle"),
            gr.Dropdown(choices=DIRECTION_VOCAB, label="Direction", value="front"),
            gr.Slider(0.1, 2.0, value=1.0, label="Temperature"),
            gr.Slider(1, 100, value=40, label="Top-K"),
            gr.Slider(0.0, 1.0, value=0.9, label="Top-P"),
        ],
        outputs=gr.Image(type="pil", label="Generated Sprite"),
        title="Single Sprite",
    )

    grid_tab = gr.Interface(
        fn=generate_grid_tab,
        inputs=[
            gr.Dropdown(choices=CLASS_VOCAB, label="Character Class", value="character"),
            gr.Dropdown(choices=ACTION_VOCAB, label="Action", value="idle"),
            gr.Dropdown(choices=DIRECTION_VOCAB, label="Direction", value="front"),
            gr.Slider(0.1, 2.0, value=1.0, label="Base Temperature"),
            gr.Slider(1, 100, value=40, label="Top-K"),
            gr.Slider(0.0, 1.0, value=0.9, label="Top-P"),
        ],
        outputs=gr.Image(type="pil", label="Generated Grid"),
        title="Grid (4x4)",
    )

    demo = gr.TabbedInterface(
        [single_tab, grid_tab],
        tab_names=["Single", "Grid"],
        title="Sprite Generator",
    )

    return demo


# Load on module import for HF Spaces
try:
    vqvae_model, transformer_model = load_models(HF_REPO, HF_TOKEN, DEVICE)
    demo = build_demo(vqvae_model, transformer_model, DEVICE)
except Exception:
    # If models aren't available (e.g. in tests), provide a fallback placeholder
    vqvae_model = None
    transformer_model = None

    def placeholder_generate(*args, **kwargs):
        return Image.new("RGBA", (128, 128), (0, 0, 0, 0))

    demo = gr.Interface(
        fn=placeholder_generate,
        inputs=[
            gr.Dropdown(choices=CLASS_VOCAB, label="Character Class", value="character"),
            gr.Dropdown(choices=ACTION_VOCAB, label="Action", value="idle"),
            gr.Dropdown(choices=DIRECTION_VOCAB, label="Direction", value="front"),
            gr.Slider(0.1, 2.0, value=1.0, label="Temperature"),
            gr.Slider(1, 100, value=40, label="Top-K"),
            gr.Slider(0.0, 1.0, value=0.9, label="Top-P"),
        ],
        outputs=gr.Image(type="pil", label="Generated Sprite"),
        title="Sprite Generator (offline)",
        description="Models not loaded. Set HF_REPO to a valid model repository.",
    )


if __name__ == "__main__":
    demo.launch()