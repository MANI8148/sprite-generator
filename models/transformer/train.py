"""
Training script for the conditional transformer prior.
Trains on VQ-VAE encoded token sequences from the dataset.
"""
import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from huggingface_hub import HfApi, login
from tqdm import tqdm
from PIL import Image
import numpy as np

from models.vqvae.model import VQVAE
from models.transformer.model import SpriteTransformer


CLASS_VOCAB = ["unknown", "character", "item", "tile", "enemy", "player", "weapon", "food",
               "vehicle", "building", "decoration", "effect", "projectile", "animal", "plant",
               "furniture", "tool", "accessory", "ui_element", "terrain"]
ACTION_VOCAB = ["idle", "walk", "run", "attack", "jump", "hurt", "death", "block", "shoot",
                "cast", "interact", "fly", "swim", "climb"]
DIRECTION_VOCAB = ["front", "back", "left", "right", "front_left", "front_right", "back_left", "back_right"]


def encode_condition(value: str, vocab: list) -> int:
    try:
        return vocab.index(value)
    except ValueError:
        return 0


class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset_path: str, vqvae: VQVAE, device: torch.device,
                 split: str = "train", image_size: int = 32):
        self.dataset = load_dataset(hf_dataset_path, split=split)
        self.vqvae = vqvae
        self.device = device
        self.image_size = image_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = item["image"].convert("RGBA")
        img = img.resize((self.image_size, self.image_size), Image.NEAREST)
        img_tensor = torch.tensor(np.array(img).astype(np.float32) / 255.0).permute(2, 0, 1)

        with torch.no_grad():
            indices = self.vqvae.encode_to_indices(img_tensor.unsqueeze(0).to(self.device))
        tokens = indices.squeeze(0).cpu()  # (seq_len,)

        class_id = encode_condition(item.get("class", "unknown"), CLASS_VOCAB)
        action_id = encode_condition(item.get("action", "idle"), ACTION_VOCAB)
        direction_id = encode_condition(item.get("direction", "front"), DIRECTION_VOCAB)

        return tokens.cpu(), torch.tensor(class_id), torch.tensor(action_id), torch.tensor(direction_id)


def main():
    parser = argparse.ArgumentParser(description="Train transformer prior")
    parser.add_argument("--dataset", required=True, help="HF Dataset path")
    parser.add_argument("--vqvae-checkpoint", required=True, help="Trained VQ-VAE checkpoint")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--checkpoint-dir", default="checkpoints/transformer")
    parser.add_argument("--hf-repo", default=None, help="HF model repo")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load VQ-VAE
    vqvae = VQVAE().to(device)
    vqvae_checkpoint = torch.load(args.vqvae_checkpoint, map_location=device)
    vqvae.load_state_dict(vqvae_checkpoint["model_state"])
    vqvae.eval()
    print("VQ-VAE loaded")

    # Token dataset
    dataset = TokenDataset(args.dataset, vqvae, device)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                            collate_fn=lambda batch: (
                                torch.stack([b[0] for b in batch]),
                                torch.stack([b[1] for b in batch]),
                                torch.stack([b[2] for b in batch]),
                                torch.stack([b[3] for b in batch]),
                            ))
    print(f"Dataset size: {len(dataset)}")

    # Get token sequence length from first sample
    sample_tokens = dataset[0][0]
    max_seq_len = sample_tokens.shape[0]

    # Model
    model = SpriteTransformer(
        vocab_size=vqvae.quantizer.num_embeddings,
        condition_vocab_size=64,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_seq_len=max_seq_len + 1,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    start_epoch = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    if args.hf_repo and args.hf_token:
        login(token=args.hf_token)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for tokens, class_ids, action_ids, direction_ids in pbar:
            tokens = tokens.to(device)
            class_ids = class_ids.to(device)
            action_ids = action_ids.to(device)
            direction_ids = direction_ids.to(device)

            optimizer.zero_grad()
            logits = model(tokens, class_ids, action_ids, direction_ids)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tokens.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(dataloader)
        scheduler.step()

        print(f"Epoch {epoch+1}: loss={avg_loss:.6f}")

        ckpt_path = checkpoint_dir / f"transformer_epoch_{epoch+1:03d}.pt"
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": avg_loss,
            "config": {
                "d_model": args.d_model,
                "n_layers": args.n_layers,
                "n_heads": args.n_heads,
            },
        }, ckpt_path)

        if args.hf_repo and args.hf_token and (epoch + 1) % 10 == 0:
            model.push_to_hub(
                repo_id=args.hf_repo,
                commit_message=f"Transformer checkpoint epoch {epoch+1}",
                token=args.hf_token,
            )

    print("Training complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
