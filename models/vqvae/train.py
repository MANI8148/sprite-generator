"""
Training script for VQ-VAE on sprite dataset.
Saves checkpoints periodically and pushes to HF Hub.
"""
import os
import sys
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from datasets import load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm

from models.vqvae.model import VQVAE


class SpriteDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset_path: str, split: str = "train", image_size: int = 32):
        self.dataset = load_dataset(hf_dataset_path, split=split)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = item["image"].convert("RGBA")
        img_tensor = self.transform(img)  # (4, H, W), values in [0,1]
        return img_tensor


def save_reconstruction_grid(model, device, sample_batch, output_path, num_samples=8):
    """Save side-by-side comparison of original vs reconstructed."""
    model.eval()
    with torch.no_grad():
        batch = sample_batch[:num_samples].to(device)
        output = model(batch)
        recon = output["recon"].cpu()

    # Create comparison grid
    import numpy as np
    imgs = []
    for i in range(num_samples):
        orig = batch[i].cpu().permute(1, 2, 0).numpy()
        rec = recon[i].permute(1, 2, 0).numpy()
        combined = np.concatenate([orig, rec], axis=1)
        imgs.append(combined)

    grid = np.concatenate(imgs, axis=0)
    grid = (grid * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(grid, "RGBA").save(output_path)
    model.train()


def main():
    parser = argparse.ArgumentParser(description="Train VQ-VAE")
    parser.add_argument("--dataset", required=True, help="HF Dataset path")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--num-embeddings", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--checkpoint-dir", default="checkpoints/vqvae")
    parser.add_argument("--hf-repo", default=None, help="HF model repo to push to")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")

    # Data
    dataset = SpriteDataset(args.dataset, image_size=args.image_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    print(f"Dataset size: {len(dataset)}")

    # Model
    model = VQVAE(
        in_channels=4,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_embeddings=args.num_embeddings,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    start_epoch = 0
    global_step = 0

    # Resume
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]
        print(f"Resumed from epoch {start_epoch}")

    # HF login — not needed, using HfApi.upload_file(token=...) directly

    # Training loop
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Get a fixed sample batch for reconstruction visualization
    vis_batch = next(iter(dataloader))

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0
        total_recon_loss = 0
        total_vq_loss = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()

            output = model(batch)
            loss = output["loss"]
            loss.backward()
            optimizer.step()

            total_loss += output["loss"].item()
            total_recon_loss += output["recon_loss"].item()
            total_vq_loss += output["vq_loss"].item()
            global_step += 1

            pbar.set_postfix({
                "loss": output["loss"].item(),
                "recon": output["recon_loss"].item(),
                "vq": output["vq_loss"].item(),
            })

        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon_loss / len(dataloader)
        avg_vq = total_vq_loss / len(dataloader)

        print(f"Epoch {epoch+1}: loss={avg_loss:.6f}, recon={avg_recon:.6f}, vq={avg_vq:.6f}")

        # Save checkpoint
        ckpt_path = checkpoint_dir / f"vqvae_epoch_{epoch+1:03d}.pt"
        torch.save({
            "epoch": epoch,
            "global_step": global_step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": avg_loss,
            "config": {
                "image_size": args.image_size,
                "hidden_dim": args.hidden_dim,
                "latent_dim": args.latent_dim,
                "num_embeddings": args.num_embeddings,
            },
        }, ckpt_path)

        # Save reconstruction visualization every 10 epochs
        if (epoch + 1) % 10 == 0:
            vis_path = checkpoint_dir / f"recon_epoch_{epoch+1:03d}.png"
            save_reconstruction_grid(model, device, vis_batch, vis_path)

        # Push to HF Hub
        if args.hf_repo and args.hf_token:
            from huggingface_hub import HfApi
            api = HfApi()
            api.upload_file(
                path_or_fileobj=str(ckpt_path),
                path_in_repo=f"vqvae_epoch_{epoch+1:03d}.pt",
                repo_id=args.hf_repo,
                repo_type="model",
                token=args.hf_token,
            )
            api.upload_file(
                path_or_fileobj=str(ckpt_path),
                path_in_repo="vqvae_latest.pt",
                repo_id=args.hf_repo,
                repo_type="model",
                token=args.hf_token,
            )

    print("Training complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
