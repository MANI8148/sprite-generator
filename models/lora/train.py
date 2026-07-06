import os
import sys
import argparse
import json
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from datasets import load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm

from models.lora.model import SpriteLoRAWrapper


CLASS_VOCAB = [
    "character", "enemy", "item", "weapon", "vehicle", "animal", "plant",
    "tile", "ui_element", "projectile", "furniture", "decoration", "food",
    "tool", "accessory", "building", "rune", "orb", "potion", "key",
]
ACTION_VOCAB = [
    "idle", "walk", "run", "attack", "jump", "hurt", "death", "block",
    "shoot", "cast", "interact", "fly", "swim", "climb",
]
DIRECTION_VOCAB = ["front", "back", "left", "right", "front_left", "front_right", "back_left", "back_right"]


class LoRASpriteDataset(torch.utils.data.Dataset):
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
        img_tensor = self.transform(img)
        return img_tensor


def main():
    parser = argparse.ArgumentParser(description="Train LoRA sprite generator")
    parser.add_argument("--dataset", required=True, help="HF Dataset path")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--checkpoint-dir", default="checkpoints/lora")
    parser.add_argument("--hf-repo", default=None, help="HF model repo to push to")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")

    dataset = LoRASpriteDataset(args.dataset, image_size=args.image_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    print(f"Dataset size: {len(dataset)}")

    model = SpriteLoRAWrapper(rank=args.rank, alpha=args.alpha).to(device)
    print(f"LoRA trainable parameters: {model.trainable_parameters():,}")

    lora_params = model.lora_parameters()
    optimizer = optim.AdamW(lora_params, lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    global_step = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]
        print(f"Resumed from epoch {start_epoch}")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    loss_fn = torch.nn.MSELoss()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()

            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            optimizer.step()

            total_loss += loss.item()
            global_step += 1
            pbar.set_postfix({"loss": loss.item()})

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}: loss={avg_loss:.6f}")

        ckpt_path = checkpoint_dir / f"lora_epoch_{epoch+1:03d}.pt"
        torch.save({
            "epoch": epoch,
            "global_step": global_step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": avg_loss,
            "config": {
                "image_size": args.image_size,
                "rank": args.rank,
                "alpha": args.alpha,
            },
        }, ckpt_path)

        if args.hf_repo and args.hf_token:
            api = HfApi()
            api.upload_file(
                path_or_fileobj=str(ckpt_path),
                path_in_repo=f"lora_epoch_{epoch+1:03d}.pt",
                repo_id=args.hf_repo,
                repo_type="model",
                token=args.hf_token,
            )
            api.upload_file(
                path_or_fileobj=str(ckpt_path),
                path_in_repo="lora_latest.pt",
                repo_id=args.hf_repo,
                repo_type="model",
                token=args.hf_token,
            )

    print("LoRA training complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
