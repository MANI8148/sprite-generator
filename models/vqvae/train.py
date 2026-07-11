import os, sys, argparse, json, math, warnings
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
import numpy as np

warnings.filterwarnings("ignore")
from models.vqvae.model import ImprovedVQVAE


class SpriteDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset_path, split="train", image_size=32, augment=True):
        self.dataset = load_dataset(hf_dataset_path, split=split)
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = item["image"].convert("RGBA")
        img = img.resize((self.image_size, self.image_size), Image.NEAREST)
        import torchvision.transforms.functional as TF
        img_t = transforms.ToTensor()(img)
        if self.augment:
            rgb, a = img_t[:3], img_t[3:]
            if torch.rand(1).item() > 0.3:
                rgb = TF.adjust_brightness(rgb, 1.0 + torch.empty(1).uniform_(-0.1, 0.1).item())
            if torch.rand(1).item() > 0.3:
                rgb = TF.adjust_contrast(rgb, 1.0 + torch.empty(1).uniform_(-0.1, 0.1).item())
            img_t = torch.cat([rgb, a])
        return img_t


def save_reconstruction_grid(model, device, sample_batch, output_path, num_samples=8):
    model.eval()
    with torch.no_grad():
        actual = min(num_samples, sample_batch.size(0))
        batch = sample_batch[:actual].to(device)
        out = model(batch)
        recon = out["recon"].cpu()
    imgs = []
    for i in range(actual):
        orig = batch[i].cpu().permute(1, 2, 0).numpy()
        rec = recon[i].permute(1, 2, 0).numpy()
        imgs.append(np.concatenate([orig, rec], axis=1))
    grid = np.concatenate(imgs, axis=0)
    grid = (grid * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(grid, "RGBA").save(output_path)
    model.train()


def load_palette(dataset_path):
    try:
        with open(dataset_path) as f:
            return json.load(f)
    except Exception:
        return []


def main():
    parser = argparse.ArgumentParser(description="Train Improved VQ-VAE")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=96)
    parser.add_argument("--num-embeddings", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--checkpoint-dir", default="checkpoints/vqvae")
    parser.add_argument("--hf-repo", default=None)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--palette-path", default="data/processed/palette.json")
    parser.add_argument("--lambda-perc", type=float, default=0.5, help="Perceptual loss weight")
    parser.add_argument("--lambda-ffl", type=float, default=0.1, help="Focal frequency loss weight")
    parser.add_argument("--lambda-edge", type=float, default=0.05, help="Sobel edge loss weight")
    parser.add_argument("--lambda-palette", type=float, default=0.1, help="Palette histogram loss weight")
    parser.add_argument("--lambda-adv", type=float, default=0.1, help="Adversarial loss weight")
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    dataset = SpriteDataset(args.dataset, image_size=args.image_size, augment=not args.no_augment)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(f"Dataset: {len(dataset)} sprites, {len(dataloader)} batches/epoch")

    model = ImprovedVQVAE(
        in_channels=4,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_embeddings=args.num_embeddings,
    ).to(device)

    palette = load_palette(args.palette_path)
    if palette:
        print(f"Loaded palette: {len(palette)} colors")

    g_params = list(model.encoder.parameters()) + list(model.decoder.parameters()) + list(model.quantizer.parameters())
    optimizer_g = optim.AdamW(g_params, lr=args.lr, weight_decay=1e-4, betas=(0.5, 0.9))
    optimizer_d = optim.AdamW(model.discriminator.parameters(), lr=args.lr * 0.5, weight_decay=1e-4, betas=(0.5, 0.9))

    def warmup_cosine(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        return 0.5 * (1 + math.cos((epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs) * math.pi))
    scheduler_g = optim.lr_scheduler.LambdaLR(optimizer_g, warmup_cosine)
    scheduler_d = optim.lr_scheduler.LambdaLR(optimizer_d, warmup_cosine)

    start_epoch = 0
    global_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer_g.load_state_dict(ckpt["optimizer_g_state"])
        optimizer_d.load_state_dict(ckpt["optimizer_d_state"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", 0)
        print(f"Resumed from epoch {start_epoch}")

    model.init_ema()
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    vis_batch = next(iter(dataloader))
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_g_loss = total_recon = total_vq = total_perc = total_ffl = total_edge = total_adv = 0
        total_d_real = total_d_fake = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            batch = batch.to(device)
            B = batch.size(0)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out = model(batch)
                recon = out["recon"]

                recon_loss = out["recon_loss"]
                vq_loss = out["vq_loss"]

                perc_loss = model.perceptual_loss(batch, recon) * args.lambda_perc
                ffl_loss = focal_frequency_loss(batch, recon) * args.lambda_ffl if args.lambda_ffl > 0 else 0
                edge_loss = sobel_edge_loss(batch, recon) * args.lambda_edge if args.lambda_edge > 0 else 0
                pal_loss = 0
                if args.lambda_palette > 0 and palette:
                    pal_loss = palette_histogram_loss(batch, recon, palette) * args.lambda_palette

                d_fake = model.discriminator(recon)
                adv_loss = torch.zeros(1, device=device)
                if args.lambda_adv > 0:
                    adv_loss = F.relu(1 - d_fake).mean() * args.lambda_adv

                g_loss = recon_loss + vq_loss + perc_loss + ffl_loss + edge_loss + pal_loss + adv_loss

            optimizer_g.zero_grad()
            scaler.scale(g_loss).backward()
            scaler.unscale_(optimizer_g)
            torch.nn.utils.clip_grad_norm_(g_params, 1.0)
            scaler.step(optimizer_g)

            if args.lambda_adv > 0 and epoch >= 2:
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    with torch.no_grad():
                        recon_detach = model(batch)["recon"].detach()
                    d_real = model.discriminator(batch)
                    d_fake_d = model.discriminator(recon_detach)
                    d_loss = F.relu(1 - d_real).mean() + F.relu(1 + d_fake_d).mean()

                optimizer_d.zero_grad()
                scaler.scale(d_loss).backward()
                scaler.step(optimizer_d)
                total_d_real += d_real.mean().item()
                total_d_fake += d_fake_d.mean().item()

            scaler.update()

            total_g_loss += g_loss.item()
            total_recon += recon_loss.item()
            total_vq += vq_loss.item()
            total_perc += perc_loss.item() if isinstance(perc_loss, torch.Tensor) else 0
            total_ffl += ffl_loss.item() if isinstance(ffl_loss, torch.Tensor) else 0
            total_edge += edge_loss.item() if isinstance(edge_loss, torch.Tensor) else 0
            total_adv += adv_loss.item() if isinstance(adv_loss, torch.Tensor) else 0
            global_step += 1

            pbar.set_postfix({
                "loss": f"{g_loss.item():.4f}",
                "recon": f"{recon_loss.item():.4f}",
                "perc": f"{perc_loss.item() if isinstance(perc_loss, torch.Tensor) else 0:.4f}",
            })

        scheduler_g.step()
        scheduler_d.step()

        n = len(dataloader)
        print(f"Epoch {epoch+1}: G={total_g_loss/n:.4f} recon={total_recon/n:.4f} "
              f"vq={total_vq/n:.4f} perc={total_perc/n:.4f} ffl={total_ffl/n:.4f} "
              f"edge={total_edge/n:.4f} adv={total_adv/n:.4f} "
              f"lr={scheduler_g.get_last_lr()[0]:.2e}")

        model.update_ema()
        model.reset_dead_codes(batch)

        ckpt_path = checkpoint_dir / f"vqvae_epoch_{epoch+1:03d}.pt"
        torch.save({
            "epoch": epoch,
            "global_step": global_step,
            "model_state": model.state_dict(),
            "optimizer_g_state": optimizer_g.state_dict(),
            "optimizer_d_state": optimizer_d.state_dict(),
            "loss": total_g_loss / n,
            "config": {
                "image_size": args.image_size,
                "hidden_dim": args.hidden_dim,
                "latent_dim": args.latent_dim,
                "num_embeddings": args.num_embeddings,
            },
        }, ckpt_path)

        if (epoch + 1) % 10 == 0:
            vis_path = checkpoint_dir / f"recon_epoch_{epoch+1:03d}.png"
            save_reconstruction_grid(model, device, vis_batch, vis_path)

        if args.hf_repo and args.hf_token:
            api = HfApi()
            api.upload_file(
                path_or_fileobj=str(ckpt_path),
                path_in_repo="vqvae_latest.pt",
                repo_id=args.hf_repo,
                repo_type="model",
                token=args.hf_token,
            )
            print(f"  -> Pushed to HF (epoch {epoch+1})")

    if args.hf_repo and args.hf_token:
        api = HfApi(token=args.hf_token)
        api.upload_file(
            path_or_fileobj=json.dumps({
                "status": "complete",
                "vqvae_epochs": args.epochs,
                "config": {"hidden_dim": args.hidden_dim, "latent_dim": args.latent_dim, "num_embeddings": args.num_embeddings}
            }).encode(),
            path_in_repo="training_complete.json",
            repo_id=args.hf_repo,
            repo_type="model",
        )
    print("Training complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
