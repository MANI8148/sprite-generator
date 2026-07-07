"""Tests for VQ-VAE training loop components."""
import torch
from pathlib import Path
from PIL import Image

from models.vqvae.model import VQVAE
from models.vqvae.train import save_reconstruction_grid


class TestSaveReconstructionGrid:
    def test_saves_png(self, tmp_path):
        model = VQVAE(in_channels=4, hidden_dim=32, latent_dim=16, num_embeddings=32)
        batch = torch.randn(4, 4, 32, 32)
        output_path = tmp_path / "recon.png"
        save_reconstruction_grid(model, "cpu", batch, output_path, num_samples=2)
        assert output_path.exists()
        img = Image.open(output_path)
        assert img.mode == "RGBA"

    def test_model_returns_to_train_mode(self, tmp_path):
        model = VQVAE(in_channels=4, hidden_dim=32, latent_dim=16, num_embeddings=32)
        model.train()
        batch = torch.randn(4, 4, 32, 32)
        output_path = tmp_path / "recon.png"
        save_reconstruction_grid(model, "cpu", batch, output_path)
        assert model.training

    def test_output_grid_size(self, tmp_path):
        model = VQVAE(in_channels=4, hidden_dim=32, latent_dim=16, num_embeddings=32)
        batch = torch.randn(8, 4, 32, 32)
        output_path = tmp_path / "recon.png"
        save_reconstruction_grid(model, "cpu", batch, output_path, num_samples=4)
        img = Image.open(output_path)
        assert img.size[0] == 64
        assert img.size[1] == 128


class TestCheckpointCycle:
    def test_save_and_load(self, tmp_path):
        model = VQVAE()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        ckpt_path = tmp_path / "test_checkpoint.pt"
        torch.save({
            "epoch": 5,
            "global_step": 100,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": 0.5,
        }, ckpt_path)

        model2 = VQVAE()
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
        checkpoint = torch.load(ckpt_path)
        model2.load_state_dict(checkpoint["model_state"])
        optimizer2.load_state_dict(checkpoint["optimizer_state"])

        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.equal(p1, p2), "Weights differ after save/load"

    def test_checkpoint_contains_metadata(self, tmp_path):
        model = VQVAE()
        ckpt_path = tmp_path / "meta.pt"
        data = {
            "epoch": 10,
            "global_step": 500,
            "model_state": model.state_dict(),
            "optimizer_state": torch.optim.Adam(model.parameters(), lr=1e-3).state_dict(),
            "loss": 0.25,
            "config": {
                "image_size": 32,
                "hidden_dim": 128,
                "latent_dim": 64,
                "num_embeddings": 256,
            },
        }
        torch.save(data, ckpt_path)
        loaded = torch.load(ckpt_path)
        assert loaded["epoch"] == 10
        assert loaded["global_step"] == 500
        assert loaded["loss"] == 0.25
        assert loaded["config"]["num_embeddings"] == 256


class TestTrainingSteps:
    def test_training_step_updates_weights(self):
        model = VQVAE(in_channels=4, hidden_dim=32, latent_dim=16, num_embeddings=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.randn(4, 4, 32, 32)

        initial_params = [p.clone() for p in model.parameters()]

        output = model(x)
        output["loss"].backward()
        optimizer.step()

        changed = any(
            not torch.equal(p1, p2)
            for p1, p2 in zip(initial_params, model.parameters())
        )
        assert changed, "No parameters changed after training step"

    def test_training_loss_decreases(self):
        model = VQVAE(in_channels=4, hidden_dim=32, latent_dim=16, num_embeddings=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.ones(4, 4, 32, 32)

        initial_loss = model(x)["loss"].item()

        for _ in range(50):
            optimizer.zero_grad()
            loss = model(x)["loss"]
            loss.backward()
            optimizer.step()

        final_loss = model(x)["loss"].item()
        assert final_loss < initial_loss, (
            f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"
        )

    def test_training_with_synthetic_dataset(self):
        class SyntheticDataset(torch.utils.data.Dataset):
            def __init__(self, size=16):
                self.size = size

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                return torch.randn(4, 32, 32)

        model = VQVAE(in_channels=4, hidden_dim=32, latent_dim=16, num_embeddings=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        dataset = SyntheticDataset(size=16)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)

        model.train()
        for batch in dataloader:
            optimizer.zero_grad()
            output = model(batch)
            output["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    def test_gradient_clipping_does_not_crash(self):
        model = VQVAE(in_channels=4, hidden_dim=32, latent_dim=16, num_embeddings=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.randn(2, 4, 32, 32)
        output = model(x)
        output["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    def test_training_with_lr_scheduler(self):
        model = VQVAE(in_channels=4, hidden_dim=32, latent_dim=16, num_embeddings=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        x = torch.ones(2, 4, 32, 32)

        initial_lr = optimizer.param_groups[0]["lr"]
        for _ in range(5):
            optimizer.zero_grad()
            loss = model(x)["loss"]
            loss.backward()
            optimizer.step()
            scheduler.step()

        assert optimizer.param_groups[0]["lr"] < initial_lr, "LR should decrease after scheduler steps"

    def test_training_with_ema(self):
        model = VQVAE(in_channels=4, hidden_dim=32, latent_dim=16, num_embeddings=32, commitment_cost=0.1)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.ones(2, 4, 32, 32)

        output = model(x)
        loss = output["loss"]
        loss.backward()
        optimizer.step()
        model.ema_update(x)

        assert model.quantizer.ema_count.sum().item() > 0, "EMA counts should be updated"

    def test_training_with_ema_and_scheduler(self):
        model = VQVAE(in_channels=4, hidden_dim=32, latent_dim=16, num_embeddings=32, commitment_cost=0.1)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        x = torch.ones(2, 4, 32, 32)

        initial_loss = model(x)["loss"].item()
        for _ in range(20):
            optimizer.zero_grad()
            output = model(x)
            output["loss"].backward()
            optimizer.step()
            model.ema_update(x)
            scheduler.step()

        final_loss = model(x)["loss"].item()
        assert final_loss < initial_loss, f"Loss did not decrease with EMA: {initial_loss:.4f} -> {final_loss:.4f}"

    def test_checkpoint_with_scheduler_state(self, tmp_path):
        model = VQVAE()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        ckpt_path = tmp_path / "scheduler_ckpt.pt"
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": 5,
            "config": {"use_ema": True, "decay": 0.99},
        }, ckpt_path)
        loaded = torch.load(ckpt_path)
        assert "scheduler_state" in loaded
        assert loaded["config"]["use_ema"] is True
