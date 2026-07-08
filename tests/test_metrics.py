import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from eval.metrics import palette_adherence_rate, grid_alignment_check, compute_reconstruction_loss
from models.vqvae.model import VQVAE


class TestPaletteAdherenceRate:
    @pytest.fixture
    def palette(self):
        return [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 255, 255),
            (0, 0, 0),
        ]

    def test_perfect_adherence(self, palette):
        arr = np.zeros((32, 32, 4), dtype=np.uint8)
        arr[:, :, :3] = [255, 0, 0]
        arr[:, :, 3] = 255
        img = Image.fromarray(arr, "RGBA")
        score = palette_adherence_rate(img, palette)
        assert score == 1.0

    def test_zero_adherence(self, palette):
        arr = np.zeros((32, 32, 4), dtype=np.uint8)
        arr[:, :, :3] = [128, 64, 32]
        arr[:, :, 3] = 255
        img = Image.fromarray(arr, "RGBA")
        score = palette_adherence_rate(img, palette)
        assert score == 0.0

    def test_partial_adherence(self, palette):
        arr = np.zeros((32, 32, 4), dtype=np.uint8)
        arr[:16, :, :3] = [255, 0, 0]
        arr[16:, :, :3] = [128, 64, 32]
        arr[:, :, 3] = 255
        img = Image.fromarray(arr, "RGBA")
        score = palette_adherence_rate(img, palette)
        assert 0.4 < score < 0.6

    def test_transparent_pixels_ignored(self, palette):
        arr = np.zeros((32, 32, 4), dtype=np.uint8)
        arr[:, :, :3] = [128, 64, 32]
        arr[:, :, 3] = 0
        img = Image.fromarray(arr, "RGBA")
        score = palette_adherence_rate(img, palette)
        assert score == 1.0

    def test_empty_image(self, palette):
        img = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
        score = palette_adherence_rate(img, palette)
        assert score == 1.0


class TestGridAlignmentCheck:
    def test_hard_edges_score_one(self):
        arr = np.zeros((32, 32, 4), dtype=np.uint8)
        arr[:, :] = [255, 0, 0, 255]
        img = Image.fromarray(arr, "RGBA")
        score = grid_alignment_check(img, 32)
        assert score == 1.0

    def test_smooth_edges_score_lower(self):
        arr = np.zeros((32, 32, 4), dtype=np.uint8)
        arr[:, :, :3] = [255, 0, 0]
        arr[5:25, 5:25, 3] = 128
        arr[5:25, 5:25, :3] = [255, 0, 0]
        arr[:, :, 3] = 255
        arr[10:20, 10:20, 3] = 200
        img = Image.fromarray(arr, "RGBA")
        score = grid_alignment_check(img, 32)
        assert score < 1.0

    def test_fully_transparent(self):
        img = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
        score = grid_alignment_check(img, 32)
        assert score == 1.0

    def test_rgb_image_no_alpha(self):
        arr = np.zeros((32, 32, 3), dtype=np.uint8)
        arr[:, :] = [255, 0, 0]
        img = Image.fromarray(arr, "RGB")
        score = grid_alignment_check(img, 32)
        assert score == 1.0


class TestComputeReconstructionLoss:
    def _make_loader(self, data_tensor, batch_size=2):
        ds = torch.utils.data.TensorDataset(data_tensor)
        orig_fn = torch.utils.data.dataloader.default_collate
        return DataLoader(ds, batch_size=batch_size, collate_fn=lambda b: orig_fn(b)[0])

    def test_returns_float(self):
        model = VQVAE(in_channels=4, hidden_dim=16, latent_dim=8, num_embeddings=16)
        data = torch.randn(4, 4, 32, 32)
        dataloader = self._make_loader(data)
        loss = compute_reconstruction_loss(model, dataloader, "cpu")
        assert isinstance(loss, float)

    def test_non_negative(self):
        model = VQVAE(in_channels=4, hidden_dim=16, latent_dim=8, num_embeddings=16)
        data = torch.randn(4, 4, 32, 32)
        dataloader = self._make_loader(data)
        loss = compute_reconstruction_loss(model, dataloader, "cpu")
        assert loss >= 0.0

    def test_identical_inputs_produce_same_loss(self):
        model = VQVAE(in_channels=4, hidden_dim=16, latent_dim=8, num_embeddings=16)
        model.eval()
        data = torch.ones(2, 4, 32, 32)
        dataloader = self._make_loader(data)
        loss1 = compute_reconstruction_loss(model, dataloader, "cpu")
        loss2 = compute_reconstruction_loss(model, dataloader, "cpu")
        assert loss1 == pytest.approx(loss2, abs=1e-6)


class TestMainEntryPoint:
    def test_main_runs_and_writes_output(self, monkeypatch, tmp_path):
        from eval.metrics import main

        output_path = tmp_path / "eval_results.json"
        checkpoint_path = tmp_path / "vqvae.pt"
        palette_path = tmp_path / "palette.json"

        model = VQVAE(num_embeddings=16)
        torch.save({
            "model_state": model.state_dict(),
            "config": {"num_embeddings": 16},
        }, checkpoint_path)

        palette = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
        with open(palette_path, "w") as f:
            json.dump(palette, f)

        class FakeItem:
            def __init__(self, idx=None):
                self.idx = idx
            def __getitem__(self, key):
                arr = np.zeros((32, 32, 4), dtype=np.uint8)
                arr[:, :, :3] = [255, 0, 0]
                arr[:, :, 3] = 255
                return Image.fromarray(arr, "RGBA")
            @property
            def image(self):
                return self.__getitem__("image")

        class FakeDataset:
            def __init__(self, n):
                self.n = n
            def __len__(self):
                return self.n
            def __getitem__(self, idx):
                return FakeItem(idx)
            def __iter__(self):
                for i in range(self.n):
                    yield FakeItem(i)

        monkeypatch.setattr(
            "eval.metrics.load_dataset",
            lambda path, split: FakeDataset(4),
        )

        test_args = [
            "prog",
            "--dataset", "fake/dataset",
            "--vqvae-checkpoint", str(checkpoint_path),
            "--palette", str(palette_path),
            "--output", str(output_path),
            "--num-samples", "4",
        ]
        monkeypatch.setattr("sys.argv", test_args)
        main()

        assert output_path.exists()
        with open(output_path) as f:
            results = json.load(f)

        assert "palette_adherence_mean" in results
        assert "grid_alignment_mean" in results
        assert "reconstruction_loss" in results
        assert results["num_samples"] == 4
        assert results["palette_size"] == 3
        assert isinstance(results["reconstruction_loss"], float)

    def test_main_without_vqvae_skips_recon_loss(self, monkeypatch, tmp_path):
        from eval.metrics import main

        output_path = tmp_path / "eval_results.json"
        palette_path = tmp_path / "palette.json"

        palette = [[255, 0, 0], [0, 255, 0]]
        with open(palette_path, "w") as f:
            json.dump(palette, f)

        class FakeItem:
            def __init__(self, idx=None):
                self.idx = idx
            def __getitem__(self, key):
                arr = np.zeros((32, 32, 4), dtype=np.uint8)
                arr[:, :, :3] = [255, 0, 0]
                arr[:, :, 3] = 255
                return Image.fromarray(arr, "RGBA")
            @property
            def image(self):
                return self.__getitem__("image")

        class FakeDataset:
            def __init__(self):
                self.items = [FakeItem() for _ in range(2)]
            def __len__(self):
                return len(self.items)
            def __getitem__(self, idx):
                return self.items[idx]
            def __iter__(self):
                return iter(self.items)

        monkeypatch.setattr(
            "eval.metrics.load_dataset",
            lambda path, split: FakeDataset(),
        )

        test_args = [
            "prog",
            "--dataset", "fake/dataset",
            "--palette", str(palette_path),
            "--output", str(output_path),
            "--num-samples", "2",
        ]
        monkeypatch.setattr("sys.argv", test_args)
        main()

        with open(output_path) as f:
            results = json.load(f)

        assert results["reconstruction_loss"] is None
        assert results["num_samples"] == 2
        assert results["palette_size"] == 2
