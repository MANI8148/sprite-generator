"""Tests for reconstruction validation (Phase 0 Item 2)."""
import json
import math
from pathlib import Path

import numpy as np
import torch
import pytest
from PIL import Image

from models.vqvae.model import VQVAE


class TestComputeMSE:
    def test_identical_arrays_zero_mse(self):
        from eval.reconstruction_validation import compute_mse
        arr = np.zeros((32, 32, 4), dtype=np.uint8)
        mse = compute_mse(arr, arr)
        assert mse == 0.0

    def test_different_arrays_positive_mse(self):
        from eval.reconstruction_validation import compute_mse
        a = np.zeros((4, 4, 3), dtype=np.uint8)
        b = np.ones((4, 4, 3), dtype=np.uint8) * 255
        mse = compute_mse(a, b)
        assert mse > 0.0

    def test_mse_scales_with_difference(self):
        from eval.reconstruction_validation import compute_mse
        a = np.zeros((4, 4, 3), dtype=np.uint8)
        b = np.ones((4, 4, 3), dtype=np.uint8) * 128
        c = np.ones((4, 4, 3), dtype=np.uint8) * 255
        mse_small = compute_mse(a, b)
        mse_large = compute_mse(a, c)
        assert mse_small < mse_large

    def test_float64_output(self):
        from eval.reconstruction_validation import compute_mse
        a = np.zeros((4, 4, 4), dtype=np.uint8)
        b = np.ones((4, 4, 4), dtype=np.uint8) * 128
        mse = compute_mse(a, b)
        assert isinstance(mse, float)


class TestComputePSNR:
    def test_identical_arrays_infinite_psnr(self):
        from eval.reconstruction_validation import compute_psnr
        arr = np.zeros((32, 32, 4), dtype=np.uint8)
        psnr = compute_psnr(arr, arr)
        assert psnr == float("inf")

    def test_different_arrays_finite_psnr(self):
        from eval.reconstruction_validation import compute_psnr
        a = np.zeros((4, 4, 3), dtype=np.uint8)
        b = np.ones((4, 4, 3), dtype=np.uint8) * 128
        psnr = compute_psnr(a, b)
        assert math.isfinite(psnr)

    def test_higher_mse_gives_lower_psnr(self):
        from eval.reconstruction_validation import compute_psnr
        a = np.zeros((4, 4, 3), dtype=np.uint8)
        b = np.ones((4, 4, 3), dtype=np.uint8) * 128
        c = np.ones((4, 4, 3), dtype=np.uint8) * 255
        psnr_128 = compute_psnr(a, b)
        psnr_255 = compute_psnr(a, c)
        assert psnr_128 > psnr_255


class TestComputeMetrics:
    def test_identical_images_perfect_metrics(self):
        from eval.reconstruction_validation import compute_metrics
        arr = np.zeros((32, 32, 4), dtype=np.uint8)
        arr[:, :, 3] = 255
        img = Image.fromarray(arr, "RGBA")
        metrics = compute_metrics(img, img)
        assert metrics["mse"] == 0.0
        assert metrics["psnr"] == float("inf")

    def test_transparent_image_zero_pixels(self):
        from eval.reconstruction_validation import compute_metrics
        img = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
        metrics = compute_metrics(img, img)
        assert metrics["pixel_count"] == 0
        assert metrics["mse"] == 0.0

    def test_different_images_non_zero_mse(self):
        from eval.reconstruction_validation import compute_metrics
        a_arr = np.zeros((32, 32, 4), dtype=np.uint8)
        a_arr[:, :, :3] = [255, 0, 0]
        a_arr[:, :, 3] = 255
        b_arr = np.zeros((32, 32, 4), dtype=np.uint8)
        b_arr[:, :, :3] = [0, 255, 0]
        b_arr[:, :, 3] = 255
        img_a = Image.fromarray(a_arr, "RGBA")
        img_b = Image.fromarray(b_arr, "RGBA")
        metrics = compute_metrics(img_a, img_b)
        assert metrics["mse"] > 0.0
        assert math.isfinite(metrics["psnr"])

    def test_returns_expected_keys(self):
        from eval.reconstruction_validation import compute_metrics
        arr = np.zeros((32, 32, 4), dtype=np.uint8)
        arr[:, :, 3] = 255
        img = Image.fromarray(arr, "RGBA")
        metrics = compute_metrics(img, img)
        assert "mse" in metrics
        assert "psnr" in metrics
        assert "pixel_count" in metrics


class TestReconstructImage:
    def test_returns_rgba_image(self):
        from eval.reconstruction_validation import reconstruct_image
        vqvae = VQVAE(in_channels=4, hidden_dim=16, latent_dim=8, num_embeddings=64)
        img = Image.new("RGBA", (32, 32), (255, 0, 0, 255))
        recon = reconstruct_image(vqvae, img, "cpu")
        assert isinstance(recon, Image.Image)
        assert recon.mode == "RGBA"

    def test_output_same_size_as_input(self):
        from eval.reconstruction_validation import reconstruct_image
        vqvae = VQVAE(in_channels=4, hidden_dim=16, latent_dim=8, num_embeddings=64)
        img = Image.new("RGBA", (64, 64), (255, 0, 0, 255))
        recon = reconstruct_image(vqvae, img, "cpu")
        assert recon.size == (32, 32)

    def test_different_inputs_produce_different_outputs(self):
        from eval.reconstruction_validation import reconstruct_image
        vqvae = VQVAE(in_channels=4, hidden_dim=16, latent_dim=8, num_embeddings=64)
        img_a = Image.new("RGBA", (32, 32), (255, 0, 0, 255))
        img_b = Image.new("RGBA", (32, 32), (0, 255, 0, 255))
        recon_a = reconstruct_image(vqvae, img_a, "cpu")
        recon_b = reconstruct_image(vqvae, img_b, "cpu")
        a_arr = np.array(recon_a)
        b_arr = np.array(recon_b)
        assert not np.array_equal(a_arr, b_arr)


class TestCreateComparisonGrid:
    def test_returns_image(self):
        from eval.reconstruction_validation import create_comparison_grid
        img = Image.new("RGBA", (32, 32), (255, 0, 0, 255))
        metrics = {"mse": 0.0, "psnr": float("inf"), "pixel_count": 1024}
        grid = create_comparison_grid([img], [img], [metrics], title="Test")
        assert isinstance(grid, Image.Image)

    def test_multiple_pairs(self):
        from eval.reconstruction_validation import create_comparison_grid
        img = Image.new("RGBA", (32, 32), (255, 0, 0, 255))
        metrics = {"mse": 0.0, "psnr": float("inf"), "pixel_count": 1024}
        grid = create_comparison_grid([img, img], [img, img], [metrics, metrics], cells_per_row=2)
        w, h = grid.size
        assert w > 0
        assert h > 0

    def test_empty_list_raises_error(self):
        from eval.reconstruction_validation import create_comparison_grid
        with pytest.raises(ValueError, match="At least one image pair is required"):
            create_comparison_grid([], [], [])

    def test_title_appears_on_grid(self):
        from eval.reconstruction_validation import create_comparison_grid
        img = Image.new("RGBA", (32, 32), (255, 0, 0, 255))
        metrics = {"mse": 0.0, "psnr": float("inf"), "pixel_count": 1024}
        grid_with = create_comparison_grid([img], [img], [metrics], title="My Report")
        grid_without = create_comparison_grid([img], [img], [metrics], title="")
        assert grid_with.size[1] > grid_without.size[1]


class TestValidateReconstructions:
    def test_returns_grid_and_metrics(self):
        from eval.reconstruction_validation import validate_reconstructions
        vqvae = VQVAE(in_channels=4, hidden_dim=16, latent_dim=8, num_embeddings=64)
        img = Image.new("RGBA", (32, 32), (255, 0, 0, 255))
        grid, metrics_list = validate_reconstructions(vqvae, [img], "cpu")
        assert isinstance(grid, Image.Image)
        assert len(metrics_list) == 1
        assert "mse" in metrics_list[0]
        assert "psnr" in metrics_list[0]

    def test_multiple_images(self):
        from eval.reconstruction_validation import validate_reconstructions
        vqvae = VQVAE(in_channels=4, hidden_dim=16, latent_dim=8, num_embeddings=64)
        imgs = [
            Image.new("RGBA", (32, 32), (255, 0, 0, 255)),
            Image.new("RGBA", (32, 32), (0, 255, 0, 255)),
            Image.new("RGBA", (32, 32), (0, 0, 255, 255)),
        ]
        grid, metrics_list = validate_reconstructions(vqvae, imgs, "cpu")
        assert len(metrics_list) == 3

    def test_saves_output_when_path_given(self, tmp_path):
        from eval.reconstruction_validation import validate_reconstructions
        vqvae = VQVAE(in_channels=4, hidden_dim=16, latent_dim=8, num_embeddings=64)
        img = Image.new("RGBA", (32, 32), (255, 0, 0, 255))
        output_path = str(tmp_path / "validation.png")
        grid, metrics_list = validate_reconstructions(vqvae, [img], "cpu", output_path=output_path)
        assert Path(output_path).exists()


class TestMainEntryPoint:
    def test_main_runs_with_image_dir(self, monkeypatch, tmp_path):
        from eval.reconstruction_validation import main

        vqvae = VQVAE(num_embeddings=64)
        checkpoint_path = tmp_path / "vqvae.pt"
        torch.save({
            "model_state": vqvae.state_dict(),
            "config": {"num_embeddings": 64},
        }, checkpoint_path)

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        for i in range(3):
            arr = np.zeros((32, 32, 4), dtype=np.uint8)
            arr[:, :, :3] = [255 if i == 0 else 0, 255 if i == 1 else 0, 255 if i == 2 else 0]
            arr[:, :, 3] = 255
            Image.fromarray(arr, "RGBA").save(str(img_dir / f"sprite_{i}.png"))

        output_path = tmp_path / "validation.png"
        metrics_path = tmp_path / "metrics.json"

        test_args = [
            "prog",
            "--vqvae-checkpoint", str(checkpoint_path),
            "--output", str(output_path),
            "--metrics-output", str(metrics_path),
            "--image-dir", str(img_dir),
            "--num-samples", "3",
        ]
        monkeypatch.setattr("sys.argv", test_args)
        result = main()
        assert result == 0
        assert output_path.exists()
        assert metrics_path.exists()

        with open(metrics_path) as f:
            data = json.load(f)
        assert data["num_samples"] == 3
        assert "avg_psnr" in data
        assert "avg_mse" in data
        assert "per_sample" in data
        assert len(data["per_sample"]) == 3

    def test_main_no_images_returns_1(self, monkeypatch, tmp_path):
        from eval.reconstruction_validation import main

        vqvae = VQVAE(num_embeddings=64)
        checkpoint_path = tmp_path / "vqvae.pt"
        torch.save({
            "model_state": vqvae.state_dict(),
            "config": {"num_embeddings": 64},
        }, checkpoint_path)

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        test_args = [
            "prog",
            "--vqvae-checkpoint", str(checkpoint_path),
            "--image-dir", str(empty_dir),
        ]
        monkeypatch.setattr("sys.argv", test_args)
        result = main()
        assert result == 1

    def test_main_no_dataset_or_image_dir_returns_1(self, monkeypatch, tmp_path):
        from eval.reconstruction_validation import main

        vqvae = VQVAE(num_embeddings=64)
        checkpoint_path = tmp_path / "vqvae.pt"
        torch.save({
            "model_state": vqvae.state_dict(),
            "config": {"num_embeddings": 64},
        }, checkpoint_path)

        test_args = [
            "prog",
            "--vqvae-checkpoint", str(checkpoint_path),
        ]
        monkeypatch.setattr("sys.argv", test_args)
        result = main()
        assert result == 1

    def test_main_with_dataset_arg(self, monkeypatch, tmp_path):
        from eval.reconstruction_validation import main

        vqvae = VQVAE(num_embeddings=64)
        checkpoint_path = tmp_path / "vqvae.pt"
        torch.save({
            "model_state": vqvae.state_dict(),
            "config": {"num_embeddings": 64},
        }, checkpoint_path)

        class FakeItem:
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
                return FakeItem()
            def __iter__(self):
                for _ in range(self.n):
                    yield FakeItem()

        monkeypatch.setattr(
            "eval.reconstruction_validation.load_dataset",
            lambda path, split: FakeDataset(2),
        )

        output_path = tmp_path / "val.png"
        metrics_path = tmp_path / "metrics.json"
        test_args = [
            "prog",
            "--vqvae-checkpoint", str(checkpoint_path),
            "--dataset", "fake/dataset",
            "--output", str(output_path),
            "--metrics-output", str(metrics_path),
            "--num-samples", "2",
        ]
        monkeypatch.setattr("sys.argv", test_args)
        result = main()
        assert result == 0
        assert output_path.exists()
        assert metrics_path.exists()
        with open(metrics_path) as f:
            data = json.load(f)
        assert data["num_samples"] == 2
