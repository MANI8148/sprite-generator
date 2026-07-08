"""Tests for Path comparison (roadmap item #5)."""
import json
import tempfile
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import pytest

from models.vqvae.model import VQVAE
from models.transformer.model import SpriteTransformer
from models.transformer.train import CLASS_VOCAB, ACTION_VOCAB, DIRECTION_VOCAB
from models.lora.model import SpriteLoRAWrapper


class TestGenerateVQVAESamples:
    def test_returns_correct_number_of_samples(self):
        vqvae = VQVAE(in_channels=4, hidden_dim=16, latent_dim=8, num_embeddings=64)
        transformer = SpriteTransformer(
            vocab_size=64, condition_vocab_size=64,
            d_model=16, n_layers=1, n_heads=1, max_seq_len=65,
        )
        conditions = [("character", "idle", "front"), ("enemy", "walk", "left")]
        from eval.compare_paths import generate_vqvae_samples
        samples = generate_vqvae_samples(vqvae, transformer, "cpu", conditions)
        assert len(samples) == 2
        for s in samples:
            assert isinstance(s, Image.Image)
            assert s.size == (32, 32)

    def test_empty_conditions_returns_empty(self):
        vqvae = VQVAE(in_channels=4, hidden_dim=16, latent_dim=8, num_embeddings=64)
        transformer = SpriteTransformer(
            vocab_size=64, condition_vocab_size=64,
            d_model=16, n_layers=1, n_heads=1, max_seq_len=65,
        )
        from eval.compare_paths import generate_vqvae_samples
        samples = generate_vqvae_samples(vqvae, transformer, "cpu", [])
        assert samples == []


class TestGenerateLoRASamples:
    def test_returns_correct_number_of_samples(self):
        lora = SpriteLoRAWrapper(rank=4)
        from eval.compare_paths import generate_lora_samples
        samples = generate_lora_samples(lora, "cpu", 3)
        assert len(samples) == 3
        for s in samples:
            assert isinstance(s, Image.Image)
            assert s.size == (32, 32)

    def test_zero_samples_returns_empty(self):
        lora = SpriteLoRAWrapper(rank=4)
        from eval.compare_paths import generate_lora_samples
        samples = generate_lora_samples(lora, "cpu", 0)
        assert samples == []


class TestComputeMetrics:
    @pytest.fixture
    def palette(self):
        return [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    def test_returns_expected_keys(self, palette):
        arr = np.zeros((32, 32, 4), dtype=np.uint8)
        arr[:, :, :3] = [255, 0, 0]
        arr[:, :, 3] = 255
        img = Image.fromarray(arr, "RGBA")
        from eval.compare_paths import compute_metrics
        metrics = compute_metrics([img, img], palette)
        assert "palette_adherence_mean" in metrics
        assert "palette_adherence_std" in metrics
        assert "grid_alignment_mean" in metrics
        assert "grid_alignment_std" in metrics

    def test_perfect_palette_adherence(self, palette):
        arr = np.zeros((32, 32, 4), dtype=np.uint8)
        arr[:, :, :3] = [255, 0, 0]
        arr[:, :, 3] = 255
        img = Image.fromarray(arr, "RGBA")
        from eval.compare_paths import compute_metrics
        metrics = compute_metrics([img], palette)
        assert metrics["palette_adherence_mean"] == 1.0

    def test_empty_samples(self, palette):
        from eval.compare_paths import compute_metrics
        metrics = compute_metrics([], palette)
        assert np.isnan(metrics["palette_adherence_mean"])
        assert np.isnan(metrics["grid_alignment_mean"])


class TestMainEntryPoint:
    def test_main_runs_with_synthetic_checkpoints(self, monkeypatch, tmp_path):
        vqvae = VQVAE(num_embeddings=64)
        transformer = SpriteTransformer(
            vocab_size=64, condition_vocab_size=64,
        )
        lora = SpriteLoRAWrapper(rank=4)

        vqvae_path = tmp_path / "vqvae.pt"
        torch.save({"model_state": vqvae.state_dict(), "config": {"num_embeddings": 64}}, vqvae_path)
        transformer_path = tmp_path / "transformer.pt"
        torch.save({"model_state": transformer.state_dict(), "config": {}}, transformer_path)
        lora_path = tmp_path / "lora.pt"
        torch.save({"model_state": lora.state_dict()}, lora_path)

        output_path = tmp_path / "results.json"
        test_args = [
            "prog",
            "--vqvae-checkpoint", str(vqvae_path),
            "--transformer-checkpoint", str(transformer_path),
            "--lora-checkpoint", str(lora_path),
            "--output", str(output_path),
            "--num-samples", "2",
            "--visualization", str(tmp_path / "viz.png"),
        ]
        monkeypatch.setattr("sys.argv", test_args)

        from eval.compare_paths import main
        main()

        assert output_path.exists()
        with open(output_path) as f:
            results = json.load(f)

        assert "comparison" in results
        assert "vqvae_transformer" in results["comparison"]
        assert "lora" in results["comparison"]
        assert "difference" in results
        assert results["num_samples"] == 2

    def test_main_without_visualization(self, monkeypatch, tmp_path):
        vqvae = VQVAE(num_embeddings=64)
        transformer = SpriteTransformer(
            vocab_size=64, condition_vocab_size=64,
        )
        lora = SpriteLoRAWrapper(rank=4)

        vqvae_path = tmp_path / "vqvae.pt"
        torch.save({"model_state": vqvae.state_dict(), "config": {"num_embeddings": 64}}, vqvae_path)
        transformer_path = tmp_path / "transformer.pt"
        torch.save({"model_state": transformer.state_dict(), "config": {}}, transformer_path)
        lora_path = tmp_path / "lora.pt"
        torch.save({"model_state": lora.state_dict()}, lora_path)

        test_args = [
            "prog",
            "--vqvae-checkpoint", str(vqvae_path),
            "--transformer-checkpoint", str(transformer_path),
            "--lora-checkpoint", str(lora_path),
            "--output", str(tmp_path / "results.json"),
            "--num-samples", "1",
        ]
        monkeypatch.setattr("sys.argv", test_args)

        from eval.compare_paths import main
        main()
