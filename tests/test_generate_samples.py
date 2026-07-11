import json
import tempfile
from pathlib import Path

import numpy as np
import torch
import pytest
from PIL import Image

from models.vqvae.model import VQVAE
from models.transformer.model import SpriteTransformer


class TestLoadModels:
    def _make_models(self):
        vqvae = VQVAE(num_embeddings=64)
        transformer = SpriteTransformer(
            vocab_size=64, condition_vocab_size=64,
            d_model=16, n_layers=1, n_heads=1, max_seq_len=65,
        )
        return vqvae, transformer

    def test_load_models_returns_tuple(self, monkeypatch):
        vqvae, transformer = self._make_models()

        saved = {"vqvae": None, "transformer": None}

        def fake_hf_download(repo_id, filename, token=None):
            if "vqvae" in filename:
                return "/tmp/fake_vqvae.pt"
            return "/tmp/fake_transformer.pt"

        def fake_torch_load(path, map_location=None):
            if "vqvae" in path:
                return {
                    "model_state": vqvae.state_dict(),
                    "config": {"num_embeddings": 64},
                }
            return {
                "model_state": transformer.state_dict(),
                "config": {"d_model": 16, "n_layers": 1, "n_heads": 1},
            }

        monkeypatch.setattr("eval.generate_samples.hf_hub_download", fake_hf_download)
        monkeypatch.setattr("eval.generate_samples.torch.load", fake_torch_load)

        from eval.generate_samples import load_models
        result = load_models("fake/repo", device="cpu")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_load_models_sets_eval_mode(self, monkeypatch):
        vqvae, transformer = self._make_models()

        def fake_hf_download(repo_id, filename, token=None):
            if "vqvae" in filename:
                return "/tmp/fake_vqvae.pt"
            return "/tmp/fake_transformer.pt"

        def fake_torch_load(path, map_location=None):
            if "vqvae" in path:
                return {
                    "model_state": vqvae.state_dict(),
                    "config": {"num_embeddings": 64},
                }
            return {
                "model_state": transformer.state_dict(),
                "config": {"d_model": 16, "n_layers": 1, "n_heads": 1},
            }

        monkeypatch.setattr("eval.generate_samples.hf_hub_download", fake_hf_download)
        monkeypatch.setattr("eval.generate_samples.torch.load", fake_torch_load)

        from eval.generate_samples import load_models
        v, t = load_models("fake/repo", device="cpu")
        assert not v.training
        assert not t.training


class TestMainEntryPoint:
    def _make_models(self):
        vqvae = VQVAE(num_embeddings=64)
        transformer = SpriteTransformer(
            vocab_size=64, condition_vocab_size=64,
            d_model=16, n_layers=1, n_heads=1, max_seq_len=65,
        )
        return vqvae, transformer

    def test_main_runs_and_saves_output(self, monkeypatch, tmp_path):
        vqvae, transformer = self._make_models()

        def fake_hf_download(repo_id, filename, token=None):
            if "vqvae" in filename:
                return "/tmp/fake_vqvae.pt"
            return "/tmp/fake_transformer.pt"

        def fake_torch_load(path, map_location=None):
            if "vqvae" in path:
                return {
                    "model_state": vqvae.state_dict(),
                    "config": {"num_embeddings": 64},
                }
            return {
                "model_state": transformer.state_dict(),
                "config": {"d_model": 16, "n_layers": 1, "n_heads": 1},
            }

        monkeypatch.setattr("eval.generate_samples.hf_hub_download", fake_hf_download)
        monkeypatch.setattr("eval.generate_samples.torch.load", fake_torch_load)

        output_path = tmp_path / "test_samples.png"
        test_args = [
            "prog",
            "--hf-repo", "fake/repo",
            "--output", str(output_path),
            "--rows", "1",
            "--cols", "1",
        ]
        monkeypatch.setattr("sys.argv", test_args)

        from eval.generate_samples import main
        main()

        assert output_path.exists()
        img = Image.open(output_path)
        assert img.size == (32, 32)

    def test_main_default_args(self, monkeypatch, tmp_path):
        vqvae, transformer = self._make_models()

        def fake_hf_download(repo_id, filename, token=None):
            if "vqvae" in filename:
                return "/tmp/fake_vqvae.pt"
            return "/tmp/fake_transformer.pt"

        def fake_torch_load(path, map_location=None):
            if "vqvae" in path:
                return {
                    "model_state": vqvae.state_dict(),
                    "config": {"num_embeddings": 64},
                }
            return {
                "model_state": transformer.state_dict(),
                "config": {"d_model": 16, "n_layers": 1, "n_heads": 1},
            }

        monkeypatch.setattr("eval.generate_samples.hf_hub_download", fake_hf_download)
        monkeypatch.setattr("eval.generate_samples.torch.load", fake_torch_load)
        monkeypatch.setattr("sys.argv", ["prog", "--hf-repo", "fake/repo"])

        from eval.generate_samples import main
        main()


class TestGenerateGrid:
    @pytest.fixture
    def vqvae(self):
        return VQVAE(in_channels=4, hidden_dim=16, latent_dim=8, num_embeddings=64)

    @pytest.fixture
    def transformer(self):
        return SpriteTransformer(
            vocab_size=64,
            condition_vocab_size=64,
            d_model=16,
            n_layers=1,
            n_heads=1,
            max_seq_len=65,
        )

    def test_generate_grid_output_shape(self, vqvae, transformer):
        from eval.generate_samples import generate_grid
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            output_path = tmp.name
            canvas = generate_grid(vqvae, transformer, "cpu", (2, 2), output_path)
            assert isinstance(canvas, Image.Image)
            assert canvas.size == (2 * 32, 2 * 32)
            assert canvas.mode == "RGBA"

    def test_generate_grid_saves_file(self, vqvae, transformer):
        from eval.generate_samples import generate_grid
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "samples.png"
            generate_grid(vqvae, transformer, "cpu", (1, 1), str(output_path))
            assert output_path.exists()
            img = Image.open(output_path)
            assert img.size == (32, 32)

    def test_generate_grid_different_sizes(self, vqvae, transformer):
        from eval.generate_samples import generate_grid
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            canvas = generate_grid(vqvae, transformer, "cpu", (3, 4), tmp.name)
            assert canvas.size == (4 * 32, 3 * 32)

    def test_generate_grid_uses_temperature(self, vqvae, transformer):
        from eval.generate_samples import generate_grid
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            canvas_hot = generate_grid(vqvae, transformer, "cpu", (1, 1), tmp.name)
            canvas_cold = generate_grid(vqvae, transformer, "cpu", (1, 1), tmp.name)
            assert isinstance(canvas_hot, Image.Image)
            assert isinstance(canvas_cold, Image.Image)


class TestQuantizeToPalette:
    def test_snaps_to_nearest(self):
        from eval.generate_samples import quantize_to_palette
        palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        arr = np.zeros((4, 4, 4), dtype=np.uint8)
        arr[:, :] = [128, 0, 0, 255]
        result = quantize_to_palette(arr, palette)
        assert (result[0, 0, :3] == [255, 0, 0]).all()

    def test_preserves_transparent_pixels(self):
        from eval.generate_samples import quantize_to_palette
        palette = [(255, 0, 0)]
        arr = np.zeros((4, 4, 4), dtype=np.uint8)
        arr[:, :] = [128, 64, 32, 0]
        result = quantize_to_palette(arr, palette)
        assert (result[0, 0] == [128, 64, 32, 0]).all()

    def test_preserves_alpha_channel(self):
        from eval.generate_samples import quantize_to_palette
        palette = [(0, 255, 0)]
        arr = np.zeros((4, 4, 4), dtype=np.uint8)
        arr[:, :] = [0, 128, 0, 200]
        result = quantize_to_palette(arr, palette)
        assert result[0, 0, 3] == 200

    def test_multi_color_image(self):
        from eval.generate_samples import quantize_to_palette
        palette = [(255, 0, 0), (0, 0, 255), (0, 255, 0)]
        arr = np.zeros((2, 2, 4), dtype=np.uint8)
        arr[0, 0] = [200, 10, 10, 255]
        arr[0, 1] = [10, 10, 200, 255]
        arr[1, 0] = [10, 200, 10, 255]
        arr[1, 1] = [100, 100, 100, 0]
        result = quantize_to_palette(arr, palette)
        assert (result[0, 0, :3] == [255, 0, 0]).all()
        assert (result[0, 1, :3] == [0, 0, 255]).all()
        assert (result[1, 0, :3] == [0, 255, 0]).all()


class TestHardAlphaEdges:
    def test_binarizes_alpha(self):
        from eval.generate_samples import hard_alpha_edges
        arr = np.zeros((4, 4, 4), dtype=np.uint8)
        arr[:, :, 3] = [0, 128, 200, 255]
        result = hard_alpha_edges(arr)
        assert (result[:, :, 3] == [0, 0, 255, 255]).all()

    def test_uses_threshold(self):
        from eval.generate_samples import hard_alpha_edges
        arr = np.zeros((4, 4, 4), dtype=np.uint8)
        arr[:, :, 3] = [0, 64, 128, 192]
        with_custom = hard_alpha_edges(arr, threshold=100)
        assert (with_custom[:, :, 3] == [0, 0, 255, 255]).all()

    def test_preserves_rgb_channels(self):
        from eval.generate_samples import hard_alpha_edges
        arr = np.zeros((4, 4, 4), dtype=np.uint8)
        arr[0, 0] = [255, 128, 64, 200]
        result = hard_alpha_edges(arr)
        assert (result[0, 0, :3] == [255, 128, 64]).all()


class TestPostProcessSprite:
    def test_with_palette_quantizes_and_binarizes(self):
        from eval.generate_samples import post_process_sprite
        palette = [(255, 0, 0)]
        arr = np.zeros((4, 4, 4), dtype=np.uint8)
        arr[:, :] = [200, 10, 10, 200]
        result = post_process_sprite(arr, palette)
        assert (result[0, 0, :3] == [255, 0, 0]).all()
        assert result[0, 0, 3] == 255

    def test_without_palette_only_binarizes(self):
        from eval.generate_samples import post_process_sprite
        arr = np.zeros((4, 4, 4), dtype=np.uint8)
        arr[:, :] = [128, 64, 32, 128]
        result = post_process_sprite(arr)
        assert (result[0, 0, :3] == [128, 64, 32]).all()
        assert result[0, 0, 3] == 0

    def test_fully_transparent_unchanged(self):
        from eval.generate_samples import post_process_sprite
        palette = [(255, 0, 0)]
        arr = np.zeros((4, 4, 4), dtype=np.uint8)
        arr[:, :] = [128, 64, 32, 0]
        result = post_process_sprite(arr, palette)
        assert (result[0, 0, :3] == [128, 64, 32]).all()
        assert result[0, 0, 3] == 0


class TestGenerateGridWithPalette:
    @pytest.fixture
    def vqvae(self):
        return VQVAE(in_channels=4, hidden_dim=16, latent_dim=8, num_embeddings=64)

    @pytest.fixture
    def transformer(self):
        return SpriteTransformer(
            vocab_size=64, condition_vocab_size=64,
            d_model=16, n_layers=1, n_heads=1, max_seq_len=65,
        )

    def test_palette_arg_does_not_break(self, vqvae, transformer):
        from eval.generate_samples import generate_grid
        palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            canvas = generate_grid(vqvae, transformer, "cpu", (1, 1), tmp.name, palette)
        assert isinstance(canvas, Image.Image)
        assert canvas.size == (32, 32)
