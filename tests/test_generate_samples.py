import tempfile
from pathlib import Path

import torch
import pytest
from PIL import Image

from models.vqvae.model import VQVAE
from models.transformer.model import SpriteTransformer


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
