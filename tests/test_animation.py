import json
from pathlib import Path

import torch
import pytest
from PIL import Image

from models.vqvae.model import VQVAE
from models.transformer.model import SpriteTransformer


class TestGenerateAnimationSequence:
    @pytest.fixture
    def vqvae(self):
        return VQVAE(in_channels=4, hidden_dim=16, latent_dim=8, num_embeddings=64)

    @pytest.fixture
    def transformer(self):
        return SpriteTransformer(
            vocab_size=64, condition_vocab_size=64,
            d_model=16, n_layers=1, n_heads=1, max_seq_len=65,
        )

    def test_returns_correct_number_of_frames(self, vqvae, transformer):
        from eval.animation import generate_animation_sequence
        directions = ["front", "left", "back", "right"]
        frames = generate_animation_sequence(
            vqvae, transformer, "cpu",
            class_name="character", action="walk",
            directions=directions, num_repeats=1,
        )
        assert len(frames) == len(directions)

    def test_frames_are_rgba_pil_images(self, vqvae, transformer):
        from eval.animation import generate_animation_sequence
        frames = generate_animation_sequence(
            vqvae, transformer, "cpu",
            directions=["front"], num_repeats=1,
        )
        assert len(frames) == 1
        img = frames[0]
        assert isinstance(img, Image.Image)
        assert img.mode == "RGBA"
        assert img.size == (32, 32)

    def test_num_repeats_multiplies_frames(self, vqvae, transformer):
        from eval.animation import generate_animation_sequence
        frames = generate_animation_sequence(
            vqvae, transformer, "cpu",
            directions=["front", "left"], num_repeats=3,
        )
        assert len(frames) == 6

    def test_default_directions(self, vqvae, transformer):
        from eval.animation import generate_animation_sequence, DEFAULT_DIRECTIONS
        frames = generate_animation_sequence(
            vqvae, transformer, "cpu", num_repeats=1,
        )
        assert len(frames) == len(DEFAULT_DIRECTIONS)

    def test_empty_directions(self, vqvae, transformer):
        from eval.animation import generate_animation_sequence
        frames = generate_animation_sequence(
            vqvae, transformer, "cpu",
            directions=[], num_repeats=1,
        )
        assert len(frames) == 0

    def test_with_palette_does_not_crash(self, vqvae, transformer):
        from eval.animation import generate_animation_sequence
        palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        frames = generate_animation_sequence(
            vqvae, transformer, "cpu",
            directions=["front"], num_repeats=1,
            palette=palette,
        )
        assert len(frames) == 1
        assert isinstance(frames[0], Image.Image)

    def test_different_temperature_does_not_crash(self, vqvae, transformer):
        from eval.animation import generate_animation_sequence
        frames = generate_animation_sequence(
            vqvae, transformer, "cpu",
            directions=["front"], num_repeats=1,
            temperature=0.5,
        )
        assert len(frames) == 1

    def test_sets_models_to_eval(self, vqvae, transformer):
        vqvae.train()
        transformer.train()
        from eval.animation import generate_animation_sequence
        generate_animation_sequence(
            vqvae, transformer, "cpu",
            directions=["front"], num_repeats=1,
        )
        assert not vqvae.training
        assert not transformer.training


class TestCreateAnimatedGif:
    def test_saves_gif_file(self, tmp_path):
        from eval.animation import create_animated_gif
        frames = [Image.new("RGBA", (32, 32), (255, 0, 0, 255))]
        output = tmp_path / "test.gif"
        result = create_animated_gif(frames, str(output))
        assert result == str(output)
        assert output.exists()
        img = Image.open(output)
        assert img.size == (32, 32)

    def test_multiple_frames(self, tmp_path):
        from eval.animation import create_animated_gif
        frames = [
            Image.new("RGBA", (32, 32), (255, 0, 0, 255)),
            Image.new("RGBA", (32, 32), (0, 255, 0, 255)),
            Image.new("RGBA", (32, 32), (0, 0, 255, 255)),
        ]
        output = tmp_path / "multi.gif"
        create_animated_gif(frames, str(output), duration=150, loop=0)
        assert output.exists()
        img = Image.open(output)
        assert img.n_frames >= 1

    def test_empty_frames_creates_placeholder(self, tmp_path):
        from eval.animation import create_animated_gif
        output = tmp_path / "empty.gif"
        result = create_animated_gif([], str(output))
        assert result == str(output)
        assert output.exists()

    def test_creates_output_directory(self, tmp_path):
        from eval.animation import create_animated_gif
        nested = tmp_path / "sub" / "dir" / "anim.gif"
        frames = [Image.new("RGBA", (16, 16), (255, 0, 0, 255))]
        create_animated_gif(frames, str(nested), duration=100)
        assert nested.exists()

    def test_custom_duration_per_frame(self, tmp_path):
        from eval.animation import create_animated_gif
        frames = [
            Image.new("RGBA", (16, 16), (255, 0, 0, 255)),
            Image.new("RGBA", (16, 16), (0, 255, 0, 255)),
        ]
        output = tmp_path / "fast.gif"
        create_animated_gif(frames, str(output), duration=50)
        assert output.exists()


class TestMainEntryPoint:
    def _make_checkpoints(self, tmp_path):
        vqvae = VQVAE(num_embeddings=64)
        transformer = SpriteTransformer(
            vocab_size=64, condition_vocab_size=64,
        )
        vqvae_path = tmp_path / "vqvae.pt"
        transformer_path = tmp_path / "transformer.pt"

        torch.save({
            "model_state": vqvae.state_dict(),
            "config": {"num_embeddings": 64},
        }, str(vqvae_path))

        torch.save({
            "model_state": transformer.state_dict(),
            "config": {"d_model": 256, "n_layers": 8, "n_heads": 4},
        }, str(transformer_path))

        return str(vqvae_path), str(transformer_path)

    def test_main_runs_and_saves_gif(self, monkeypatch, tmp_path):
        vqvae_path, transformer_path = self._make_checkpoints(tmp_path)
        output = tmp_path / "anim.gif"
        test_args = [
            "prog",
            "--vqvae-checkpoint", vqvae_path,
            "--transformer-checkpoint", transformer_path,
            "--output", str(output),
            "--class-name", "character",
            "--action", "walk",
            "--directions", "front", "left",
            "--num-repeats", "1",
        ]
        monkeypatch.setattr("sys.argv", test_args)
        from eval.animation import main
        main()
        assert output.exists()
        img = Image.open(output)
        assert img.size == (32, 32)

    def test_main_with_palette(self, monkeypatch, tmp_path):
        vqvae_path, transformer_path = self._make_checkpoints(tmp_path)
        palette_path = tmp_path / "palette.json"
        palette = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
        with open(palette_path, "w") as f:
            json.dump(palette, f)

        output = tmp_path / "palette_anim.gif"
        test_args = [
            "prog",
            "--vqvae-checkpoint", vqvae_path,
            "--transformer-checkpoint", transformer_path,
            "--output", str(output),
            "--class-name", "character",
            "--directions", "front",
            "--palette", str(palette_path),
        ]
        monkeypatch.setattr("sys.argv", test_args)
        from eval.animation import main
        main()
        assert output.exists()

    def test_main_default_args(self, monkeypatch, tmp_path):
        vqvae_path, transformer_path = self._make_checkpoints(tmp_path)
        output = Path.cwd() / "animation.gif"
        if output.exists():
            output.unlink()
        test_args = [
            "prog",
            "--vqvae-checkpoint", vqvae_path,
            "--transformer-checkpoint", transformer_path,
        ]
        monkeypatch.setattr("sys.argv", test_args)
        from eval.animation import main
        main()
        assert output.exists()
