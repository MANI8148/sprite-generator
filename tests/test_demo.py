"""Tests for the Gradio demo (roadmap item #6)."""
import os
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import pytest

from models.vqvae.model import VQVAE
from models.transformer.model import SpriteTransformer
from models.transformer.train import CLASS_VOCAB, ACTION_VOCAB, DIRECTION_VOCAB


class TestEncodeCondition:
    def test_known_value_returns_index(self):
        from demo.app import encode_condition
        assert encode_condition("character", CLASS_VOCAB) == CLASS_VOCAB.index("character")

    def test_unknown_value_returns_zero(self):
        from demo.app import encode_condition
        assert encode_condition("nonexistent", CLASS_VOCAB) == 0

    def test_empty_string_returns_zero(self):
        from demo.app import encode_condition
        assert encode_condition("", CLASS_VOCAB) == 0


class TestGenerateSprite:
    @pytest.fixture
    def vqvae(self):
        return VQVAE(in_channels=4, hidden_dim=16, latent_dim=8, num_embeddings=64)

    @pytest.fixture
    def transformer(self):
        return SpriteTransformer(
            vocab_size=64, condition_vocab_size=64,
            d_model=16, n_layers=1, n_heads=1, max_seq_len=65,
        )

    def test_generates_rgba_image(self, vqvae, transformer):
        from demo.app import generate_sprite
        img = generate_sprite(
            vqvae, transformer,
            "character", "idle", "front",
            temperature=1.0, top_k=40, top_p=0.9,
            device="cpu",
        )
        assert isinstance(img, Image.Image)
        assert img.mode == "RGBA"
        assert img.size == (32, 32)

    def test_different_conditions_produce_different_images(self, vqvae, transformer):
        from demo.app import generate_sprite
        img1 = generate_sprite(vqvae, transformer, "character", "idle", "front", device="cpu")
        img2 = generate_sprite(vqvae, transformer, "enemy", "attack", "left", device="cpu")
        arr1 = np.array(img1)
        arr2 = np.array(img2)
        assert not np.array_equal(arr1, arr2)

    def test_all_vocab_values_produce_valid_images(self, vqvae, transformer):
        from demo.app import generate_sprite
        for cls in CLASS_VOCAB[:3]:
            for act in ACTION_VOCAB[:3]:
                for dire in DIRECTION_VOCAB[:3]:
                    img = generate_sprite(vqvae, transformer, cls, act, dire, device="cpu")
                    assert img.size == (32, 32)
                    assert img.mode == "RGBA"


class TestBuildDemo:
    def test_build_demo_returns_interface(self):
        from demo.app import build_demo
        vqvae = VQVAE(in_channels=4, hidden_dim=16, latent_dim=8, num_embeddings=64)
        transformer = SpriteTransformer(
            vocab_size=64, condition_vocab_size=64,
            d_model=16, n_layers=1, n_heads=1, max_seq_len=65,
        )
        iface = build_demo(vqvae, transformer, "cpu")
        assert iface.title == "Sprite Generator"
        assert len(iface.input_components) == 6
        assert len(iface.output_components) == 1

    def test_generate_function_in_demo_produces_128px(self):
        from demo.app import build_demo
        vqvae = VQVAE(in_channels=4, hidden_dim=16, latent_dim=8, num_embeddings=64)
        transformer = SpriteTransformer(
            vocab_size=64, condition_vocab_size=64,
            d_model=16, n_layers=1, n_heads=1, max_seq_len=65,
        )
        iface = build_demo(vqvae, transformer, "cpu")
        result = iface.fn("character", "idle", "front", 1.0, 40, 0.9)
        assert isinstance(result, Image.Image)
        assert result.size == (128, 128)
        assert result.mode == "RGBA"


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
        cfg = {"d_model": 16, "n_layers": 1, "n_heads": 1}

        def fake_hf_download(repo_id, filename, token=None):
            return f"/tmp/fake_{filename}"

        call_count = [0]

        def fake_torch_load(path, map_location=None):
            call_count[0] += 1
            if "vqvae" in path:
                return {
                    "model_state": vqvae.state_dict(),
                    "config": {"num_embeddings": 64},
                }
            return {
                "model_state": transformer.state_dict(),
                "config": cfg,
            }

        monkeypatch.setattr("demo.app.hf_hub_download", fake_hf_download)
        monkeypatch.setattr("demo.app.torch.load", fake_torch_load)

        from demo.app import load_models
        result = load_models("fake/repo", device="cpu")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert call_count[0] == 2

    def test_load_models_infers_config_from_state_dict(self, monkeypatch):
        vqvae = VQVAE(num_embeddings=64)
        transformer = SpriteTransformer(
            vocab_size=64, condition_vocab_size=64,
            d_model=16, n_layers=1, n_heads=1, max_seq_len=65,
        )

        def fake_hf_download(repo_id, filename, token=None):
            return f"/tmp/fake_{filename}"

        def fake_torch_load(path, map_location=None):
            if "vqvae" in path:
                return {
                    "model_state": vqvae.state_dict(),
                    "config": {},
                }
            return {
                "model_state": transformer.state_dict(),
                "config": {},
            }

        monkeypatch.setattr("demo.app.hf_hub_download", fake_hf_download)
        monkeypatch.setattr("demo.app.torch.load", fake_torch_load)

        from demo.app import load_models
        v, t = load_models("fake/repo", device="cpu")
        assert not v.training
        assert not t.training

    def test_load_models_sets_eval_mode(self, monkeypatch):
        vqvae, transformer = self._make_models()
        cfg = {"d_model": 16, "n_layers": 1, "n_heads": 1}

        def fake_hf_download(repo_id, filename, token=None):
            return f"/tmp/fake_{filename}"

        def fake_torch_load(path, map_location=None):
            if "vqvae" in path:
                return {
                    "model_state": vqvae.state_dict(),
                    "config": {"num_embeddings": 64},
                }
            return {
                "model_state": transformer.state_dict(),
                "config": cfg,
            }

        monkeypatch.setattr("demo.app.hf_hub_download", fake_hf_download)
        monkeypatch.setattr("demo.app.torch.load", fake_torch_load)

        from demo.app import load_models
        v, t = load_models("fake/repo", device="cpu")
        assert not v.training
        assert not t.training


class TestDemoFallback:
    def test_demo_exists_as_module_level(self):
        import demo.app
        assert hasattr(demo.app, "demo")
        assert demo.app.demo is not None

    def test_fallback_generates_placeholder(self):
        import demo.app
        if demo.app.vqvae_model is None:
            result = demo.app.demo.fn("character", "idle", "front", 1.0, 40, 0.9)
            assert isinstance(result, Image.Image)
            assert result.size == (128, 128)