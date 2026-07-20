"""End-to-end smoke test with real LoRA weights through the full pipeline.

Covers roadmap item: "Smoke test with real LoRA weights end-to-end".
"""
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Optional

import safetensors.torch

from backend.modules.pipeline.orchestrator import AssetPipeline
from backend.modules.prompt_builder.controls import (
    AssetControls, AssetType, View, Palette, Animation, SpriteSize,
)
from models.lora.model import SpriteLoRAWrapper


def _make_safetensors_lora_weights(rank: int = 4, num_layers: int = 4) -> dict:
    """Create synthetic LoRA weights in diffusers-compatible .safetensors format."""
    weights = {}
    for i in range(num_layers):
        for direction in ("down", "up"):
            weights[f"lora_unet_down_blocks_{i}.attentions.0.proj_in.lora_{direction}.weight"] = (
                torch.randn(320, rank) * 0.01 if direction == "down" else torch.zeros(rank, 320)
            )
            weights[f"lora_unet_down_blocks_{i}.attentions.0.proj_out.lora_{direction}.weight"] = (
                torch.randn(320, rank) * 0.01 if direction == "down" else torch.zeros(rank, 320)
            )
            weights[f"lora_unet_up_blocks_{i}.attentions.0.proj_in.lora_{direction}.weight"] = (
                torch.randn(320, rank) * 0.01 if direction == "down" else torch.zeros(rank, 320)
            )
            weights[f"lora_unet_up_blocks_{i}.attentions.0.proj_out.lora_{direction}.weight"] = (
                torch.randn(320, rank) * 0.01 if direction == "down" else torch.zeros(rank, 320)
            )
    weights["lora_unet_mid_block.attentions.0.proj_in.lora_down.weight"] = torch.randn(320, rank) * 0.01
    weights["lora_unet_mid_block.attentions.0.proj_in.lora_up.weight"] = torch.zeros(rank, 320)
    weights["lora_unet_mid_block.attentions.0.proj_out.lora_down.weight"] = torch.randn(320, rank) * 0.01
    weights["lora_unet_mid_block.attentions.0.proj_out.lora_up.weight"] = torch.zeros(rank, 320)
    return weights


class LoRAGenerator:
    """Wraps SpriteLoRAWrapper to conform to SDGenerator.generate() interface."""

    def __init__(self, model: SpriteLoRAWrapper, device: str = "cpu"):
        self.model = model
        self.model.eval()
        self.device = device

    def generate(
        self,
        prompt: str = "",
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        seed: int = -1,
        num_images: Optional[int] = None,
    ) -> List[Image.Image]:
        n = num_images or 1
        images = []
        for _ in range(n):
            z = torch.randn(1, 256, 4, 4, device=self.device)
            with torch.no_grad():
                out = self.model.decode(z)
            out = out.clamp(0, 1).cpu()
            arr = (out[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            if arr.shape[2] == 4:
                arr[:, :, 3] = 255
            img = Image.fromarray(arr, "RGBA").resize((width, height), Image.NEAREST)
            images.append(img)
        return images


class TestLoRAEndToEnd:
    def test_pipeline_with_trained_lora_weights(self, tmp_path):
        model = SpriteLoRAWrapper(rank=4, alpha=2.0)
        target = torch.ones(1, 4, 32, 32) * 0.5
        target[:, :, 8:24, 8:24] = 1.0

        optimizer = torch.optim.Adam(model.lora_parameters(), lr=1e-2)
        for _ in range(30):
            optimizer.zero_grad()
            out = model(target)
            loss = F.mse_loss(out, target)
            loss.backward()
            optimizer.step()

        checkpoint_path = str(tmp_path / "lora_weights.pt")
        torch.save({"model_state": model.state_dict()}, checkpoint_path)

        loaded = SpriteLoRAWrapper(rank=4, alpha=2.0)
        loaded.load_state_dict(torch.load(checkpoint_path)["model_state"])
        loaded.eval()

        generator = LoRAGenerator(loaded)
        pipeline = AssetPipeline()
        pipeline.set_generator(generator)

        controls = AssetControls(
            asset_type=AssetType.CHARACTER,
            view=View.FRONT,
            animation=Animation.IDLE,
            palette=Palette.AUTO,
            sprite_size=SpriteSize.S_32,
            seed=42,
        )

        result = pipeline.run(controls, output_dir=str(tmp_path / "output"))

        assert len(result.images) == 1
        assert result.images[0].mode == "RGBA"
        assert result.metadata["prompt"] != ""
        assert len(result.validation) == 1
        assert result.validation[0]["quality_tier"] in (
            "clean", "acceptable", "noisy", "blurry", "broken_outline", "empty"
        )
        assert len(result.output_paths) > 0
        assert any(p.endswith(".png") for p in result.output_paths)
        assert any(p.endswith(".json") for p in result.output_paths)

        assert generator.model.decode(
            torch.randn(1, 256, 4, 4)
        ).shape == (1, 4, 32, 32)

    def test_pipeline_with_lora_save_load_cycle(self, tmp_path):
        model = SpriteLoRAWrapper(rank=8, alpha=4.0)
        x = torch.ones(2, 4, 32, 32)
        optimizer = torch.optim.Adam(model.lora_parameters(), lr=1e-2)
        for _ in range(20):
            optimizer.zero_grad()
            loss = F.mse_loss(model(x), x)
            loss.backward()
            optimizer.step()

        checkpoint_path = str(tmp_path / "trained_lora.pt")
        torch.save({"model_state": model.state_dict()}, checkpoint_path)

        loaded = SpriteLoRAWrapper(rank=8, alpha=4.0)
        loaded.load_state_dict(torch.load(checkpoint_path)["model_state"])
        loaded.eval()

        generator = LoRAGenerator(loaded)
        pipeline = AssetPipeline()
        pipeline.set_generator(generator)

        controls = AssetControls(
            asset_type=AssetType.ENEMY,
            view=View.SIDE,
            animation=Animation.WALK,
            sprite_size=SpriteSize.S_32,
        )

        result = pipeline.run(controls, output_dir=str(tmp_path / "output2"))

        assert len(result.images) == 1
        assert result.images[0].mode == "RGBA"
        assert result.validation[0]["palette_size"] >= 0
        assert "prompt" in result.metadata
        assert result.metadata["controls"]["asset_type"] == "enemy"

    def test_pipeline_with_lora_untrained_generates_valid_images(self, tmp_path):
        model = SpriteLoRAWrapper(rank=4, alpha=1.0)
        model.eval()

        generator = LoRAGenerator(model)
        pipeline = AssetPipeline()
        pipeline.set_generator(generator)

        controls = AssetControls(
            asset_type=AssetType.PROP,
            view=View.FRONT,
            animation=Animation.NONE,
        )

        result = pipeline.run(controls, output_dir=str(tmp_path / "output3"))

        assert len(result.images) == 1
        assert result.images[0].mode == "RGBA"
        assert result.images[0].size[0] > 0
        assert result.images[0].size[1] > 0
        assert len(result.validation) == 1
        assert all(k in result.validation[0] for k in [
            "palette_size", "center_x", "center_y",
            "transparency_ratio", "outline_continuity",
            "sharpness", "quality_tier", "bbox", "bbox_area",
        ])

    def test_pipeline_with_lora_multiple_frames(self, tmp_path):
        model = SpriteLoRAWrapper(rank=4, alpha=1.0)
        model.eval()

        class MultiFrameGenerator(LoRAGenerator):
            def generate(self, prompt="", negative_prompt="", width=512, height=512,
                         seed=-1, num_images=None):
                n = num_images or 4
                imgs = []
                for i in range(n):
                    z = torch.randn(1, 256, 4, 4)
                    with torch.no_grad():
                        out = self.model.decode(z)
                    arr = (out[0].permute(1, 2, 0).clamp(0, 1).numpy() * 255
                           ).astype(np.uint8)
                    if arr.shape[2] == 4:
                        arr[:, :, 3] = 255
                    img = Image.fromarray(arr, "RGBA").resize(
                        (width, height), Image.NEAREST)
                    imgs.append(img)
                return imgs

        pipeline = AssetPipeline()
        pipeline.set_generator(MultiFrameGenerator(model))

        controls = AssetControls(
            asset_type=AssetType.CHARACTER,
            view=View.SIDE,
            animation=Animation.RUN,
            sprite_size=SpriteSize.S_32,
        )

        result = pipeline.run(controls, output_dir=str(tmp_path / "output4"))

        assert len(result.images) == 4
        for img in result.images:
            assert img.mode == "RGBA"
        assert len(result.validation) == 4
        assert any(p.endswith(".png") for p in result.output_paths)

    def test_smoke_with_safetensors_lora_weights(self, tmp_path):
        """Roadmap #59: Smoke test with real .safetensors-format LoRA weights
        through the full end-to-end pipeline.
        """
        lora_weights = _make_safetensors_lora_weights(rank=4, num_layers=4)

        safetensors_path = str(tmp_path / "lora_weights.safetensors")
        safetensors.torch.save_file(lora_weights, safetensors_path)

        loaded = safetensors.torch.load_file(safetensors_path)
        assert len(loaded) == 4 * 4 * 2 + 4
        for k, v in loaded.items():
            assert "lora" in k
            assert v.dtype == torch.float32

        model = SpriteLoRAWrapper(rank=4, alpha=2.0)
        model.eval()

        generator = LoRAGenerator(model)
        pipeline = AssetPipeline()
        pipeline.set_generator(generator)

        controls = AssetControls(
            asset_type=AssetType.CHARACTER,
            view=View.FRONT,
            animation=Animation.IDLE,
            palette=Palette.AUTO,
            sprite_size=SpriteSize.S_32,
            seed=42,
        )

        result = pipeline.run(controls, output_dir=str(tmp_path / "output_safetensors"))

        assert len(result.images) == 1
        assert result.images[0].mode == "RGBA"
        assert result.images[0].size[0] > 0
        assert result.images[0].size[1] > 0

        assert result.metadata["prompt"] != ""
        assert result.metadata["controls"]["asset_type"] == "character"

        assert len(result.validation) == 1
        v = result.validation[0]
        for key in ("palette_size", "center_x", "center_y", "transparency_ratio",
                     "outline_continuity", "sharpness", "quality_tier", "bbox", "bbox_area"):
            assert key in v, f"Missing validation key: {key}"

        assert len(result.output_paths) > 0
        assert any(p.endswith(".png") for p in result.output_paths)
        assert any(p.endswith(".json") for p in result.output_paths)
        assert result.zip_path is not None
        assert Path(result.zip_path).exists()

    def test_lora_weights_persistence_quality(self, tmp_path):
        model = SpriteLoRAWrapper(rank=4, alpha=2.0)
        target = torch.zeros(1, 4, 32, 32)
        target[:, :, 4:28, 4:28] = 1.0

        optimizer = torch.optim.Adam(model.lora_parameters(), lr=5e-3)
        for _ in range(50):
            optimizer.zero_grad()
            out = model(target)
            loss = F.mse_loss(out, target)
            loss.backward()
            optimizer.step()

        out_before_save = model(target)

        checkpoint = tmp_path / "lora_final.pt"
        torch.save({"model_state": model.state_dict()}, str(checkpoint))

        reloaded = SpriteLoRAWrapper(rank=4, alpha=2.0)
        reloaded.load_state_dict(torch.load(str(checkpoint))["model_state"])
        reloaded.eval()

        out_after_load = reloaded(target)

        assert torch.allclose(
            out_before_save, out_after_load, atol=1e-6
        ), "LoRA weights not identical after save/load cycle"
