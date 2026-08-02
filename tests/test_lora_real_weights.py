"""Real-LoRA-weights end-to-end smoke test (ROADMAP MVP item #9).

Implements: "Smoke test with real LoRA weights end-to-end".

Unlike the toy tests in test_lora_end_to_end.py (which drive the pipeline
with a small autoencoder wrapper and synthetic tensors), this test exercises
the REAL Stable Diffusion inference path with REAL LoRA weights:

  1. loads a real (tiny) Stable Diffusion 1.x checkpoint,
  2. trains a genuine LoRA adapter on it with peft + the standard diffusers
     training objective (noise prediction on VAE latents),
  3. exports the trained adapter to the canonical diffusers
     ``.safetensors`` format via ``StableDiffusionPipeline.save_lora_weights``,
  4. loads it back through the real ``SDGenerator`` (which calls
     ``StableDiffusionPipeline.from_pretrained`` + ``load_lora_weights``),
  5. runs the full asset pipeline end-to-end with those real LoRA weights.

The tiny checkpoint (``hf-internal-testing/tiny-stable-diffusion-torch``,
~9 MB) is the same model diffusers uses in its own CI, so this stays fast
enough to run on CPU. If the model cannot be downloaded the tests are
skipped rather than failed.
"""
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

TINY_SD_MODEL_ID = "hf-internal-testing/tiny-stable-diffusion-torch"


def _load_sd_components(model_id):
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    return tokenizer, text_encoder, vae, unet, noise_scheduler


def _train_and_save_real_lora(model_id, output_dir, seed=1234, steps=20, rank=4):
    """Train a real LoRA adapter and export it to diffusers .safetensors format."""
    torch.manual_seed(seed)
    from diffusers import StableDiffusionPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict

    tokenizer, text_encoder, vae, unet, noise_scheduler = _load_sd_components(model_id)
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    unet.add_adapter(
        LoraConfig(
            r=rank,
            lora_alpha=rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
    )
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()), lr=1e-3
    )
    text_encoder.eval()
    vae.eval()
    unet.train()

    input_ids = tokenizer(
        ["a pixel art sprite of a knight"],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids
    with torch.no_grad():
        encoder_hidden_states = text_encoder(input_ids)[0]
        latents = (
            vae.encode(torch.randn(1, 3, 128, 128)).latent_dist.sample()
            * vae.config.scaling_factor
        )

    for _ in range(steps):
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (1,), device=latents.device
        ).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states)[0]
        loss = F.mse_loss(noise_pred.float(), noise.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    unet.to(torch.float32)
    lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionPipeline.save_lora_weights(
        output_dir, unet_lora_layers=lora_state_dict, safe_serialization=True
    )
    return output_dir


@pytest.fixture(scope="module")
def real_lora_dir(tmp_path_factory):
    """Train + export real LoRA weights once for the whole module."""
    output_dir = str(tmp_path_factory.mktemp("real_sd_lora"))
    try:
        return _train_and_save_real_lora(TINY_SD_MODEL_ID, output_dir)
    except Exception as exc:  # noqa: BLE001 - network/model unavailability
        pytest.skip(f"Real SD LoRA smoke test unavailable: {exc}")


class TestRealSDLoRAEndToEnd:
    def test_real_lora_weights_are_valid_safetensors(self, real_lora_dir):
        import safetensors.torch

        weights_file = Path(real_lora_dir) / "pytorch_lora_weights.safetensors"
        assert weights_file.exists()
        assert weights_file.stat().st_size > 0

        state_dict = safetensors.torch.load_file(str(weights_file))
        assert len(state_dict) > 0
        assert all("lora" in key for key in state_dict)
        assert all(key.startswith("unet.") for key in state_dict)

    def test_real_lora_weights_load_through_sdgenerator(self, real_lora_dir):
        from backend.modules.generator.sd_generator import SDGenerator

        generator = SDGenerator(
            model_id=TINY_SD_MODEL_ID,
            lora_path=real_lora_dir,
            device="cpu",
            torch_dtype=torch.float32,
        )
        images = generator.generate(
            prompt="a pixel art sprite of a knight",
            seed=42,
            num_images=1,
            num_inference_steps=8,
            width=128,
            height=128,
        )
        assert len(images) == 1
        assert isinstance(images[0], Image.Image)
        assert images[0].size == (128, 128)
        assert images[0].mode == "RGB"

        bare = SDGenerator(
            model_id=TINY_SD_MODEL_ID,
            device="cpu",
            torch_dtype=torch.float32,
        )
        bare_images = bare.generate(
            prompt="a pixel art sprite of a knight",
            seed=42,
            num_images=1,
            num_inference_steps=8,
            width=128,
            height=128,
        )
        assert not np.array_equal(
            np.array(images[0]), np.array(bare_images[0])
        ), "Loaded real LoRA weights must change generation output"

    def test_full_pipeline_end_to_end_with_real_lora_weights(
        self, real_lora_dir, tmp_path
    ):
        from backend.modules.generator.sd_generator import SDGenerator
        from backend.modules.pipeline.orchestrator import AssetPipeline, PipelineConfig
        from backend.modules.prompt_builder.controls import (
            AssetControls,
            Animation,
            AssetType,
            Palette,
            SpriteSize,
            View,
        )

        generator = SDGenerator(
            model_id=TINY_SD_MODEL_ID,
            lora_path=real_lora_dir,
            device="cpu",
            torch_dtype=torch.float32,
        )
        pipeline = AssetPipeline(config=PipelineConfig(export_engine="godot"))
        pipeline.set_generator(generator)

        controls = AssetControls(
            asset_type=AssetType.CHARACTER,
            view=View.FRONT,
            animation=Animation.IDLE,
            palette=Palette.AUTO,
            sprite_size=SpriteSize.S_16,
            seed=42,
        )

        result = pipeline.run(controls, output_dir=str(tmp_path))

        assert len(result.images) == 1
        assert result.images[0].mode == "RGBA"
        assert result.metadata["prompt"] != ""
        assert result.metadata["controls"]["asset_type"] == "character"

        assert len(result.validation) == 1
        validation = result.validation[0]
        assert validation["quality_tier"] in (
            "clean",
            "acceptable",
            "noisy",
            "blurry",
            "broken_outline",
            "empty",
            "extreme_aspect",
        )
        assert all(k in validation for k in [
            "palette_size",
            "center_x",
            "center_y",
            "transparency_ratio",
            "outline_continuity",
            "sharpness",
            "quality_tier",
            "aspect_ratio",
            "bbox",
            "bbox_area",
        ])

        assert len(result.output_paths) > 0
        assert any(p.endswith(".png") for p in result.output_paths)
        assert any(p.endswith(".json") for p in result.output_paths)
        assert result.zip_path is not None
        assert Path(result.zip_path).exists()
