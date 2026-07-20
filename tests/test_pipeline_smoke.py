"""End-to-end smoke test for the full asset pipeline (roadmap: MVP)."""

import os
import json
import zipfile
from pathlib import Path
from typing import List, Optional

from PIL import Image
import numpy as np

from backend.modules.pipeline.orchestrator import AssetPipeline, PipelineConfig
from backend.modules.prompt_builder.controls import (
    AssetControls, AssetType, View, Palette, Animation, SpriteSize,
)


class FakeGenerator:
    def __init__(self, num_images: int = 1, size: int = 64):
        self.num_images = num_images
        self.size = size

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        seed: int = -1,
        num_images: Optional[int] = None,
    ) -> List[Image.Image]:
        n = num_images or self.num_images
        images = []
        for i in range(n):
            arr = np.zeros((height, width, 4), dtype=np.uint8)
            cy, cx = height // 4, width // 4
            hh, hw = height // 2, width // 2
            r, g, b = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)][i % 4]
            arr[cy:cy + hh, cx:cx + hw, 0] = r
            arr[cy:cy + hh, cx:cx + hw, 1] = g
            arr[cy:cy + hh, cx:cx + hw, 2] = b
            arr[cy:cy + hh, cx:cx + hw, 3] = 255
            images.append(Image.fromarray(arr, "RGBA"))
        return images


class TestPipelineSmoke:
    def test_pipeline_runs_end_to_end_with_single_image(self, tmp_path):
        pipeline = AssetPipeline()
        pipeline.set_generator(FakeGenerator(num_images=1))

        controls = AssetControls(
            asset_type=AssetType.CHARACTER,
            view=View.FRONT,
            animation=Animation.IDLE,
            palette=Palette.AUTO,
            sprite_size=SpriteSize.S_32,
            seed=42,
        )

        result = pipeline.run(controls, output_dir=str(tmp_path))

        assert len(result.images) == 1
        assert result.images[0].mode == "RGBA"
        assert result.metadata["prompt"] != ""
        assert len(result.validation) == 1
        assert result.validation[0]["quality_tier"] in ("clean", "acceptable", "noisy", "blurry", "broken_outline", "empty")
        assert len(result.output_paths) > 0
        assert any(p.endswith(".png") for p in result.output_paths)
        assert any(p.endswith(".json") for p in result.output_paths)

    def test_pipeline_runs_with_multiple_frames(self, tmp_path):
        pipeline = AssetPipeline()
        pipeline.set_generator(FakeGenerator(num_images=4))

        controls = AssetControls(
            asset_type=AssetType.CHARACTER,
            view=View.SIDE,
            animation=Animation.WALK,
            sprite_size=SpriteSize.S_32,
        )

        result = pipeline.run(controls, output_dir=str(tmp_path))

        assert len(result.images) == 4
        for img in result.images:
            assert img.mode == "RGBA"
        assert len(result.validation) == 4
        assert any(p.endswith(".tres") for p in result.output_paths)

    def test_pipeline_creates_zip_package(self, tmp_path):
        pipeline = AssetPipeline()
        pipeline.set_generator(FakeGenerator(num_images=2))

        controls = AssetControls(
            asset_type=AssetType.ENEMY,
            view=View.TOP,
            animation=Animation.ATTACK,
        )

        result = pipeline.run(controls, output_dir=str(tmp_path))

        assert result.zip_path is not None
        assert os.path.isfile(result.zip_path)
        with zipfile.ZipFile(result.zip_path, "r") as zf:
            names = zf.namelist()
            assert any(n.endswith(".png") for n in names)
            assert "metadata.json" in names

    def test_pipeline_metadata_is_valid_json(self, tmp_path):
        pipeline = AssetPipeline()
        pipeline.set_generator(FakeGenerator(num_images=1))

        controls = AssetControls(
            asset_type=AssetType.BUILDING,
            view=View.ISOMETRIC,
            palette=Palette.RETRO_16,
        )

        result = pipeline.run(controls, output_dir=str(tmp_path))

        meta_paths = [p for p in result.output_paths if p.endswith("metadata.json")]
        assert len(meta_paths) == 1
        with open(meta_paths[0]) as f:
            meta = json.load(f)
        assert "prompt" in meta
        assert "controls" in meta
        assert meta["controls"]["asset_type"] == "building"
        assert meta["controls"]["view"] == "isometric"
        assert meta["controls"]["palette"] == "retro_16"
        assert "validation" in meta
        assert "outputs" in meta

    def test_pipeline_with_all_postprocessing_disabled(self, tmp_path):
        config = PipelineConfig(
            remove_bg=False,
            reduce_palette=False,
            pixel_cleanup=False,
            auto_center=False,
            auto_pad=False,
            normalize_size=False,
            upscale=1,
            outline_cleanup=False,
        )
        pipeline = AssetPipeline(config=config)
        pipeline.set_generator(FakeGenerator(num_images=1))

        controls = AssetControls(
            asset_type=AssetType.PROP,
            view=View.FRONT,
            animation=Animation.NONE,
        )

        result = pipeline.run(controls, output_dir=str(tmp_path))

        assert len(result.images) == 1
        assert result.images[0].mode == "RGBA"

    def test_pipeline_with_various_asset_types(self, tmp_path):
        pipeline = AssetPipeline()
        pipeline.set_generator(FakeGenerator(num_images=1))

        for asset_type in [AssetType.CHARACTER, AssetType.VEHICLE, AssetType.TREE, AssetType.PROP]:
            controls = AssetControls(
                asset_type=asset_type,
                view=View.FRONT,
                animation=Animation.IDLE,
            )
            result = pipeline.run(controls, output_dir=str(tmp_path / asset_type.value))
            assert len(result.images) == 1
            assert result.images[0].mode == "RGBA"

    def test_pipeline_with_seed_reproducibility(self, tmp_path):
        pipeline = AssetPipeline()
        pipeline.set_generator(FakeGenerator(num_images=1))

        controls = AssetControls(
            asset_type=AssetType.CHARACTER,
            view=View.FRONT,
            seed=42,
        )

        result1 = pipeline.run(controls, output_dir=str(tmp_path / "run1"))
        result2 = pipeline.run(controls, output_dir=str(tmp_path / "run2"))

        assert result1.metadata["controls"]["seed"] == 42
        assert result2.metadata["controls"]["seed"] == 42

    def test_pipeline_without_generator_raises_error(self):
        pipeline = AssetPipeline()
        controls = AssetControls()

        import pytest
        with pytest.raises(RuntimeError, match="Generator not set"):
            pipeline.run(controls)

    def test_pipeline_validation_metrics_are_present(self, tmp_path):
        pipeline = AssetPipeline()
        pipeline.set_generator(FakeGenerator(num_images=2))

        controls = AssetControls(
            asset_type=AssetType.CHARACTER,
            view=View.FRONT,
            animation=Animation.IDLE,
        )

        result = pipeline.run(controls, output_dir=str(tmp_path))

        for v in result.validation:
            assert "palette_size" in v
            assert "center_x" in v
            assert "center_y" in v
            assert "transparency_ratio" in v
            assert "outline_continuity" in v
            assert "sharpness" in v
            assert "quality_tier" in v
            assert "bbox" in v
            assert "bbox_area" in v

    def test_pipeline_duplicate_detection(self, tmp_path):
        pipeline = AssetPipeline()
        pipeline.set_generator(FakeGenerator(num_images=4))

        controls = AssetControls(
            asset_type=AssetType.CHARACTER,
            view=View.FRONT,
            animation=Animation.WALK,
        )

        result = pipeline.run(controls, output_dir=str(tmp_path))

        for v in result.validation:
            assert "duplicates" in v
