"""Tests for palette lock — Phase 2 style consistency engine."""

import numpy as np
from PIL import Image

from backend.modules.postprocess.processor import palette_lock
from backend.modules.postprocess.palettes import KNOWN_PALETTES, get_palette
from backend.modules.validation.metrics import palette_consistency
from backend.modules.pipeline.orchestrator import AssetPipeline, PipelineConfig
from backend.modules.prompt_builder.controls import AssetControls, AssetType, View, Animation, SpriteSize


def _make_test_image(width=32, height=32):
    arr = np.zeros((height, width, 4), dtype=np.uint8)
    arr[4:28, 4:28, :3] = [200, 100, 50]
    arr[4:28, 4:28, 3] = 255
    return Image.fromarray(arr)


class TestPalettesModule:
    def test_known_palettes_are_defined(self):
        assert "retro_8" in KNOWN_PALETTES
        assert "retro_16" in KNOWN_PALETTES
        assert "retro_32" in KNOWN_PALETTES
        assert "monochrome" in KNOWN_PALETTES
        assert "gameboy" in KNOWN_PALETTES
        assert "snes" in KNOWN_PALETTES

    def test_get_palette_returns_correct_colors(self):
        pal = get_palette("gameboy")
        assert len(pal) == 4
        assert pal[0] == (15, 56, 15)

    def test_get_palette_fallback_on_unknown(self):
        pal = get_palette("nonexistent")
        assert pal == KNOWN_PALETTES["retro_16"]

    def test_get_palette_normalizes_name(self):
        pal1 = get_palette("RETRO_16")
        pal2 = get_palette("retro-16")
        assert pal1 == pal2 == KNOWN_PALETTES["retro_16"]


class TestPaletteLock:
    def test_palette_lock_maps_colors_to_nearest(self):
        img = _make_test_image()
        locked = palette_lock(img, palette_name="gameboy")
        assert locked.mode == "RGBA"
        assert locked.size == img.size
        arr = np.array(locked)
        opaque = arr[:, :, 3] > 128
        locked_colors = set(tuple(c) for c in arr[opaque][:, :3])
        gameboy_set = set(KNOWN_PALETTES["gameboy"])
        assert locked_colors.issubset(gameboy_set)

    def test_palette_lock_preserves_transparency(self):
        img = _make_test_image()
        locked = palette_lock(img, palette_name="retro_8")
        arr_in = np.array(img)
        arr_out = np.array(locked)
        assert np.array_equal(arr_in[:, :, 3], arr_out[:, :, 3])

    def test_palette_lock_fully_transparent_image(self):
        arr = np.zeros((16, 16, 4), dtype=np.uint8)
        img = Image.fromarray(arr)
        locked = palette_lock(img, palette_name="retro_16")
        assert np.array_equal(np.array(img), np.array(locked))

    def test_palette_lock_monochrome(self):
        img = _make_test_image()
        locked = palette_lock(img, palette_name="monochrome")
        arr = np.array(locked)
        opaque = arr[:, :, 3] > 128
        locked_colors = set(tuple(c) for c in arr[opaque][:, :3])
        monochrome_set = set(KNOWN_PALETTES["monochrome"])
        assert locked_colors.issubset(monochrome_set)

    def test_palette_lock_different_palettes_produce_different_results(self):
        img = _make_test_image()
        locked_a = palette_lock(img, palette_name="gameboy")
        locked_b = palette_lock(img, palette_name="snes")
        arr_a = np.array(locked_a)
        arr_b = np.array(locked_b)
        assert not np.array_equal(arr_a, arr_b)


class TestPaletteConsistencyMetric:
    def test_palette_consistency_perfect_score(self):
        img = _make_test_image()
        locked = palette_lock(img, palette_name="retro_16")
        score = palette_consistency(locked, palette_name="retro_16")
        assert score > 0.99

    def test_palette_consistency_low_score_for_mismatched(self):
        img = _make_test_image()
        score = palette_consistency(img, palette_name="gameboy")
        assert score < 0.99

    def test_palette_consistency_transparent_image(self):
        arr = np.zeros((16, 16, 4), dtype=np.uint8)
        img = Image.fromarray(arr)
        score = palette_consistency(img, palette_name="retro_16")
        assert score == 1.0


class FakeGenerator:
    def __init__(self, num_images=1, size=64):
        self.num_images = num_images
        self.size = size

    def generate(self, prompt="", negative_prompt="", width=512, height=512, seed=-1, num_images=None):
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


class TestPipelinePaletteLock:
    def test_palette_lock_in_pipeline_with_gameboy(self, tmp_path):
        config = PipelineConfig(palette_lock=True, palette_name="gameboy")
        pipeline = AssetPipeline(config=config)
        pipeline.set_generator(FakeGenerator(num_images=2))
        controls = AssetControls(
            asset_type=AssetType.CHARACTER,
            view=View.FRONT,
        )
        result = pipeline.run(controls, output_dir=str(tmp_path))
        for img in result.images:
            arr = np.array(img)
            opaque = arr[:, :, 3] > 128
            if opaque.any():
                colors = set(tuple(c) for c in arr[opaque][:, :3])
                assert colors.issubset(set(KNOWN_PALETTES["gameboy"]))

    def test_palette_lock_disabled_by_default(self, tmp_path):
        pipeline = AssetPipeline()
        pipeline.set_generator(FakeGenerator(num_images=1))
        controls = AssetControls(
            asset_type=AssetType.CHARACTER,
            view=View.FRONT,
        )
        result = pipeline.run(controls, output_dir=str(tmp_path))
        arr = np.array(result.images[0])
        opaque = arr[:, :, 3] > 128
        if opaque.any():
            colors = set(tuple(c) for c in arr[opaque][:, :3])
            has_non_locked = not colors.issubset(set(KNOWN_PALETTES["gameboy"]))
            assert has_non_locked

    def test_palette_lock_with_snes_in_pipeline(self, tmp_path):
        config = PipelineConfig(palette_lock=True, palette_name="snes")
        pipeline = AssetPipeline(config=config)
        pipeline.set_generator(FakeGenerator(num_images=1))
        controls = AssetControls(
            asset_type=AssetType.ENEMY,
            view=View.SIDE,
        )
        result = pipeline.run(controls, output_dir=str(tmp_path))
        arr = np.array(result.images[0])
        opaque = arr[:, :, 3] > 128
        if opaque.any():
            colors = set(tuple(c) for c in arr[opaque][:, :3])
            assert colors.issubset(set(KNOWN_PALETTES["snes"]))

    def test_palette_lock_metadata_contains_palette_name(self, tmp_path):
        config = PipelineConfig(palette_lock=True, palette_name="monochrome")
        pipeline = AssetPipeline(config=config)
        pipeline.set_generator(FakeGenerator(num_images=1))
        controls = AssetControls(
            asset_type=AssetType.PROP,
            view=View.FRONT,
        )
        result = pipeline.run(controls, output_dir=str(tmp_path))
        assert result.metadata["controls"]["palette_name"] == "monochrome"
