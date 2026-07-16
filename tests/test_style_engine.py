import numpy as np
from PIL import Image

from backend.modules.style_engine import StyleEngine, StylePreset, STYLE_PRESETS
from backend.modules.postprocess.palettes import KNOWN_PALETTES, get_palette
from backend.modules.pipeline.orchestrator import AssetPipeline, PipelineConfig
from backend.modules.prompt_builder.controls import AssetControls, AssetType, View


def _make_test_image(width=32, height=32):
    arr = np.zeros((height, width, 4), dtype=np.uint8)
    arr[4:28, 4:28, :3] = [200, 100, 50]
    arr[4:28, 4:28, 3] = 255
    return Image.fromarray(arr)


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


class TestStyleEngineInit:
    def test_engine_creates_successfully(self):
        engine = StyleEngine()
        assert engine is not None

    def test_get_available_palettes(self):
        engine = StyleEngine()
        palettes = engine.get_available_palettes()
        assert "retro_8" in palettes
        assert "retro_16" in palettes
        assert "gameboy" in palettes
        assert "snes" in palettes
        assert "monochrome" in palettes

    def test_get_palette_colors(self):
        engine = StyleEngine()
        colors = engine.get_palette_colors("gameboy")
        assert len(colors) == 4
        assert colors[0] == (15, 56, 15)

    def test_get_palette_colors_fallback(self):
        engine = StyleEngine()
        colors = engine.get_palette_colors("nonexistent")
        assert colors == KNOWN_PALETTES["retro_16"]


class TestStyleEnginePaletteLock:
    def test_apply_palette_lock(self):
        engine = StyleEngine()
        img = _make_test_image()
        locked = engine.apply_palette_lock(img, palette_name="gameboy")
        assert locked.mode == "RGBA"
        assert locked.size == img.size
        arr = np.array(locked)
        opaque = arr[:, :, 3] > 128
        locked_colors = set(tuple(c) for c in arr[opaque][:, :3])
        assert locked_colors.issubset(set(KNOWN_PALETTES["gameboy"]))

    def test_apply_palette_lock_preserves_transparency(self):
        engine = StyleEngine()
        img = _make_test_image()
        locked = engine.apply_palette_lock(img, palette_name="retro_8")
        assert np.array_equal(np.array(img)[:, :, 3], np.array(locked)[:, :, 3])

    def test_apply_palette_lock_fully_transparent(self):
        engine = StyleEngine()
        arr = np.zeros((16, 16, 4), dtype=np.uint8)
        img = Image.fromarray(arr)
        locked = engine.apply_palette_lock(img, palette_name="retro_16")
        assert np.array_equal(np.array(img), np.array(locked))

    def test_apply_palette_lock_monochrome(self):
        engine = StyleEngine()
        img = _make_test_image()
        locked = engine.apply_palette_lock(img, palette_name="monochrome")
        arr = np.array(locked)
        opaque = arr[:, :, 3] > 128
        locked_colors = set(tuple(c) for c in arr[opaque][:, :3])
        assert locked_colors.issubset(set(KNOWN_PALETTES["monochrome"]))

    def test_apply_palette_lock_different_palettes(self):
        engine = StyleEngine()
        img = _make_test_image()
        locked_a = engine.apply_palette_lock(img, palette_name="gameboy")
        locked_b = engine.apply_palette_lock(img, palette_name="snes")
        assert not np.array_equal(np.array(locked_a), np.array(locked_b))


class TestStyleEngineConsistency:
    def test_check_consistency_perfect(self):
        engine = StyleEngine()
        img = _make_test_image()
        locked = engine.apply_palette_lock(img, palette_name="retro_16")
        score = engine.check_consistency(locked, palette_name="retro_16")
        assert score > 0.99

    def test_check_consistency_low_for_mismatch(self):
        engine = StyleEngine()
        img = _make_test_image()
        score = engine.check_consistency(img, palette_name="gameboy")
        assert score < 0.99

    def test_check_consistency_transparent(self):
        engine = StyleEngine()
        arr = np.zeros((16, 16, 4), dtype=np.uint8)
        img = Image.fromarray(arr)
        score = engine.check_consistency(img, palette_name="retro_16")
        assert score == 1.0


class TestStylePresets:
    def test_presets_defined(self):
        assert "retro_8bit" in STYLE_PRESETS
        assert "retro_16bit" in STYLE_PRESETS
        assert "retro_32bit" in STYLE_PRESETS
        assert "gameboy" in STYLE_PRESETS
        assert "monochrome" in STYLE_PRESETS
        assert "snes" in STYLE_PRESETS

    def test_preset_attributes(self):
        preset = STYLE_PRESETS["gameboy"]
        assert preset.palette_name == "gameboy"
        assert preset.apply_palette_lock is True
        assert len(preset.description) > 0

    def test_get_style_presets(self):
        engine = StyleEngine()
        presets = engine.get_style_presets()
        assert "gameboy" in presets
        assert presets["gameboy"].palette_name == "gameboy"

    def test_apply_style_preset_gameboy(self):
        engine = StyleEngine()
        img = _make_test_image()
        result = engine.apply_style_preset(img, "gameboy")
        arr = np.array(result)
        opaque = arr[:, :, 3] > 128
        colors = set(tuple(c) for c in arr[opaque][:, :3])
        assert colors.issubset(set(KNOWN_PALETTES["gameboy"]))

    def test_apply_style_preset_snes(self):
        engine = StyleEngine()
        img = _make_test_image()
        result = engine.apply_style_preset(img, "snes")
        arr = np.array(result)
        opaque = arr[:, :, 3] > 128
        colors = set(tuple(c) for c in arr[opaque][:, :3])
        assert colors.issubset(set(KNOWN_PALETTES["snes"]))

    def test_apply_style_preset_invalid_falls_back(self):
        engine = StyleEngine()
        img = _make_test_image()
        result = engine.apply_style_preset(img, "nonexistent_preset")
        arr = np.array(result)
        opaque = arr[:, :, 3] > 128
        colors = set(tuple(c) for c in arr[opaque][:, :3])
        assert colors.issubset(set(KNOWN_PALETTES["retro_16"]))

    def test_apply_style_preset_case_insensitive(self):
        engine = StyleEngine()
        img = _make_test_image()
        result = engine.apply_style_preset(img, "GAMEBOY")
        arr = np.array(result)
        opaque = arr[:, :, 3] > 128
        colors = set(tuple(c) for c in arr[opaque][:, :3])
        assert colors.issubset(set(KNOWN_PALETTES["gameboy"]))


class TestExtractPaletteFromReference:
    def test_extract_palette_from_ref_image(self):
        engine = StyleEngine()
        ref = Image.new("RGB", (64, 64), (100, 150, 200))
        palette = engine.extract_palette_from_reference(ref, num_colors=4)
        assert len(palette) <= 4
        assert all(len(c) == 3 for c in palette)
        assert (0, 0, 0) in palette

    def test_extract_palette_multicolor(self):
        engine = StyleEngine()
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        arr[:32, :32] = [255, 0, 0]
        arr[:32, 32:] = [0, 255, 0]
        arr[32:, :32] = [0, 0, 255]
        arr[32:, 32:] = [255, 255, 0]
        ref = Image.fromarray(arr)
        palette = engine.extract_palette_from_reference(ref, num_colors=8)
        assert len(palette) >= 3

    def test_extract_palette_fully_transparent(self):
        engine = StyleEngine()
        arr = np.zeros((64, 64, 4), dtype=np.uint8)
        ref = Image.fromarray(arr)
        palette = engine.extract_palette_from_reference(ref, num_colors=8)
        assert len(palette) >= 2


class TestApplyReferenceStyle:
    def test_apply_reference_style(self):
        engine = StyleEngine()
        ref = Image.new("RGB", (64, 64), (50, 100, 150))
        images = [_make_test_image(), _make_test_image()]
        result = engine.apply_reference_style(images, ref, num_colors=8)
        assert len(result) == len(images)
        assert all(img.mode == "RGBA" for img in result)
        assert all(img.size == images[0].size for img in result)

    def test_apply_reference_style_preserves_transparency(self):
        engine = StyleEngine()
        ref = Image.new("RGB", (64, 64), (50, 100, 150))
        images = [_make_test_image()]
        result = engine.apply_reference_style(images, ref, num_colors=8)
        orig_alpha = np.array(images[0])[:, :, 3]
        res_alpha = np.array(result[0])[:, :, 3]
        assert np.array_equal(orig_alpha, res_alpha)

    def test_apply_reference_style_does_not_mutate_input(self):
        engine = StyleEngine()
        ref = Image.new("RGB", (64, 64), (50, 100, 150))
        images = [_make_test_image()]
        original = np.array(images[0])
        engine.apply_reference_style(images, ref, num_colors=8)
        assert np.array_equal(np.array(images[0]), original)


class TestPipelineWithStyleEngine:
    def test_pipeline_palette_lock_true(self, tmp_path):
        config = PipelineConfig(palette_lock=True, palette_name="gameboy")
        pipeline = AssetPipeline(config=config)
        pipeline.set_generator(FakeGenerator(num_images=2))
        controls = AssetControls(asset_type=AssetType.CHARACTER, view=View.FRONT)
        result = pipeline.run(controls, output_dir=str(tmp_path))
        for img in result.images:
            arr = np.array(img)
            opaque = arr[:, :, 3] > 128
            if opaque.any():
                colors = set(tuple(c) for c in arr[opaque][:, :3])
                assert colors.issubset(set(KNOWN_PALETTES["gameboy"]))

    def test_pipeline_metadata_has_palette_name(self, tmp_path):
        config = PipelineConfig(palette_lock=True, palette_name="monochrome")
        pipeline = AssetPipeline(config=config)
        pipeline.set_generator(FakeGenerator(num_images=1))
        controls = AssetControls(asset_type=AssetType.PROP, view=View.FRONT)
        result = pipeline.run(controls, output_dir=str(tmp_path))
        assert result.metadata["controls"]["palette_name"] == "monochrome"
