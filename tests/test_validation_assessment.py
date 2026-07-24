"""Tests for backend validation metrics (Phase 0: validate generated samples)."""

import numpy as np
from PIL import Image
import pytest

from backend.modules.validation.metrics import (
    assess_all,
    palette_size,
    sprite_centering,
    sprite_aspect_ratio,
    bounding_box,
    transparency_coverage,
    outline_continuity,
    pixel_sharpness,
    duplicate_detection,
    palette_consistency,
)


def _solid(rgb, size=(64, 64)):
    w, h = size
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[:, :, :3] = rgb
    arr[:, :, 3] = 255
    return Image.fromarray(arr, "RGBA")


def _sprite(rgb, size=(64, 64), sprite_size=(24, 24)):
    w, h = size
    sw, sh = sprite_size
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    ox = (w - sw) // 2
    oy = (h - sh) // 2
    arr[oy:oy+sh, ox:ox+sw, :3] = rgb
    arr[oy:oy+sh, ox:ox+sw, 3] = 255
    return Image.fromarray(arr, "RGBA")


def _checkerboard(size=(64, 64), tile=4):
    arr = np.zeros((*size, 4), dtype=np.uint8)
    arr[:, :, 3] = 255
    for y in range(0, size[0], tile):
        for x in range(0, size[1], tile):
            if (x // tile + y // tile) % 2 == 0:
                arr[y:y+tile, x:x+tile, :3] = [255, 255, 255]
            else:
                arr[y:y+tile, x:x+tile, :3] = [0, 0, 0]
    return Image.fromarray(arr, "RGBA")


class TestAssessAll:
    def test_clean_sprite(self):
        img = _sprite((100, 50, 200))
        result = assess_all(img)
        assert result["quality_tier"] == "clean"
        assert result["palette_size"] == 1
        assert 0.4 < result["center_x"] < 0.6
        assert 0.4 < result["center_y"] < 0.6
        assert "bbox" in result
        assert "transparency_ratio" in result
        assert result["outline_continuity"] == 1.0
        assert result["sharpness"] > 50
        assert result["palette_consistency"] >= 0

    def test_noisy_sprite_many_colors(self):
        arr = np.zeros((64, 64, 4), dtype=np.uint8)
        arr[:, :, 3] = 255
        for y in range(64):
            for x in range(64):
                arr[y, x, :3] = [y * 4 % 256, x * 4 % 256, (y + x) * 2 % 256]
        img = Image.fromarray(arr, "RGBA")
        result = assess_all(img)
        assert result["quality_tier"] == "noisy"
        assert result["palette_size"] > 128

    def test_blurry_sprite(self):
        from PIL import ImageFilter
        img = _checkerboard(size=(64, 64), tile=2).filter(
            ImageFilter.GaussianBlur(radius=12)
        )
        result = assess_all(img)
        assert result["sharpness"] < 50

    def test_empty_transparent_sprite(self):
        arr = np.zeros((64, 64, 4), dtype=np.uint8)
        img = Image.fromarray(arr, "RGBA")
        result = assess_all(img)
        assert result["palette_size"] == 0
        assert result["transparency_ratio"] == 1.0
        assert result["quality_tier"] == "empty"

    def test_outline_continuity_perfect(self):
        img = _sprite((200, 100, 50))
        result = assess_all(img)
        assert result["outline_continuity"] == 1.0

    def test_bbox_present(self):
        img = _sprite((255, 0, 0))
        result = assess_all(img)
        assert "bbox" in result
        assert result["bbox"]["w"] > 0
        assert result["bbox"]["h"] > 0
        assert result["bbox_area"] > 0

    def test_duplicate_detection(self):
        img = _sprite((100, 200, 50))
        different = _sprite((200, 100, 50))
        result_no_batch = assess_all(img)
        assert "duplicates" not in result_no_batch
        result_with_batch = assess_all(img, batch=[img, different])
        assert "duplicates" in result_with_batch

    def test_duplicate_detection_identical(self):
        img = _sprite((100, 200, 50))
        result = assess_all(img, batch=[img, img])
        assert result["duplicates"] == 1

    def test_duplicate_detection_all_unique(self):
        a = _solid((255, 0, 0))
        b = _solid((0, 255, 0))
        c = _solid((0, 0, 255))
        result = assess_all(a, batch=[a, b, c])
        assert result["duplicates"] == 0

    def test_all_metric_keys_present(self):
        img = _sprite((50, 100, 200))
        result = assess_all(img)
        expected = {
            "palette_size", "center_x", "center_y",
            "transparency_ratio", "outline_continuity",
            "sharpness", "quality_tier", "palette_consistency",
            "aspect_ratio",
        }
        assert expected.issubset(result.keys())

    def test_result_is_serializable(self):
        import json
        img = _sprite((50, 100, 200), size=(32, 32))
        result = assess_all(img)
        dumped = json.dumps(result)
        loaded = json.loads(dumped)
        assert loaded["quality_tier"] == result["quality_tier"]

    def test_different_sizes_and_aspect_ratios(self):
        cases = [(16, 16, 8), (32, 64, 20), (128, 32, 30), (64, 64, 32)]
        for w, h, sw in cases:
            img = _sprite((0, 200, 100), size=(w, h), sprite_size=(sw, sw))
            result = assess_all(img)
            assert "bbox" in result, f"bbox missing for size={w}x{h}, sprite={sw}"
            assert result["quality_tier"] in (
                "clean", "acceptable", "noisy", "blurry", "broken_outline", "empty", "extreme_aspect"
            )


class TestPaletteSize:
    def test_single_color(self):
        img = _solid((255, 0, 0))
        assert palette_size(img) == 1

    def test_multi_color(self):
        img = _checkerboard()
        assert palette_size(img) == 2

    def test_empty_alpha(self):
        arr = np.zeros((64, 64, 4), dtype=np.uint8)
        img = Image.fromarray(arr, "RGBA")
        assert palette_size(img) == 0

    def test_transparent_pixels_excluded(self):
        arr = np.zeros((32, 32, 4), dtype=np.uint8)
        arr[0, 0, :3] = [255, 0, 0]
        arr[0, 0, 3] = 255
        arr[1, 1, :3] = [0, 255, 0]
        arr[1, 1, 3] = 0
        img = Image.fromarray(arr, "RGBA")
        assert palette_size(img) == 1


class TestSpriteCentering:
    def test_centered(self):
        arr = np.zeros((64, 64, 4), dtype=np.uint8)
        arr[16:48, 16:48, :3] = [255, 0, 0]
        arr[16:48, 16:48, 3] = 255
        img = Image.fromarray(arr, "RGBA")
        cx, cy = sprite_centering(img)
        assert cx == pytest.approx(0.5, abs=0.05)
        assert cy == pytest.approx(0.5, abs=0.05)

    def test_top_left(self):
        arr = np.zeros((64, 64, 4), dtype=np.uint8)
        arr[0:16, 0:16, :3] = [0, 255, 0]
        arr[0:16, 0:16, 3] = 255
        img = Image.fromarray(arr, "RGBA")
        cx, cy = sprite_centering(img)
        assert cx < 0.3
        assert cy < 0.3

    def test_empty(self):
        arr = np.zeros((64, 64, 4), dtype=np.uint8)
        img = Image.fromarray(arr, "RGBA")
        cx, cy = sprite_centering(img)
        assert cx == 0.5
        assert cy == 0.5


class TestBoundingBox:
    def test_fills_frame(self):
        img = _solid((255, 0, 0))
        bbox = bounding_box(img)
        assert bbox is not None
        x, y, x2, y2 = bbox
        assert x == 0
        assert y == 0
        assert x2 == img.size[0] - 1
        assert y2 == img.size[1] - 1

    def test_empty(self):
        arr = np.zeros((64, 64, 4), dtype=np.uint8)
        img = Image.fromarray(arr, "RGBA")
        assert bounding_box(img) is None

    def test_small_sprite_large_canvas(self):
        arr = np.zeros((128, 128, 4), dtype=np.uint8)
        arr[32:64, 32:64, :3] = [0, 0, 255]
        arr[32:64, 32:64, 3] = 255
        img = Image.fromarray(arr, "RGBA")
        bbox = bounding_box(img)
        assert bbox is not None
        x, y, x2, y2 = bbox
        assert x >= 0
        assert y >= 0
        assert x2 > x
        assert y2 > y


class TestTransparencyCoverage:
    def test_opaque(self):
        img = _solid((128, 128, 128))
        assert transparency_coverage(img) == 0.0

    def test_transparent(self):
        arr = np.zeros((64, 64, 4), dtype=np.uint8)
        img = Image.fromarray(arr, "RGBA")
        assert transparency_coverage(img) == 1.0

    def test_half_transparent(self):
        arr = np.zeros((64, 64, 4), dtype=np.uint8)
        arr[:32, :, :3] = [255, 255, 255]
        arr[:32, :, 3] = 255
        img = Image.fromarray(arr, "RGBA")
        ratio = transparency_coverage(img)
        assert 0.45 < ratio < 0.55


class TestPixelSharpness:
    def test_sharp_edges(self):
        img = _checkerboard(size=(32, 32), tile=2)
        assert pixel_sharpness(img) > 50

    def test_uniform(self):
        img = _solid((128, 128, 128))
        assert pixel_sharpness(img) == 0.0


class TestPaletteConsistency:
    def test_perfect_match_retro16(self):
        from backend.modules.postprocess.palettes import get_palette
        pal = get_palette("retro_16")
        arr = np.zeros((32, 32, 4), dtype=np.uint8)
        arr[:, :, :3] = pal[0]
        arr[:, :, 3] = 255
        img = Image.fromarray(arr, "RGBA")
        assert palette_consistency(img, "retro_16") == 1.0

    def test_low_score_for_random(self):
        arr = np.random.randint(0, 256, (32, 32, 4), dtype=np.uint8)
        arr[:, :, 3] = 255
        img = Image.fromarray(arr, "RGBA")
        assert palette_consistency(img, "retro_16") < 0.99

    def test_transparent_image(self):
        arr = np.zeros((32, 32, 4), dtype=np.uint8)
        img = Image.fromarray(arr, "RGBA")
        assert palette_consistency(img, "retro_16") == 1.0

    def test_accepts_palette_name(self):
        img = _solid((128, 128, 128))
        score = palette_consistency(img, "gameboy")
        assert 0.0 <= score <= 1.0


class TestSpriteAspectRatio:
    def test_square_sprite(self):
        img = _sprite((255, 0, 0), size=(64, 64), sprite_size=(32, 32))
        ratio = sprite_aspect_ratio(img)
        assert ratio == 1.0

    def test_wide_sprite(self):
        arr = np.zeros((64, 64, 4), dtype=np.uint8)
        arr[16:48, 4:60, :3] = [255, 0, 0]
        arr[16:48, 4:60, 3] = 255
        img = Image.fromarray(arr, "RGBA")
        ratio = sprite_aspect_ratio(img)
        assert ratio > 1.0

    def test_tall_sprite(self):
        arr = np.zeros((64, 64, 4), dtype=np.uint8)
        arr[4:60, 16:48, :3] = [0, 255, 0]
        arr[4:60, 16:48, 3] = 255
        img = Image.fromarray(arr, "RGBA")
        ratio = sprite_aspect_ratio(img)
        assert ratio > 1.0

    def test_empty_image_returns_one(self):
        arr = np.zeros((64, 64, 4), dtype=np.uint8)
        img = Image.fromarray(arr, "RGBA")
        ratio = sprite_aspect_ratio(img)
        assert ratio == 1.0

    def test_single_pixel(self):
        arr = np.zeros((64, 64, 4), dtype=np.uint8)
        arr[32, 32, :3] = [255, 255, 255]
        arr[32, 32, 3] = 255
        img = Image.fromarray(arr, "RGBA")
        ratio = sprite_aspect_ratio(img)
        assert ratio == 1.0

    def test_extreme_aspect_ratio_triggers_tier(self):
        arr = np.zeros((64, 64, 4), dtype=np.uint8)
        arr[28:36, 2:62, :3] = [255, 0, 0]
        arr[28:36, 2:62, 3] = 255
        img = Image.fromarray(arr, "RGBA")
        result = assess_all(img)
        assert result["aspect_ratio"] > 4.0
        assert result["quality_tier"] == "extreme_aspect"


class TestOutlineContinuity:
    def test_perfect_continuity(self):
        img = _solid((100, 150, 200))
        assert outline_continuity(img) == 1.0

    def test_fragmented_outline(self):
        arr = np.zeros((64, 64, 4), dtype=np.uint8)
        arr[4:12, 4:12, :3] = [255, 0, 0]
        arr[4:12, 4:12, 3] = 255
        arr[4:12, 20:28, :3] = [255, 0, 0]
        arr[4:12, 20:28, 3] = 255
        img = Image.fromarray(arr, "RGBA")
        score = outline_continuity(img)
        assert score < 1.0


class TestDuplicateDetection:
    def test_no_duplicates(self):
        a = _solid((255, 0, 0))
        b = _solid((0, 255, 0))
        c = _solid((0, 0, 255))
        assert duplicate_detection([a, b, c]) == 0

    def test_all_duplicates(self):
        a = _solid((255, 0, 0))
        assert duplicate_detection([a, a, a]) == 3

    def test_single_image(self):
        a = _solid((255, 0, 0))
        assert duplicate_detection([a]) == 0

    def test_empty(self):
        assert duplicate_detection([]) == 0
