"""
Tests for data pipeline scripts (roadmap item #1).
Covers clean_normalize, caption_ai, and augment_dataset utilities.
"""
import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from data.scripts.clean_normalize import (
    remove_background,
    find_content_bbox,
    center_on_canvas,
    quantize_to_palette,
    build_global_palette,
)
from data.scripts.caption_ai import caption_locally
from data.scripts.augment_dataset import (
    color_jitter,
    random_translate,
    horizontal_flip,
    DIRECTION_SWAP,
)


class TestRemoveBackground:
    def test_removes_known_bg_color(self):
        img = Image.new("RGBA", (32, 32), (255, 0, 0, 255))
        result = remove_background(img, bg_color=(255, 0, 0, 255))
        arr = np.array(result)
        assert arr[:, :, 3].sum() == 0

    def test_preserves_non_bg_pixels(self):
        img = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
        result = remove_background(img)
        assert result.mode == "RGBA"

    def test_auto_detect_bg(self):
        arr = np.zeros((32, 32, 4), dtype=np.uint8)
        arr[:, :, :3] = [255, 255, 255]
        arr[:, :, 3] = 255
        arr[16, 16, :3] = [128, 64, 32]
        img = Image.fromarray(arr, "RGBA")
        result = remove_background(img)
        r_arr = np.array(result)
        assert r_arr[0, 0, 3] == 0

    def test_rgb_input(self):
        arr = np.zeros((32, 32, 3), dtype=np.uint8)
        arr[:, :] = [0, 255, 0]
        img = Image.fromarray(arr, "RGB")
        result = remove_background(img)
        assert result.mode == "RGBA"


class TestFindContentBBox:
    def test_full_content(self):
        arr = np.zeros((32, 32, 4), dtype=np.uint8)
        arr[:, :, 3] = 255
        arr[:, :, :3] = [255, 0, 0]
        img = Image.fromarray(arr, "RGBA")
        bbox = find_content_bbox(img)
        assert bbox == (0, 0, 32, 32)

    def test_empty_content(self):
        img = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
        bbox = find_content_bbox(img)
        assert bbox == (0, 0, 32, 32)

    def test_partial_content(self):
        arr = np.zeros((32, 32, 4), dtype=np.uint8)
        arr[4:20, 8:24, 3] = 255
        arr[4:20, 8:24, :3] = [255, 0, 0]
        img = Image.fromarray(arr, "RGBA")
        bbox = find_content_bbox(img)
        assert bbox[0] == 8
        assert bbox[1] == 4
        assert bbox[2] == 24
        assert bbox[3] == 20


class TestCenterOnCanvas:
    def test_output_size(self):
        img = Image.new("RGBA", (16, 16), (255, 0, 0, 255))
        result = center_on_canvas(img, canvas_size=32)
        assert result.size == (32, 32)

    def test_output_mode(self):
        img = Image.new("RGBA", (16, 16), (255, 0, 0, 255))
        result = center_on_canvas(img)
        assert result.mode == "RGBA"

    def test_content_centered(self):
        arr = np.zeros((64, 64, 4), dtype=np.uint8)
        arr[16:48, 16:48, :3] = [255, 0, 0]
        arr[16:48, 16:48, 3] = 255
        img = Image.fromarray(arr, "RGBA")
        result = center_on_canvas(img, canvas_size=32)
        r_arr = np.array(result)
        assert r_arr.shape == (32, 32, 4)

    def test_rgb_input(self):
        img = Image.new("RGB", (16, 16), (255, 0, 0))
        result = center_on_canvas(img, canvas_size=32)
        assert result.size == (32, 32)


class TestQuantizeToPalette:
    @pytest.fixture
    def palette(self):
        return [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0)]

    def test_quantize_to_exact_color(self, palette):
        arr = np.zeros((16, 16, 4), dtype=np.uint8)
        arr[:, :, :3] = [255, 0, 0]
        arr[:, :, 3] = 255
        img = Image.fromarray(arr, "RGBA")
        result = quantize_to_palette(img, palette)
        r_arr = np.array(result)
        assert np.all(r_arr[:, :, :3] == [255, 0, 0])

    def test_quantize_to_nearest(self, palette):
        arr = np.zeros((16, 16, 4), dtype=np.uint8)
        arr[:, :, :3] = [254, 1, 2]
        arr[:, :, 3] = 255
        img = Image.fromarray(arr, "RGBA")
        result = quantize_to_palette(img, palette)
        r_arr = np.array(result)
        assert np.all(r_arr[:, :, :3] == [255, 0, 0])

    def test_preserves_alpha(self, palette):
        arr = np.zeros((16, 16, 4), dtype=np.uint8)
        arr[:, :, :3] = [255, 0, 0]
        arr[:, :, 3] = 128
        img = Image.fromarray(arr, "RGBA")
        result = quantize_to_palette(img, palette)
        r_arr = np.array(result)
        assert np.all(r_arr[:, :, 3] == 128)


class TestBuildGlobalPalette:
    def test_single_color(self):
        img = Image.new("RGBA", (8, 8), (255, 0, 0, 255))
        palette = build_global_palette([img], n_colors=1)
        assert len(palette) == 1
        assert palette[0] == (255, 0, 0)

    def test_empty_images(self):
        palette = build_global_palette([], n_colors=4)
        assert len(palette) == 1
        assert palette[0] == (0, 0, 0)

    def test_transparent_images(self):
        img = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
        palette = build_global_palette([img], n_colors=4)
        assert len(palette) == 1
        assert palette[0] == (0, 0, 0)

    def test_palette_size(self):
        imgs = []
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        for c in colors:
            arr = np.zeros((4, 4, 4), dtype=np.uint8)
            arr[:, :, :3] = c
            arr[:, :, 3] = 255
            imgs.append(Image.fromarray(arr, "RGBA"))
        palette = build_global_palette(imgs, n_colors=4)
        assert len(palette) == 4


class TestCaptionLocally:
    def test_tall_sprite_is_character(self):
        img = Image.new("RGBA", (16, 32), (255, 0, 0, 255))
        result = caption_locally(img)
        assert result["class"] == "character"
        assert result["action"] == "idle"
        assert result["direction"] == "front"

    def test_wide_sprite_is_item(self):
        img = Image.new("RGBA", (32, 16), (255, 0, 0, 255))
        result = caption_locally(img)
        assert result["class"] == "item"

    def test_square_sprite_is_tile(self):
        img = Image.new("RGBA", (32, 32), (255, 0, 0, 255))
        result = caption_locally(img)
        assert result["class"] == "tile"


class TestAugmentHorizontalFlip:
    def test_flip_output_shape(self):
        img = Image.new("RGBA", (32, 32), (255, 0, 0, 255))
        flipped = horizontal_flip(img)
        assert flipped.size == img.size

    def test_flip_changes_content(self):
        arr = np.zeros((32, 32, 4), dtype=np.uint8)
        arr[:, :16, :3] = [255, 0, 0]
        arr[:, :16, 3] = 255
        arr[:, 16:, :3] = [0, 255, 0]
        arr[:, 16:, 3] = 255
        img = Image.fromarray(arr, "RGBA")
        flipped = horizontal_flip(img)
        f_arr = np.array(flipped)
        assert np.all(f_arr[0, 0, :3] == [0, 255, 0])
        assert np.all(f_arr[0, -1, :3] == [255, 0, 0])


class TestAugmentRandomTranslate:
    def test_output_shape(self):
        img = Image.new("RGBA", (32, 32), (255, 0, 0, 255))
        translated = random_translate(img, max_shift=2)
        assert translated.size == (32, 32)

    def test_max_shift_respected(self):
        img = Image.new("RGBA", (32, 32), (255, 0, 0, 255))
        for _ in range(20):
            translated = random_translate(img, max_shift=1)
            assert translated.size == (32, 32)

    def test_output_mode(self):
        img = Image.new("RGBA", (32, 32), (255, 0, 0, 255))
        translated = random_translate(img)
        assert translated.mode == "RGBA"


class TestAugmentColorJitter:
    def test_output_shape(self):
        img = Image.new("RGBA", (32, 32), (128, 128, 128, 255))
        jittered = color_jitter(img)
        assert jittered.size == (32, 32)

    def test_output_mode(self):
        img = Image.new("RGBA", (32, 32), (128, 128, 128, 255))
        jittered = color_jitter(img)
        assert jittered.mode == "RGBA"


class TestDirectionSwap:
    def test_left_becomes_right(self):
        assert DIRECTION_SWAP["left"] == "right"

    def test_right_becomes_left(self):
        assert DIRECTION_SWAP["right"] == "left"

    def test_front_unchanged(self):
        assert "front" not in DIRECTION_SWAP

    def test_round_trip(self):
        assert DIRECTION_SWAP[DIRECTION_SWAP["left"]] == "left"
        assert DIRECTION_SWAP[DIRECTION_SWAP["right"]] == "right"
        assert DIRECTION_SWAP[DIRECTION_SWAP["front_left"]] == "front_left"
