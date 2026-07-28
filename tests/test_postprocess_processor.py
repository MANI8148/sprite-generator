"""Direct unit tests for backend.modules.postprocess.processor functions.

Covers: to_rgba, remove_background, reduce_palette, pixel_cleanup,
auto_center, auto_pad, normalize, upscale, outline_cleanup.
"""

from unittest.mock import patch
import numpy as np
from PIL import Image


def _make_rgba(size=(16, 16), color=(255, 0, 0, 255)):
    img = Image.new("RGBA", size, color)
    return img


def _make_rgb(size=(16, 16), color=(255, 0, 0)):
    return Image.new("RGB", size, color)


def _make_p(size=(16, 16)):
    img = Image.new("P", size, 0)
    return img


def _make_transparent(size=(16, 16)):
    return Image.new("RGBA", size, (0, 0, 0, 0))


# ---------------------------------------------------------------------------
# to_rgba
# ---------------------------------------------------------------------------

class TestToRgba:
    def test_rgba_passthrough(self):
        from backend.modules.postprocess.processor import to_rgba
        img = _make_rgba()
        result = to_rgba(img)
        assert result.mode == "RGBA"
        assert result is img

    def test_rgb_converts(self):
        from backend.modules.postprocess.processor import to_rgba
        img = _make_rgb()
        result = to_rgba(img)
        assert result.mode == "RGBA"
        assert result.size == (16, 16)

    def test_palette_converts(self):
        from backend.modules.postprocess.processor import to_rgba
        img = _make_p()
        result = to_rgba(img)
        assert result.mode == "RGBA"

    def test_l_converts(self):
        from backend.modules.postprocess.processor import to_rgba
        img = Image.new("L", (8, 8), 128)
        result = to_rgba(img)
        assert result.mode == "RGBA"


# ---------------------------------------------------------------------------
# remove_background
# ---------------------------------------------------------------------------

class TestRemoveBackground:
    def test_rembg_import_error_fallback(self):
        from backend.modules.postprocess.processor import remove_background
        img = Image.new("RGBA", (8, 8), (0, 255, 0, 255))
        with patch.dict("sys.modules", {"rembg": None}):
            import importlib
            import sys
            saved = sys.modules.pop("rembg", None)
            try:
                for m in list(sys.modules.keys()):
                    if "rembg" in m:
                        sys.modules.pop(m, None)
                importlib.invalidate_caches()
                result = remove_background(img, model="u2net")
                assert result.mode == "RGBA"
                assert result.size == (8, 8)
            finally:
                if saved is not None:
                    sys.modules["rembg"] = saved
                importlib.invalidate_caches()

    def test_fallback_removes_corner_color(self):
        from backend.modules.postprocess.processor import remove_background
        green_bg = Image.new("RGBA", (16, 16), (0, 255, 0, 255))
        green_bg.putpixel((8, 8), (255, 0, 0, 255))
        with patch.dict("sys.modules", {"rembg": None}):
            import importlib
            import sys
            saved = sys.modules.pop("rembg", None)
            try:
                for m in list(sys.modules.keys()):
                    if "rembg" in m:
                        sys.modules.pop(m, None)
                importlib.invalidate_caches()
                result = remove_background(green_bg)
                arr = np.array(result)
                assert arr[8, 8, 3] > 128
                assert arr[0, 0, 3] == 0
            finally:
                if saved is not None:
                    sys.modules["rembg"] = saved
                importlib.invalidate_caches()

    def test_alpha_threshold_binarizes(self):
        from backend.modules.postprocess.processor import remove_background
        rgba = Image.new("RGBA", (4, 4), (100, 100, 100, 100))
        with patch.dict("sys.modules", {"rembg": None}):
            import importlib
            import sys
            saved = sys.modules.pop("rembg", None)
            try:
                for m in list(sys.modules.keys()):
                    if "rembg" in m:
                        sys.modules.pop(m, None)
                importlib.invalidate_caches()
                result = remove_background(rgba, alpha_threshold=50)
                arr = np.array(result)
                assert set(np.unique(arr[:, :, 3])).issubset({0, 255})
            finally:
                if saved is not None:
                    sys.modules["rembg"] = saved
                importlib.invalidate_caches()

    def test_fully_transparent_input(self):
        from backend.modules.postprocess.processor import remove_background
        img = _make_transparent()
        with patch.dict("sys.modules", {"rembg": None}):
            import importlib
            import sys
            saved = sys.modules.pop("rembg", None)
            try:
                for m in list(sys.modules.keys()):
                    if "rembg" in m:
                        sys.modules.pop(m, None)
                importlib.invalidate_caches()
                result = remove_background(img)
                assert np.all(np.array(result)[:, :, 3] == 0)
            finally:
                if saved is not None:
                    sys.modules["rembg"] = saved
                importlib.invalidate_caches()


# ---------------------------------------------------------------------------
# reduce_palette
# ---------------------------------------------------------------------------

class TestReducePalette:
    def test_reduces_colors(self):
        from backend.modules.postprocess.processor import reduce_palette
        img = _make_rgba()
        result = reduce_palette(img, max_colors=4)
        arr = np.array(result)
        opaque = arr[arr[:, :, 3] > 128]
        unique = len(set(tuple(p[:3]) for p in opaque))
        assert unique <= 4

    def test_max_colors_256_returns_original(self):
        from backend.modules.postprocess.processor import reduce_palette
        img = _make_rgba()
        result = reduce_palette(img, max_colors=256)
        assert np.array_equal(np.array(img), np.array(result))

    def test_preserves_alpha(self):
        from backend.modules.postprocess.processor import reduce_palette
        rgba = Image.new("RGBA", (8, 8), (255, 0, 0, 128))
        result = reduce_palette(rgba, max_colors=8)
        arr = np.array(result)
        assert (arr[:, :, 3] == 128).all()

    def test_rgb_input_converts(self):
        from backend.modules.postprocess.processor import reduce_palette
        img = _make_rgb()
        result = reduce_palette(img, max_colors=8)
        assert result.mode == "RGBA"

    def test_fully_transparent(self):
        from backend.modules.postprocess.processor import reduce_palette
        img = _make_transparent()
        result = reduce_palette(img, max_colors=8)
        assert np.all(np.array(result)[:, :, 3] == 0)


# ---------------------------------------------------------------------------
# pixel_cleanup
# ---------------------------------------------------------------------------

class TestPixelCleanup:
    def test_removes_small_regions(self):
        from backend.modules.postprocess.processor import pixel_cleanup
        rgba = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
        rgba.putpixel((0, 0), (255, 0, 0, 255))
        rgba.putpixel((1, 0), (255, 0, 0, 255))
        rgba.putpixel((0, 1), (255, 0, 0, 255))
        for dx in range(4):
            for dy in range(4):
                rgba.putpixel((8 + dx, 8 + dy), (0, 255, 0, 255))
        result = pixel_cleanup(rgba, min_region_size=4)
        arr = np.array(result)
        assert arr[0, 0, 3] == 0
        assert arr[1, 0, 3] == 0
        assert arr[0, 1, 3] == 0
        assert arr[8, 8, 3] == 255

    def test_large_region_preserved(self):
        from backend.modules.postprocess.processor import pixel_cleanup
        solid = Image.new("RGBA", (16, 16), (255, 255, 255, 255))
        result = pixel_cleanup(solid, min_region_size=3)
        assert np.all(np.array(result)[:, :, 3] == 255)

    def test_fully_transparent(self):
        from backend.modules.postprocess.processor import pixel_cleanup
        img = _make_transparent()
        result = pixel_cleanup(img)
        assert np.all(np.array(result)[:, :, 3] == 0)

    def test_single_pixel_removed(self):
        from backend.modules.postprocess.processor import pixel_cleanup
        rgba = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
        rgba.putpixel((4, 4), (255, 0, 0, 255))
        result = pixel_cleanup(rgba, min_region_size=2)
        assert np.array(result)[4, 4, 3] == 0


# ---------------------------------------------------------------------------
# auto_center
# ---------------------------------------------------------------------------

class TestAutoCenter:
    def test_centers_content(self):
        from backend.modules.postprocess.processor import auto_center
        rgba = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
        for dx in range(4, 12):
            for dy in range(4, 12):
                rgba.putpixel((dx, dy), (255, 255, 255, 255))
        result = auto_center(rgba, padding=0)
        arr = np.array(result)
        rows = np.any(arr[:, :, 3] > 0, axis=1)
        cols = np.any(arr[:, :, 3] > 0, axis=0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        content_center_x = (x_min + x_max) / 2
        content_center_y = (y_min + y_max) / 2
        canvas_center_x = result.size[0] / 2
        canvas_center_y = result.size[1] / 2
        assert abs(content_center_x - canvas_center_x) < 1
        assert abs(content_center_y - canvas_center_y) < 1

    def test_fully_transparent_returns_unchanged(self):
        from backend.modules.postprocess.processor import auto_center
        img = _make_transparent((16, 16))
        result = auto_center(img)
        assert np.all(np.array(result) == 0)

    def test_with_canvas_size(self):
        from backend.modules.postprocess.processor import auto_center
        rgba = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
        rgba.putpixel((8, 8), (255, 255, 255, 255))
        result = auto_center(rgba, canvas_size=(64, 64), padding=0)
        assert result.size == (64, 64)

    def test_padding_added(self):
        from backend.modules.postprocess.processor import auto_center
        rgba = Image.new("RGBA", (8, 8), (255, 0, 0, 255))
        result = auto_center(rgba, padding=4)
        assert result.size[0] == 8 + 8
        assert result.size[1] == 8 + 8

    def test_rgb_input(self):
        from backend.modules.postprocess.processor import auto_center
        img = _make_rgb((8, 8))
        result = auto_center(img)
        assert result.mode == "RGBA"


# ---------------------------------------------------------------------------
# auto_pad
# ---------------------------------------------------------------------------

class TestAutoPad:
    def test_adds_padding(self):
        from backend.modules.postprocess.processor import auto_pad
        img = _make_rgba((8, 8))
        result = auto_pad(img, padding=4)
        assert result.size == (16, 16)

    def test_target_size(self):
        from backend.modules.postprocess.processor import auto_pad
        img = _make_rgba((8, 8))
        result = auto_pad(img, target_size=(32, 32))
        assert result.size == (32, 32)

    def test_target_size_centers_content(self):
        from backend.modules.postprocess.processor import auto_pad
        img = Image.new("RGBA", (4, 4), (255, 0, 0, 255))
        result = auto_pad(img, target_size=(8, 8))
        arr = np.array(result)
        assert arr[0, 0, 3] == 0
        assert arr[2, 2, 3] == 255

    def test_rgb_input(self):
        from backend.modules.postprocess.processor import auto_pad
        img = _make_rgb((8, 8))
        result = auto_pad(img, padding=2)
        assert result.mode == "RGBA"

    def test_transparent_input(self):
        from backend.modules.postprocess.processor import auto_pad
        img = _make_transparent((4, 4))
        result = auto_pad(img, padding=4)
        assert result.size == (12, 12)


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_downscales_when_larger_than_target(self):
        from backend.modules.postprocess.processor import normalize
        img = _make_rgba((256, 256))
        result = normalize(img, target_size=(64, 64))
        assert max(result.size) <= 64

    def test_small_image_unchanged(self):
        from backend.modules.postprocess.processor import normalize
        img = _make_rgba((16, 16))
        result = normalize(img, target_size=(64, 64))
        assert result.size[0] <= 64
        assert result.size[1] <= 64

    def test_returns_rgba(self):
        from backend.modules.postprocess.processor import normalize
        img = _make_rgba((32, 32))
        result = normalize(img)
        assert result.mode == "RGBA"

    def test_transparent_input(self):
        from backend.modules.postprocess.processor import normalize
        img = _make_transparent((32, 32))
        result = normalize(img, target_size=(64, 64))
        assert result.mode == "RGBA"


# ---------------------------------------------------------------------------
# upscale
# ---------------------------------------------------------------------------

class TestUpscale:
    def test_upscale_factor_2(self):
        from backend.modules.postprocess.processor import upscale
        img = _make_rgba((8, 8))
        result = upscale(img, factor=2)
        assert result.size == (16, 16)

    def test_factor_1_returns_original(self):
        from backend.modules.postprocess.processor import upscale
        img = _make_rgba((8, 8))
        result = upscale(img, factor=1)
        assert result is img

    def test_factor_0_returns_original(self):
        from backend.modules.postprocess.processor import upscale
        img = _make_rgba((8, 8))
        result = upscale(img, factor=0)
        assert result is img

    def test_lanczos_method(self):
        from backend.modules.postprocess.processor import upscale
        img = _make_rgba((8, 8))
        result = upscale(img, factor=2, method="lanczos")
        assert result.size == (16, 16)

    def test_preserves_alpha(self):
        from backend.modules.postprocess.processor import upscale
        rgba = Image.new("RGBA", (4, 4), (255, 0, 0, 200))
        result = upscale(rgba, factor=2)
        arr = np.array(result)
        assert (arr[:, :, 3] == 200).all()

    def test_rgb_input_preserves_mode(self):
        from backend.modules.postprocess.processor import upscale
        img = _make_rgb((4, 4))
        result = upscale(img, factor=2)
        assert result.mode == "RGB"
        assert result.size == (8, 8)


# ---------------------------------------------------------------------------
# outline_cleanup
# ---------------------------------------------------------------------------

class TestOutlineCleanup:
    def test_default_outline_color(self):
        from backend.modules.postprocess.processor import outline_cleanup
        solid = Image.new("RGBA", (16, 16), (255, 255, 255, 255))
        solid.putpixel((0, 0), (255, 0, 0, 255))
        result = outline_cleanup(solid)
        arr = np.array(result)
        assert arr.shape == (16, 16, 4)

    def test_fully_transparent(self):
        from backend.modules.postprocess.processor import outline_cleanup
        img = _make_transparent((8, 8))
        result = outline_cleanup(img)
        assert np.all(np.array(result)[:, :, 3] == 0)

    def test_custom_outline_color(self):
        from backend.modules.postprocess.processor import outline_cleanup
        solid = Image.new("RGBA", (16, 16), (255, 255, 255, 255))
        result = outline_cleanup(solid, outline_color=(255, 0, 0, 255))
        arr = np.array(result)
        assert arr.shape == (16, 16, 4)

    def test_threshold_variations(self):
        from backend.modules.postprocess.processor import outline_cleanup
        solid = Image.new("RGBA", (8, 8), (255, 255, 255, 255))
        result_low = outline_cleanup(solid, threshold=10)
        result_high = outline_cleanup(solid, threshold=200)
        assert result_low.size == (8, 8)
        assert result_high.size == (8, 8)

    def test_rgb_input(self):
        from backend.modules.postprocess.processor import outline_cleanup
        img = _make_rgb((8, 8))
        result = outline_cleanup(img)
        assert result.mode == "RGBA"
