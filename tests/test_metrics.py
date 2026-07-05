import numpy as np
from PIL import Image
import pytest

from eval.metrics import palette_adherence_rate, grid_alignment_check


class TestPaletteAdherenceRate:
    @pytest.fixture
    def palette(self):
        return [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 255, 255),
            (0, 0, 0),
        ]

    def test_perfect_adherence(self, palette):
        arr = np.zeros((32, 32, 4), dtype=np.uint8)
        arr[:, :, :3] = [255, 0, 0]
        arr[:, :, 3] = 255
        img = Image.fromarray(arr, "RGBA")
        score = palette_adherence_rate(img, palette)
        assert score == 1.0

    def test_zero_adherence(self, palette):
        arr = np.zeros((32, 32, 4), dtype=np.uint8)
        arr[:, :, :3] = [128, 64, 32]
        arr[:, :, 3] = 255
        img = Image.fromarray(arr, "RGBA")
        score = palette_adherence_rate(img, palette)
        assert score == 0.0

    def test_partial_adherence(self, palette):
        arr = np.zeros((32, 32, 4), dtype=np.uint8)
        arr[:16, :, :3] = [255, 0, 0]
        arr[16:, :, :3] = [128, 64, 32]
        arr[:, :, 3] = 255
        img = Image.fromarray(arr, "RGBA")
        score = palette_adherence_rate(img, palette)
        assert 0.4 < score < 0.6

    def test_transparent_pixels_ignored(self, palette):
        arr = np.zeros((32, 32, 4), dtype=np.uint8)
        arr[:, :, :3] = [128, 64, 32]
        arr[:, :, 3] = 0
        img = Image.fromarray(arr, "RGBA")
        score = palette_adherence_rate(img, palette)
        assert score == 1.0

    def test_empty_image(self, palette):
        img = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
        score = palette_adherence_rate(img, palette)
        assert score == 1.0


class TestGridAlignmentCheck:
    def test_hard_edges_score_one(self):
        arr = np.zeros((32, 32, 4), dtype=np.uint8)
        arr[:, :] = [255, 0, 0, 255]
        img = Image.fromarray(arr, "RGBA")
        score = grid_alignment_check(img, 32)
        assert score == 1.0

    def test_smooth_edges_score_lower(self):
        arr = np.zeros((32, 32, 4), dtype=np.uint8)
        arr[:, :, :3] = [255, 0, 0]
        arr[5:25, 5:25, 3] = 128
        arr[5:25, 5:25, :3] = [255, 0, 0]
        arr[:, :, 3] = 255
        arr[10:20, 10:20, 3] = 200
        img = Image.fromarray(arr, "RGBA")
        score = grid_alignment_check(img, 32)
        assert score < 1.0

    def test_fully_transparent(self):
        img = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
        score = grid_alignment_check(img, 32)
        assert score == 1.0

    def test_rgb_image_no_alpha(self):
        arr = np.zeros((32, 32, 3), dtype=np.uint8)
        arr[:, :] = [255, 0, 0]
        img = Image.fromarray(arr, "RGB")
        score = grid_alignment_check(img, 32)
        assert score == 1.0
