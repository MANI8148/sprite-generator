import numpy as np

from tests.test_generator_modules import FakeGenerator

from backend.modules.generator.tileset_generator import TilesetGenerator
from backend.modules.generator.environment_generator import EnvironmentGenerator
from backend.modules.generator.ui_generator import UIGenerator, slice_grid
from backend.modules.generator.prop_generator import PropGenerator, isolate


class TestTilesetSeamless:
    def setup_method(self):
        self.fake = FakeGenerator()
        self.gen = TilesetGenerator(self.fake)

    def test_seamless_makes_left_and_right_edges_continuous(self):
        img = self.fake.generate(num_images=1)[0]
        seam = self.gen.make_seamless(img, band=12)
        arr = np.asarray(seam)
        # Left column must equal right column so the horizontal wrap is seamless.
        assert np.array_equal(arr[:, 0], arr[:, -1])

    def test_seamless_makes_top_and_bottom_edges_continuous(self):
        img = self.fake.generate(num_images=1)[0]
        seam = self.gen.make_seamless(img, band=12)
        arr = np.asarray(seam)
        assert np.array_equal(arr[0, :], arr[-1, :])

    def test_seamless_preserves_size_and_mode(self):
        img = self.fake.generate(num_images=1)[0]
        seam = self.gen.make_seamless(img, band=8)
        assert seam.size == img.size
        assert seam.mode == "RGBA"

    def test_generate_applies_seamless_by_default(self):
        images = self.gen.generate(prompt="ground", num_images=1)
        arr = np.asarray(images[0])
        assert np.array_equal(arr[:, 0], arr[:, -1])

    def test_generate_can_skip_seamless(self):
        base = self.fake.generate(prompt="x", num_images=1)[0]
        # A raw fake image has nothing at the borders, so edges differ before blending.
        base_arr = np.asarray(base)
        edges_differ = not np.array_equal(base_arr[:, 0], base_arr[:, -1])
        images = self.gen.generate(prompt="ground", num_images=1, seamless=False)
        arr = np.asarray(images[0])
        if edges_differ:
            assert not np.array_equal(arr[:, 0], arr[:, -1])

    def test_tiled_grid_has_continuous_horizontal_seam(self):
        img = self.fake.generate(num_images=1)[0]
        seam = self.gen.make_seamless(img, band=12)
        arr = np.asarray(seam)
        # When two seamless tiles sit side by side, the right edge of the left
        # tile content matches the left edge of the right tile.
        assert np.array_equal(arr[:, -1], arr[:, 0])


class TestEnvironmentParallax:
    def setup_method(self):
        self.fake = FakeGenerator()
        self.gen = EnvironmentGenerator(self.fake)

    def test_generate_layer_returns_single_image(self):
        layer = self.gen.generate_layer(prompt="sky", depth=0.2, width=32, height=64)
        assert isinstance(layer.size, tuple)
        assert layer.mode == "RGBA"

    def test_generate_layer_depth_scales_height(self):
        far = self.gen.generate_layer(prompt="sky", depth=0.1, width=32, height=100)
        near = self.gen.generate_layer(prompt="foreground", depth=0.9, width=32, height=100)
        assert far.size[1] < near.size[1]

    def test_generate_layer_clamps_depth(self):
        out = self.gen.generate_layer(prompt="x", depth=5.0, width=32, height=100)
        assert 0 < out.size[1] <= 100

    def test_generate_panorama_stacks_layers(self):
        pan = self.gen.generate_panorama(
            prompt="valley", layers=3, width=128, height=64,
            depth_start=0.2, depth_end=0.9,
        )
        assert pan.mode == "RGBA"
        assert pan.size[0] == 128
        # Height grows with the number of stacked depth layers.
        assert pan.size[1] > 64

    def test_generate_panorama_returns_one_image(self):
        pan = self.gen.generate_panorama(prompt="valley", layers=1, width=128, height=64)
        assert pan.size[0] == 128
        assert pan.size[1] > 0


class TestUISlicing:
    def setup_method(self):
        self.fake = FakeGenerator()
        self.gen = UIGenerator(self.fake)

    def test_slice_grid_splits_into_cells(self):
        base = self.fake.generate(num_images=1)[0].resize((128, 64))
        cells = slice_grid(base, cols=4, rows=1)
        assert len(cells) == 4
        assert all(c.size == (32, 64) for c in cells)

    def test_slice_grid_multi_row(self):
        base = self.fake.generate(num_images=1)[0].resize((128, 64))
        cells = slice_grid(base, cols=2, rows=2)
        assert len(cells) == 4

    def test_slice_grid_reassembles_original(self):
        base = self.fake.generate(num_images=1)[0].resize((128, 64))
        cells = slice_grid(base, cols=4, rows=1)
        canvas = np.hstack([np.asarray(c) for c in cells])
        assert np.array_equal(canvas, np.asarray(base))

    def test_slice_grid_rejects_bad_args(self):
        import pytest
        base = self.fake.generate(num_images=1)[0]
        with pytest.raises(ValueError):
            slice_grid(base, cols=0, rows=1)

    def test_generate_sheet_returns_sliced_cells(self):
        cells = self.gen.generate_sheet(prompt="icons", cols=4, rows=2, width=128, height=64)
        assert len(cells) == 8


class TestPropIsolation:
    def setup_method(self):
        self.fake = FakeGenerator()
        self.gen = PropGenerator(self.fake)

    def test_isolate_centers_and_pads(self):
        base = self.fake.generate(num_images=1)[0].resize((128, 128))
        out = isolate(base, size=256, padding=0)
        arr = np.asarray(out)
        assert out.size == (256, 256)
        # Left/right and top/bottom transparent margins are symmetric.
        rows = np.where(np.any(arr[:, :, 3] > 0, axis=1))[0]
        cols = np.where(np.any(arr[:, :, 3] > 0, axis=0))[0]
        assert rows[0] + rows[-1] == arr.shape[0] - 1
        assert cols[0] + cols[-1] == arr.shape[1] - 1

    def test_isolate_keeps_content_alpha(self):
        base = self.fake.generate(num_images=1)[0]
        out = isolate(base, size=512)
        arr = np.asarray(out)
        assert (arr[:, :, 3] > 0).any()

    def test_generate_isolated_by_default_is_square(self):
        images = self.gen.generate(prompt="chest", num_images=1)
        assert images[0].size[0] == images[0].size[1]

    def test_generate_skips_isolate(self):
        images = self.gen.generate(prompt="chest", num_images=1, isolated=False)
        assert len(images) == 1
        assert images[0].mode == "RGBA"