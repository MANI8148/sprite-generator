from PIL import Image
import json
import os
from backend.modules.packing.packer import (
    sprite_sheet,
    tileset,
    animation_strip,
    individual_pngs,
)


class TestSpriteSheet:
    def test_raises_on_empty(self):
        try:
            sprite_sheet([])
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_single_image(self):
        img = Image.new("RGBA", (16, 16), (255, 0, 0, 255))
        sheet, meta = sprite_sheet([img], cols=1, rows=1, padding=0)
        assert sheet.size == (16, 16)
        assert meta["type"] == "sprite_sheet"
        assert len(meta["frames"]) == 1
        assert meta["frames"][0]["w"] == 16

    def test_grid_layout(self):
        imgs = [Image.new("RGBA", (8, 8), (255, 0, 0, 255)) for _ in range(4)]
        sheet, meta = sprite_sheet(imgs, cols=2, padding=0)
        assert sheet.size == (16, 16)
        assert meta["cols"] == 2
        assert meta["rows"] == 2
        assert len(meta["frames"]) == 4

    def test_auto_compute_cols_rows(self):
        imgs = [Image.new("RGBA", (8, 8), (255, 0, 0, 255)) for _ in range(6)]
        sheet, meta = sprite_sheet(imgs, padding=0)
        assert meta["cols"] >= 2
        assert meta["rows"] >= 2
        assert meta["cols"] * meta["rows"] >= 6

    def test_padding(self):
        img = Image.new("RGBA", (8, 8), (255, 0, 0, 255))
        sheet, meta = sprite_sheet([img, img], cols=2, padding=4)
        f0 = meta["frames"][0]
        f1 = meta["frames"][1]
        assert f1["x"] - (f0["x"] + f0["w"]) == 4

    def test_metadata_structure(self):
        img = Image.new("RGBA", (12, 24), (255, 0, 0, 255))
        sheet, meta = sprite_sheet([img], cols=1, rows=1, padding=0)
        assert "type" in meta
        assert "size" in meta
        assert "cell_size" in meta
        assert "frames" in meta
        assert meta["frames"][0]["source_w"] == 12
        assert meta["frames"][0]["source_h"] == 24

    def test_different_image_sizes(self):
        imgs = [
            Image.new("RGBA", (8, 16), (255, 0, 0, 255)),
            Image.new("RGBA", (16, 8), (0, 255, 0, 255)),
        ]
        sheet, meta = sprite_sheet(imgs, cols=2, padding=0)
        assert sheet.size == (32, 16)
        assert meta["cell_size"] == {"w": 16, "h": 16}


class TestTileset:
    def test_raises_on_empty(self):
        try:
            tileset([])
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_default_tile_size_from_first_image(self):
        img = Image.new("RGBA", (16, 16), (255, 0, 0, 255))
        sheet, meta = tileset([img], cols=1, padding=0)
        assert meta["type"] == "tileset"
        assert meta["tile_size"] == {"w": 16, "h": 16}
        assert sheet.size == (16, 16)

    def test_resizes_images_to_tile_size(self):
        imgs = [
            Image.new("RGBA", (8, 8), (255, 0, 0, 255)),
            Image.new("RGBA", (16, 16), (0, 255, 0, 255)),
        ]
        sheet, meta = tileset(imgs, tile_size=(16, 16), cols=2, padding=0)
        assert meta["tile_size"] == {"w": 16, "h": 16}
        assert sheet.size == (32, 16)
        for f in meta["frames"]:
            assert f["w"] == 16
            assert f["h"] == 16

    def test_grid_layout(self):
        imgs = [Image.new("RGBA", (16, 16), (255, 0, 0, 255)) for _ in range(4)]
        sheet, meta = tileset(imgs, cols=2, padding=0)
        assert sheet.size == (32, 32)
        assert meta["cols"] == 2
        assert meta["rows"] == 2
        assert meta["frame_count"] == 4

    def test_auto_compute_cols(self):
        imgs = [Image.new("RGBA", (16, 16), (255, 0, 0, 255)) for _ in range(5)]
        sheet, meta = tileset(imgs, padding=0)
        assert meta["cols"] >= 3
        assert meta["rows"] >= 2
        assert meta["frame_count"] == 5

    def test_padding(self):
        img = Image.new("RGBA", (16, 16), (255, 0, 0, 255))
        sheet, meta = tileset([img, img], cols=2, padding=4)
        f0 = meta["frames"][0]
        f1 = meta["frames"][1]
        assert f1["x"] - (f0["x"] + f0["w"]) == 4

    def test_preserves_rgba(self):
        img = Image.new("RGBA", (16, 16), (128, 64, 32, 255))
        sheet, meta = tileset([img], cols=1, padding=0)
        assert sheet.mode == "RGBA"
        pixel = sheet.getpixel((0, 0))
        assert pixel[:3] == (128, 64, 32)

    def test_frame_has_row_and_col(self):
        imgs = [Image.new("RGBA", (16, 16), (255, 0, 0, 255)) for _ in range(4)]
        sheet, meta = tileset(imgs, cols=2, padding=0)
        for f in meta["frames"]:
            assert "col" in f
            assert "row" in f
        assert meta["frames"][0]["col"] == 0
        assert meta["frames"][0]["row"] == 0
        assert meta["frames"][1]["col"] == 1
        assert meta["frames"][2]["col"] == 0
        assert meta["frames"][2]["row"] == 1


class TestAnimationStrip:
    def test_raises_on_empty(self):
        try:
            animation_strip([])
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_horizontal_layout(self):
        imgs = [Image.new("RGBA", (16, 16), (255, 0, 0, 255)) for _ in range(3)]
        strip, meta = animation_strip(imgs, direction="horizontal", padding=0)
        assert meta["type"] == "animation_strip"
        assert meta["direction"] == "horizontal"
        assert meta["frame_count"] == 3
        assert strip.size == (48, 16)

    def test_vertical_layout(self):
        imgs = [Image.new("RGBA", (16, 16), (255, 0, 0, 255)) for _ in range(3)]
        strip, meta = animation_strip(imgs, direction="vertical", padding=0)
        assert meta["direction"] == "vertical"
        assert strip.size == (16, 48)

    def test_default_direction(self):
        img = Image.new("RGBA", (16, 16), (255, 0, 0, 255))
        strip, meta = animation_strip([img], padding=0)
        assert meta["direction"] == "horizontal"

    def test_padding(self):
        imgs = [Image.new("RGBA", (16, 16), (255, 0, 0, 255)) for _ in range(2)]
        strip, meta = animation_strip(imgs, direction="horizontal", padding=4)
        assert strip.size == (44, 24)

    def test_metadata(self):
        img = Image.new("RGBA", (16, 16), (255, 0, 0, 255))
        strip, meta = animation_strip([img], padding=0)
        assert meta["frame_size"] == {"w": 16, "h": 16}
        assert meta["total_size"] == {"w": 16, "h": 16}

    def test_preserves_rgba(self):
        img = Image.new("RGBA", (8, 8), (64, 128, 192, 255))
        strip, meta = animation_strip([img], padding=0)
        assert strip.mode == "RGBA"
        pixel = strip.getpixel((4, 4))
        assert pixel[:3] == (64, 128, 192)


class TestIndividualPngs:
    def test_saves_files(self, tmp_path):
        imgs = [
            Image.new("RGBA", (8, 8), (255, 0, 0, 255)),
            Image.new("RGBA", (8, 8), (0, 255, 0, 255)),
        ]
        paths = individual_pngs(imgs, ["sprite_a", "sprite_b"], str(tmp_path))
        assert len(paths) == 2
        for p in paths:
            assert os.path.isfile(p)
        assert os.path.isfile(os.path.join(tmp_path, "sprite_a.png"))
        assert os.path.isfile(os.path.join(tmp_path, "sprite_b.png"))

    def test_sanitizes_names(self, tmp_path):
        img = Image.new("RGBA", (8, 8), (255, 0, 0, 255))
        paths = individual_pngs([img], ["bad/name:char*"], str(tmp_path))
        assert os.path.isfile(paths[0])
        assert "bad_name_char_" in paths[0]

    def test_creates_output_dir(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        img = Image.new("RGBA", (8, 8), (255, 0, 0, 255))
        paths = individual_pngs([img], ["test"], str(nested))
        assert nested.exists()
        assert os.path.isfile(paths[0])
