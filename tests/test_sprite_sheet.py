import json
from pathlib import Path
from PIL import Image

from eval.sprite_sheet import pack_sprite_sheet, save_sprite_sheet


class TestPackSpriteSheet:
    def test_empty_sprites(self):
        sheet, data = pack_sprite_sheet([])
        assert sheet.size == (0, 0)
        assert data == {"frames": {}, "meta": {}}

    def test_single_sprite(self):
        img = Image.new("RGBA", (16, 16), (255, 0, 0, 255))
        sheet, data = pack_sprite_sheet([("sprite1", img)], max_width=32)
        assert sheet.size == (32, 16)
        assert "sprite1" in data["frames"]
        f = data["frames"]["sprite1"]["frame"]
        assert f == {"x": 0, "y": 0, "w": 16, "h": 16}

    def test_two_sprites_side_by_side(self):
        imgs = [
            ("a", Image.new("RGBA", (8, 8), (255, 0, 0, 255))),
            ("b", Image.new("RGBA", (8, 8), (0, 255, 0, 255))),
        ]
        sheet, data = pack_sprite_sheet(imgs, max_width=32, padding=0)
        assert sheet.size == (32, 8)
        assert data["frames"]["a"]["frame"]["x"] == 0
        assert data["frames"]["b"]["frame"]["x"] == 8

    def test_padding_between_sprites(self):
        imgs = [
            ("a", Image.new("RGBA", (8, 8), (255, 0, 0, 255))),
            ("b", Image.new("RGBA", (8, 8), (0, 255, 0, 255))),
        ]
        sheet, data = pack_sprite_sheet(imgs, max_width=32, padding=2)
        assert data["frames"]["a"]["frame"]["x"] == 0
        assert data["frames"]["b"]["frame"]["x"] == 10

    def test_wraps_to_next_row(self):
        imgs = [
            ("wide", Image.new("RGBA", (60, 10), (255, 0, 0, 255))),
            ("tall", Image.new("RGBA", (10, 60), (0, 255, 0, 255))),
        ]
        sheet, data = pack_sprite_sheet(imgs, max_width=64, padding=1)
        assert data["frames"]["wide"]["frame"]["y"] == 0
        assert data["frames"]["tall"]["frame"]["y"] > 0

    def test_exceeds_max_height_truncated(self):
        imgs = [
            ("a", Image.new("RGBA", (20, 100), (255, 0, 0, 255))),
            ("b", Image.new("RGBA", (20, 100), (0, 255, 0, 255))),
        ]
        sheet, data = pack_sprite_sheet(imgs, max_width=25, max_height=150, padding=0)
        assert len(data["frames"]) < 2
        assert data["meta"].get("truncated", 0) >= 1

    def test_varying_sprite_sizes(self):
        imgs = [
            ("small", Image.new("RGBA", (4, 4), (255, 0, 0, 255))),
            ("medium", Image.new("RGBA", (16, 16), (0, 255, 0, 255))),
            ("large", Image.new("RGBA", (32, 32), (0, 0, 255, 255))),
        ]
        sheet, data = pack_sprite_sheet(imgs, max_width=64, padding=1)
        assert len(data["frames"]) == 3
        for name in ("small", "medium", "large"):
            assert name in data["frames"]

    def test_frame_rectangles_do_not_overlap(self):
        imgs = [
            (f"s{i}", Image.new("RGBA", (8, 8), (255, 0, 0, 255)))
            for i in range(20)
        ]
        sheet, data = pack_sprite_sheet(imgs, max_width=32, padding=1)
        rects = []
        for name, fdata in data["frames"].items():
            f = fdata["frame"]
            r = (f["x"], f["y"], f["x"] + f["w"], f["y"] + f["h"])
            for other in rects:
                assert not (
                    r[0] < other[2] and r[2] > other[0]
                    and r[1] < other[3] and r[3] > other[1]
                ), f"{r} overlaps {other}"
            rects.append(r)

    def test_meta_structure(self):
        img = Image.new("RGBA", (16, 16), (255, 0, 0, 255))
        sheet, data = pack_sprite_sheet([("sprite1", img)], max_width=32)
        meta = data["meta"]
        assert meta["app"] == "sprite-generator"
        assert meta["version"] == "1.0"
        assert meta["image"] == "sprite_sheet.png"
        assert "size" in meta
        assert "scale" in meta

    def test_rgba_preserved(self):
        img = Image.new("RGBA", (8, 8), (128, 64, 32, 200))
        sheet, data = pack_sprite_sheet([("sprite1", img)], max_width=16)
        pixel = sheet.getpixel((0, 0))
        assert pixel == (128, 64, 32, 200)

    def test_sprite_source_size_in_frames(self):
        img = Image.new("RGBA", (12, 24), (255, 0, 0, 255))
        sheet, data = pack_sprite_sheet([("s", img)], max_width=32)
        f = data["frames"]["s"]
        assert f["spriteSourceSize"] == {"x": 0, "y": 0, "w": 12, "h": 24}
        assert f["sourceSize"] == {"w": 12, "h": 24}


class TestSaveSpriteSheet:
    def test_saves_png_and_json(self, tmp_path):
        imgs = [
            ("a", Image.new("RGBA", (8, 8), (255, 0, 0, 255))),
            ("b", Image.new("RGBA", (8, 8), (0, 255, 0, 255))),
        ]
        png_path, json_path = save_sprite_sheet(
            imgs, output_stem="test_sheet", output_dir=str(tmp_path), padding=0,
        )
        assert png_path.exists()
        assert json_path.exists()
        img = Image.open(png_path)
        assert img.mode == "RGBA"
        with open(json_path) as f:
            data = json.load(f)
        assert "frames" in data
        assert "meta" in data

    def test_creates_output_directory(self, tmp_path):
        nested = tmp_path / "sub" / "dir"
        imgs = [
            ("a", Image.new("RGBA", (4, 4), (255, 0, 0, 255))),
        ]
        png_path, json_path = save_sprite_sheet(
            imgs, output_stem="sheet", output_dir=str(nested), padding=0,
        )
        assert nested.exists()
        assert png_path.exists()
        assert json_path.exists()

    def test_default_args(self, tmp_path):
        origin = Path.cwd()
        try:
            import os
            os.chdir(str(tmp_path))
            imgs = [
                ("a", Image.new("RGBA", (4, 4), (255, 0, 0, 255))),
            ]
            png_path, json_path = save_sprite_sheet(imgs)
            assert png_path.name == "sprite_sheet.png"
            assert json_path.name == "sprite_sheet.json"
        finally:
            os.chdir(str(origin))
