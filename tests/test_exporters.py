import json
import os
from pathlib import Path

from PIL import Image

from backend.modules.exporters.exporter import (
    godot,
    unity,
    gamemaker,
    phaser,
    export_animation,
    zip_package,
)


def _make_sample_atlas() -> Image.Image:
    img = Image.new("RGBA", (64, 32), (0, 0, 0, 0))
    for x in range(0, 64, 32):
        for y in range(0, 32, 32):
            for py in range(16):
                for px in range(16):
                    img.putpixel((x + px, y + py), (255, 0, 0, 255))
    return img


def _make_metadata(name: str = "sprite", num_frames: int = 2) -> dict:
    frames = []
    for i in range(num_frames):
        frames.append({"index": i, "x": i * 32, "y": 0, "w": 32, "h": 32})
    return {"name": name, "frames": frames, "width": 64, "height": 32}


def _make_sample_images(num: int = 2) -> list:
    images = []
    for i in range(num):
        img = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
        for y in range(16):
            for x in range(16):
                img.putpixel((x + 8, y + 8), (255, 0, 0, 255))
        images.append(img)
    return images


class TestGodotExporter:
    def test_creates_png_and_tres(self, tmp_path):
        atlas = _make_sample_atlas()
        meta = _make_metadata()
        paths = godot(atlas, meta, str(tmp_path))
        assert len(paths) == 2
        assert Path(paths[0]).suffix == ".png"
        assert Path(paths[1]).suffix == ".tres"

    def test_tres_contains_sprite_frames(self, tmp_path):
        atlas = _make_sample_atlas()
        meta = _make_metadata("hero", 1)
        paths = godot(atlas, meta, str(tmp_path))
        text = Path(paths[1]).read_text()
        assert "SpriteFrames" in text
        assert "hero" in text

    def test_tres_has_frame_regions(self, tmp_path):
        atlas = _make_sample_atlas()
        meta = _make_metadata("char", 2)
        paths = godot(atlas, meta, str(tmp_path))
        text = Path(paths[1]).read_text()
        assert "Rect2(0, 0, 32, 32)" in text
        assert "Rect2(32, 0, 32, 32)" in text


class TestUnityExporter:
    def test_creates_png_and_meta(self, tmp_path):
        atlas = _make_sample_atlas()
        meta = _make_metadata()
        paths = unity(atlas, meta, str(tmp_path))
        assert len(paths) == 2
        assert Path(paths[0]).suffix == ".png"
        assert Path(paths[1]).suffix == ".meta"

    def test_meta_contains_sprite_data(self, tmp_path):
        atlas = _make_sample_atlas()
        meta = _make_metadata("player", 2)
        paths = unity(atlas, meta, str(tmp_path))
        text = Path(paths[1]).read_text()
        assert "fileFormatVersion: 2" in text
        assert "frame_0" in text
        assert "frame_1" in text

    def test_meta_has_rect_coordinates(self, tmp_path):
        atlas = _make_sample_atlas()
        meta = _make_metadata()
        meta["frames"][0] = {"index": 0, "x": 10, "y": 20, "w": 16, "h": 32}
        paths = unity(atlas, meta, str(tmp_path))
        text = Path(paths[1]).read_text()
        assert "x: 10" in text
        assert "y: 20" in text
        assert "width: 16" in text
        assert "height: 32" in text


class TestGameMakerExporter:
    def test_creates_png_and_yy(self, tmp_path):
        atlas = _make_sample_atlas()
        meta = _make_metadata()
        paths = gamemaker(atlas, meta, str(tmp_path))
        assert len(paths) == 2
        assert Path(paths[0]).suffix == ".png"
        assert Path(paths[1]).suffix == ".yy"

    def test_yy_contains_gmsprite(self, tmp_path):
        atlas = _make_sample_atlas()
        meta = _make_metadata("hero", 1)
        paths = gamemaker(atlas, meta, str(tmp_path))
        data = json.loads(Path(paths[1]).read_text())
        assert "$GMSprite" in data
        assert data["$GMSprite"]["name"] == "hero"

    def test_yy_has_all_frames(self, tmp_path):
        atlas = _make_sample_atlas()
        meta = _make_metadata("char", 3)
        paths = gamemaker(atlas, meta, str(tmp_path))
        data = json.loads(Path(paths[1]).read_text())
        assert len(data["$GMSprite"]["frames"]) == 3

    def test_yy_frame_coordinates(self, tmp_path):
        atlas = _make_sample_atlas()
        meta = _make_metadata()
        meta["frames"] = [
            {"index": 0, "x": 5, "y": 10, "w": 16, "h": 24},
        ]
        paths = gamemaker(atlas, meta, str(tmp_path))
        data = json.loads(Path(paths[1]).read_text())
        f = data["$GMSprite"]["frames"][0]
        assert f["x"] == 5
        assert f["y"] == 10
        assert f["width"] == 16
        assert f["height"] == 24

    def test_yy_has_sprite_dimensions(self, tmp_path):
        atlas = _make_sample_atlas()
        meta = _make_metadata("sprite", 1)
        paths = gamemaker(atlas, meta, str(tmp_path))
        data = json.loads(Path(paths[1]).read_text())
        assert data["$GMSprite"]["width"] == 64
        assert data["$GMSprite"]["height"] == 32

    def test_yy_empty_frames(self, tmp_path):
        atlas = _make_sample_atlas()
        meta = _make_metadata("empty", 0)
        meta["frames"] = []
        paths = gamemaker(atlas, meta, str(tmp_path))
        data = json.loads(Path(paths[1]).read_text())
        assert data["$GMSprite"]["frames"] == []


class TestPhaserExporter:
    def test_creates_png_and_json(self, tmp_path):
        atlas = _make_sample_atlas()
        meta = _make_metadata()
        paths = phaser(atlas, meta, str(tmp_path))
        assert len(paths) == 2
        assert Path(paths[0]).suffix == ".png"
        assert Path(paths[1]).suffix == ".json"

    def test_json_has_frames_and_meta(self, tmp_path):
        atlas = _make_sample_atlas()
        meta = _make_metadata("hero", 1)
        paths = phaser(atlas, meta, str(tmp_path))
        data = json.loads(Path(paths[1]).read_text())
        assert "frames" in data
        assert "meta" in data
        assert data["meta"]["image"] == "hero.png"

    def test_json_frame_count(self, tmp_path):
        atlas = _make_sample_atlas()
        meta = _make_metadata("char", 4)
        paths = phaser(atlas, meta, str(tmp_path))
        data = json.loads(Path(paths[1]).read_text())
        assert len(data["frames"]) == 4

    def test_json_frame_coordinates(self, tmp_path):
        atlas = _make_sample_atlas()
        meta = _make_metadata("sprite", 1)
        meta["frames"] = [
            {"index": 0, "x": 8, "y": 16, "w": 24, "h": 32},
        ]
        paths = phaser(atlas, meta, str(tmp_path))
        data = json.loads(Path(paths[1]).read_text())
        key = list(data["frames"].keys())[0]
        f = data["frames"][key]["frame"]
        assert f["x"] == 8
        assert f["y"] == 16
        assert f["w"] == 24
        assert f["h"] == 32

    def test_json_meta_size(self, tmp_path):
        atlas = _make_sample_atlas()
        meta = _make_metadata("sprite", 1)
        paths = phaser(atlas, meta, str(tmp_path))
        data = json.loads(Path(paths[1]).read_text())
        assert data["meta"]["size"]["w"] == 64
        assert data["meta"]["size"]["h"] == 32

    def test_json_phaser_flags(self, tmp_path):
        atlas = _make_sample_atlas()
        meta = _make_metadata("sprite", 1)
        paths = phaser(atlas, meta, str(tmp_path))
        data = json.loads(Path(paths[1]).read_text())
        key = list(data["frames"].keys())[0]
        assert data["frames"][key]["rotated"] is False
        assert data["frames"][key]["trimmed"] is False

    def test_json_empty_frames(self, tmp_path):
        atlas = _make_sample_atlas()
        meta = _make_metadata("empty", 0)
        meta["frames"] = []
        paths = phaser(atlas, meta, str(tmp_path))
        data = json.loads(Path(paths[1]).read_text())
        assert data["frames"] == {}


class TestExportAnimation:
    def test_godot_engine(self, tmp_path):
        images = _make_sample_images(2)
        paths = export_animation(images, str(tmp_path), "hero", engine="godot")
        assert len(paths) == 2
        assert Path(paths[0]).suffix == ".png"
        assert Path(paths[1]).suffix == ".tres"

    def test_unity_engine(self, tmp_path):
        images = _make_sample_images(2)
        paths = export_animation(images, str(tmp_path), "hero", engine="unity")
        assert len(paths) == 2
        assert Path(paths[0]).suffix == ".png"
        assert paths[1].endswith(".png.meta")

    def test_gamemaker_engine(self, tmp_path):
        images = _make_sample_images(2)
        paths = export_animation(images, str(tmp_path), "hero", engine="gamemaker")
        assert len(paths) == 2
        assert Path(paths[0]).suffix == ".png"
        assert Path(paths[1]).suffix == ".yy"

    def test_phaser_engine(self, tmp_path):
        images = _make_sample_images(2)
        paths = export_animation(images, str(tmp_path), "hero", engine="phaser")
        assert len(paths) == 2
        assert Path(paths[0]).suffix == ".png"
        assert Path(paths[1]).suffix == ".json"

    def test_default_engine_is_godot(self, tmp_path):
        images = _make_sample_images(1)
        paths = export_animation(images, str(tmp_path), "sprite")
        assert Path(paths[1]).suffix == ".tres"

    def test_unknown_engine_falls_back_to_generic(self, tmp_path):
        images = _make_sample_images(2)
        paths = export_animation(images, str(tmp_path), "sprite", engine="unknown")
        assert len(paths) == 2
        assert Path(paths[0]).suffix == ".png"
        assert Path(paths[1]).suffix == ".json"

    def test_export_creates_output_dir(self, tmp_path):
        images = _make_sample_images(1)
        nested = tmp_path / "sub" / "dir"
        paths = export_animation(images, str(nested), "sprite", engine="phaser")
        assert Path(paths[0]).exists()

    def test_single_frame_export(self, tmp_path):
        images = _make_sample_images(1)
        paths = export_animation(images, str(tmp_path), "single", engine="gamemaker")
        assert len(paths) == 2
        data = json.loads(Path(paths[1]).read_text())
        assert len(data["$GMSprite"]["frames"]) == 1

    def test_multiple_frames_phaser(self, tmp_path):
        images = _make_sample_images(4)
        paths = export_animation(images, str(tmp_path), "multi", engine="phaser")
        data = json.loads(Path(paths[1]).read_text())
        assert len(data["frames"]) == 4


class TestZipPackage:
    def test_creates_zip_with_files(self, tmp_path):
        a = tmp_path / "a.txt"
        a.write_text("hello")
        b = tmp_path / "b.txt"
        b.write_text("world")
        zip_path = str(tmp_path / "out.zip")
        result = zip_package([str(a), str(b)], zip_path)
        assert Path(result).exists()
        assert Path(result).suffix == ".zip"

    def test_zip_contains_all_files(self, tmp_path):
        a = tmp_path / "a.txt"
        a.write_text("hello")
        b = tmp_path / "sub" / "b.txt"
        b.parent.mkdir()
        b.write_text("world")
        zip_path = str(tmp_path / "out.zip")
        result = zip_package([str(a), str(b)], zip_path)
        import zipfile
        with zipfile.ZipFile(result) as zf:
            names = zf.namelist()
        assert "a.txt" in names
        assert "b.txt" in names

    def test_zip_skips_missing_files(self, tmp_path):
        a = tmp_path / "exists.txt"
        a.write_text("hello")
        missing = tmp_path / "missing.txt"
        zip_path = str(tmp_path / "out.zip")
        result = zip_package([str(a), str(missing)], zip_path)
        import zipfile
        with zipfile.ZipFile(result) as zf:
            names = zf.namelist()
        assert "exists.txt" in names
        assert "missing.txt" not in names
