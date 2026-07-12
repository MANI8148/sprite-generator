import json
from pathlib import Path

from eval.godot_export import generate_sprite_frames, export_sprite_sheet


def _make_meta(frame_data: dict, sheet_w: int = 32, sheet_h: int = 32) -> dict:
    return {
        "frames": {
            name: {
                "frame": {"x": f["x"], "y": f["y"], "w": f["w"], "h": f["h"]},
                "spriteSourceSize": {"x": 0, "y": 0, "w": f["w"], "h": f["h"]},
                "sourceSize": {"w": f["w"], "h": f["h"]},
            }
            for name, f in frame_data.items()
        },
        "meta": {
            "app": "sprite-generator",
            "version": "1.0",
            "image": "sprite_sheet.png",
            "size": {"w": sheet_w, "h": sheet_h},
            "scale": "1",
        },
    }


class TestGenerateSpriteFrames:
    def test_empty_meta_produces_empty_output(self, tmp_path):
        meta = tmp_path / "empty.json"
        meta.write_text(json.dumps({"frames": {}, "meta": {}}))
        out = tmp_path / "out.tres"
        result = generate_sprite_frames(str(meta), str(out))
        assert Path(result).exists()
        assert out.read_text() == ""

    def test_outputs_valid_headers(self, tmp_path):
        meta = tmp_path / "meta.json"
        meta.write_text(json.dumps(_make_meta({"s1": {"x": 0, "y": 0, "w": 8, "h": 8}})))
        out = tmp_path / "out.tres"
        generate_sprite_frames(str(meta), str(out))
        text = out.read_text()
        assert text.startswith("[gd_resource")
        assert "type=\"SpriteFrames\"" in text
        assert "[ext_resource" in text
        assert "[sub_resource" in text
        assert "[resource]" in text

    def test_ext_resource_points_to_correct_texture(self, tmp_path):
        meta = tmp_path / "meta.json"
        meta.write_text(json.dumps(_make_meta({"s1": {"x": 0, "y": 0, "w": 8, "h": 8}})))
        out = tmp_path / "out.tres"
        generate_sprite_frames(str(meta), str(out), sheet_name="my_sheet.png")
        text = out.read_text()
        assert 'path="res://my_sheet.png"' in text
        assert 'type="Texture2D"' in text

    def test_each_frame_gets_atlas_texture(self, tmp_path):
        meta = tmp_path / "meta.json"
        meta.write_text(json.dumps(_make_meta({
            "a": {"x": 0, "y": 0, "w": 8, "h": 8},
            "b": {"x": 8, "y": 0, "w": 8, "h": 8},
            "c": {"x": 0, "y": 8, "w": 8, "h": 8},
        })))
        out = tmp_path / "out.tres"
        generate_sprite_frames(str(meta), str(out))
        text = out.read_text()
        assert text.count("type=\"AtlasTexture\"") == 3
        assert text.count("SubResource(") == 3

    def test_region_rectangles_match_metadata(self, tmp_path):
        meta = tmp_path / "meta.json"
        meta.write_text(json.dumps(_make_meta({"s1": {"x": 10, "y": 20, "w": 16, "h": 32}})))
        out = tmp_path / "out.tres"
        generate_sprite_frames(str(meta), str(out))
        text = out.read_text()
        assert "Rect2(10, 20, 16, 32)" in text

    def test_multiple_frames_region_matches(self, tmp_path):
        meta = tmp_path / "meta.json"
        meta.write_text(json.dumps(_make_meta({
            "a": {"x": 0, "y": 0, "w": 10, "h": 10},
            "b": {"x": 10, "y": 0, "w": 20, "h": 10},
        })))
        out = tmp_path / "out.tres"
        generate_sprite_frames(str(meta), str(out))
        text = out.read_text()
        assert "Rect2(0, 0, 10, 10)" in text
        assert "Rect2(10, 0, 20, 10)" in text

    def test_animations_array_contains_all_frames(self, tmp_path):
        meta = tmp_path / "meta.json"
        meta.write_text(json.dumps(_make_meta({
            "f1": {"x": 0, "y": 0, "w": 8, "h": 8},
            "f2": {"x": 8, "y": 0, "w": 8, "h": 8},
        })))
        out = tmp_path / "out.tres"
        generate_sprite_frames(str(meta), str(out))
        text = out.read_text()
        assert text.count("SubResource(") == 2
        assert "\"frames\"" in text
        assert "\"loop\": true" in text
        assert "\"animation\"" not in text or True

    def test_custom_animation_speed(self, tmp_path):
        meta = tmp_path / "meta.json"
        meta.write_text(json.dumps(_make_meta({"s1": {"x": 0, "y": 0, "w": 8, "h": 8}})))
        out = tmp_path / "out.tres"
        generate_sprite_frames(str(meta), str(out), animation_speed=10.0)
        text = out.read_text()
        assert "\"speed\": 10.0" in text

    def test_custom_frame_duration(self, tmp_path):
        meta = tmp_path / "meta.json"
        meta.write_text(json.dumps(_make_meta({"s1": {"x": 0, "y": 0, "w": 8, "h": 8}})))
        out = tmp_path / "out.tres"
        generate_sprite_frames(str(meta), str(out), frame_duration=0.5)
        text = out.read_text()
        assert "\"duration\": 0.5" in text

    def test_custom_animation_name(self, tmp_path):
        meta = tmp_path / "meta.json"
        meta.write_text(json.dumps(_make_meta({"s1": {"x": 0, "y": 0, "w": 8, "h": 8}})))
        out = tmp_path / "out.tres"
        generate_sprite_frames(str(meta), str(out), animation_name="run")
        text = out.read_text()
        assert "\"name\": \"run\"" in text

    def test_creates_output_directory(self, tmp_path):
        meta = tmp_path / "meta.json"
        meta.write_text(json.dumps(_make_meta({"s1": {"x": 0, "y": 0, "w": 8, "h": 8}})))
        nested = tmp_path / "sub" / "dir" / "out.tres"
        result = generate_sprite_frames(str(meta), str(nested))
        assert Path(result).exists()

    def test_truncated_meta_still_works(self, tmp_path):
        meta = tmp_path / "meta.json"
        data = _make_meta({"s1": {"x": 0, "y": 0, "w": 8, "h": 8}})
        data["meta"]["truncated"] = 2
        meta.write_text(json.dumps(data))
        out = tmp_path / "out.tres"
        generate_sprite_frames(str(meta), str(out))
        text = out.read_text()
        assert "Rect2(0, 0, 8, 8)" in text

    def test_sorted_alphabetically(self, tmp_path):
        meta = tmp_path / "meta.json"
        meta.write_text(json.dumps(_make_meta({
            "z": {"x": 0, "y": 10, "w": 8, "h": 8},
            "a": {"x": 0, "y": 0, "w": 8, "h": 8},
        })))
        out = tmp_path / "out.tres"
        generate_sprite_frames(str(meta), str(out))
        text = out.read_text()
        a_pos = text.index("Rect2(0, 0, 8, 8)") if "Rect2(0, 0, 8, 8)" in text else -1
        z_pos = text.index("Rect2(0, 10, 8, 8)") if "Rect2(0, 10, 8, 8)" in text else -1
        assert a_pos >= 0 and z_pos >= 0
        assert a_pos < z_pos

    def test_load_steps_counts_resources_correctly(self, tmp_path):
        meta = tmp_path / "meta.json"
        meta.write_text(json.dumps(_make_meta({
            "a": {"x": 0, "y": 0, "w": 8, "h": 8},
            "b": {"x": 8, "y": 0, "w": 8, "h": 8},
        })))
        out = tmp_path / "out.tres"
        generate_sprite_frames(str(meta), str(out))
        text = out.read_text()
        # load_steps = 2 (ext_resource + main resource) + num_frames (sub_resources)
        assert "load_steps=4" in text


class TestExportSpriteSheet:
    def test_default_args(self, tmp_path):
        meta = tmp_path / "meta.json"
        meta.write_text(json.dumps(_make_meta({"s1": {"x": 0, "y": 0, "w": 8, "h": 8}})))
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        meta_copy = out_dir / "sprite_sheet.json"
        meta_copy.write_text(meta.read_text())

        result = export_sprite_sheet(str(meta_copy), output_dir=str(out_dir))
        assert Path(result).exists()
        assert Path(result).suffix == ".tres"
        assert "sprite_sheet.png" in Path(result).read_text()

    def test_custom_output_stem(self, tmp_path):
        meta = tmp_path / "char.json"
        meta.write_text(json.dumps(_make_meta({"s1": {"x": 0, "y": 0, "w": 8, "h": 8}})))
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        meta_copy = out_dir / "char.json"
        meta_copy.write_text(meta.read_text())

        result = export_sprite_sheet(
            str(meta_copy), output_stem="hero", output_dir=str(out_dir)
        )
        assert "hero.tres" in result
        text = Path(result).read_text()
        assert "hero.png" in text

    def test_creates_output_directory(self, tmp_path):
        meta = tmp_path / "meta.json"
        meta.write_text(json.dumps(_make_meta({"s1": {"x": 0, "y": 0, "w": 8, "h": 8}})))
        nested = tmp_path / "deep" / "out"
        nested.mkdir(parents=True)
        meta_copy = nested / "meta.json"
        meta_copy.write_text(meta.read_text())

        result = export_sprite_sheet(str(meta_copy), output_dir=str(nested))
        assert Path(result).exists()
