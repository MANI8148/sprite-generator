from PIL import Image
import json
import os
import zipfile
import io
from typing import List, Optional
from ..packing.packer import sprite_sheet, animation_strip, individual_pngs
from ..postprocess.processor import normalize


def _format_metadata(metadata: dict) -> str:
    return json.dumps(metadata, indent=2)


def godot(atlas: Image.Image, metadata: dict, output_dir: str) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    name = metadata.get("name", "sprite")
    atlas_path = os.path.join(output_dir, f"{name}.png")
    atlas.save(atlas_path)
    tres_path = os.path.join(output_dir, f"{name}.tres")
    frames = metadata.get("frames", [])

    load_steps = 2 + len(frames)
    tres_lines = [
        f"[gd_resource type=\"SpriteFrames\" load_steps={load_steps} format=3 uid=\"uid://{name}_sprite\"]",
        "",
        f"[ext_resource type=\"Texture2D\" path=\"res://{os.path.basename(atlas_path)}\" id=\"1_{name}\"]",
        "",
    ]
    for i, f in enumerate(frames):
        sub_id = f"AtlasTexture_{name}_{i}"
        tres_lines.append(f"[sub_resource type=\"AtlasTexture\" id=\"{sub_id}\"]")
        tres_lines.append(f"atlas = ExtResource(\"1_{name}\")")
        tres_lines.append(f"region = Rect2({f['x']}, {f['y']}, {f['w']}, {f['h']})")
        tres_lines.append("")
    tres_lines.append("[resource]")
    tres_lines.append("animations = [{")
    tres_lines.append(f"\"name\": &\"{name}\",")
    tres_lines.append("\"speed\": 5.0,")
    tres_lines.append("\"loop\": true,")
    tres_lines.append("\"frames\": [")
    for i, f in enumerate(frames):
        sub_id = f"AtlasTexture_{name}_{i}"
        tres_lines.append("{")
        tres_lines.append("\"duration\": 0.2,")
        tres_lines.append(f"\"region\": SubResource(\"{sub_id}\")")
        tres_lines.append("}," if i < len(frames) - 1 else "}")
    tres_lines.append("]")
    tres_lines.append("}]")
    with open(tres_path, "w") as f:
        f.write("\n".join(tres_lines))
    return [atlas_path, tres_path]


def unity(atlas: Image.Image, metadata: dict, output_dir: str) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    name = metadata.get("name", "sprite")
    atlas_path = os.path.join(output_dir, f"{name}.png")
    atlas.save(atlas_path)
    meta_path = os.path.join(output_dir, f"{name}.png.meta")
    frames = metadata.get("frames", [])
    meta_lines = [
        "fileFormatVersion: 2",
        "guid: {}".format(hash(name) % (2**32)),
        "SpriteMetaData:",
    ]
    for f in frames:
        meta_lines.extend([
            "- name: frame_{}".format(f["index"]),
            "  rect:",
            "    serializedVersion: 2",
            "    x: {}".format(f["x"]),
            "    y: {}".format(f["y"]),
            "    width: {}".format(f["w"]),
            "    height: {}".format(f["h"]),
        ])
    with open(meta_path, "w") as f:
        f.write("\n".join(meta_lines))
    return [atlas_path, meta_path]


def generic_png(images: List[Image.Image], names: List[str], output_dir: str, atlas: bool = True) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    paths = individual_pngs(images, names, output_dir)
    if atlas and len(images) > 1:
        sheet, metadata = sprite_sheet(images, padding=2)
        metadata["name"] = "atlas"
        atlas_path = os.path.join(output_dir, "atlas.png")
        sheet.save(atlas_path)
        meta_path = os.path.join(output_dir, "atlas.json")
        with open(meta_path, "w") as f:
            f.write(_format_metadata(metadata))
        paths.extend([atlas_path, meta_path])
    return paths


def gamemaker(atlas: Image.Image, metadata: dict, output_dir: str) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    name = metadata.get("name", "sprite")
    atlas_path = os.path.join(output_dir, f"{name}.png")
    atlas.save(atlas_path)
    yy_path = os.path.join(output_dir, f"{name}.yy")
    frames = metadata.get("frames", [])
    yy_data = {
        "$GMSprite": {
            "name": name,
            "width": atlas.width,
            "height": atlas.height,
            "frames": [
                {
                    "frame": i,
                    "x": f["x"],
                    "y": f["y"],
                    "width": f["w"],
                    "height": f["h"],
                }
                for i, f in enumerate(frames)
            ],
        }
    }
    with open(yy_path, "w") as f:
        json.dump(yy_data, f, indent=2)
    return [atlas_path, yy_path]


def phaser(atlas: Image.Image, metadata: dict, output_dir: str) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    name = metadata.get("name", "sprite")
    atlas_path = os.path.join(output_dir, f"{name}.png")
    atlas.save(atlas_path)
    json_path = os.path.join(output_dir, f"{name}.json")
    frames = metadata.get("frames", [])
    phaser_frames = {}
    for i, f in enumerate(frames):
        key = f"sprite_{name}_{i}"
        phaser_frames[key] = {
            "frame": {"x": f["x"], "y": f["y"], "w": f["w"], "h": f["h"]},
            "rotated": False,
            "trimmed": False,
            "spriteSourceSize": {"x": 0, "y": 0, "w": f["w"], "h": f["h"]},
            "sourceSize": {"w": f["w"], "h": f["h"]},
        }
    atlas_data = {
        "frames": phaser_frames,
        "meta": {
            "app": "sprite-generator",
            "version": "1.0",
            "image": f"{name}.png",
            "size": {"w": atlas.width, "h": atlas.height},
            "scale": "1",
        },
    }
    with open(json_path, "w") as f:
        json.dump(atlas_data, f, indent=2)
    return [atlas_path, json_path]


def _engine_export(
    sheet: Image.Image,
    meta: dict,
    output_dir: str,
    engine: str,
) -> List[str]:
    """Dispatch a packed sheet + frame metadata to an engine-specific exporter.

    Returns the engine artifact paths, or ``[]`` for unrecognized engines so
    callers can fall back to a generic PNG + JSON export.
    """
    export_meta = dict(meta)
    export_meta["frames"] = [
        {"index": i, "x": f["x"], "y": f["y"], "w": f["w"], "h": f["h"]}
        for i, f in enumerate(meta.get("frames", []))
    ]
    if engine == "godot":
        return godot(sheet, export_meta, output_dir)
    elif engine == "unity":
        return unity(sheet, export_meta, output_dir)
    elif engine == "gamemaker":
        return gamemaker(sheet, export_meta, output_dir)
    elif engine == "phaser":
        return phaser(sheet, export_meta, output_dir)
    return []


def export_sprite(
    images: List[Image.Image],
    output_dir: str,
    name: str,
    engine: str = "godot",
    atlas: bool = True,
) -> List[str]:
    """Pack a list of images into a sprite sheet and export in the requested engine format.

    Unlike ``export_animation`` (which always builds a horizontal strip), this
    packs the images as a grid sprite sheet and honors the selected ``engine``
    (godot / unity / gamemaker / phaser), falling back to a generic PNG + JSON
    export for unknown engines. This makes engine selection work for static
    multi-frame assets too, not just animation strips.
    """
    if not atlas or len(images) == 1:
        names = [f"{name}_{i}" for i in range(len(images))]
        return individual_pngs(images, names, output_dir)
    norm = [normalize(img, target_size=(512, 512)) for img in images]
    sheet, meta = sprite_sheet(norm, padding=2)
    meta["name"] = name
    engine_paths = _engine_export(sheet, meta, output_dir, engine)
    if engine_paths:
        return engine_paths
    names = [f"{name}_{i}" for i in range(len(images))]
    return generic_png(norm, names, output_dir, atlas=True)


def _build_frames(meta: dict) -> list:
    fw = meta["frame_size"]["w"]
    fh = meta["frame_size"]["h"]
    pad = meta["padding"]
    frames = []
    for i in range(meta["frame_count"]):
        x = pad + i * (fw + pad)
        y = pad
        frames.append({"index": i, "x": x, "y": y, "w": fw, "h": fh})
    return frames


def export_animation(images: List[Image.Image], output_dir: str, name: str, engine: str = "godot") -> List[str]:
    norm = [normalize(img, target_size=(512, 512)) for img in images]
    strip, meta = animation_strip(norm, direction="horizontal", padding=2)
    meta["name"] = name
    meta["frames"] = _build_frames(meta)
    engine_paths = _engine_export(strip, meta, output_dir, engine)
    if engine_paths:
        return engine_paths
    os.makedirs(output_dir, exist_ok=True)
    strip_path = os.path.join(output_dir, f"{name}_strip.png")
    strip.save(strip_path)
    meta_path = os.path.join(output_dir, f"{name}_strip.json")
    with open(meta_path, "w") as f:
        f.write(_format_metadata(meta))
    return [strip_path, meta_path]


def zip_package(file_paths: List[str], output_path: str) -> str:
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in file_paths:
            if os.path.isfile(fp):
                zf.write(fp, os.path.basename(fp))
    return output_path
