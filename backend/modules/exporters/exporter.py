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
    tres_lines = [
        "[gd_resource type=\"SpriteFrames\" load_steps=2 format=3 uid=\"uid://{}_{}\"]".format(name, "sprite"),
        "",
        "[ext_resource type=\"Texture2D\" path=\"res://{}\"]".format(os.path.basename(atlas_path)),
        "",
        "[resource]",
        "animations = [{",
        "\"name\": &\"{}\",".format(name),
        "\"speed\": 5.0,",
        "\"loop\": true,",
    ]
    for f in frames:
        tres_lines.append("\"frames\": [{")
        tres_lines.append("\"duration\": 0.2,")
        tres_lines.append("\"region\": Rect2({}, {}, {}, {})".format(f["x"], f["y"], f["w"], f["h"]))
        tres_lines.append("}],")
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


def export_animation(images: List[Image.Image], output_dir: str, name: str, engine: str = "godot") -> List[str]:
    norm = [normalize(img, target_size=(512, 512)) for img in images]
    strip, meta = animation_strip(norm, direction="horizontal", padding=2)
    meta["name"] = name
    if engine == "godot":
        return godot(strip, meta, output_dir)
    elif engine == "unity":
        return unity(strip, meta, output_dir)
    else:
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
