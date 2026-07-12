import json
from pathlib import Path
from typing import Optional


def generate_sprite_frames(
    meta_path: str,
    output_path: str,
    sheet_name: str = "sprite_sheet.png",
    animation_speed: float = 5.0,
    frame_duration: float = 1.0,
    animation_name: str = "default",
) -> str:
    with open(meta_path) as f:
        data = json.load(f)

    frames = data.get("frames", {})
    if not frames:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("")
        return str(out)

    sorted_names = sorted(frames.keys())
    num_frames = len(sorted_names)
    load_steps = 2 + num_frames

    lines = [
        f'[gd_resource type="SpriteFrames" load_steps={load_steps} format=3]',
        "",
        f'[ext_resource type="Texture2D" path="res://{sheet_name}" id="1"]',
        "",
    ]

    sub_ids = []
    for i, name in enumerate(sorted_names):
        sid = i + 1
        sub_ids.append(sid)
        f = frames[name]["frame"]
        lines.append(f"[sub_resource type=\"AtlasTexture\" id={sid}]")
        lines.append(f'atlas = ExtResource("1")')
        lines.append(f"region = Rect2({f['x']}, {f['y']}, {f['w']}, {f['h']})")
        lines.append("")

    frame_entries = ",\n".join(
        f'{{"duration": {frame_duration}, "texture": SubResource({sid})}}'
        for sid in sub_ids
    )

    lines.append("[resource]")
    lines.append(f"animations = [{{\n\"frames\": [\n{frame_entries}\n],\n\"loop\": true,\n\"name\": \"{animation_name}\",\n\"speed\": {animation_speed}\n}}]")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return str(out)


def export_sprite_sheet(
    meta_path: str,
    output_stem: str = "sprite_sheet",
    output_dir: str = ".",
    sheet_name: Optional[str] = None,
    animation_speed: float = 5.0,
    frame_duration: float = 1.0,
    animation_name: str = "default",
) -> str:
    if sheet_name is None:
        sheet_name = f"{output_stem}.png"
    tres_path = str(Path(output_dir) / f"{output_stem}.tres")
    return generate_sprite_frames(
        meta_path=meta_path,
        output_path=tres_path,
        sheet_name=sheet_name,
        animation_speed=animation_speed,
        frame_duration=frame_duration,
        animation_name=animation_name,
    )
