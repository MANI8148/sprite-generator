from .controls import AssetControls, AssetType, View, Palette, Animation, SpriteSize

STYLE_MAP = {
    "pixel art": "pixel art, hard edges, flat colors, no anti-aliasing, sprite sheet style",
    "rpg": "RPG pixel art style, detailed, classic JRPG aesthetic",
    "retro": "retro 8-bit pixel art, chunky pixels, limited palette",
    "modern": "modern pixel art, clean edges, smooth pixel shading",
}

THEME_PROMPTS = {
    "fantasy": "fantasy theme, medieval",
    "sci-fi": "sci-fi theme, futuristic",
    "forest": "forest theme, nature, woodland",
    "dungeon": "dungeon theme, dark, underground",
    "desert": "desert theme, sandy, arid",
    "cyberpunk": "cyberpunk theme, neon, dark",
    "cave": "cave theme, rocky, underground",
    "castle": "castle theme, stone, royal",
}

SIZE_KEYWORDS = {
    SpriteSize.S_16: "tiny 16x16 sprite, small",
    SpriteSize.S_32: "32x32 sprite, medium",
    SpriteSize.S_64: "64x64 sprite, large",
    SpriteSize.S_128: "128x64 sprite, detailed",
}

VIEW_KEYWORDS = {
    View.FRONT: "front view, facing forward",
    View.SIDE: "side view, profile",
    View.TOP: "top-down view, bird's eye",
    View.ISOMETRIC: "isometric view, 3/4 perspective",
    View.THREE_QUARTER: "three-quarter view, angled",
    View.BACK: "back view, from behind",
}

PALETTE_KEYWORDS = {
    Palette.RETRO_8: "8-color palette, limited colors, NES style",
    Palette.RETRO_16: "16-color palette, GameBoy Advance style",
    Palette.RETRO_32: "32-color palette, SNES style",
    Palette.MONOCHROME: "monochrome, single color with shades",
    Palette.GAMEBOY: "GameBoy palette, 4-shade green",
    Palette.SNES: "SNES palette, vibrant 16-bit colors",
}


def build_prompt(controls: AssetControls) -> str:
    parts = []

    # Asset type
    if controls.asset_type == AssetType.CHARACTER:
        parts.append(f"a pixel art {controls.asset_type.value}")
    else:
        parts.append(f"a pixel art {controls.asset_type.value}")

    # Animation / action
    if controls.animation != Animation.NONE:
        anim_word = controls.animation.value
        parts.append(f"{anim_word} pose")

    # View
    view_str = VIEW_KEYWORDS.get(controls.view, controls.view.value)
    parts.append(view_str)

    # Theme
    if controls.theme:
        theme_str = THEME_PROMPTS.get(controls.theme, controls.theme)
        parts.append(theme_str)

    # Palette
    if controls.palette != Palette.AUTO:
        pal_str = PALETTE_KEYWORDS.get(controls.palette)
        if pal_str:
            parts.append(pal_str)

    # Size guidance
    size_str = SIZE_KEYWORDS.get(controls.sprite_size)
    if size_str:
        parts.append(size_str)

    # Style
    parts.append("pixel art style, hard edges, flat colors, clean outlines")

    # Background
    if controls.background == "transparent":
        parts.append("transparent background, no background")

    prompt = ", ".join(parts)

    # Custom prompt overrides or appends
    if controls.custom_prompt:
        prompt = f"{prompt}, {controls.custom_prompt}"

    return prompt
