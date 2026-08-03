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
    SpriteSize.S_128: "128x128 sprite, detailed",
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
        if controls.palette == Palette.CUSTOM:
            if controls.custom_palette_description:
                parts.append(f"{controls.custom_palette_description}, custom color palette")
        else:
            pal_str = PALETTE_KEYWORDS.get(controls.palette)
            if pal_str:
                parts.append(pal_str)

    # Size guidance
    size_str = SIZE_KEYWORDS.get(controls.sprite_size)
    if size_str:
        parts.append(size_str)

    # Style
    if controls.style and controls.style.lower() != "pixel art":
        style_str = STYLE_MAP.get(controls.style.lower())
        if style_str:
            parts.append(style_str)
        else:
            parts.append(f"{controls.style} style")

    parts.append("pixel art style, hard edges, flat colors, clean outlines")

    # Background
    if controls.background == "transparent":
        parts.append("transparent background, no background")

    prompt = ", ".join(parts)

    # Custom prompt overrides or appends
    if controls.custom_prompt:
        prompt = f"{prompt}, {controls.custom_prompt}"

    return prompt


def build_negative_prompt(controls: AssetControls) -> str:
    parts = ["blurry, low quality, distorted, ugly, bad anatomy"]

    if controls.asset_type == AssetType.CHARACTER:
        parts.append("extra limbs, missing limbs, deformed face")
    elif controls.asset_type in (AssetType.TILESET, AssetType.BUILDING):
        parts.append("seams, borders, uneven edges, misaligned tiles")
    elif controls.asset_type in (AssetType.UI, AssetType.ICON):
        parts.append("text, words, letters, complex background, scene")
    elif controls.asset_type == AssetType.ENEMY:
        parts.append("extra limbs, malformed, inconsistent style")
    elif controls.asset_type in (AssetType.VEHICLE, AssetType.PROP):
        parts.append("deformed shape, broken perspective, floating parts")
    elif controls.asset_type == AssetType.TREE:
        parts.append("unnatural colors, deformed canopy, floating, bare branches")
    elif controls.asset_type == AssetType.ROAD:
        parts.append("gaps, misaligned segments, uneven edges")
    elif controls.asset_type == AssetType.PROJECTILE:
        parts.append("blurry trail, wrong trajectory, static")
    elif controls.asset_type == AssetType.EFFECT:
        parts.append("flat, static, low energy, no glow")

    if controls.animation in (Animation.WALK, Animation.RUN):
        parts.append("static pose, stiff legs, no motion")
    elif controls.animation == Animation.ATTACK:
        parts.append("no weapon, static pose, missing action")
    elif controls.animation == Animation.JUMP:
        parts.append("standing on ground, no vertical motion")

    if controls.palette == Palette.MONOCHROME:
        parts.append("multiple colors, bright colors, gradients")
    elif controls.palette in (Palette.RETRO_8, Palette.GAMEBOY):
        parts.append("too many colors, smooth gradients, anti-aliasing")
    elif controls.palette == Palette.CUSTOM:
        parts.append("too many colors, rainbow, smooth gradients, anti-aliasing")

    if controls.sprite_size == SpriteSize.S_16:
        parts.append("details too small for 16x16, over-detailed")
    elif controls.sprite_size == SpriteSize.S_128:
        parts.append("too simple for 128x128, under-detailed, empty space")

    return ", ".join(parts)
