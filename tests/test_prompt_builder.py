from backend.modules.prompt_builder.builder import build_prompt
from backend.modules.prompt_builder.controls import (
    AssetControls, AssetType, View, Palette, Animation, SpriteSize,
)


class TestAssetType:
    def test_character(self):
        controls = AssetControls(asset_type=AssetType.CHARACTER, animation=Animation.NONE)
        prompt = build_prompt(controls)
        assert "character" in prompt

    def test_building(self):
        controls = AssetControls(asset_type=AssetType.BUILDING, animation=Animation.NONE)
        prompt = build_prompt(controls)
        assert "building" in prompt

    def test_weapon(self):
        controls = AssetControls(asset_type=AssetType.WEAPON, animation=Animation.NONE)
        prompt = build_prompt(controls)
        assert "weapon" in prompt

    def test_vehicle(self):
        controls = AssetControls(asset_type=AssetType.VEHICLE, animation=Animation.NONE)
        prompt = build_prompt(controls)
        assert "vehicle" in prompt

    def test_enemy(self):
        controls = AssetControls(asset_type=AssetType.ENEMY, animation=Animation.NONE)
        prompt = build_prompt(controls)
        assert "enemy" in prompt

    def test_tileset(self):
        controls = AssetControls(asset_type=AssetType.TILESET, animation=Animation.NONE)
        prompt = build_prompt(controls)
        assert "tileset" in prompt

    def test_prop(self):
        controls = AssetControls(asset_type=AssetType.PROP, animation=Animation.NONE)
        prompt = build_prompt(controls)
        assert "prop" in prompt

    def test_tree(self):
        controls = AssetControls(asset_type=AssetType.TREE, animation=Animation.NONE)
        prompt = build_prompt(controls)
        assert "tree" in prompt

    def test_ui(self):
        controls = AssetControls(asset_type=AssetType.UI, animation=Animation.NONE)
        prompt = build_prompt(controls)
        assert "ui" in prompt

    def test_icon(self):
        controls = AssetControls(asset_type=AssetType.ICON, animation=Animation.NONE)
        prompt = build_prompt(controls)
        assert "icon" in prompt

    def test_road(self):
        controls = AssetControls(asset_type=AssetType.ROAD, animation=Animation.NONE)
        prompt = build_prompt(controls)
        assert "road" in prompt


class TestView:
    def test_front(self):
        controls = AssetControls(view=View.FRONT, animation=Animation.NONE)
        prompt = build_prompt(controls)
        assert "front view" in prompt

    def test_side(self):
        controls = AssetControls(view=View.SIDE, animation=Animation.NONE)
        prompt = build_prompt(controls)
        assert "side view" in prompt

    def test_top(self):
        controls = AssetControls(view=View.TOP, animation=Animation.NONE)
        prompt = build_prompt(controls)
        assert "top-down" in prompt

    def test_isometric(self):
        controls = AssetControls(view=View.ISOMETRIC, animation=Animation.NONE)
        prompt = build_prompt(controls)
        assert "isometric" in prompt

    def test_three_quarter(self):
        controls = AssetControls(view=View.THREE_QUARTER, animation=Animation.NONE)
        prompt = build_prompt(controls)
        assert "three-quarter" in prompt

    def test_back(self):
        controls = AssetControls(view=View.BACK, animation=Animation.NONE)
        prompt = build_prompt(controls)
        assert "back view" in prompt


class TestAnimation:
    def test_idle(self):
        controls = AssetControls(animation=Animation.IDLE)
        prompt = build_prompt(controls)
        assert "idle" in prompt

    def test_walk(self):
        controls = AssetControls(animation=Animation.WALK)
        prompt = build_prompt(controls)
        assert "walk" in prompt

    def test_run(self):
        controls = AssetControls(animation=Animation.RUN)
        prompt = build_prompt(controls)
        assert "run" in prompt

    def test_attack(self):
        controls = AssetControls(animation=Animation.ATTACK)
        prompt = build_prompt(controls)
        assert "attack" in prompt

    def test_hurt(self):
        controls = AssetControls(animation=Animation.HURT)
        prompt = build_prompt(controls)
        assert "hurt" in prompt

    def test_none_excludes_animation(self):
        controls = AssetControls(animation=Animation.NONE)
        prompt = build_prompt(controls)
        assert "pose" not in prompt


class TestSpritesize:
    def test_s_16(self):
        controls = AssetControls(sprite_size=SpriteSize.S_16, animation=Animation.NONE)
        prompt = build_prompt(controls)
        assert "16x16" in prompt

    def test_s_32(self):
        controls = AssetControls(sprite_size=SpriteSize.S_32, animation=Animation.NONE)
        prompt = build_prompt(controls)
        assert "32x32" in prompt

    def test_s_64(self):
        controls = AssetControls(sprite_size=SpriteSize.S_64, animation=Animation.NONE)
        prompt = build_prompt(controls)
        assert "64x64" in prompt

    def test_s_128(self):
        controls = AssetControls(sprite_size=SpriteSize.S_128, animation=Animation.NONE)
        prompt = build_prompt(controls)
        assert "128x64" in prompt


class TestPalette:
    def test_auto_excludes_palette_keyword(self):
        controls = AssetControls(palette=Palette.AUTO, animation=Animation.NONE)
        prompt = build_prompt(controls)
        assert "palette" not in prompt.lower()

    def test_retro_8(self):
        controls = AssetControls(palette=Palette.RETRO_8, animation=Animation.NONE)
        prompt = build_prompt(controls)
        assert "8-color" in prompt or "NES" in prompt

    def test_retro_16(self):
        controls = AssetControls(palette=Palette.RETRO_16, animation=Animation.NONE)
        prompt = build_prompt(controls)
        assert "16-color" in prompt or "GameBoy Advance" in prompt

    def test_gameboy(self):
        controls = AssetControls(palette=Palette.GAMEBOY, animation=Animation.NONE)
        prompt = build_prompt(controls)
        assert "GameBoy" in prompt

    def test_monochrome(self):
        controls = AssetControls(palette=Palette.MONOCHROME, animation=Animation.NONE)
        prompt = build_prompt(controls)
        assert "monochrome" in prompt

    def test_snes(self):
        controls = AssetControls(palette=Palette.SNES, animation=Animation.NONE)
        prompt = build_prompt(controls)
        assert "SNES" in prompt or "16-bit" in prompt


class TestTheme:
    def test_empty_theme_excluded(self):
        controls = AssetControls(theme="", animation=Animation.NONE)
        prompt = build_prompt(controls)
        assert ", ," not in prompt

    def test_fantasy_theme(self):
        controls = AssetControls(theme="fantasy", animation=Animation.NONE)
        prompt = build_prompt(controls)
        assert "fantasy" in prompt

    def test_custom_theme(self):
        controls = AssetControls(theme="underwater", animation=Animation.NONE)
        prompt = build_prompt(controls)
        assert "underwater" in prompt


class TestBackground:
    def test_transparent_background(self):
        controls = AssetControls(background="transparent", animation=Animation.NONE)
        prompt = build_prompt(controls)
        assert "transparent background" in prompt

    def test_non_transparent_background(self):
        controls = AssetControls(background="white", animation=Animation.NONE)
        prompt = build_prompt(controls)
        assert "transparent background" not in prompt


class TestCustomPrompt:
    def test_custom_prompt_appended(self):
        controls = AssetControls(custom_prompt="highly detailed", animation=Animation.NONE)
        prompt = build_prompt(controls)
        assert "highly detailed" in prompt

    def test_empty_custom_prompt(self):
        controls = AssetControls(custom_prompt="", animation=Animation.NONE)
        prompt = build_prompt(controls)
        assert "highly detailed" not in prompt


class TestStyleConsistency:
    def test_pixel_art_style_always_included(self):
        controls = AssetControls(animation=Animation.NONE)
        prompt = build_prompt(controls)
        assert "pixel art" in prompt
        assert "hard edges" in prompt
        assert "flat colors" in prompt

    def test_prompt_is_comma_separated(self):
        controls = AssetControls(
            asset_type=AssetType.CHARACTER,
            view=View.FRONT,
            animation=Animation.IDLE,
            theme="forest",
        )
        prompt = build_prompt(controls)
        parts = [p.strip() for p in prompt.split(",")]
        assert len(parts) >= 4


class TestPromptComposition:
    def test_full_controls_generate_long_prompt(self):
        controls = AssetControls(
            asset_type=AssetType.ENEMY,
            view=View.SIDE,
            animation=Animation.ATTACK,
            palette=Palette.GAMEBOY,
            sprite_size=SpriteSize.S_64,
            theme="dungeon",
            background="transparent",
            custom_prompt="glowing eyes",
        )
        prompt = build_prompt(controls)
        assert "enemy" in prompt
        assert "attack" in prompt
        assert "side view" in prompt
        assert "dungeon" in prompt
        assert "GameBoy" in prompt
        assert "64x64" in prompt
        assert "transparent background" in prompt
        assert "glowing eyes" in prompt

    def test_prompt_does_not_end_with_comma(self):
        controls = AssetControls(animation=Animation.NONE)
        prompt = build_prompt(controls)
        assert not prompt.endswith(", ")

    def test_minimal_controls(self):
        controls = AssetControls(animation=Animation.NONE)
        prompt = build_prompt(controls)
        assert isinstance(prompt, str)
        assert len(prompt) > 10


class TestAssetTypeVariants:
    def test_projectile(self):
        controls = AssetControls(asset_type=AssetType.PROJECTILE, animation=Animation.NONE)
        prompt = build_prompt(controls)
        assert "projectile" in prompt

    def test_effect(self):
        controls = AssetControls(asset_type=AssetType.EFFECT, animation=Animation.NONE)
        prompt = build_prompt(controls)
        assert "effect" in prompt
