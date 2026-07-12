from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class AssetType(str, Enum):
    CHARACTER = "character"
    BUILDING = "building"
    WEAPON = "weapon"
    VEHICLE = "vehicle"
    TREE = "tree"
    ROAD = "road"
    UI = "ui"
    ICON = "icon"
    ENEMY = "enemy"
    PROP = "prop"
    TILESET = "tileset"
    PROJECTILE = "projectile"
    EFFECT = "effect"


class View(str, Enum):
    FRONT = "front"
    SIDE = "side"
    TOP = "top"
    ISOMETRIC = "isometric"
    THREE_QUARTER = "3/4"
    BACK = "back"


class Palette(str, Enum):
    AUTO = "auto"
    RETRO_8 = "retro_8"
    RETRO_16 = "retro_16"
    RETRO_32 = "retro_32"
    MONOCHROME = "monochrome"
    GAMEBOY = "gameboy"
    SNES = "snes"
    CUSTOM = "custom"


class Animation(str, Enum):
    IDLE = "idle"
    WALK = "walk"
    RUN = "run"
    ATTACK = "attack"
    HURT = "hurt"
    DEATH = "death"
    JUMP = "jump"
    SHOOT = "shoot"
    CAST = "cast"
    NONE = "none"


class SpriteSize(str, Enum):
    S_16 = "16x16"
    S_32 = "32x32"
    S_64 = "64x64"
    S_128 = "128x128"


@dataclass
class AssetControls:
    asset_type: AssetType = AssetType.CHARACTER
    view: View = View.FRONT
    palette: Palette = Palette.AUTO
    animation: Animation = Animation.IDLE
    sprite_size: SpriteSize = SpriteSize.S_32
    theme: str = ""
    style: str = "pixel art"
    background: str = "transparent"
    custom_prompt: str = ""
    seed: int = -1
