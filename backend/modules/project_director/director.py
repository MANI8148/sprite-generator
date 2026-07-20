import re
from typing import List, Optional
from dataclasses import dataclass, field

from ..prompt_builder.controls import AssetType, View, Animation, Palette, SpriteSize


ASSET_KEYWORDS = {
    "character": AssetType.CHARACTER,
    "hero": AssetType.CHARACTER,
    "player": AssetType.CHARACTER,
    "enemy": AssetType.ENEMY,
    "monster": AssetType.ENEMY,
    "building": AssetType.BUILDING,
    "house": AssetType.BUILDING,
    "weapon": AssetType.WEAPON,
    "sword": AssetType.WEAPON,
    "vehicle": AssetType.VEHICLE,
    "car": AssetType.VEHICLE,
    "tree": AssetType.TREE,
    "tileset": AssetType.TILESET,
    "tile": AssetType.TILESET,
    "tiles": AssetType.TILESET,
    "environment": AssetType.TREE,
    "background": AssetType.TREE,
    "prop": AssetType.PROP,
    "object": AssetType.PROP,
    "item": AssetType.PROP,
    "ui": AssetType.UI,
    "icon": AssetType.ICON,
    "projectile": AssetType.PROJECTILE,
    "bullet": AssetType.PROJECTILE,
    "effect": AssetType.EFFECT,
    "particle": AssetType.EFFECT,
}

VIEW_KEYWORDS = {
    "front": View.FRONT,
    "side": View.SIDE,
    "top": View.TOP,
    "isometric": View.ISOMETRIC,
    "iso": View.ISOMETRIC,
    "3/4": View.THREE_QUARTER,
    "three-quarter": View.THREE_QUARTER,
    "back": View.BACK,
    "rear": View.BACK,
}

ANIMATION_KEYWORDS = {
    "idle": Animation.IDLE,
    "standing": Animation.IDLE,
    "walk": Animation.WALK,
    "walking": Animation.WALK,
    "run": Animation.RUN,
    "running": Animation.RUN,
    "attack": Animation.ATTACK,
    "attacking": Animation.ATTACK,
    "hurt": Animation.HURT,
    "damage": Animation.HURT,
    "death": Animation.DEATH,
    "dying": Animation.DEATH,
    "jump": Animation.JUMP,
    "jumping": Animation.JUMP,
    "shoot": Animation.SHOOT,
    "shooting": Animation.SHOOT,
    "cast": Animation.CAST,
    "casting": Animation.CAST,
}

PALETTE_KEYWORDS = {
    "retro_8": Palette.RETRO_8,
    "8 color": Palette.RETRO_8,
    "8-colour": Palette.RETRO_8,
    "retro_16": Palette.RETRO_16,
    "16 color": Palette.RETRO_16,
    "16-colour": Palette.RETRO_16,
    "retro_32": Palette.RETRO_32,
    "32 color": Palette.RETRO_32,
    "32-colour": Palette.RETRO_32,
    "monochrome": Palette.MONOCHROME,
    "mono": Palette.MONOCHROME,
    "gameboy": Palette.GAMEBOY,
    "gb": Palette.GAMEBOY,
    "snes": Palette.SNES,
    "auto": Palette.AUTO,
}

SIZE_KEYWORDS = {
    "16x16": SpriteSize.S_16,
    "32x32": SpriteSize.S_32,
    "64x64": SpriteSize.S_64,
    "128x128": SpriteSize.S_128,
}

FRAME_KEYWORDS = {
    r"(\d+)[ -]?frame": 1,
    r"(\d+)[ -]?direction": 1,
}


@dataclass
class PlanStep:
    asset_type: AssetType
    view: View = View.FRONT
    animation: Animation = Animation.IDLE
    palette: Palette = Palette.AUTO
    sprite_size: SpriteSize = SpriteSize.S_32
    theme: str = ""
    num_frames: int = 1
    seed: int = -1

    def to_dict(self) -> dict:
        return {
            "asset_type": self.asset_type.value,
            "view": self.view.value,
            "animation": self.animation.value,
            "palette": self.palette.value,
            "sprite_size": self.sprite_size.value,
            "theme": self.theme,
            "num_frames": self.num_frames,
            "seed": self.seed,
        }


@dataclass
class ProjectPlan:
    title: str
    steps: List[PlanStep] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "steps": [s.to_dict() for s in self.steps],
            "total_steps": len(self.steps),
        }


class ProjectDirector:
    def parse(self, request: str) -> ProjectPlan:
        request_lower = request.lower()
        steps: List[PlanStep] = []

        segments = _split_request(request)

        for seg in segments:
            asset_type = _detect_asset_type(seg)
            view = _detect_view(seg)
            animation = _detect_animation(seg)
            palette = _detect_palette(seg)
            sprite_size = _detect_size(seg)
            num_frames = _detect_num_frames(seg)
            theme = _extract_theme(seg, asset_type)

            step = PlanStep(
                asset_type=asset_type,
                view=view,
                animation=animation,
                palette=palette,
                sprite_size=sprite_size,
                theme=theme,
                num_frames=num_frames,
            )
            steps.append(step)

        if not steps:
            steps.append(PlanStep(asset_type=AssetType.CHARACTER))

        title = _generate_title(request, steps)
        return ProjectPlan(title=title, steps=steps)


def _split_request(request: str) -> List[str]:
    separators = [r"\band\b", r"\bplus\b", r"\balso\b", r"[,;]"]
    parts = [request]
    for sep in separators:
        new_parts = []
        for p in parts:
            new_parts.extend(re.split(sep, p, flags=re.IGNORECASE))
        parts = [p.strip() for p in new_parts if p.strip()]
    return parts if len(parts) > 1 else [request]


def _detect_asset_type(text: str) -> AssetType:
    for keyword, atype in ASSET_KEYWORDS.items():
        if keyword in text:
            return atype
    return AssetType.CHARACTER


def _detect_view(text: str) -> View:
    for keyword, view in VIEW_KEYWORDS.items():
        if keyword in text:
            return view
    return View.FRONT


def _detect_animation(text: str) -> Animation:
    for keyword, anim in ANIMATION_KEYWORDS.items():
        if keyword in text:
            return anim
    return Animation.IDLE


def _detect_palette(text: str) -> Palette:
    for keyword, pal in PALETTE_KEYWORDS.items():
        if keyword in text:
            return pal
    return Palette.AUTO


def _detect_size(text: str) -> SpriteSize:
    for keyword, size in SIZE_KEYWORDS.items():
        if keyword in text:
            return size
    return SpriteSize.S_32


def _detect_num_frames(text: str) -> int:
    for pattern, group in FRAME_KEYWORDS.items():
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            try:
                return int(m.group(group))
            except (ValueError, IndexError):
                pass
    return 1


def _extract_theme(text: str, asset_type: AssetType) -> str:
    theme_words = []
    skip_words = set()
    for keyword, _ in ASSET_KEYWORDS.items():
        skip_words.add(keyword)
    for keyword, _ in VIEW_KEYWORDS.items():
        skip_words.add(keyword)
    for keyword, _ in ANIMATION_KEYWORDS.items():
        skip_words.add(keyword)
    for keyword, _ in PALETTE_KEYWORDS.items():
        skip_words.add(keyword)
    for keyword, _ in SIZE_KEYWORDS.items():
        skip_words.add(keyword)
    skip_words.update({"a", "an", "the", "with", "in", "and", "or", "but", "of", "for", "to", "is", "that", "this", "create", "make", "generate", "need", "want", "please", "would", "like"})

    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    for w in words:
        if w not in skip_words and len(w) > 2:
            theme_words.append(w)

    return " ".join(theme_words[:8]) if theme_words else ""


def _generate_title(request: str, steps: List[PlanStep]) -> str:
    if len(steps) == 1:
        return f"Generate {steps[0].asset_type.value}"
    types = [s.asset_type.value for s in steps]
    unique = list(dict.fromkeys(types))
    if len(unique) <= 2:
        return f"Generate {' and '.join(unique)}"
    return f"Multi-asset project ({', '.join(unique[:3])})..."
