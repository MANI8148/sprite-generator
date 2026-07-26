import random
from typing import List, Optional


CLASS_VOCAB = ["character", "enemy", "item", "tile", "player", "weapon", "food",
               "vehicle", "building", "decoration", "effect", "projectile", "animal",
               "plant", "furniture", "tool", "accessory", "ui_element", "terrain",
               "environment", "prop", "icon", "portrait"]

ACTION_VOCAB = ["idle", "walk", "run", "attack", "jump", "hurt", "death", "block",
                "shoot", "cast", "interact", "fly", "swim", "climb", "slide", "dash",
                "crouch", "open", "close", "destroy"]

DIRECTION_VOCAB = ["front", "back", "left", "right", "front_left", "front_right",
                   "back_left", "back_right"]

_TEMPLATES = [
    "a pixel art sprite of a {cls}, {act} pose, {dir} view, high quality, transparent background",
    "pixel art {cls} performing {act}, seen from the {dir}, clean edges, game asset",
    "a {act} {cls} sprite, {dir} perspective, pixelated, video game character",
    "pixel art sprite of a {cls}, {act} animation, {dir} facing, retro game style",
    "{cls} pixel art, {act} action, {dir} view, transparent, sprite sheet quality",
    "retro pixel art {cls} in {act} state, {dir} orientation, game ready asset",
    "a game sprite of a {cls}, {act} movement, {dir} direction, 2D pixel art",
    "pixelated {cls} character, {act} pose, {dir} angle, clean pixel outline",
]

_SHORT_TEMPLATES = [
    "{cls} {act} {dir}",
    "pixel art {cls} {act} {dir}",
    "sprite {cls} {act} {dir}",
]


def _normalize(value: str, vocab: List[str]) -> str:
    normalized = value.lower().replace(" ", "_")
    if normalized in vocab:
        return normalized
    for v in vocab:
        if normalized == v or normalized.startswith(v) or v.startswith(normalized):
            return v
    return normalized if normalized else (vocab[0] if vocab else value)


def generate_caption(
    cls: str = "character",
    action: str = "idle",
    direction: str = "front",
    use_short: bool = False,
    templates: Optional[List[str]] = None,
) -> str:
    cls_norm = _normalize(cls, CLASS_VOCAB)
    act_norm = _normalize(action, ACTION_VOCAB)
    dir_norm = _normalize(direction, DIRECTION_VOCAB)

    pool = _SHORT_TEMPLATES if use_short else (templates or _TEMPLATES)
    template = random.choice(pool)
    return template.format(cls=cls_norm, act=act_norm, dir=dir_norm)


def generate_captions_batch(
    metadata: List[dict],
    use_short: bool = False,
    seed: Optional[int] = None,
) -> List[str]:
    if seed is not None:
        random.seed(seed)
    captions = []
    for item in metadata:
        captions.append(
            generate_caption(
                cls=item.get("class", "character"),
                action=item.get("action", "idle"),
                direction=item.get("direction", "front"),
                use_short=use_short,
            )
        )
    return captions


def generate_caption_from_hf_item(
    item: dict,
    use_short: bool = False,
    cls_field: str = "class",
    action_field: str = "action",
    direction_field: str = "direction",
) -> str:
    return generate_caption(
        cls=str(item.get(cls_field, "character")),
        action=str(item.get(action_field, "idle")),
        direction=str(item.get(direction_field, "front")),
        use_short=use_short,
    )
