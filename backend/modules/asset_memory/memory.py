import hashlib
import json
from typing import List, Optional, Tuple

from ..prompt_builder.controls import (
    AssetControls, AssetType, View, Palette, Animation, SpriteSize,
)
from ..pipeline.orchestrator import PipelineConfig, PipelineResult
from ..storage.asset_library import AssetLibrary, AssetRecord


def compute_generation_hash(
    controls: AssetControls,
    config: PipelineConfig,
) -> str:
    raw = json.dumps(
        {
            "asset_type": controls.asset_type.value,
            "view": controls.view.value,
            "animation": controls.animation.value,
            "palette": controls.palette.value,
            "sprite_size": controls.sprite_size.value,
            "theme": controls.theme,
            "style": controls.style,
            "seed": controls.seed,
            "remove_bg": config.remove_bg,
            "remove_bg_model": config.remove_bg_model,
            "remove_bg_alpha_threshold": config.remove_bg_alpha_threshold,
            "reduce_palette": config.reduce_palette,
            "max_colors": config.max_colors,
            "pixel_cleanup": config.pixel_cleanup,
            "auto_center": config.auto_center,
            "auto_pad": config.auto_pad,
            "normalize_size": config.normalize_size,
            "target_size": list(config.target_size),
            "upscale": config.upscale,
            "use_realesrgan": config.use_realesrgan,
            "outline_cleanup": config.outline_cleanup,
            "palette_lock": config.palette_lock,
            "palette_name": config.palette_name,
            "ip_adapter": config.ip_adapter,
            "ip_adapter_scale": config.ip_adapter_scale,
            "reference_image": config.reference_image,
            "pack_sheet": config.pack_sheet,
            "pack_tileset": config.pack_tileset,
            "export_engine": config.export_engine,
            "export_zip": config.export_zip,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def seed_from_generation_hash(generation_hash: str) -> int:
    """Derive a deterministic base seed from a generation hash.

    Lets a stored asset anchor the seed context for its variants without
    persisting the seed separately: the same asset always regenerates the
    same seed, and every distinct asset yields a distinct base seed.
    """
    if not generation_hash:
        return 1
    return int(generation_hash[:16], 16) % (2**31)


def build_variant_controls(
    base: AssetRecord,
    variant_index: int,
    seed_stride: int = 1,
    theme_override: str = "",
) -> AssetControls:
    """Derive controls for one variant of ``base``.

    Incremental regeneration keeps the base asset's style context
    (asset type / view / animation / palette / sprite size / style / theme)
    while varying the seed deterministically, so ``variant_index`` 0 of an
    asset always maps to the same seed and subsequent indexes stride forward.
    """
    meta = base.metadata or {}
    snapshot = meta.get("control_snapshot") or {}

    def _pick(key: str, default: str) -> str:
        return snapshot.get(key) or meta.get(key) or default

    asset_type = _pick("asset_type", base.asset_type or AssetType.CHARACTER.value)
    view = _pick("view", View.FRONT.value)
    animation = _pick("animation", Animation.IDLE.value)
    palette = _pick("palette", Palette.AUTO.value)
    sprite_size = _pick("sprite_size", SpriteSize.S_32.value)
    style = _pick("style", "pixel art")
    theme = theme_override if theme_override else _pick("theme", "")
    base_seed = seed_from_generation_hash(base.generation_hash or "")
    raw_seed = snapshot.get("seed", meta.get("seed"))
    if raw_seed is not None:
        try:
            base_seed = int(raw_seed)
        except (TypeError, ValueError):
            pass
    variant_seed = (base_seed + variant_index * max(1, seed_stride)) % (2**31)

    return AssetControls(
        asset_type=AssetType(asset_type),
        view=View(view),
        animation=Animation(animation),
        palette=Palette(palette),
        sprite_size=SpriteSize(sprite_size),
        theme=theme,
        style=style,
        seed=variant_seed,
    )


def cache_result_key(
    pipeline_result: PipelineResult,
) -> str:
    meta = pipeline_result.metadata
    controls_dict = meta.get("controls", {})
    raw = json.dumps(controls_dict, sort_keys=True, separators=(",", ":"))
    raw += json.dumps(
        [v.get("quality_tier", "") for v in pipeline_result.validation],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class AssetMemory:
    def __init__(self, library: AssetLibrary):
        self._library = library

    def lookup(self, generation_hash: str) -> Optional[AssetRecord]:
        return self._library.find_by_generation_hash(generation_hash)

    def store(
        self,
        generation_hash: str,
        record: AssetRecord,
    ):
        record.metadata["generation_hash"] = generation_hash
        self._library.add_asset(record)

    def plan_variants(
        self,
        base_id: str,
        num_variants: int = 4,
        seed_stride: int = 1,
        theme_override: str = "",
    ) -> Tuple[AssetRecord, List[AssetControls]]:
        """Plan ``num_variants`` regeneration jobs for an existing asset.

        Returns the base :class:`AssetRecord` and the derived variant
        controls. Raises ``KeyError`` when ``base_id`` is unknown.
        """
        base = self._library.get_asset(base_id)
        if base is None:
            raise KeyError(f"Base asset '{base_id}' not found")
        variants = [
            build_variant_controls(base, i, seed_stride=seed_stride, theme_override=theme_override)
            for i in range(max(1, num_variants))
        ]
        return base, variants
