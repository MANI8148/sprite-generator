import hashlib
import json
from typing import Optional

from ..prompt_builder.controls import AssetControls
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
            "seed": controls.seed,
            "remove_bg": config.remove_bg,
            "reduce_palette": config.reduce_palette,
            "max_colors": config.max_colors,
            "pixel_cleanup": config.pixel_cleanup,
            "auto_center": config.auto_center,
            "auto_pad": config.auto_pad,
            "normalize_size": config.normalize_size,
            "target_size": list(config.target_size),
            "upscale": config.upscale,
            "outline_cleanup": config.outline_cleanup,
            "palette_lock": config.palette_lock,
            "palette_name": config.palette_name,
            "ip_adapter": config.ip_adapter,
            "ip_adapter_scale": config.ip_adapter_scale,
            "reference_image": config.reference_image,
            "pack_sheet": config.pack_sheet,
            "export_engine": config.export_engine,
            "export_zip": config.export_zip,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


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
