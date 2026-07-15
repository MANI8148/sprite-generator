from PIL import Image
from typing import Optional, List, Dict
from dataclasses import dataclass, field
import tempfile
import os
import json
import zipfile

from ..prompt_builder.controls import AssetControls, AssetType, View, Animation
from ..prompt_builder.builder import build_prompt
from ..postprocess.processor import (
    remove_background,
    reduce_palette,
    pixel_cleanup,
    auto_center,
    auto_pad,
    normalize,
    upscale,
    outline_cleanup,
    palette_lock,
)
from ..validation.metrics import assess_all
from ..packing.packer import sprite_sheet, animation_strip, individual_pngs
from ..exporters.exporter import export_animation, zip_package
from ..generator.sd_generator import SDGenerator


@dataclass
class PipelineConfig:
    remove_bg: bool = True
    reduce_palette: bool = True
    max_colors: int = 32
    pixel_cleanup: bool = True
    auto_center: bool = True
    auto_pad: bool = True
    normalize_size: bool = True
    target_size: tuple = (512, 512)
    upscale: int = 1
    outline_cleanup: bool = False
    palette_lock: bool = False
    palette_name: str = "retro_16"
    pack_sheet: bool = True
    export_engine: str = "godot"
    export_zip: bool = True


@dataclass
class PipelineResult:
    images: List[Image.Image]
    metadata: Dict
    validation: List[Dict]
    output_paths: List[str] = field(default_factory=list)
    zip_path: Optional[str] = None


class AssetPipeline:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.generator: Optional[SDGenerator] = None

    def set_generator(self, generator: SDGenerator):
        self.generator = generator

    def run(
        self,
        controls: AssetControls,
        output_dir: Optional[str] = None,
    ) -> PipelineResult:
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="sprite_gen_")
        os.makedirs(output_dir, exist_ok=True)

        # 1. Build prompt
        prompt = build_prompt(controls)
        neg_prompt = "blurry, low quality, distorted, bad anatomy, ugly"

        size_map = {"16x16": 128, "32x32": 256, "64x64": 512, "128x128": 1024}
        gen_size = size_map.get(controls.sprite_size.value, 512)

        # 2. Generate
        if self.generator is None:
            raise RuntimeError("Generator not set. Call set_generator() first.")
        images = self.generator.generate(
            prompt=prompt,
            negative_prompt=neg_prompt,
            width=gen_size,
            height=gen_size,
            seed=controls.seed,
        )

        # 3. Post-process each image
        processed = []
        for img in images:
            p = img
            if self.config.remove_bg:
                p = remove_background(p)
            if self.config.pixel_cleanup:
                p = pixel_cleanup(p)
            if self.config.auto_center:
                p = auto_center(p)
            if self.config.auto_pad:
                p = auto_pad(p)
            if self.config.reduce_palette:
                p = reduce_palette(p, max_colors=self.config.max_colors)
            if self.config.normalize_size:
                p = normalize(p, target_size=self.config.target_size)
            if self.config.upscale > 1:
                p = upscale(p, factor=self.config.upscale)
            if self.config.outline_cleanup:
                p = outline_cleanup(p)
            if self.config.palette_lock:
                p = palette_lock(p, palette_name=self.config.palette_name)
            processed.append(p)

        # 4. Validate
        validation = [assess_all(img, batch=processed) for img in processed]

        # 5. Export
        paths = []
        if self.config.pack_sheet and len(processed) > 1:
            names = [f"{controls.asset_type.value}_{controls.animation.value}_{i}" for i in range(len(processed))]
            if controls.animation != Animation.NONE:
                paths = export_animation(
                    processed, output_dir, names[0], engine=self.config.export_engine
                )
            else:
                from ..exporters.exporter import generic_png
                paths = generic_png(processed, names, output_dir)
        else:
            from ..exporters.exporter import generic_png
            names = [f"{controls.asset_type.value}_{i}" for i in range(len(processed))]
            paths = generic_png(processed, names, output_dir, atlas=False)

        # 6. ZIP
        zip_path = None
        if self.config.export_zip:
            zip_path = os.path.join(output_dir, "sprite_package.zip")
            zip_package(paths, zip_path)

        # 7. Save metadata
        meta = {
            "prompt": prompt,
            "controls": {
                "asset_type": controls.asset_type.value,
                "view": controls.view.value,
                "animation": controls.animation.value,
                "palette": controls.palette.value,
                "palette_name": self.config.palette_name,
                "sprite_size": controls.sprite_size.value,
                "theme": controls.theme,
                "seed": controls.seed,
            },
            "validation": validation,
            "outputs": paths,
        }
        meta_path = os.path.join(output_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        paths.append(meta_path)

        if zip_path:
            with zipfile.ZipFile(zip_path, "a", zipfile.ZIP_DEFLATED) as zf:
                if os.path.isfile(meta_path):
                    zf.write(meta_path, os.path.basename(meta_path))

        return PipelineResult(
            images=processed,
            metadata=meta,
            validation=validation,
            output_paths=paths,
            zip_path=zip_path,
        )
