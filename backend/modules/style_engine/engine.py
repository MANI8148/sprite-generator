from PIL import Image
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import numpy as np

from ..postprocess.processor import palette_lock as _palette_lock
from ..postprocess.palettes import KNOWN_PALETTES, get_palette
from ..validation.metrics import palette_consistency


@dataclass
class StylePreset:
    name: str
    description: str
    palette_name: str
    apply_palette_lock: bool = True


STYLE_PRESETS: Dict[str, StylePreset] = {
    "retro_8bit": StylePreset(
        name="retro_8bit",
        description="8-color retro palette, ideal for classic NES-style sprites",
        palette_name="retro_8",
    ),
    "retro_16bit": StylePreset(
        name="retro_16bit",
        description="16-color retro palette, balanced for most pixel art",
        palette_name="retro_16",
    ),
    "retro_32bit": StylePreset(
        name="retro_32bit",
        description="32-color palette, richer but still constrained",
        palette_name="retro_32",
    ),
    "gameboy": StylePreset(
        name="gameboy",
        description="4-shade Gameboy palette (greenish)",
        palette_name="gameboy",
    ),
    "monochrome": StylePreset(
        name="monochrome",
        description="5-shade grayscale",
        palette_name="monochrome",
    ),
    "snes": StylePreset(
        name="snes",
        description="16-color SNES-inspired palette",
        palette_name="snes",
    ),
}


class StyleEngine:
    def apply_palette_lock(
        self,
        image: Image.Image,
        palette_name: str = "retro_16",
    ) -> Image.Image:
        return _palette_lock(image, palette_name=palette_name)

    def get_available_palettes(self) -> List[str]:
        return list(KNOWN_PALETTES.keys())

    def get_palette_colors(self, palette_name: str) -> List[Tuple[int, int, int]]:
        return get_palette(palette_name)

    def check_consistency(
        self,
        image: Image.Image,
        palette_name: str = "retro_16",
    ) -> float:
        return palette_consistency(image, palette_name=palette_name)

    def get_style_presets(self) -> Dict[str, StylePreset]:
        return dict(STYLE_PRESETS)

    def apply_style_preset(
        self,
        image: Image.Image,
        preset_name: str,
    ) -> Image.Image:
        preset = STYLE_PRESETS.get(preset_name.lower())
        if preset is None:
            preset = STYLE_PRESETS["retro_16bit"]
        if preset.apply_palette_lock:
            return _palette_lock(image, palette_name=preset.palette_name)
        return image

    def extract_palette_from_reference(
        self,
        reference_image: Image.Image,
        num_colors: int = 16,
    ) -> List[Tuple[int, int, int]]:
        rgba = reference_image.convert("RGBA")
        arr = np.array(rgba)
        alpha = arr[:, :, 3]
        opaque = alpha > 128
        if not opaque.any():
            return [(0, 0, 0), (255, 255, 255)]
        small = reference_image.resize((64, 64), Image.NEAREST).convert("RGB")
        quantized = small.quantize(colors=min(num_colors, 256), method=Image.Quantize.MEDIANCUT)
        palette = quantized.getpalette()[:num_colors * 3]
        palette = [tuple(palette[i:i + 3]) for i in range(0, len(palette), 3)]
        if (0, 0, 0) not in palette:
            palette.insert(0, (0, 0, 0))
        return palette[:num_colors]

    def apply_reference_style(
        self,
        images: List[Image.Image],
        reference_image: Image.Image,
        num_colors: int = 16,
    ) -> List[Image.Image]:
        palette = self.extract_palette_from_reference(reference_image, num_colors=num_colors)
        from ..postprocess.palettes import KNOWN_PALETTES
        temp_name = "_reference_derived"
        KNOWN_PALETTES[temp_name] = palette
        result = []
        for img in images:
            locked = _palette_lock(img, palette_name=temp_name)
            result.append(locked)
        del KNOWN_PALETTES[temp_name]
        return result
