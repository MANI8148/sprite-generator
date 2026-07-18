from PIL import Image
from typing import List, Tuple, Optional, Dict, Set
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


def _get_opaque_colors(image: Image.Image) -> np.ndarray:
    rgba = image.convert("RGBA")
    arr = np.array(rgba)
    alpha = arr[:, :, 3]
    opaque = alpha > 128
    if not opaque.any():
        return np.zeros((0, 3), dtype=np.uint8)
    return arr[opaque][:, :3]


def _extract_used_colors(image: Image.Image) -> Set[Tuple[int, int, int]]:
    colors = _get_opaque_colors(image)
    return set(tuple(c) for c in colors)


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

    def color_statistics(
        self,
        image: Image.Image,
    ) -> Dict[str, float]:
        colors = _get_opaque_colors(image)
        if len(colors) == 0:
            return {"mean_r": 0.0, "mean_g": 0.0, "mean_b": 0.0,
                    "std_r": 0.0, "std_g": 0.0, "std_b": 0.0,
                    "num_colors": 0, "colorfulness": 0.0}
        mean = colors.mean(axis=0)
        std = colors.std(axis=0)
        unique = len(set(tuple(c) for c in colors))
        rg = colors[:, 0].astype(np.float32) - colors[:, 1].astype(np.float32)
        yb = 0.5 * (colors[:, 0].astype(np.float32) + colors[:, 1].astype(np.float32)) - colors[:, 2].astype(np.float32)
        colorfulness = float(np.sqrt(rg.var() + yb.var()))
        return {
            "mean_r": float(mean[0]), "mean_g": float(mean[1]), "mean_b": float(mean[2]),
            "std_r": float(std[0]), "std_g": float(std[1]), "std_b": float(std[2]),
            "num_colors": unique, "colorfulness": round(colorfulness, 1),
        }

    def cross_image_palette_agreement(
        self,
        images: List[Image.Image],
    ) -> float:
        if len(images) < 2:
            return 1.0
        palettes = [_extract_used_colors(img) for img in images]
        total_pairs = 0
        total_intersection = 0
        for i in range(len(palettes)):
            for j in range(i + 1, len(palettes)):
                if not palettes[i] or not palettes[j]:
                    continue
                intersection = palettes[i] & palettes[j]
                union = palettes[i] | palettes[j]
                total_intersection += len(intersection)
                total_pairs += 1
        if total_pairs == 0:
            return 1.0
        return total_intersection / (len(images) * total_pairs) if (len(images) * total_pairs) > 0 else 1.0

    def compute_style_similarity(
        self,
        images: List[Image.Image],
    ) -> float:
        if len(images) < 2:
            return 1.0
        stats = [self.color_statistics(img) for img in images]
        scores = []
        for i in range(len(stats)):
            for j in range(i + 1, len(stats)):
                m1 = np.array([stats[i]["mean_r"], stats[i]["mean_g"], stats[i]["mean_b"]])
                m2 = np.array([stats[j]["mean_r"], stats[j]["mean_g"], stats[j]["mean_b"]])
                s1 = np.array([stats[i]["std_r"], stats[i]["std_g"], stats[i]["std_b"]])
                s2 = np.array([stats[j]["std_r"], stats[j]["std_g"], stats[j]["std_b"]])
                mean_diff = float(np.linalg.norm(m1 - m2)) / 442.0
                std_diff = float(np.linalg.norm(s1 - s2)) / 442.0
                sim = 1.0 - 0.5 * mean_diff - 0.5 * std_diff
                scores.append(max(0.0, min(1.0, sim)))
        return float(np.mean(scores)) if scores else 1.0

    def batch_consistency_score(
        self,
        images: List[Image.Image],
        palette_name: Optional[str] = None,
    ) -> Dict[str, float]:
        if not images:
            return {"mean_consistency": 1.0, "std_consistency": 0.0,
                    "palette_agreement": 1.0, "style_similarity": 1.0,
                    "overall": 1.0}
        individual = []
        for img in images:
            if palette_name:
                individual.append(self.check_consistency(img, palette_name=palette_name))
            else:
                individual.append(self.check_consistency(img))
        mean_cons = float(np.mean(individual)) if individual else 1.0
        std_cons = float(np.std(individual)) if individual else 0.0
        agreement = self.cross_image_palette_agreement(images)
        style_sim = self.compute_style_similarity(images)
        palette_lock_score = mean_cons
        overall = 0.4 * palette_lock_score + 0.3 * agreement + 0.3 * style_sim
        return {
            "mean_consistency": round(mean_cons, 4),
            "std_consistency": round(std_cons, 4),
            "palette_agreement": round(agreement, 4),
            "style_similarity": round(style_sim, 4),
            "overall": round(overall, 4),
        }

    def extract_batch_palette(
        self,
        images: List[Image.Image],
        num_colors: int = 16,
    ) -> List[Tuple[int, int, int]]:
        if not images:
            return get_palette("retro_16")
        all_colors = []
        for img in images:
            colors = _get_opaque_colors(img)
            if len(colors) > 0:
                all_colors.append(colors)
        if not all_colors:
            return get_palette("retro_16")
        composite = np.vstack(all_colors)
        if len(composite) == 0:
            return get_palette("retro_16")
        h, w = composite.shape
        if h * w == 0:
            return get_palette("retro_16")
        import math
        size = int(math.ceil(math.sqrt(len(composite))))
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        idx = 0
        for i in range(size):
            for j in range(size):
                if idx < len(composite):
                    canvas[i, j] = composite[idx]
                    idx += 1
        img = Image.fromarray(canvas, "RGB")
        quantized = img.quantize(colors=min(num_colors, 256), method=Image.Quantize.MEDIANCUT)
        palette = quantized.getpalette()[:num_colors * 3]
        palette = [tuple(palette[i:i + 3]) for i in range(0, len(palette), 3)]
        return palette[:num_colors]

    def harmonize_batch(
        self,
        images: List[Image.Image],
        palette_name: str = "retro_16",
    ) -> List[Image.Image]:
        return [_palette_lock(img, palette_name=palette_name) for img in images]

    def apply_reference_color_transfer(
        self,
        images: List[Image.Image],
        reference_image: Image.Image,
        strength: float = 0.5,
    ) -> List[Image.Image]:
        ref_stats = self.color_statistics(reference_image)
        result = []
        for img in images:
            rgba = img.convert("RGBA")
            arr = np.array(rgba, dtype=np.float32)
            alpha = arr[:, :, 3]
            opaque_mask = alpha > 128
            if not opaque_mask.any():
                result.append(img)
                continue
            pixels = arr[opaque_mask, :3]
            orig_mean = pixels.mean(axis=0)
            orig_std = pixels.std(axis=0)
            ref_mean = np.array([ref_stats["mean_r"], ref_stats["mean_g"], ref_stats["mean_b"]])
            ref_std = np.array([ref_stats["std_r"], ref_stats["std_g"], ref_stats["std_b"]])
            orig_std = np.where(orig_std < 1.0, 1.0, orig_std)
            normalized = (pixels - orig_mean) / orig_std
            transferred = normalized * ref_std + ref_mean
            blended = (1.0 - strength) * pixels + strength * transferred
            blended = np.clip(blended, 0, 255).astype(np.uint8)
            arr[opaque_mask, :3] = blended
            result.append(Image.fromarray(arr.astype(np.uint8), "RGBA"))
        return result
