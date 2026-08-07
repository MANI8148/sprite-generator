from typing import List
import numpy as np
from PIL import Image

from .base import BaseGenerator


def _make_seamless(image: Image.Image, band: int = 12) -> Image.Image:
    """Make an image tile without visible seams.

    Standard technique: cross-fade the left/right and top/bottom edge bands so
    the left column matches the right column and the top row matches the bottom
    row. Any single tile can then be repeated in a grid with no edge break.
    """
    img = image.convert("RGBA")
    arr = np.asarray(img).astype(np.float32)
    h, w = arr.shape[:2]
    band = max(1, min(band, w // 2, h // 2))
    out = arr.copy()

    ramp = np.linspace(0.0, 1.0, band, dtype=np.float32)

    # Horizontal seam: blend the left band into the right band and vice versa
    left = out[:, :band].copy()
    right = out[:, w - band:].copy()
    # force right edge == left edge content so the horizontal wrap is continuous
    out[:, :band] = left * (1 - ramp)[None, :, None] + right * ramp[None, :, None]
    out[:, w - band:] = out[:, :band][:, ::-1]

    # Vertical seam: blend the top band into the bottom band
    top = out[:band, :].copy()
    bottom = out[h - band:, :].copy()
    out[:band] = top * (1 - ramp)[:, None, None] + bottom * ramp[:, None, None]
    out[h - band:] = out[:band][::-1]

    return Image.fromarray(out.astype(np.uint8), "RGBA")


class TilesetGenerator(BaseGenerator):
    def __init__(self, base_generator: BaseGenerator):
        self._gen = base_generator

    def load(self):
        self._gen.load()

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 28,
        guidance_scale: float = 7.0,
        seed: int = -1,
        num_images: int = 1,
        seamless: bool = True,
        seamless_band: int = 12,
    ) -> List[Image.Image]:
        tileset_prompt = f"tileset, seamless, {prompt}"
        tileset_neg = f"seams, borders, edges, {negative_prompt}"
        images = self._gen.generate(
            prompt=tileset_prompt,
            negative_prompt=tileset_neg,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            num_images=num_images,
        )
        if seamless:
            images = [_make_seamless(img, seamless_band) for img in images]
        return images

    def make_seamless(self, image: Image.Image, band: int = 12) -> Image.Image:
        return _make_seamless(image, band)

    def unload(self):
        self._gen.unload()

    def get_defaults(self) -> dict:
        return {"width": 256, "height": 256}