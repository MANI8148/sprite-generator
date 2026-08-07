from typing import List, Optional, Tuple
import numpy as np
from PIL import Image

from .base import BaseGenerator


def isolate(image: Image.Image, size: Optional[int] = None, padding: int = 8) -> Image.Image:
    """Center the opaque content of an image on a transparent square canvas.

    Trims to the alpha bounding box, then places the trimmed sprite dead-center
    on a transparent square of ``size`` (defaults to the largest sprite side plus
    padding). Outside the sprite the alpha is fully transparent.
    """
    rgba = image.convert("RGBA")
    arr = np.asarray(rgba)
    alpha = arr[:, :, 3]
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)
    if not rows.any() or not cols.any():
        return rgba
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    sprite = rgba.crop((x0, y0, x1 + 1, y1 + 1))
    if size is None:
        size = max(sprite.size) + padding * 2
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ox = (size - sprite.size[0]) // 2
    oy = (size - sprite.size[1]) // 2
    canvas.paste(sprite, (ox, oy), sprite)
    return canvas


class PropGenerator(BaseGenerator):
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
        isolated: bool = True,
        padding: int = 8,
    ) -> List[Image.Image]:
        prop_prompt = f"game prop item, isolated on transparent background, {prompt}"
        images = self._gen.generate(
            prompt=prop_prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            num_images=num_images,
        )
        if isolated:
            images = [isolate(img, padding=padding) for img in images]
        return images

    def isolate(self, image: Image.Image, size: Optional[int] = None, padding: int = 8) -> Image.Image:
        return isolate(image, size=size, padding=padding)

    def unload(self):
        self._gen.unload()

    def get_defaults(self) -> dict:
        return {"width": 256, "height": 256}
