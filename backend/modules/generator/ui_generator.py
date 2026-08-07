from typing import List, Optional, Tuple
from PIL import Image

from .base import BaseGenerator


def slice_grid(image: Image.Image, cols: int = 4, rows: int = 1) -> List[Image.Image]:
    """Split an image into a regular grid of equally-sized UI element cells.

    Divides the canvas into ``cols*rows`` tiles. Non-perfect divisions round
    the cell width/height up and center each full-width cell on the source.
    """
    if cols < 1 or rows < 1:
        raise ValueError("cols and rows must be >= 1")
    rgba = image.convert("RGBA")
    w, h = rgba.size
    cell_w = (w + cols - 1) // cols
    cell_h = (h + rows - 1) // rows
    cells = []
    for r in range(rows):
        for c in range(cols):
            x0 = c * cell_w
            y0 = r * cell_h
            x1 = min(x0 + cell_w, w)
            y1 = min(y0 + cell_h, h)
            cells.append(rgba.crop((x0, y0, x1, y1)))
    return cells


class UIGenerator(BaseGenerator):
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
    ) -> List[Image.Image]:
        ui_prompt = f"game UI element, user interface, {prompt}"
        ui_neg = f"character, background, scene, {negative_prompt}"
        return self._gen.generate(
            prompt=ui_prompt,
            negative_prompt=ui_neg,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            num_images=num_images,
        )

    def generate_sheet(
        self,
        prompt: str,
        negative_prompt: str = "",
        cols: int = 4,
        rows: int = 1,
        width: int = 256,
        height: int = 128,
        seed: int = -1,
    ) -> List[Image.Image]:
        """Generate one UI atlas and slice it into ``cols*rows`` icon cells."""
        image = self._gen.generate(
            prompt=f"UI icon sheet, {prompt}",
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=24,
            guidance_scale=7.0,
            seed=seed,
            num_images=1,
        )[0]
        return slice_grid(image, cols=cols, rows=rows)

    def unload(self):
        self._gen.unload()

    def get_defaults(self) -> dict:
        return {"width": 128, "height": 128}
