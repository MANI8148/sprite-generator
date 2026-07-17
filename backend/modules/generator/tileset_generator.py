from typing import Optional, List
from PIL import Image

from .base import BaseGenerator


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
    ) -> List[Image.Image]:
        tileset_prompt = f"tileset, seamless, {prompt}"
        tileset_neg = f"seams, borders, edges, {negative_prompt}"
        return self._gen.generate(
            prompt=tileset_prompt,
            negative_prompt=tileset_neg,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            num_images=num_images,
        )

    def unload(self):
        self._gen.unload()

    def get_defaults(self) -> dict:
        return {"width": 256, "height": 256}
