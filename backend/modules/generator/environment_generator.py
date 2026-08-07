from typing import List
from PIL import Image

from .base import BaseGenerator


class EnvironmentGenerator(BaseGenerator):
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
        env_prompt = f"game background environment, {prompt}"
        env_neg = f"character, sprite, ui, text, {negative_prompt}"
        return self._gen.generate(
            prompt=env_prompt,
            negative_prompt=env_neg,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            num_images=num_images,
        )

    def generate_layer(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        depth: float = 0.5,
        seed: int = -1,
    ) -> Image.Image:
        """Generate a single parallax layer at a given view depth.

        ``depth`` in [0, 1]: 0 is the far (sky/back) layer, 1 is the near
        foreground. The depth is baked in as a per-layer offset so the returned
        height is reduced for distant layers, mimicking parallax scale.
        """
        depth = max(0.0, min(1.0, depth))
        depth_prompt = f"{depth:.2f} parallax depth, {prompt}"
        layer_height = max(1, int(height * (0.5 + 0.5 * depth)))
        images = self._gen.generate(
            prompt=depth_prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=layer_height,
            num_inference_steps=24,
            guidance_scale=7.0,
            seed=seed,
            num_images=1,
        )
        return images[0]

    def generate_panorama(
        self,
        prompt: str,
        negative_prompt: str = "",
        depth_start: float = 0.2,
        depth_end: float = 0.9,
        layers: int = 3,
        width: int = 512,
        height: int = 512,
        seed: int = -1,
    ) -> Image.Image:
        """Generate a layered parallax background as a single composite image.

        Renders ``layers`` depth slices and vertically stacks them (nearest
        layer drawn on top of the horizon), producing one wide scene ready for
        side-scrolling parallax.
        """
        depths = []
        if layers == 1:
            depths = [depth_start]
        else:
            step = (depth_end - depth_start) / (layers - 1)
            depths = [depth_start + step * i for i in range(layers)]
        layers_list = [
            self.generate_layer(
                prompt, negative_prompt=negative_prompt,
                width=width, height=height,
                depth=d, seed=seed,
            )
            for d in reversed(depths)
        ]
        total_h = sum(img.size[1] for img in layers_list)
        canvas = Image.new("RGBA", (width, total_h), (0, 0, 0, 0))
        y = 0
        for img in layers_list:
            canvas.paste(img, (0, y), img)
            y += img.size[1]
        return canvas

    def unload(self):
        self._gen.unload()

    def get_defaults(self) -> dict:
        return {"width": 512, "height": 512}
