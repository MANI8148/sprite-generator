from diffusers import StableDiffusionPipeline
import torch
from typing import Optional, List
from PIL import Image

from .base import BaseGenerator


class SDGenerator(BaseGenerator):
    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        lora_path: Optional[str] = None,
        device: str = "cuda",
        torch_dtype=torch.float16,
    ):
        self.model_id = model_id
        self.lora_path = lora_path
        self.device = device
        self.torch_dtype = torch_dtype
        self.pipe = None

    def load(self):
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            safety_checker=None,
        )
        if self.lora_path:
            self.pipe.load_lora_weights(self.lora_path)
        self.pipe.to(self.device)

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
        if self.pipe is None:
            self.load()
        generator = None
        if seed >= 0:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        images = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images,
            generator=generator,
        ).images
        return images

    def unload(self):
        if self.pipe is not None:
            self.pipe.to("cpu")
            torch.cuda.empty_cache()
            self.pipe = None
