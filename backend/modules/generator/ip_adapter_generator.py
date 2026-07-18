from typing import Optional, List
from PIL import Image
import torch

from diffusers import StableDiffusionPipeline

from .base import BaseGenerator


class IPAdapterGenerator(BaseGenerator):
    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        ip_adapter_model_id: str = "h94/IP-Adapter",
        ip_adapter_subfolder: str = "models",
        ip_adapter_weight_name: str = "ip-adapter_sd15.safetensors",
        ip_adapter_scale: float = 0.6,
        lora_path: Optional[str] = None,
        device: str = "cuda",
        torch_dtype=torch.float16,
    ):
        self.model_id = model_id
        self.ip_adapter_model_id = ip_adapter_model_id
        self.ip_adapter_subfolder = ip_adapter_subfolder
        self.ip_adapter_weight_name = ip_adapter_weight_name
        self.ip_adapter_scale = ip_adapter_scale
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
        self.pipe.load_ip_adapter(
            self.ip_adapter_model_id,
            subfolder=self.ip_adapter_subfolder,
            weight_name=self.ip_adapter_weight_name,
        )
        self.pipe.set_ip_adapter_scale(self.ip_adapter_scale)
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
        ip_adapter_image: Optional[Image.Image] = None,
    ) -> List[Image.Image]:
        if self.pipe is None:
            self.load()
        generator = None
        if seed >= 0:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images,
            generator=generator,
        )
        if ip_adapter_image is not None:
            kwargs["ip_adapter_image"] = ip_adapter_image
        images = self.pipe(**kwargs).images
        return images

    def unload(self):
        if self.pipe is not None:
            self.pipe.to("cpu")
            torch.cuda.empty_cache()
            self.pipe = None
