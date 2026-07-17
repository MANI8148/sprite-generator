from abc import ABC, abstractmethod
from typing import List, Optional
from PIL import Image


class BaseGenerator(ABC):
    @abstractmethod
    def load(self):
        ...

    @abstractmethod
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
        ...

    @abstractmethod
    def unload(self):
        ...

    def get_defaults(self) -> dict:
        return {}
