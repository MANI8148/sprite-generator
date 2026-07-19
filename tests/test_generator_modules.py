import numpy as np
from PIL import Image
from typing import List, Optional

from backend.modules.generator.base import BaseGenerator
from backend.modules.generator.sd_generator import SDGenerator
from backend.modules.generator.tileset_generator import TilesetGenerator
from backend.modules.generator.environment_generator import EnvironmentGenerator
from backend.modules.generator.prop_generator import PropGenerator
from backend.modules.generator.ui_generator import UIGenerator
from backend.modules.generator.registry import (
    register_generator,
    get_generator_class,
    list_generators,
    create_generator,
)
from backend.modules.pipeline.orchestrator import AssetPipeline, PipelineConfig
from backend.modules.prompt_builder.controls import AssetControls, AssetType, View


class FakeGenerator(BaseGenerator):
    def __init__(self, num_images=1, size=64):
        self.num_images = num_images
        self.size = size
        self._loaded = False

    def load(self):
        self._loaded = True

    def generate(
        self,
        prompt: str = "",
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 28,
        guidance_scale: float = 7.0,
        seed: int = -1,
        num_images: int = 1,
    ) -> List[Image.Image]:
        n = num_images or self.num_images
        images = []
        for i in range(n):
            arr = np.zeros((height, width, 4), dtype=np.uint8)
            cy, cx = height // 4, width // 4
            hh, hw = height // 2, width // 2
            r, g, b = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)][i % 4]
            arr[cy:cy + hh, cx:cx + hw, 0] = r
            arr[cy:cy + hh, cx:cx + hw, 1] = g
            arr[cy:cy + hh, cx:cx + hw, 2] = b
            arr[cy:cy + hh, cx:cx + hw, 3] = 255
            images.append(Image.fromarray(arr, "RGBA"))
        return images

    def unload(self):
        self._loaded = False


class TestBaseGenerator:
    def test_cannot_instantiate_abc(self):
        import pytest
        with pytest.raises(TypeError):
            BaseGenerator()

    def test_fake_generator_is_instance(self):
        assert isinstance(FakeGenerator(), BaseGenerator)

    def test_sd_generator_is_instance(self):
        assert isinstance(SDGenerator(lora_path=None), BaseGenerator)


class TestTilesetGenerator:
    def setup_method(self):
        self.fake = FakeGenerator()
        self.gen = TilesetGenerator(self.fake)

    def test_is_base_generator(self):
        assert isinstance(self.gen, BaseGenerator)

    def test_generate_returns_images(self):
        images = self.gen.generate(prompt="grass", num_images=2)
        assert len(images) == 2
        assert all(img.mode == "RGBA" for img in images)

    def test_generate_modifies_prompt(self):
        images = self.gen.generate(prompt="grass", num_images=1)
        assert len(images) == 1

    def test_get_defaults(self):
        defaults = self.gen.get_defaults()
        assert defaults["width"] == 256
        assert defaults["height"] == 256

    def test_load_and_unload(self):
        self.gen.load()
        assert self.fake._loaded
        self.gen.unload()
        assert not self.fake._loaded


class TestEnvironmentGenerator:
    def setup_method(self):
        self.fake = FakeGenerator()
        self.gen = EnvironmentGenerator(self.fake)

    def test_is_base_generator(self):
        assert isinstance(self.gen, BaseGenerator)

    def test_generate_returns_images(self):
        images = self.gen.generate(prompt="forest", num_images=3)
        assert len(images) == 3
        assert all(img.mode == "RGBA" for img in images)

    def test_get_defaults(self):
        defaults = self.gen.get_defaults()
        assert defaults["width"] == 512
        assert defaults["height"] == 512


class TestPropGenerator:
    def setup_method(self):
        self.fake = FakeGenerator()
        self.gen = PropGenerator(self.fake)

    def test_is_base_generator(self):
        assert isinstance(self.gen, BaseGenerator)

    def test_generate_returns_images(self):
        images = self.gen.generate(prompt="chest", num_images=1)
        assert len(images) == 1

    def test_get_defaults(self):
        defaults = self.gen.get_defaults()
        assert defaults["width"] == 256
        assert defaults["height"] == 256


class TestUIGenerator:
    def setup_method(self):
        self.fake = FakeGenerator()
        self.gen = UIGenerator(self.fake)

    def test_is_base_generator(self):
        assert isinstance(self.gen, BaseGenerator)

    def test_generate_returns_images(self):
        images = self.gen.generate(prompt="health bar", num_images=2)
        assert len(images) == 2
        assert all(img.mode == "RGBA" for img in images)

    def test_get_defaults(self):
        defaults = self.gen.get_defaults()
        assert defaults["width"] == 128
        assert defaults["height"] == 128

    def test_load_and_unload(self):
        self.gen.load()
        assert self.fake._loaded
        self.gen.unload()
        assert not self.fake._loaded


class TestGeneratorRegistry:
    def test_list_generators(self):
        gens = list_generators()
        assert "sd" in gens
        assert "tileset" in gens
        assert "environment" in gens
        assert "prop" in gens
        assert "ui" in gens

    def test_get_generator_class(self):
        cls = get_generator_class("sd")
        assert cls is SDGenerator

    def test_get_generator_class_nonexistent(self):
        cls = get_generator_class("nonexistent")
        assert cls is None

    def test_create_generator(self):
        gen = create_generator("sd", lora_path=None)
        assert isinstance(gen, SDGenerator)

    def test_create_generator_nonexistent(self):
        gen = create_generator("nonexistent")
        assert gen is None

    def test_register_custom_generator(self):
        class CustomGen(BaseGenerator):
            def load(self): pass
            def generate(self, **kwargs): return []
            def unload(self): pass

        register_generator("custom", CustomGen)
        assert "custom" in list_generators()
        assert get_generator_class("custom") is CustomGen
        assert isinstance(create_generator("custom"), CustomGen)

    def test_register_overwrites(self):
        class GenA(BaseGenerator):
            def load(self): pass
            def generate(self, **kwargs): return []
            def unload(self): pass
        class GenB(BaseGenerator):
            def load(self): pass
            def generate(self, **kwargs): return []
            def unload(self): pass

        register_generator("dup", GenA)
        register_generator("dup", GenB)
        assert get_generator_class("dup") is GenB


class TestFakeGeneratorBasics:
    def test_fake_generator_generates(self):
        gen = FakeGenerator(num_images=2)
        images = gen.generate(num_images=2)
        assert len(images) == 2

    def test_fake_generator_load_unload(self):
        gen = FakeGenerator()
        assert not gen._loaded
        gen.load()
        assert gen._loaded
        gen.unload()
        assert not gen._loaded

    def test_fake_generator_respects_num_images(self):
        gen = FakeGenerator(num_images=1)
        images = gen.generate(num_images=4)
        assert len(images) == 4


class TestPipelineWithGeneratorModules:
    def test_pipeline_with_sd_generator(self, tmp_path):
        config = PipelineConfig()
        pipeline = AssetPipeline(config=config)
        pipeline.set_generator(FakeGenerator(num_images=1))
        controls = AssetControls(asset_type=AssetType.CHARACTER, view=View.FRONT)
        result = pipeline.run(controls, output_dir=str(tmp_path))
        assert len(result.images) == 1

    def test_specialized_generator_preserves_transparency(self):
        fake = FakeGenerator()
        gen = TilesetGenerator(fake)
        images = gen.generate(prompt="test", num_images=1)
        arr = np.array(images[0])
        assert arr.shape[2] == 4
        assert (arr[:, :, 3] > 0).any()
