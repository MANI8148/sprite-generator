import numpy as np
from PIL import Image
from unittest.mock import MagicMock, patch, PropertyMock

from backend.modules.generator.ip_adapter_generator import IPAdapterGenerator
from backend.modules.generator.registry import create_generator, list_generators
from backend.modules.style_engine import StyleEngine
from backend.modules.pipeline.orchestrator import AssetPipeline, PipelineConfig
from backend.modules.prompt_builder.controls import AssetControls, AssetType, View


def _make_ref_image(width=64, height=64):
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, :, 0] = 100
    arr[:, :, 1] = 150
    arr[:, :, 2] = 200
    return Image.fromarray(arr)


class FakeIPAdapterPipe:
    def __init__(self):
        self.unet = MagicMock()
        self.tokenizer = MagicMock()
        self.text_encoder = MagicMock()
        self.vae = MagicMock()
        self.scheduler = MagicMock()
        self.safety_checker = None
        self.feature_extractor = MagicMock()
        self.image_encoder = MagicMock()

    def to(self, device):
        return self

    def __call__(self, **kwargs):
        n = kwargs.get("num_images_per_prompt", 1)
        images = []
        for i in range(n):
            arr = np.zeros((kwargs.get("height", 512), kwargs.get("width", 512), 3), dtype=np.uint8)
            arr[:, :, 0] = 50 + i * 50
            arr[:, :, 1] = 100 + i * 30
            arr[:, :, 2] = 150 + i * 20
            images.append(Image.fromarray(arr, "RGB"))
        return MagicMock(images=images)


class FakeIPAdapterGen:
    def __init__(self, num_images=1):
        self.num_images = num_images
        self.pipe = FakeIPAdapterPipe()

    def generate(self, prompt="", negative_prompt="", width=512, height=512,
                 seed=-1, num_images=None, ip_adapter_image=None):
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

    def load(self):
        self.pipe = FakeIPAdapterPipe()

    def unload(self):
        self.pipe = None


class TestIPAdapterGenerator:
    def test_creation_default(self):
        gen = IPAdapterGenerator()
        assert gen is not None
        assert gen.model_id == "runwayml/stable-diffusion-v1-5"
        assert gen.ip_adapter_model_id == "h94/IP-Adapter"
        assert gen.ip_adapter_scale == 0.6

    def test_creation_custom(self):
        gen = IPAdapterGenerator(
            model_id="custom/model",
            ip_adapter_model_id="custom/ip-adapter",
            ip_adapter_scale=0.8,
            lora_path="path/to/lora",
        )
        assert gen.model_id == "custom/model"
        assert gen.ip_adapter_model_id == "custom/ip-adapter"
        assert gen.ip_adapter_scale == 0.8
        assert gen.lora_path == "path/to/lora"

    def test_registered(self):
        generators = list_generators()
        assert "ip_adapter" in generators
        gen = create_generator("ip_adapter")
        assert isinstance(gen, IPAdapterGenerator)

    def test_generate_with_reference_image(self):
        gen = FakeIPAdapterGen(num_images=2)
        ref = _make_ref_image()
        images = gen.generate(
            prompt="test sprite",
            ip_adapter_image=ref,
            num_images=2,
        )
        assert len(images) == 2
        assert all(img.mode == "RGBA" for img in images)

    def test_generate_without_reference_image(self):
        gen = FakeIPAdapterGen(num_images=1)
        images = gen.generate(prompt="test sprite")
        assert len(images) == 1


class TestStyleEngineIPAdapter:
    def test_generate_with_ip_adapter_uses_fake_gen(self):
        engine = StyleEngine()
        ref = _make_ref_image()
        fake_gen = FakeIPAdapterGen(num_images=3)
        images = engine.generate_with_ip_adapter(
            prompt="test sprite",
            reference_image=ref,
            generator=fake_gen,
            num_images=3,
        )
        assert len(images) == 3
        assert all(img.mode == "RGBA" for img in images)

    def test_generate_with_ip_adapter_single_image(self):
        engine = StyleEngine()
        ref = _make_ref_image()
        fake_gen = FakeIPAdapterGen(num_images=1)
        images = engine.generate_with_ip_adapter(
            prompt="test sprite",
            reference_image=ref,
            generator=fake_gen,
            num_images=1,
        )
        assert len(images) == 1

    def test_generate_ip_adapter_consistent_batch(self):
        engine = StyleEngine()
        ref = _make_ref_image()
        fake_gen = FakeIPAdapterGen(num_images=1)
        prompts = ["sprite 1", "sprite 2", "sprite 3"]
        results = engine.generate_ip_adapter_consistent_batch(
            prompts=prompts,
            reference_image=ref,
            generator=fake_gen,
        )
        assert len(results) == 3
        for i, images in enumerate(results):
            assert len(images) == 1
            assert images[0].mode == "RGBA"

    def test_generate_ip_adapter_consistent_batch_single_prompt(self):
        engine = StyleEngine()
        ref = _make_ref_image()
        fake_gen = FakeIPAdapterGen(num_images=1)
        results = engine.generate_ip_adapter_consistent_batch(
            prompts=["only one"],
            reference_image=ref,
            generator=fake_gen,
        )
        assert len(results) == 1
        assert len(results[0]) == 1

    def test_generate_with_ip_adapter_uses_pipe_if_available(self):
        engine = StyleEngine()
        ref = _make_ref_image()
        fake_gen = FakeIPAdapterGen(num_images=2)
        fake_gen.pipe = FakeIPAdapterPipe()
        images = engine.generate_with_ip_adapter(
            prompt="test sprite",
            reference_image=ref,
            generator=fake_gen,
            num_images=2,
        )
        assert len(images) == 2


class TestPipelineIPAdapter:
    def test_pipeline_config_has_ip_adapter_fields(self):
        config = PipelineConfig()
        assert hasattr(config, "ip_adapter")
        assert hasattr(config, "ip_adapter_scale")
        assert hasattr(config, "reference_image")

    def test_pipeline_config_ip_adapter_defaults(self):
        config = PipelineConfig()
        assert config.ip_adapter is False
        assert config.ip_adapter_scale == 0.6
        assert config.reference_image is None

    def test_pipeline_config_ip_adapter_custom(self):
        config = PipelineConfig(
            ip_adapter=True,
            ip_adapter_scale=0.8,
            reference_image="/path/to/ref.png",
        )
        assert config.ip_adapter is True
        assert config.ip_adapter_scale == 0.8
        assert config.reference_image == "/path/to/ref.png"

    def test_pipeline_metadata_includes_ip_adapter(self, tmp_path):
        from backend.modules.pipeline.orchestrator import AssetPipeline, PipelineConfig

        class FakeGenerator:
            def generate(self, prompt="", negative_prompt="", width=512, height=512,
                         seed=-1, ip_adapter_image=None):
                arr = np.zeros((64, 64, 4), dtype=np.uint8)
                arr[8:56, 8:56, :3] = [100, 150, 200]
                arr[8:56, 8:56, 3] = 255
                return [Image.fromarray(arr)]

        config = PipelineConfig(ip_adapter=True, ip_adapter_scale=0.7)
        pipeline = AssetPipeline(config=config)
        pipeline.set_generator(FakeGenerator())
        controls = AssetControls(asset_type=AssetType.CHARACTER, view=View.FRONT)
        result = pipeline.run(controls, output_dir=str(tmp_path))
        assert result.metadata["controls"]["ip_adapter"] is True

    def test_pipeline_generate_with_ip_adapter_image(self, tmp_path):
        class FakeGenerator:
            def __init__(self):
                self.last_kwargs = None

            def generate(self, **kwargs):
                self.last_kwargs = kwargs
                arr = np.zeros((64, 64, 4), dtype=np.uint8)
                arr[8:56, 8:56, :3] = [100, 150, 200]
                arr[8:56, 8:56, 3] = 255
                return [Image.fromarray(arr)]

        gen = FakeGenerator()
        ref_path = str(tmp_path / "ref.png")
        _make_ref_image().save(ref_path)

        config = PipelineConfig(ip_adapter=True, reference_image=ref_path)
        pipeline = AssetPipeline(config=config)
        pipeline.set_generator(gen)
        controls = AssetControls(asset_type=AssetType.CHARACTER, view=View.FRONT)
        result = pipeline.run(controls, output_dir=str(tmp_path))
        assert "ip_adapter_image" in gen.last_kwargs
        assert gen.last_kwargs["ip_adapter_image"] is not None

    def test_pipeline_generate_without_ip_adapter(self, tmp_path):
        class FakeGenerator:
            def __init__(self):
                self.last_kwargs = None

            def generate(self, **kwargs):
                self.last_kwargs = kwargs
                arr = np.zeros((64, 64, 4), dtype=np.uint8)
                arr[8:56, 8:56, :3] = [100, 150, 200]
                arr[8:56, 8:56, 3] = 255
                return [Image.fromarray(arr)]

        gen = FakeGenerator()
        config = PipelineConfig(ip_adapter=False)
        pipeline = AssetPipeline(config=config)
        pipeline.set_generator(gen)
        controls = AssetControls(asset_type=AssetType.CHARACTER, view=View.FRONT)
        result = pipeline.run(controls, output_dir=str(tmp_path))
        assert "ip_adapter_image" not in gen.last_kwargs
