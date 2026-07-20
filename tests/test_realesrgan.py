"""Tests for Real-ESRGAN upscaling (roadmap: Explicitly Deferred -> Real-ESRGAN upscaling)."""

import os
import tempfile
from unittest.mock import patch, MagicMock

from PIL import Image
import numpy as np
import pytest


class TestRealESRGNUpscalerModule:
    def test_is_available_returns_false_by_default(self):
        from backend.modules.postprocess.realesrgan_upscaler import is_available
        assert is_available() is False

    def test_load_model_returns_false_when_import_fails(self):
        from backend.modules.postprocess.realesrgan_upscaler import load_model
        with patch("backend.modules.postprocess.realesrgan_upscaler._REALESRGAN_AVAILABLE", False):
            result = load_model()
            assert result is False

    def test_upscale_with_realesrgan_fallback_on_unavailable(self):
        from backend.modules.postprocess.realesrgan_upscaler import upscale_with_realesrgan
        img = Image.new("RGBA", (16, 16), (255, 0, 0, 255))
        result = upscale_with_realesrgan(img, scale=2)
        assert result.size == (32, 32)
        assert result.mode == "RGBA"


class TestRealESRGANProcessor:
    def test_realesrgan_upscale_default_factor(self):
        from backend.modules.postprocess.processor import realesrgan_upscale
        img = Image.new("RGBA", (16, 16), (255, 0, 0, 255))
        result = realesrgan_upscale(img)
        assert result.size == (64, 64)
        assert result.mode == "RGBA"

    def test_realesrgan_upscale_factor_2(self):
        from backend.modules.postprocess.processor import realesrgan_upscale
        img = Image.new("RGBA", (32, 32), (0, 255, 0, 255))
        result = realesrgan_upscale(img, factor=2)
        assert result.size == (64, 64)

    def test_realesrgan_upscale_factor_1_returns_original(self):
        from backend.modules.postprocess.processor import realesrgan_upscale
        img = Image.new("RGBA", (16, 16), (0, 0, 255, 255))
        result = realesrgan_upscale(img, factor=1)
        assert result.size == (16, 16)
        assert result is img

    def test_realesrgan_upscale_preserves_alpha(self):
        from backend.modules.postprocess.processor import realesrgan_upscale
        arr = np.zeros((16, 16, 4), dtype=np.uint8)
        arr[:, :, :3] = [128, 64, 32]
        arr[4:12, 4:12, 3] = 255
        img = Image.fromarray(arr, "RGBA")
        result = realesrgan_upscale(img, factor=2)
        result_arr = np.array(result)
        assert result_arr.shape == (32, 32, 4)
        assert result_arr[0, 0, 3] == 0
        assert result_arr[8, 8, 3] > 0

    def test_realesrgan_upscale_with_rgba_input(self):
        from backend.modules.postprocess.processor import realesrgan_upscale
        img = Image.new("RGBA", (16, 16), (255, 128, 64, 200))
        result = realesrgan_upscale(img, factor=3)
        assert result.size == (48, 48)


class TestRealESRGANPipelineConfig:
    def test_pipeline_config_has_realesrgan_field(self):
        from backend.modules.pipeline.orchestrator import PipelineConfig
        config = PipelineConfig()
        assert hasattr(config, "use_realesrgan")
        assert config.use_realesrgan is False

    def test_pipeline_with_realesrgan_config(self, tmp_path):
        from backend.modules.pipeline.orchestrator import AssetPipeline, PipelineConfig
        from backend.modules.prompt_builder.controls import (
            AssetControls, AssetType, View, Animation,
        )

        config = PipelineConfig(upscale=2, use_realesrgan=True)
        pipeline = AssetPipeline(config=config)
        from tests.test_pipeline_smoke import FakeGenerator
        pipeline.set_generator(FakeGenerator(num_images=1, size=32))

        controls = AssetControls(
            asset_type=AssetType.CHARACTER,
            view=View.FRONT,
            animation=Animation.IDLE,
        )

        result = pipeline.run(controls, output_dir=str(tmp_path))
        assert len(result.images) == 1
        assert result.images[0].mode == "RGBA"

    def test_pipeline_with_realesrgan_disabled(self, tmp_path):
        from backend.modules.pipeline.orchestrator import AssetPipeline, PipelineConfig
        from backend.modules.prompt_builder.controls import (
            AssetControls, AssetType, View, Animation,
        )

        config = PipelineConfig(upscale=2, use_realesrgan=False)
        pipeline = AssetPipeline(config=config)
        from tests.test_pipeline_smoke import FakeGenerator
        pipeline.set_generator(FakeGenerator(num_images=1, size=32))

        controls = AssetControls(
            asset_type=AssetType.CHARACTER,
            view=View.FRONT,
            animation=Animation.IDLE,
        )

        result = pipeline.run(controls, output_dir=str(tmp_path))
        assert len(result.images) == 1
        assert result.images[0].mode == "RGBA"


class TestRealESRGANAPI:
    def test_generate_request_has_realesrgan_field(self):
        from backend.api.routes import GenerateRequest
        req = GenerateRequest()
        assert hasattr(req, "use_realesrgan")
        assert req.use_realesrgan is False

    def test_batch_item_has_realesrgan_field(self):
        from backend.api.routes import BatchItem
        item = BatchItem()
        assert hasattr(item, "use_realesrgan")
        assert item.use_realesrgan is False

    def test_generate_with_realesrgan(self, tmp_path):
        import tempfile
        from fastapi.testclient import TestClient
        from backend.main import app
        from backend.api.routes import (
            set_pipeline, set_generator_loaded, set_storage, set_library,
            _batch_jobs,
        )
        from backend.modules.pipeline.orchestrator import AssetPipeline
        from backend.modules.storage.file_storage import FileStorage
        from backend.modules.storage.asset_library import AssetLibrary
        from backend.modules.rate_limiter import RateLimiter, set_rate_limiter, get_rate_limiter
        from backend.modules.tasks.queue import TaskQueue, set_task_queue, get_task_queue
        from tests.test_api import FakeGenerator, poll_job

        set_generator_loaded(False)
        tmp = tempfile.mkdtemp()
        set_storage(FileStorage(base_dir=tmp))
        set_library(AssetLibrary(base_dir=os.path.join(tmp, "lib")))
        set_rate_limiter(RateLimiter(max_requests=100, window_seconds=60))
        set_task_queue(TaskQueue(max_workers=4))
        _batch_jobs.clear()

        pipe = AssetPipeline()
        pipe.set_generator(FakeGenerator(num_images=1))
        set_pipeline(pipe)
        set_generator_loaded(True)

        client = TestClient(app)
        resp = client.post("/generate", json={
            "asset_type": "character",
            "view": "front",
            "upscale": 2,
            "use_realesrgan": True,
        })
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]
        data = poll_job(client, job_id)
        assert data["status"] == "done"
