"""Tests for rembg background removal.

Implements the detailed roadmap Phase 1 item:
"Add background removal (rembg) so sprites have proper transparency".

Covers the real rembg code path (previously only the ImportError fallback
was tested), robustness against non-RGBA rembg output, and end-to-end
config passthrough through the pipeline config and the /generate API.

rembg requires onnxruntime at import time, so we inject a fake ``rembg``
module into ``sys.modules`` (same technique as test_postprocess_processor.py)
to exercise the real rembg branch of ``remove_background`` without pulling
in the onnxruntime dependency.
"""

import importlib
import os
import sys
import tempfile
import types
from typing import List, Optional

import numpy as np
import pytest
from PIL import Image


def _opaque_image(size=(32, 32)):
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    for y in range(size[1] // 4, 3 * size[1] // 4):
        for x in range(size[0] // 4, 3 * size[0] // 4):
            img.putpixel((x, y), (255, 0, 0, 255))
    return img


def _corner_rembg(image: Image.Image, **kwargs) -> Image.Image:
    """Stand-in for rembg: drops the corner-color background."""
    arr = np.array(image.convert("RGBA"))
    bg = arr[0, 0, :3].copy()
    mask = np.all(arr[:, :, :3] == bg, axis=2)
    arr[..., 3] = np.where(mask, 0, 255)
    return Image.fromarray(arr)


@pytest.fixture
def fake_rembg():
    """Inject a controllable fake ``rembg`` module into sys.modules."""
    saved = sys.modules.get("rembg")
    module = types.ModuleType("rembg")
    holder = {"remove": _corner_rembg, "calls": []}

    def remove(image, **kwargs):
        holder["calls"].append(kwargs)
        return holder["remove"](image, **kwargs)

    module.remove = remove
    sys.modules["rembg"] = module
    for m in list(sys.modules):
        if m.startswith("rembg."):
            del sys.modules[m]
    importlib.invalidate_caches()
    yield holder
    if saved is not None:
        sys.modules["rembg"] = saved
    else:
        sys.modules.pop("rembg", None)
    importlib.invalidate_caches()


class FakeGenerator:
    def __init__(self, num_images: int = 1, size: int = 64):
        self.num_images = num_images
        self.size = size

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        seed: int = -1,
        num_images: Optional[int] = None,
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


# ---------------------------------------------------------------------------
# remove_background — real rembg code path
# ---------------------------------------------------------------------------

class TestRemoveBackgroundRembgPath:
    def test_rembg_output_is_binarized_to_transparency(self, fake_rembg):
        from backend.modules.postprocess.processor import remove_background

        soft = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
        soft.putpixel((8, 8), (120, 120, 120, 200))
        fake_rembg["remove"] = lambda image, **kw: soft
        result = remove_background(_opaque_image(), model="u2net")

        assert result.mode == "RGBA"
        arr = np.array(result)
        assert arr[8, 8, 3] == 255
        assert arr[0, 0, 3] == 0
        assert fake_rembg["calls"]
        assert fake_rembg["calls"][0].get("model_name") == "u2net"

    def test_rembg_model_name_is_forwarded(self, fake_rembg):
        from backend.modules.postprocess.processor import remove_background

        remove_background(_opaque_image(), model="isnet-general-use")

        assert fake_rembg["calls"]
        assert fake_rembg["calls"][0].get("model_name") == "isnet-general-use"

    def test_rembg_rgb_output_is_converted_to_rgba(self, fake_rembg):
        from backend.modules.postprocess.processor import remove_background

        fake_rembg["remove"] = lambda image, **kw: Image.new("RGB", (16, 16), (200, 200, 200))
        result = remove_background(_opaque_image())

        assert result.mode == "RGBA"
        assert np.array(result).shape[2] == 4

    def test_alpha_threshold_is_applied(self, fake_rembg):
        from backend.modules.postprocess.processor import remove_background

        low_alpha = Image.new("RGBA", (8, 8), (0, 0, 0, 100))
        fake_rembg["remove"] = lambda image, **kw: low_alpha
        kept = remove_background(_opaque_image(), alpha_threshold=50)
        assert np.array(kept)[0, 0, 3] == 255

        dropped = remove_background(_opaque_image(), alpha_threshold=200)
        assert np.array(dropped)[0, 0, 3] == 0

    def test_import_error_fallback_still_works(self):
        from backend.modules.postprocess.processor import remove_background

        green_bg = Image.new("RGBA", (16, 16), (0, 255, 0, 255))
        green_bg.putpixel((8, 8), (255, 0, 0, 255))
        saved = sys.modules.get("rembg")
        sys.modules["rembg"] = None
        try:
            for m in list(sys.modules):
                if m == "rembg" or m.startswith("rembg."):
                    sys.modules.pop(m, None)
            importlib.invalidate_caches()
            result = remove_background(green_bg)
            arr = np.array(result)
            assert arr[8, 8, 3] > 128
            assert arr[0, 0, 3] == 0
        finally:
            if saved is not None:
                sys.modules["rembg"] = saved
            importlib.invalidate_caches()


# ---------------------------------------------------------------------------
# Pipeline config passthrough
# ---------------------------------------------------------------------------

class TestPipelineConfigPassthrough:
    def _build_pipeline(self, **overrides):
        from backend.modules.pipeline.orchestrator import AssetPipeline, PipelineConfig
        config = PipelineConfig(**overrides)
        pipeline = AssetPipeline(config=config)
        pipeline.set_generator(FakeGenerator(num_images=1))
        return pipeline

    def _controls(self):
        from backend.modules.prompt_builder.controls import (
            AssetControls, AssetType, View, Animation, Palette, SpriteSize,
        )
        return AssetControls(
            asset_type=AssetType.CHARACTER,
            view=View.FRONT,
            animation=Animation.IDLE,
            palette=Palette.AUTO,
            sprite_size=SpriteSize.S_32,
        )

    def test_orchestrator_forwards_rembg_settings(self, tmp_path, fake_rembg):
        pipeline = self._build_pipeline(
            remove_bg_model="isnet-general-use",
            remove_bg_alpha_threshold=64,
        )
        result = pipeline.run(self._controls(), output_dir=str(tmp_path))

        assert fake_rembg["calls"]
        assert all(
            c.get("model_name") == "isnet-general-use"
            for c in fake_rembg["calls"]
        )
        assert len(result.images) == 1
        assert result.images[0].mode == "RGBA"

    def test_orchestrator_skips_rembg_when_disabled(self, tmp_path, fake_rembg):
        pipeline = self._build_pipeline(remove_bg=False)
        result = pipeline.run(self._controls(), output_dir=str(tmp_path))

        assert fake_rembg["calls"] == []
        assert len(result.images) == 1
        assert result.images[0].mode == "RGBA"

    def test_default_config_matches_request_defaults(self):
        from backend.modules.pipeline.orchestrator import PipelineConfig
        config = PipelineConfig()
        assert config.remove_bg is True
        assert config.remove_bg_model == "u2net"
        assert config.remove_bg_alpha_threshold == 128

    def test_generation_hash_includes_rembg_settings(self):
        from backend.modules.asset_memory.memory import compute_generation_hash
        from backend.modules.pipeline.orchestrator import PipelineConfig
        from backend.modules.prompt_builder.controls import AssetControls

        controls = AssetControls()
        base = compute_generation_hash(controls, PipelineConfig())
        different_model = compute_generation_hash(
            controls, PipelineConfig(remove_bg_model="isnet-general-use")
        )
        different_threshold = compute_generation_hash(
            controls, PipelineConfig(remove_bg_alpha_threshold=64)
        )
        assert base != different_model
        assert base != different_threshold


# ---------------------------------------------------------------------------
# API /generate — new fields accepted and honored
# ---------------------------------------------------------------------------

class TestApiBackgroundRemoval:
    @pytest.fixture(autouse=True)
    def reset_state(self):
        from backend.api.routes import (
            set_pipeline, set_generator_loaded, set_storage, set_library,
            _batch_jobs, set_job_store,
        )
        from backend.modules.storage.file_storage import FileStorage
        from backend.modules.storage.asset_library import AssetLibrary
        from backend.modules.rate_limiter import RateLimiter, set_rate_limiter
        from backend.modules.tasks.queue import TaskQueue, set_task_queue

        set_generator_loaded(False)
        tmp = tempfile.mkdtemp()
        set_storage(FileStorage(base_dir=tmp))
        set_library(AssetLibrary(base_dir=os.path.join(tmp, "lib")))
        set_rate_limiter(RateLimiter(max_requests=100, window_seconds=60))
        set_task_queue(TaskQueue(max_workers=4))
        set_job_store(None)
        _batch_jobs.clear()
        yield

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from backend.api.routes import set_pipeline, set_generator_loaded
        from backend.modules.pipeline.orchestrator import AssetPipeline
        from backend.main import app

        pipe = AssetPipeline()
        pipe.set_generator(FakeGenerator(num_images=1))
        set_pipeline(pipe)
        set_generator_loaded(True)
        return TestClient(app)

    def _poll(self, client, job_id, timeout=10):
        import time
        deadline = time.time() + timeout
        while time.time() < deadline:
            resp = client.get(f"/status/{job_id}")
            assert resp.status_code == 200
            data = resp.json()
            if data["status"] in ("done", "failed"):
                return data
            time.sleep(0.01)
        raise TimeoutError(f"Job {job_id} did not complete")

    def test_generate_honors_rembg_settings(self, client, fake_rembg):
        resp = client.post("/generate", json={
            "asset_type": "character",
            "view": "front",
            "animation": "idle",
            "remove_bg": True,
            "remove_bg_model": "isnet-general-use",
            "remove_bg_alpha_threshold": 64,
        })
        assert resp.status_code == 202, resp.text
        job_id = resp.json()["job_id"]

        status = self._poll(client, job_id)
        assert status["status"] == "done", status.get("error")

        assert fake_rembg["calls"]
        assert all(
            c.get("model_name") == "isnet-general-use"
            for c in fake_rembg["calls"]
        )

        png_paths = [p for p in status["output_paths"] if p.endswith(".png")]
        assert png_paths, "expected a generated PNG in output_paths"
        img = Image.open(png_paths[0])
        assert img.mode == "RGBA"
        arr = np.array(img)
        assert (arr[..., 3] == 0).any(), "expected transparent pixels"
        assert (arr[..., 3] == 255).any(), "expected opaque pixels"

    def test_generate_accepts_defaults(self, client, fake_rembg):
        resp = client.post("/generate", json={
            "asset_type": "character",
            "view": "front",
        })
        assert resp.status_code == 202, resp.text
        status = self._poll(client, resp.json()["job_id"])
        assert status["status"] == "done", status.get("error")
        assert fake_rembg["calls"]
