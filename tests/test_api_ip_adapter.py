"""IP-Adapter must be reachable through the public API (roadmap: Phase 2
"Style consistency engine" — the pipeline and module already support it, the
request models did not expose it)."""

import os
import tempfile
from unittest.mock import patch

from PIL import Image
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.api.routes import (
    set_pipeline, set_generator_loaded,
    set_storage, set_library, GenerateRequest, BatchItem,
    _batch_jobs, set_job_store,
)
from backend.modules.pipeline.orchestrator import AssetPipeline, PipelineConfig
from backend.modules.storage.file_storage import FileStorage
from backend.modules.storage.asset_library import AssetLibrary
from backend.modules.rate_limiter import RateLimiter, set_rate_limiter
from backend.modules.tasks.queue import TaskQueue, set_task_queue
from backend.main import app


class RecordingGenerator:
    """A fake generator that supports IP-Adapter and records the kwargs it saw."""

    def __init__(self, num_images: int = 1, size: int = 64):
        self.num_images = num_images
        self.size = size
        self.last_kwargs = None
        self.all_kwargs = []

    def supports_ip_adapter(self) -> bool:
        return True

    def generate(self, **kwargs):
        self.last_kwargs = kwargs
        self.all_kwargs.append(kwargs)
        n = kwargs.get("num_images") or self.num_images
        images = []
        for i in range(n):
            arr = np.zeros((self.size, self.size, 4), dtype=np.uint8)
            arr[8:56, 8:56, :3] = [100, 150, 200]
            arr[8:56, 8:56, 3] = 255
            images.append(Image.fromarray(arr, "RGBA"))
        return images


@pytest.fixture(autouse=True)
def reset_state():
    set_generator_loaded(False)
    tmp = tempfile.mkdtemp()
    set_storage(FileStorage(base_dir=tmp))
    set_library(AssetLibrary(base_dir=os.path.join(tmp, "lib")))
    set_rate_limiter(RateLimiter(max_requests=100, window_seconds=60))
    set_task_queue(TaskQueue(max_workers=4))
    set_job_store(None)
    _batch_jobs.clear()


@pytest.fixture
def client():
    pipe = AssetPipeline()
    pipe.set_generator(RecordingGenerator(num_images=1))
    set_pipeline(pipe)
    set_generator_loaded(True)
    return TestClient(app)


@pytest.fixture
def reference_image(tmp_path):
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[:, :, 0] = 80
    arr[:, :, 1] = 120
    arr[:, :, 2] = 200
    path = tmp_path / "reference.png"
    Image.fromarray(arr, "RGB").save(path)
    return str(path)


def poll_job(client, job_id, timeout=10):
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(f"/status/{job_id}")
        assert resp.status_code == 200
        data = resp.json()
        if data["status"] in ("done", "failed"):
            return data
        time.sleep(0.01)
    raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")


class TestGenerateRequestModel:
    def test_ip_adapter_fields_defaults(self):
        req = GenerateRequest()
        assert req.ip_adapter is False
        assert req.ip_adapter_scale == 0.6
        assert req.reference_image is None

    def test_ip_adapter_fields_set(self):
        req = GenerateRequest(ip_adapter=True, ip_adapter_scale=0.8, reference_image="/tmp/ref.png")
        assert req.ip_adapter is True
        assert req.ip_adapter_scale == 0.8
        assert req.reference_image == "/tmp/ref.png"

    def test_ip_adapter_requires_reference_image(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            GenerateRequest(ip_adapter=True)

    def test_batch_item_fields_defaults(self):
        item = BatchItem()
        assert item.ip_adapter is False
        assert item.ip_adapter_scale == 0.6
        assert item.reference_image is None

    def test_batch_item_requires_reference_image(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            BatchItem(ip_adapter=True)


class TestGenerateIPAdapterAPI:
    def test_generate_passes_reference_image_to_generator(self, client, reference_image):
        resp = client.post("/generate", json={
            "asset_type": "character",
            "view": "front",
            "ip_adapter": True,
            "ip_adapter_scale": 0.8,
            "reference_image": reference_image,
        })
        assert resp.status_code == 202
        data = resp.json()
        result = poll_job(client, data["job_id"])
        assert result["status"] == "done"

        gen = get_pipeline_generator()
        assert gen.last_kwargs is not None
        assert "ip_adapter_image" in gen.last_kwargs
        assert gen.last_kwargs["ip_adapter_image"] is not None

    def test_generate_without_ip_adapter_has_no_reference_image(self, client):
        resp = client.post("/generate", json={"asset_type": "character"})
        assert resp.status_code == 202
        data = resp.json()
        result = poll_job(client, data["job_id"])
        assert result["status"] == "done"

        gen = get_pipeline_generator()
        assert "ip_adapter_image" not in gen.last_kwargs

    def test_generate_ip_adapter_without_reference_image_rejected(self, client):
        resp = client.post("/generate", json={
            "asset_type": "character",
            "ip_adapter": True,
        })
        assert resp.status_code == 422

    def test_generate_metadata_records_ip_adapter(self, client, reference_image):
        resp = client.post("/generate", json={
            "asset_type": "character",
            "ip_adapter": True,
            "ip_adapter_scale": 0.7,
            "reference_image": reference_image,
        })
        assert resp.status_code == 202
        data = resp.json()
        poll_job(client, data["job_id"])

        lib = get_library()
        assets = lib.list_assets()
        assert len(assets) == 1
        meta = assets[0].metadata
        assert meta["generation_hash"] != ""
        assert assets[0].quality_tier in ("clean", "acceptable", "noisy", "blurry", "broken_outline", "empty", "extreme_aspect")

    def test_generate_ip_adapter_affects_generation_hash(self, client, reference_image):
        """ip_adapter config must participate in the asset-memory generation
        hash so IP-Adapter runs are not served from a non-IP-Adapter cache."""
        from backend.modules.prompt_builder.controls import AssetControls, AssetType
        from backend.modules.pipeline.orchestrator import PipelineConfig
        from backend.modules.asset_memory import compute_generation_hash

        controls = AssetControls(asset_type=AssetType.CHARACTER)
        base = PipelineConfig()
        with_adapter = PipelineConfig(ip_adapter=True, ip_adapter_scale=0.7, reference_image=reference_image)
        assert compute_generation_hash(controls, base) != compute_generation_hash(controls, with_adapter)

    def test_generate_routes_plain_generator_to_ip_adapter(self, client, reference_image):
        """If the configured generator cannot accept ip_adapter_image, the
        orchestrator must fall back to an IPAdapter generator instead of failing."""
        class PlainGenerator:
            def __init__(self):
                self.last_kwargs = None

            def supports_ip_adapter(self):
                return False

            def generate(self, **kwargs):
                self.last_kwargs = kwargs
                arr = np.zeros((64, 64, 4), dtype=np.uint8)
                arr[8:56, 8:56, :3] = [10, 20, 30]
                arr[8:56, 8:56, 3] = 255
                return [Image.fromarray(arr, "RGBA")]

        pipe = AssetPipeline()
        plain = PlainGenerator()
        pipe.set_generator(plain)
        set_pipeline(pipe)
        set_generator_loaded(True)
        tc = TestClient(app)

        fake_fallback = RecordingGenerator(num_images=1)
        with patch("backend.modules.generator.registry.create_generator") as mock_create:
            mock_create.return_value = fake_fallback
            resp = tc.post("/generate", json={
                "asset_type": "character",
                "ip_adapter": True,
                "ip_adapter_scale": 0.9,
                "reference_image": reference_image,
            })
        assert resp.status_code == 202
        data = resp.json()
        result = poll_job(tc, data["job_id"])
        assert result["status"] == "done"
        mock_create.assert_called_once_with(
            "ip_adapter", ip_adapter_scale=0.9, lora_path=None
        )
        assert fake_fallback.last_kwargs is not None
        assert "ip_adapter_image" in fake_fallback.last_kwargs


class TestBatchIPAdapterAPI:
    def test_batch_item_passes_reference_image(self, client, reference_image):
        resp = client.post("/generate/batch", json={
            "items": [
                {"asset_type": "character", "view": "front",
                 "ip_adapter": True, "ip_adapter_scale": 0.75,
                 "reference_image": reference_image},
            ]
        })
        assert resp.status_code == 202
        data = resp.json()
        assert data["total"] == 1

        batch = poll_batch(client, data["batch_id"])
        assert batch["completed"] == 1
        assert batch["failed"] == 0

        gen = get_pipeline_generator()
        assert gen.last_kwargs is not None
        assert "ip_adapter_image" in gen.last_kwargs
        assert gen.last_kwargs["ip_adapter_image"] is not None

    def test_batch_item_ip_adapter_without_reference_rejected(self, client):
        resp = client.post("/generate/batch", json={
            "items": [
                {"asset_type": "character", "ip_adapter": True},
            ]
        })
        assert resp.status_code == 422

    def test_batch_without_ip_adapter_no_reference_image(self, client):
        resp = client.post("/generate/batch", json={
            "items": [{"asset_type": "character"}]
        })
        assert resp.status_code == 202
        data = resp.json()
        batch = poll_batch(client, data["batch_id"])
        assert batch["completed"] == 1

        gen = get_pipeline_generator()
        assert gen.all_kwargs
        assert all("ip_adapter_image" not in k for k in gen.all_kwargs)


def poll_batch(client, batch_id, timeout=30):
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(f"/batch-status/{batch_id}")
        assert resp.status_code == 200
        data = resp.json()
        if data["status"] in ("done", "partial_failure"):
            return data
        time.sleep(0.05)
    raise TimeoutError(f"Batch {batch_id} did not complete within {timeout}s")


def get_pipeline_generator():
    from backend.api.routes import get_pipeline
    return get_pipeline().generator


def get_library():
    from backend.api.routes import get_library
    return get_library()
