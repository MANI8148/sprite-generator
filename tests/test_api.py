"""Tests for FastAPI backend routes (roadmap: Phase 1 Item 1)."""

import os
import json
import time
import tempfile
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock

from PIL import Image
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.api.routes import (
    router, set_pipeline, set_generator_loaded,
    get_pipeline, set_storage, _generator_loaded,
)
from backend.modules.pipeline.orchestrator import AssetPipeline, PipelineConfig
from backend.modules.prompt_builder.controls import AssetControls
from backend.modules.storage.file_storage import FileStorage
from backend.modules.rate_limiter import RateLimiter, set_rate_limiter, get_rate_limiter, EXEMPT_PATHS
from backend.main import app


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


@pytest.fixture(autouse=True)
def reset_state():
    set_generator_loaded(False)
    tmp = tempfile.mkdtemp()
    set_storage(FileStorage(base_dir=tmp))
    set_rate_limiter(RateLimiter(max_requests=100, window_seconds=60))


@pytest.fixture
def client():
    pipe = AssetPipeline()
    pipe.set_generator(FakeGenerator(num_images=1))
    set_pipeline(pipe)
    set_generator_loaded(True)
    return TestClient(app)


class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["generator_loaded"] is True

    def test_health_reports_generator_not_loaded(self):
        pipe = AssetPipeline()
        set_pipeline(pipe)
        set_generator_loaded(False)
        tc = TestClient(app)
        resp = tc.get("/health")
        assert resp.status_code == 200
        assert resp.json()["generator_loaded"] is False


class TestGenerate:
    def test_generate_returns_images(self, client):
        resp = client.post("/generate", json={
            "asset_type": "character",
            "view": "front",
            "animation": "idle",
            "palette": "auto",
            "sprite_size": "32x32",
            "num_frames": 1,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        assert data["prompt"] != ""
        assert "quality_tier" in data
        assert "validation" in data
        assert isinstance(data["output_paths"], list)
        assert len(data["output_paths"]) > 0

    def test_generate_with_multiple_frames(self, client):
        pipe = AssetPipeline()
        pipe.set_generator(FakeGenerator(num_images=4))
        set_pipeline(pipe)
        set_generator_loaded(True)
        tc = TestClient(app)

        resp = tc.post("/generate", json={
            "asset_type": "character",
            "view": "side",
            "animation": "walk",
            "num_frames": 4,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["output_paths"]) > 0

    def test_generate_without_loaded_generator_returns_503(self):
        pipe = AssetPipeline()
        set_pipeline(pipe)
        set_generator_loaded(False)
        tc = TestClient(app)

        resp = tc.post("/generate", json={})
        assert resp.status_code == 503
        assert "Generator not set" in resp.json()["detail"]

    def test_generate_with_various_asset_types(self, client):
        for asset_type in ["character", "building", "vehicle", "enemy", "prop"]:
            resp = client.post("/generate", json={
                "asset_type": asset_type,
                "view": "front",
            })
            assert resp.status_code == 200, f"Failed for asset_type={asset_type}"
            data = resp.json()
            assert data["quality_tier"] in ("clean", "acceptable", "noisy", "blurry", "broken_outline")

    def test_generate_metadata_includes_controls(self, client):
        resp = client.post("/generate", json={
            "asset_type": "building",
            "view": "isometric",
            "palette": "retro_16",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "validation" in data
        assert "output_paths" in data


class TestDownload:
    def test_download_existing_job_returns_zip(self, client):
        resp = client.post("/generate", json={"asset_type": "character"})
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]

        resp = client.get(f"/download/{job_id}")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/zip"

    def test_download_nonexistent_job_returns_404(self, client):
        resp = client.get("/download/nonexistent")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()


class TestHistory:
    def test_history_returns_list(self, client):
        resp = client.get("/history")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
        assert len(resp.json()) == 0

    def test_history_records_generation(self, client):
        client.post("/generate", json={"asset_type": "character"})
        client.post("/generate", json={"asset_type": "enemy"})

        resp = client.get("/history")
        data = resp.json()
        assert len(data) == 2
        assert data[0]["job_id"] != data[1]["job_id"]
        assert "prompt" in data[0]
        assert "quality_tier" in data[0]
        assert "outputs" in data[0]

    def test_history_isolation_between_requests(self, client):
        resp = client.get("/history")
        assert len(resp.json()) == 0

        client.post("/generate", json={"asset_type": "character"})
        resp = client.get("/history")
        assert len(resp.json()) == 1

        client.post("/generate", json={"asset_type": "vehicle"})
        resp = client.get("/history")
        assert len(resp.json()) == 2


class TestLoadModel:
    @pytest.mark.skip(reason="Requires GPU with CUDA to load SD model")
    def test_load_model_endpoint_exists(self, client):
        resp = client.post("/load-model", json={})
        assert resp.status_code in (200, 500)


class TestRateLimiter:
    def test_unit_block_when_exceeded(self):
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        ip = "192.168.1.1"
        assert limiter.check(ip) is True
        assert limiter.check(ip) is True
        assert limiter.check(ip) is True
        assert limiter.check(ip) is False

    def test_unit_allows_after_window_expires(self):
        limiter = RateLimiter(max_requests=1, window_seconds=0)
        ip = "192.168.1.2"
        assert limiter.check(ip) is True
        assert limiter.check(ip) is True

    def test_unit_separate_ips_independent(self):
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        assert limiter.check("ip-a") is True
        assert limiter.check("ip-b") is True
        assert limiter.check("ip-a") is True
        assert limiter.check("ip-b") is True
        assert limiter.check("ip-a") is False
        assert limiter.check("ip-b") is False

    def test_integration_health_exempt_from_rate_limit(self, client):
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        set_rate_limiter(limiter)
        for _ in range(5):
            resp = client.get("/health")
            assert resp.status_code == 200

    def test_integration_generate_blocked_when_rate_exceeded(self, client):
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        set_rate_limiter(limiter)
        resp1 = client.post("/generate", json={"asset_type": "character"})
        assert resp1.status_code == 200
        resp2 = client.post("/generate", json={"asset_type": "enemy"})
        assert resp2.status_code == 200
        resp3 = client.post("/generate", json={"asset_type": "vehicle"})
        assert resp3.status_code == 429
        data = resp3.json()
        assert "Rate limit exceeded" in data["detail"]

    def test_headers_present_on_success(self, client):
        limiter = RateLimiter(max_requests=100, window_seconds=60)
        set_rate_limiter(limiter)
        resp = client.post("/generate", json={"asset_type": "character"})
        assert resp.status_code == 200
        assert "X-RateLimit-Limit" in resp.headers
        assert "X-RateLimit-Remaining" in resp.headers
        assert "X-RateLimit-Reset" in resp.headers
        assert resp.headers["X-RateLimit-Limit"] == "100"
        assert int(resp.headers["X-RateLimit-Remaining"]) <= 99

    def test_headers_present_on_429(self, client):
        limiter = RateLimiter(max_requests=1, window_seconds=60)
        set_rate_limiter(limiter)
        resp1 = client.post("/generate", json={"asset_type": "character"})
        assert resp1.status_code == 200
        resp2 = client.post("/generate", json={"asset_type": "enemy"})
        assert resp2.status_code == 429
        assert resp2.headers["X-RateLimit-Remaining"] == "0"
        assert resp2.headers["X-RateLimit-Limit"] == "1"

    def test_remaining_decrements_correctly(self):
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        ip = "10.0.0.1"
        assert limiter.remaining(ip) == 5
        limiter.check(ip)
        assert limiter.remaining(ip) == 4
        limiter.check(ip)
        assert limiter.remaining(ip) == 3
        limiter.check(ip)
        assert limiter.remaining(ip) == 2
        limiter.check(ip)
        assert limiter.remaining(ip) == 1
        limiter.check(ip)
        assert limiter.remaining(ip) == 0
        assert limiter.check(ip) is False

    def test_reset_time_returns_future(self):
        limiter = RateLimiter(max_requests=1, window_seconds=10)
        ip = "10.0.0.2"
        limiter.check(ip)
        rt = limiter.reset_time(ip)
        now = time.time()
        assert rt > now
        assert rt <= now + 10

    def test_reset_time_when_no_requests(self):
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        ip = "10.0.0.3"
        rt = limiter.reset_time(ip)
        assert rt <= time.time()

    def test_env_var_configuration(self):
        os.environ["RATE_LIMIT_MAX_REQUESTS"] = "25"
        os.environ["RATE_LIMIT_WINDOW_SECONDS"] = "120"
        limiter = RateLimiter()
        assert limiter.max_requests == 25
        assert limiter.window_seconds == 120
        del os.environ["RATE_LIMIT_MAX_REQUESTS"]
        del os.environ["RATE_LIMIT_WINDOW_SECONDS"]

    def test_env_var_configuration_fallback(self):
        limiter = RateLimiter()
        assert limiter.max_requests == 10
        assert limiter.window_seconds == 60


class TestBatchGenerate:
    def test_batch_generate_two_items(self, client):
        resp = client.post("/generate/batch", json={
            "items": [
                {"asset_type": "character", "view": "front"},
                {"asset_type": "building", "view": "isometric"},
            ]
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert data["succeeded"] == 2
        assert data["failed"] == 0
        assert len(data["results"]) == 2
        assert data["results"][0]["quality_tier"] in ("clean", "acceptable", "noisy", "blurry", "broken_outline")
        assert data["results"][1]["quality_tier"] in ("clean", "acceptable", "noisy", "blurry", "broken_outline")

    def test_batch_generate_empty_items(self, client):
        resp = client.post("/generate/batch", json={"items": []})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["succeeded"] == 0
        assert data["failed"] == 0
        assert len(data["results"]) == 0

    def test_batch_generate_single_item(self, client):
        resp = client.post("/generate/batch", json={
            "items": [
                {"asset_type": "enemy", "view": "side", "animation": "walk", "num_frames": 4}
            ]
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["succeeded"] == 1
        assert len(data["results"]) == 1
        assert data["results"][0]["job_id"] != ""

    def test_batch_generate_custom_batch_id(self, client):
        resp = client.post("/generate/batch", json={
            "batch_id": "my_batch_001",
            "items": [
                {"asset_type": "character"},
            ]
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["batch_id"] == "my_batch_001"
        assert data["results"][0]["job_id"].startswith("my_batch_001")

    def test_batch_generate_without_loaded_generator_returns_503(self):
        pipe = AssetPipeline()
        set_pipeline(pipe)
        set_generator_loaded(False)
        tc = TestClient(app)

        resp = tc.post("/generate/batch", json={
            "items": [{"asset_type": "character"}]
        })
        assert resp.status_code == 503
        assert "Generator not set" in resp.json()["detail"]

    def test_batch_generate_multiple_asset_types(self, client):
        items = [
            {"asset_type": t, "view": "front"}
            for t in ["character", "building", "enemy", "vehicle", "prop"]
        ]
        resp = client.post("/generate/batch", json={"items": items})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 5
        assert data["succeeded"] == 5
        for r in data["results"]:
            assert r["quality_tier"] in ("clean", "acceptable", "noisy", "blurry", "broken_outline")

    def test_batch_generate_different_views(self, client):
        items = [
            {"asset_type": "character", "view": v}
            for v in ["front", "side", "isometric", "back"]
        ]
        resp = client.post("/generate/batch", json={"items": items})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 4
        assert data["succeeded"] == 4
