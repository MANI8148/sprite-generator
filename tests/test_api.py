"""Tests for FastAPI backend routes (roadmap: Phase 1 Item 1)."""

import os
import json
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
