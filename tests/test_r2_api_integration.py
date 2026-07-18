"""Integration tests for Cloudflare R2 cloud storage in the API (roadmap: Phase 3 Item 1)."""

import os
import json
import tempfile
from unittest.mock import patch, MagicMock
from typing import Optional, List
from pathlib import Path

from PIL import Image
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.api.routes import (
    router, set_pipeline, set_generator_loaded,
    set_storage, set_library, set_r2_storage, _batch_jobs,
)
from backend.modules.pipeline.orchestrator import AssetPipeline
from backend.modules.storage.file_storage import FileStorage
from backend.modules.storage.r2_storage import R2Storage
from backend.modules.storage.asset_library import AssetLibrary
from backend.modules.rate_limiter import RateLimiter, set_rate_limiter
from backend.modules.tasks.queue import TaskQueue, set_task_queue, JobStatus
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


@pytest.fixture(autouse=True)
def reset_state():
    set_generator_loaded(False)
    tmp = tempfile.mkdtemp()
    set_storage(FileStorage(base_dir=tmp))
    set_library(AssetLibrary(base_dir=os.path.join(tmp, "lib")))
    set_rate_limiter(RateLimiter(max_requests=100, window_seconds=60))
    set_task_queue(TaskQueue(max_workers=4))
    set_r2_storage(R2Storage())
    _batch_jobs.clear()


@pytest.fixture
def mock_s3_client():
    with patch("boto3.client") as mock_client:
        client = MagicMock()
        mock_client.return_value = client
        yield client


@pytest.fixture
def client_with_r2(mock_s3_client):
    """Set up app with R2 storage enabled."""
    r2 = R2Storage(
        bucket="test-bucket",
        endpoint="https://test.r2.cloudflarestorage.com",
        access_key_id="test-key",
        secret_access_key="test-secret",
    )
    set_r2_storage(r2)
    pipe = AssetPipeline()
    pipe.set_generator(FakeGenerator(num_images=1))
    set_pipeline(pipe)
    set_generator_loaded(True)
    return TestClient(app)


class TestR2StatusEndpoint:
    def test_r2_status_not_available_by_default(self, client_with_r2):
        resp = client_with_r2.get("/storage/r2-status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["available"] is True
        assert data["bucket"] == "test-bucket"

    def test_r2_status_not_available_without_config(self):
        r2 = R2Storage()
        set_r2_storage(r2)
        pipe = AssetPipeline()
        pipe.set_generator(FakeGenerator())
        set_pipeline(pipe)
        set_generator_loaded(True)
        tc = TestClient(app)
        resp = tc.get("/storage/r2-status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["available"] is False
        assert data["bucket"] == ""


class TestR2IntegrationGenerate:
    def test_generate_uploads_to_r2_when_available(self, client_with_r2, mock_s3_client):
        mock_s3_client.get_object.side_effect = Exception("No history")
        paginator = MagicMock()
        mock_s3_client.get_paginator.return_value = paginator
        paginator.paginate.return_value = [{"Contents": []}]

        resp = client_with_r2.post("/generate", json={
            "asset_type": "character",
            "view": "front",
            "animation": "idle",
            "palette": "auto",
            "sprite_size": "32x32",
            "num_frames": 1,
        })
        assert resp.status_code == 202
        data = resp.json()
        result = poll_job(client_with_r2, data["job_id"])
        assert result["status"] == "done"

        assert mock_s3_client.upload_file.call_count >= 1
        call_args_list = mock_s3_client.upload_file.call_args_list
        r2_keys = [call[1]["Key"] for call in call_args_list]
        assert any(f"jobs/{data['job_id']}/" in key for key in r2_keys)

    def test_generate_uploads_zip_to_r2(self, client_with_r2, mock_s3_client):
        mock_s3_client.get_object.side_effect = Exception("No history")
        paginator = MagicMock()
        mock_s3_client.get_paginator.return_value = paginator
        paginator.paginate.return_value = [{"Contents": []}]

        resp = client_with_r2.post("/generate", json={
            "asset_type": "character",
            "view": "front",
        })
        assert resp.status_code == 202
        data = resp.json()
        result = poll_job(client_with_r2, data["job_id"])
        assert result["status"] == "done"

        call_args_list = mock_s3_client.upload_file.call_args_list
        r2_keys = [call[1]["Key"] for call in call_args_list]
        assert any(key.endswith("sprite_package.zip") for key in r2_keys)

    def test_generate_skips_r2_when_not_available(self, mock_s3_client):
        r2 = R2Storage()
        set_r2_storage(r2)
        pipe = AssetPipeline()
        pipe.set_generator(FakeGenerator())
        set_pipeline(pipe)
        set_generator_loaded(True)
        tc = TestClient(app)

        resp = tc.post("/generate", json={"asset_type": "character"})
        assert resp.status_code == 202
        data = resp.json()
        result = poll_job(tc, data["job_id"])
        assert result["status"] == "done"
        mock_s3_client.upload_file.assert_not_called()


class TestR2IntegrationBatch:
    def test_batch_generate_uploads_to_r2(self, client_with_r2, mock_s3_client):
        mock_s3_client.get_object.side_effect = Exception("No history")
        paginator = MagicMock()
        mock_s3_client.get_paginator.return_value = paginator
        paginator.paginate.return_value = [{"Contents": []}]

        resp = client_with_r2.post("/generate/batch", json={
            "items": [
                {"asset_type": "character", "view": "front"},
                {"asset_type": "building", "view": "isometric"},
            ]
        })
        assert resp.status_code == 202
        data = resp.json()

        def poll_batch(batch_id, timeout=30):
            import time
            deadline = time.time() + timeout
            while time.time() < deadline:
                r = client_with_r2.get(f"/batch-status/{batch_id}")
                assert r.status_code == 200
                d = r.json()
                if d["status"] in ("done", "partial_failure"):
                    return d
                time.sleep(0.05)
            raise TimeoutError(f"Batch {batch_id} not done in {timeout}s")

        result = poll_batch(data["batch_id"])
        assert result["completed"] == 2

        assert mock_s3_client.upload_file.call_count >= 1

    def test_batch_skips_r2_when_not_available(self, mock_s3_client):
        r2 = R2Storage()
        set_r2_storage(r2)
        pipe = AssetPipeline()
        pipe.set_generator(FakeGenerator())
        set_pipeline(pipe)
        set_generator_loaded(True)
        tc = TestClient(app)

        resp = tc.post("/generate/batch", json={
            "items": [{"asset_type": "character"}]
        })
        assert resp.status_code == 202
        data = resp.json()

        def poll_batch(batch_id, timeout=30):
            import time
            deadline = time.time() + timeout
            while time.time() < deadline:
                r = tc.get(f"/batch-status/{batch_id}")
                assert r.status_code == 200
                d = r.json()
                if d["status"] in ("done", "partial_failure"):
                    return d
                time.sleep(0.05)
            raise TimeoutError(f"Batch {batch_id} not done in {timeout}s")

        result = poll_batch(data["batch_id"])
        assert result["completed"] == 1
        mock_s3_client.upload_file.assert_not_called()


class TestR2IntegrationConfig:
    def test_r2_configured_via_env(self, mock_s3_client):
        with patch.dict(os.environ, {
            "R2_BUCKET": "env-bucket",
            "R2_ENDPOINT": "https://env.example.com",
            "R2_ACCESS_KEY_ID": "env-key",
            "R2_SECRET_ACCESS_KEY": "env-secret",
        }):
            r2 = R2Storage()
            set_r2_storage(r2)
            assert r2.available is True
            tc = TestClient(app)
            resp = tc.get("/storage/r2-status")
            assert resp.status_code == 200
            data = resp.json()
            assert data["available"] is True
            assert data["bucket"] == "env-bucket"


class TestR2UploadHelper:
    def test_upload_to_r2_skips_when_not_available(self):
        from backend.api.routes import _upload_to_r2
        r2 = R2Storage()
        with patch("logging.Logger.warning") as mock_warn:
            _upload_to_r2(r2, "test-job", [])
            mock_warn.assert_not_called()

    def test_upload_to_r2_uploads_files(self, mock_s3_client):
        r2 = R2Storage(
            bucket="b",
            endpoint="https://e.com",
            access_key_id="k",
            secret_access_key="s",
        )
        from backend.api.routes import _upload_to_r2
        tmp = tempfile.mkdtemp()
        path1 = os.path.join(tmp, "test.png")
        with open(path1, "w") as f:
            f.write("fake png")
        zip_path = os.path.join(tmp, "package.zip")
        with open(zip_path, "w") as f:
            f.write("fake zip")

        _upload_to_r2(r2, "myjob", [path1], zip_path)

        assert mock_s3_client.upload_file.call_count == 2
        keys = [call[1]["Key"] for call in mock_s3_client.upload_file.call_args_list]
        assert any("myjob" in k for k in keys)
        assert any(k.endswith("package.zip") for k in keys)

    def test_upload_to_r2_handles_missing_files(self, mock_s3_client):
        r2 = R2Storage(
            bucket="b",
            endpoint="https://e.com",
            access_key_id="k",
            secret_access_key="s",
        )
        from backend.api.routes import _upload_to_r2
        _upload_to_r2(r2, "myjob", ["/nonexistent/path.png"])
        mock_s3_client.upload_file.assert_not_called()
