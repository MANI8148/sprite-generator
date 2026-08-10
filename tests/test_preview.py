"""Tests for GET /preview/{job_id} — the frontend image-preview endpoint.

Covers the AI_GAME_STUDIO Phase 1 roadmap item "Build a minimal frontend ...
image preview, download button": the backend must serve generated PNG frames
to the browser (not just the ZIP), so the frontend can render real previews.
"""

import os
import tempfile
from typing import List, Optional

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.api.routes import (
    router, set_pipeline, set_generator_loaded,
    get_pipeline, set_storage, set_library, set_job_store,
    _batch_jobs,
)
from backend.modules.pipeline.orchestrator import AssetPipeline
from backend.modules.storage.file_storage import FileStorage
from backend.modules.storage.asset_library import AssetLibrary
from backend.modules.rate_limiter import RateLimiter, set_rate_limiter
from backend.modules.tasks.queue import TaskQueue, set_task_queue
from backend.main import app


class FakeGenerator:
    num_images = 1

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
            arr[cy:cy + height // 2, cx:cx + width // 2, 0] = 255
            arr[cy:cy + height // 2, cx:cx + width // 2, 3] = 255
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
    set_job_store(None)
    _batch_jobs.clear()


@pytest.fixture
def client():
    pipe = AssetPipeline()
    pipe.set_generator(FakeGenerator())
    set_pipeline(pipe)
    set_generator_loaded(True)
    return TestClient(app)


class TestPreview:
    def test_preview_returns_png_for_generated_job(self, client):
        resp = client.post("/generate", json={
            "asset_type": "character",
            "view": "front",
            "animation": "idle",
        })
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]
        result = poll_job(client, job_id)
        assert result["status"] == "done"
        assert len(result["output_paths"]) > 0

        preview = client.get(f"/preview/{job_id}")
        assert preview.status_code == 200
        assert preview.headers["content-type"].startswith("image/png")
        assert preview.content.startswith(b"\x89PNG")

    def test_preview_defaults_to_first_frame(self, client):
        pipe = AssetPipeline()
        pipe.set_generator(FakeGenerator())
        set_pipeline(pipe)
        set_generator_loaded(True)
        tc = TestClient(app)

        resp = tc.post("/generate", json={
            "asset_type": "character",
            "view": "side",
            "animation": "walk",
            "num_frames": 4,
        })
        data = resp.json()
        result = poll_job(tc, data["job_id"])
        assert result["status"] == "done"

        first = tc.get(f"/preview/{data['job_id']}")
        assert first.status_code == 200

    def test_preview_index_out_of_range_returns_422(self, client):
        resp = client.post("/generate", json={
            "asset_type": "character",
            "view": "front",
        })
        data = resp.json()
        job_id = data["job_id"]
        poll_job(client, job_id)

        bad = client.get(f"/preview/{job_id}?index=99")
        assert bad.status_code == 422

    def test_preview_unknown_job_returns_404(self, client):
        resp = client.get("/preview/does-not-exist")
        assert resp.status_code == 404

    def test_preview_negative_index_returns_422(self, client):
        resp = client.post("/generate", json={
            "asset_type": "character",
            "view": "front",
        })
        data = resp.json()
        job_id = data["job_id"]
        poll_job(client, job_id)

        resp = client.get(f"/preview/{job_id}?index=-1")
        assert resp.status_code == 422

    def test_preview_ignores_non_image_outputs(self, tmp_path):
        storage = FileStorage(base_dir=str(tmp_path))
        job_id = "job_no_images"
        storage.add_job(job_id, {
            "outputs": [str(tmp_path / "metadata.json")],
            "zip_path": str(tmp_path / "sprite_package.zip"),
        })
        set_storage(storage)
        set_generator_loaded(True)
        tc = TestClient(app)
        resp = tc.get(f"/preview/{job_id}")
        assert resp.status_code == 404
        assert "No image outputs" in resp.json()["detail"]

    def test_preview_uses_library_snapshot_path(self, tmp_path):
        from backend.api.routes import set_library as _set_library
        lib = AssetLibrary(base_dir=str(tmp_path / "lib"))
        _set_library(lib)
        storage = FileStorage(base_dir=str(tmp_path))
        png_path = tmp_path / "frame_0.png"
        Image.new("RGBA", (8, 8), (255, 0, 0, 255)).save(png_path)
        job_id = "job_with_images"
        storage.add_job(job_id, {
            "outputs": [str(png_path)],
        })
        set_storage(storage)
        set_generator_loaded(True)
        tc = TestClient(app)
        resp = tc.get(f"/preview/{job_id}")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("image/png")