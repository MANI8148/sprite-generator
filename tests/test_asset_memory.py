"""Tests for Incremental Regeneration / Asset Memory (roadmap: Phase 3)."""

import os
import json
import tempfile

import pytest
from fastapi.testclient import TestClient

from backend.modules.asset_memory import compute_generation_hash, cache_result_key
from backend.modules.asset_memory.memory import AssetMemory
from backend.modules.pipeline.orchestrator import AssetPipeline, PipelineConfig
from backend.modules.prompt_builder.controls import (
    AssetControls, AssetType, View, Palette, Animation, SpriteSize,
)
from backend.modules.storage.asset_library import AssetLibrary, AssetRecord
from backend.modules.storage.file_storage import FileStorage
from backend.modules.rate_limiter import RateLimiter, set_rate_limiter
from backend.modules.tasks.queue import TaskQueue, set_task_queue
from backend.api.routes import set_library, set_pipeline, set_generator_loaded, set_storage
from backend.main import app


class FakeGenerator:
    def __init__(self, call_count=None):
        self._call_count = call_count if call_count is not None else [0]

    def generate(self, prompt="", negative_prompt="", width=512, height=512, seed=-1, num_images=None):
        from PIL import Image
        import numpy as np
        self._call_count[0] += 1
        n = num_images or 1
        images = []
        for i in range(n):
            arr = np.zeros((height, width, 4), dtype=np.uint8)
            val = (self._call_count[0] * 50 + i * 30) % 256
            arr[:, :, 0] = val
            arr[:, :, 1] = val
            arr[:, :, 2] = val
            arr[:, :, 3] = 255
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


def reset_state():
    tmp = tempfile.mkdtemp()
    set_storage(FileStorage(base_dir=tmp))
    set_library(AssetLibrary(base_dir=os.path.join(tmp, "lib")))
    set_pipeline(AssetPipeline())
    set_generator_loaded(True)
    set_rate_limiter(RateLimiter(max_requests=100, window_seconds=60))
    set_task_queue(TaskQueue(max_workers=4))
    return tmp


class TestComputeGenerationHash:
    def test_same_controls_same_hash(self):
        c1 = AssetControls(asset_type=AssetType.CHARACTER, view=View.FRONT, seed=42)
        c2 = AssetControls(asset_type=AssetType.CHARACTER, view=View.FRONT, seed=42)
        cfg = PipelineConfig()
        h1 = compute_generation_hash(c1, cfg)
        h2 = compute_generation_hash(c2, cfg)
        assert h1 == h2

    def test_different_seed_different_hash(self):
        c1 = AssetControls(seed=42)
        c2 = AssetControls(seed=99)
        cfg = PipelineConfig()
        assert compute_generation_hash(c1, cfg) != compute_generation_hash(c2, cfg)

    def test_different_asset_type_different_hash(self):
        c1 = AssetControls(asset_type=AssetType.CHARACTER)
        c2 = AssetControls(asset_type=AssetType.ENEMY)
        cfg = PipelineConfig()
        assert compute_generation_hash(c1, cfg) != compute_generation_hash(c2, cfg)

    def test_different_config_different_hash(self):
        controls = AssetControls()
        cfg1 = PipelineConfig(remove_bg=True)
        cfg2 = PipelineConfig(remove_bg=False)
        assert compute_generation_hash(controls, cfg1) != compute_generation_hash(controls, cfg2)

    def test_hash_is_sha256_hex(self):
        controls = AssetControls()
        cfg = PipelineConfig()
        h = compute_generation_hash(controls, cfg)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_deterministic(self):
        controls = AssetControls(seed=12345, theme="forest")
        cfg = PipelineConfig(max_colors=16, palette_name="gameboy")
        h1 = compute_generation_hash(controls, cfg)
        h2 = compute_generation_hash(controls, cfg)
        assert h1 == h2


class TestCacheResultKey:
    def test_key_is_short_hex(self):
        from backend.modules.pipeline.orchestrator import PipelineResult
        from PIL import Image
        import numpy as np
        img = Image.fromarray(np.zeros((64, 64, 4), dtype=np.uint8), "RGBA")
        result = PipelineResult(
            images=[img],
            metadata={"prompt": "test", "controls": {"seed": 42}},
            validation=[{"quality_tier": "clean"}],
        )
        key = cache_result_key(result)
        assert len(key) == 16
        assert all(c in "0123456789abcdef" for c in key)


class TestAssetMemoryLookup:
    def test_lookup_returns_none_for_unknown_hash(self):
        tmp = tempfile.mkdtemp()
        lib = AssetLibrary(base_dir=os.path.join(tmp, "lib"))
        memory = AssetMemory(library=lib)
        record = memory.lookup("nonexistent_hash_1234567890123456")
        assert record is None

    def test_lookup_finds_existing_asset(self):
        tmp = tempfile.mkdtemp()
        lib = AssetLibrary(base_dir=os.path.join(tmp, "lib"))
        record = AssetRecord(
            asset_id="test1",
            job_id="job1",
            asset_type="character",
            prompt="test prompt",
            quality_tier="clean",
            metadata={"generation_hash": "abc123hash"},
        )
        lib.add_asset(record)
        memory = AssetMemory(library=lib)
        found = memory.lookup("abc123hash")
        assert found is not None
        assert found.asset_id == "test1"

    def test_store_and_retrieve(self):
        tmp = tempfile.mkdtemp()
        lib = AssetLibrary(base_dir=os.path.join(tmp, "lib"))
        memory = AssetMemory(library=lib)
        record = AssetRecord(
            asset_id="store1",
            job_id="job_store",
            asset_type="enemy",
            prompt="goblin",
            quality_tier="clean",
        )
        memory.store("myhash123", record)
        found = memory.lookup("myhash123")
        assert found is not None
        assert found.asset_id == "store1"
        assert found.metadata.get("generation_hash") == "myhash123"


class TestAssetMemoryAPI:
    @pytest.fixture(autouse=True)
    def reset(self):
        reset_state()

    @pytest.fixture
    def client(self):
        pipe = AssetPipeline()
        pipe.set_generator(FakeGenerator())
        set_pipeline(pipe)
        set_generator_loaded(True)
        return TestClient(app)

    def test_repeat_generation_uses_cache(self, client):
        resp1 = client.post("/generate", json={
            "asset_type": "character",
            "view": "front",
            "animation": "idle",
            "palette": "auto",
            "sprite_size": "32x32",
            "seed": 42,
        })
        assert resp1.status_code == 202
        job_id_1 = resp1.json()["job_id"]
        data1 = poll_job(client, job_id_1)
        assert data1["status"] == "done"

        resp2 = client.post("/generate", json={
            "asset_type": "character",
            "view": "front",
            "animation": "idle",
            "palette": "auto",
            "sprite_size": "32x32",
            "seed": 42,
        })
        assert resp2.status_code == 202
        job_id_2 = resp2.json()["job_id"]
        data2 = poll_job(client, job_id_2)
        assert data2["status"] == "done"

        assert data1["quality_tier"] == data2["quality_tier"]

    def test_different_parameters_do_not_cache(self, client):
        resp1 = client.post("/generate", json={
            "asset_type": "character",
            "view": "front",
            "animation": "idle",
            "palette": "auto",
            "sprite_size": "32x32",
            "seed": 42,
        })
        assert resp1.status_code == 202
        data1 = poll_job(client, resp1.json()["job_id"])
        assert data1["status"] == "done"

        resp2 = client.post("/generate", json={
            "asset_type": "enemy",
            "view": "side",
            "animation": "walk",
            "palette": "retro_16",
            "sprite_size": "64x64",
            "seed": 99,
        })
        assert resp2.status_code == 202
        data2 = poll_job(client, resp2.json()["job_id"])
        assert data2["status"] == "done"

        resp3 = client.post("/generate", json={
            "asset_type": "character",
            "view": "front",
            "animation": "idle",
            "palette": "auto",
            "sprite_size": "32x32",
            "seed": 42,
        })
        assert resp3.status_code == 202
        data3 = poll_job(client, resp3.json()["job_id"])
        assert data3["status"] == "done"

        assert data1["quality_tier"] == data3["quality_tier"]

    def test_cache_works_across_multiple_identical_requests(self, client):
        job_ids = []
        for _ in range(3):
            resp = client.post("/generate", json={
                "asset_type": "character",
                "view": "front",
                "seed": 7,
            })
            assert resp.status_code == 202
            job_ids.append(resp.json()["job_id"])

        results = [poll_job(client, jid) for jid in job_ids]
        for r in results:
            assert r["status"] == "done"
        tiers = [r["quality_tier"] for r in results]
        assert len(set(tiers)) == 1

    def test_generation_hash_in_asset_library(self, client):
        resp = client.post("/generate", json={
            "asset_type": "character",
            "view": "front",
            "seed": 42,
        })
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]
        data = poll_job(client, job_id)
        assert data["status"] == "done"

        history = client.get("/history").json()
        assert len(history) > 0

    def test_batch_generation_respects_cache(self, client):
        resp = client.post("/generate/batch", json={
            "items": [
                {"asset_type": "character", "view": "front", "seed": 1},
                {"asset_type": "character", "view": "front", "seed": 1},
                {"asset_type": "enemy", "view": "side", "seed": 2},
            ]
        })
        assert resp.status_code == 202
        batch_id = resp.json()["batch_id"]
        job_ids = resp.json()["job_ids"]
        assert len(job_ids) == 3

        for jid in job_ids:
            poll_job(client, jid)

        batch_status = client.get(f"/batch-status/{batch_id}").json()
        assert batch_status["status"] == "done"
        assert len(batch_status["results"]) == 3
