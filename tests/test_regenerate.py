"""Tests for Incremental Regeneration (asset memory variant generation).

Implements the ROADMAP "Incremental regeneration ('asset memory')" priority:
re-generate one or more variants of an existing asset while keeping its
style/seed context, exposed as ``POST /api/regenerate``.
"""

import os
import tempfile
import time

import pytest
from fastapi.testclient import TestClient

from backend.modules.asset_memory import AssetMemory
from backend.modules.asset_memory.memory import (
    seed_from_generation_hash,
    build_variant_controls,
)
from backend.modules.pipeline.orchestrator import AssetPipeline
from backend.modules.prompt_builder.controls import (
    AssetType, View, Palette, Animation, SpriteSize,
)
from backend.modules.storage.asset_library import AssetLibrary, AssetRecord
from backend.modules.storage.file_storage import FileStorage
from backend.modules.rate_limiter import RateLimiter, set_rate_limiter
from backend.modules.tasks.queue import TaskQueue, set_task_queue
from backend.api.routes import set_library, set_pipeline, set_generator_loaded, set_storage
from backend.main import app


class FakeGenerator:
    def __init__(self):
        self._call_count = [0]

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


def add_base(library, asset_id="base1", seed=42):
    record = AssetRecord(
        asset_id=asset_id,
        job_id="job_base",
        asset_type="character",
        prompt="pixel art character",
        quality_tier="clean",
        metadata={
            "view": "front",
            "animation": "idle",
            "palette": "retro_16",
            "sprite_size": "32x32",
            "theme": "forest",
            "style": "pixel art",
            "seed": seed,
            "control_snapshot": {
                "asset_type": "character",
                "view": "front",
                "animation": "idle",
                "palette": "retro_16",
                "sprite_size": "32x32",
                "theme": "forest",
                "style": "pixel art",
                "seed": seed,
            },
        },
        generation_hash="abcdef0123456789",
    )
    library.add_asset(record)
    return library.get_asset(asset_id)


class TestSeedFromGenerationHash:
    def test_deterministic(self):
        assert seed_from_generation_hash("ffeeddccbbaa9988") == seed_from_generation_hash("ffeeddccbbaa9988")

    def test_different_hashes_mostly_differ(self):
        seeds = {
            seed_from_generation_hash(h)
            for h in ("ffffffffffffffff", "0000000000000000", "deadbeefdeadbeef", "1234567890abcdef")
        }
        assert len(seeds) == 4

    def test_in_range(self):
        for h in ("ffffffffffffffff", "0000000000000000", "cafebabe", "8000000000000000"):
            s = seed_from_generation_hash(h)
            assert 0 <= s < 2**31

    def test_empty_hash_defaults_to_one(self):
        assert seed_from_generation_hash("") == 1


class TestBuildVariantControls:
    def setup_method(self):
        self.library = AssetLibrary(base_dir=tempfile.mkdtemp())
        self.base = add_base(self.library)

    def test_preserves_style_context(self):
        ctrl = build_variant_controls(self.base, 0)
        assert ctrl.asset_type == AssetType.CHARACTER
        assert ctrl.view == View.FRONT
        assert ctrl.animation == Animation.IDLE
        assert ctrl.palette == Palette.RETRO_16
        assert ctrl.sprite_size == SpriteSize.S_32
        assert ctrl.theme == "forest"
        assert ctrl.style == "pixel art"

    def test_distinct_seeds_per_index(self):
        c0 = build_variant_controls(self.base, 0).seed
        c1 = build_variant_controls(self.base, 1).seed
        c2 = build_variant_controls(self.base, 2).seed
        assert len({c0, c1, c2}) == 3

    def test_deterministic(self):
        assert build_variant_controls(self.base, 2).seed == build_variant_controls(self.base, 2).seed

    def test_theme_override_wins(self):
        ctrl = build_variant_controls(self.base, 1, theme_override="dungeon")
        assert ctrl.theme == "dungeon"
        assert ctrl.asset_type == AssetType.CHARACTER

    def test_stride_changes_seed(self):
        a0 = build_variant_controls(self.base, 0, seed_stride=1).seed
        b0 = build_variant_controls(self.base, 0, seed_stride=5).seed
        # index 0 should equal the base seed regardless of stride
        assert a0 == b0
        assert build_variant_controls(self.base, 1, seed_stride=5).seed == (b0 + 5) % (2**31)

    def test_anchors_to_generation_hash_without_snapshot_seed(self):
        base = AssetRecord(
            asset_id="nohash",
            job_id="j",
            asset_type="enemy",
            prompt="goblin",
            quality_tier="clean",
            metadata={"view": "side", "style": "pixel art"},
            generation_hash="deadbeef00000000",
        )
        c1 = build_variant_controls(base, 0)
        c2 = build_variant_controls(base, 0)
        assert c1.seed == c2.seed == seed_from_generation_hash("deadbeef00000000")


class TestAssetMemoryPlanVariants:
    def test_missing_base_raises_key_error(self):
        memory = AssetMemory(library=AssetLibrary(base_dir=tempfile.mkdtemp()))
        with pytest.raises(KeyError):
            memory.plan_variants("unknown_asset", num_variants=2)

    def test_plan_variants_returns_base_and_controls(self):
        library = AssetLibrary(base_dir=tempfile.mkdtemp())
        base = add_base(library)
        memory = AssetMemory(library=library)
        base_rec, controls = memory.plan_variants(base.asset_id, num_variants=3)
        assert base_rec.asset_id == base.asset_id
        assert len(controls) == 3
        assert all(c.asset_type == AssetType.CHARACTER for c in controls)
        assert len({c.seed for c in controls}) == 3


class TestRegenerateAPI:
    @pytest.fixture(autouse=True)
    def reset(self):
        reset_state()

    @pytest.fixture
    def fake_gen(self):
        return FakeGenerator()

    @pytest.fixture
    def client(self, fake_gen):
        pipe = AssetPipeline()
        pipe.set_generator(fake_gen)
        set_pipeline(pipe)
        set_generator_loaded(True)
        return TestClient(app)

    def generate_base(self, client, **overrides):
        payload = {
            "asset_type": "character",
            "view": "front",
            "animation": "idle",
            "palette": "auto",
            "sprite_size": "32x32",
            "seed": 42,
        }
        payload.update(overrides)
        resp = client.post("/generate", json=payload)
        assert resp.status_code == 202
        data = poll_job(client, resp.json()["job_id"])
        assert data["status"] == "done"
        return poll_history(client)[0]["asset_id"]

    def test_503_when_generator_not_loaded(self):
        set_generator_loaded(False)
        client = TestClient(app)
        resp = client.post("/regenerate", json={"base_id": "whatever", "num_variants": 2})
        assert resp.status_code == 503

    def test_404_for_unknown_base(self):
        client, _ = TestClient(app), None
        resp = client.post("/regenerate", json={"base_id": "missing", "num_variants": 2})
        assert resp.status_code == 404

    def test_422_for_invalid_num_variants(self):
        library = AssetLibrary(base_dir=tempfile.mkdtemp())
        base = add_base(library)
        import backend.api.routes as routes
        routes.set_library(library)
        client = TestClient(app)
        resp = client.post("/regenerate", json={"base_id": base.asset_id, "num_variants": 0})
        assert resp.status_code == 422

    def test_regenerate_creates_variants(self, client):
        library = AssetLibrary(base_dir=tempfile.mkdtemp())
        base = add_base(library)
        import backend.api.routes as routes
        routes.set_library(library)

        resp = client.post("/regenerate", json={"base_id": base.asset_id, "num_variants": 3})
        assert resp.status_code == 202
        body = resp.json()
        assert body["base_id"] == base.asset_id
        assert body["num_variants"] == 3
        assert len(body["job_ids"]) == 3

        for jid in body["job_ids"]:
            poll_job(client, jid)

        batch = client.get(f"/batch-status/{body['batch_id']}").json()
        assert batch["status"] == "done"
        assert len(batch["results"]) == 3

    def test_regenerate_variants_generated_once_each(self, client, fake_gen):
        library = AssetLibrary(base_dir=tempfile.mkdtemp())
        base = add_base(library)
        import backend.api.routes as routes
        routes.set_library(library)

        resp = client.post("/regenerate", json={"base_id": base.asset_id, "num_variants": 3})
        body = resp.json()
        for jid in body["job_ids"]:
            poll_job(client, jid)

        # 3 distinct variant seeds => 3 generator calls, none served from cache
        assert fake_gen._call_count[0] == 3

    def test_repeat_regenerate_hits_cache(self, client, fake_gen):
        library = AssetLibrary(base_dir=tempfile.mkdtemp())
        base = add_base(library)
        import backend.api.routes as routes
        routes.set_library(library)

        body1 = client.post("/regenerate", json={"base_id": base.asset_id, "num_variants": 3}).json()
        for jid in body1["job_ids"]:
            poll_job(client, jid)
        calls = fake_gen._call_count[0]
        assert calls == 3

        body2 = client.post("/regenerate", json={"base_id": base.asset_id, "num_variants": 3}).json()
        for jid in body2["job_ids"]:
            poll_job(client, jid)

        assert fake_gen._call_count[0] == calls, (
            "identical regenerate must be served entirely from cache"
        )
        batch = client.get(f"/batch-status/{body2['batch_id']}").json()
        assert all(r["cached"] for r in batch["results"])

    def test_theme_override_keeps_asset_type(self, client, fake_gen):
        library = AssetLibrary(base_dir=tempfile.mkdtemp())
        base = add_base(library)
        import backend.api.routes as routes
        routes.set_library(library)

        resp = client.post("/regenerate", json={
            "base_id": base.asset_id,
            "num_variants": 1,
            "theme_override": "dungeon",
        })
        body = resp.json()
        jid = body["job_ids"][0]
        data = poll_job(client, jid)
        assert data["status"] == "done"