"""Tests for tileset packing integrated into the asset pipeline.

Roadmap: MVP "Sprite packing (sprite sheet, animation strip)" — the packing
module spec (sprite sheet, tileset, animation strip) — and Phase 3 tileset
generator support. Tileset assets must be packed as proper tilesets with
grid metadata, not generic sprite sheets.
"""

import json
import os
import tempfile
from typing import List, Optional

import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from backend.modules.pipeline.orchestrator import AssetPipeline, PipelineConfig
from backend.modules.prompt_builder.controls import (
    AssetControls, AssetType, View, Palette, Animation, SpriteSize,
)
from backend.modules.asset_memory import compute_generation_hash
from backend.api.routes import set_library, set_pipeline, set_generator_loaded, set_storage
from backend.modules.rate_limiter import RateLimiter, set_rate_limiter
from backend.modules.tasks.queue import TaskQueue, set_task_queue
from backend.modules.storage.file_storage import FileStorage
from backend.modules.storage.asset_library import AssetLibrary
from backend.main import app


class FakeGenerator:
    def __init__(self, num_images: int = 4, size: int = 32):
        self.num_images = num_images
        self.size = size

    def generate(
        self,
        prompt: str = "",
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        seed: int = -1,
        num_images: Optional[int] = None,
    ) -> List[Image.Image]:
        n = num_images or self.num_images
        images = []
        for i in range(n):
            arr = np.zeros((self.size, self.size, 4), dtype=np.uint8)
            arr[:, :, 0] = (i * 40) % 256
            arr[:, :, 3] = 255
            images.append(Image.fromarray(arr, "RGBA"))
        return images


def reset_state():
    tmp = tempfile.mkdtemp()
    set_storage(FileStorage(base_dir=tmp))
    set_library(AssetLibrary(base_dir=os.path.join(tmp, "lib")))
    set_pipeline(AssetPipeline())
    set_generator_loaded(True)
    set_rate_limiter(RateLimiter(max_requests=100, window_seconds=60))
    set_task_queue(TaskQueue(max_workers=4))
    return tmp


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


class TestTilesetPipeline:
    def test_tileset_asset_type_packs_as_tileset(self, tmp_path):
        pipeline = AssetPipeline()
        pipeline.set_generator(FakeGenerator(num_images=4))

        controls = AssetControls(
            asset_type=AssetType.TILESET,
            view=View.TOP,
            animation=Animation.NONE,
            palette=Palette.RETRO_16,
            sprite_size=SpriteSize.S_32,
        )

        result = pipeline.run(controls, output_dir=str(tmp_path))

        assert len(result.images) == 4
        tileset_pngs = [p for p in result.output_paths if p.endswith("_tileset.png")]
        assert len(tileset_pngs) == 1
        tileset_jsons = [p for p in result.output_paths if p.endswith("_tileset.json")]
        assert len(tileset_jsons) == 1

        with open(tileset_jsons[0]) as f:
            tmeta = json.load(f)
        assert tmeta["type"] == "tileset"
        assert tmeta["frame_count"] == 4
        assert "tile_size" in tmeta
        assert "cols" in tmeta
        assert "rows" in tmeta
        assert len(tmeta["frames"]) == 4
        for fr in tmeta["frames"]:
            assert "col" in fr
            assert "row" in fr
            assert "x" in fr
            assert "y" in fr

        sheet = Image.open(tileset_pngs[0])
        assert sheet.mode == "RGBA"
        tw = tmeta["tile_size"]["w"]
        th = tmeta["tile_size"]["h"]
        assert sheet.size[0] >= tw * tmeta["cols"]
        assert sheet.size[1] >= th * tmeta["rows"]

    def test_tileset_metadata_recorded_in_metadata_json(self, tmp_path):
        pipeline = AssetPipeline()
        pipeline.set_generator(FakeGenerator(num_images=4))

        controls = AssetControls(
            asset_type=AssetType.TILESET,
            view=View.TOP,
            animation=Animation.NONE,
        )

        result = pipeline.run(controls, output_dir=str(tmp_path))

        meta_path = os.path.join(str(tmp_path), "metadata.json")
        assert os.path.isfile(meta_path)
        with open(meta_path) as f:
            meta = json.load(f)
        assert "tileset" in meta
        assert meta["tileset"]["type"] == "tileset"
        assert meta["tileset"]["frame_count"] == 4
        assert meta["controls"]["asset_type"] == "tileset"

    def test_pack_tileset_config_flag(self, tmp_path):
        config = PipelineConfig(pack_tileset=True)
        pipeline = AssetPipeline(config=config)
        pipeline.set_generator(FakeGenerator(num_images=4))

        controls = AssetControls(
            asset_type=AssetType.CHARACTER,
            view=View.FRONT,
            animation=Animation.WALK,
        )

        result = pipeline.run(controls, output_dir=str(tmp_path))

        assert any(p.endswith("_tileset.png") for p in result.output_paths)
        assert any(p.endswith("_tileset.json") for p in result.output_paths)

    def test_default_pipeline_does_not_tileset_non_tileset_assets(self, tmp_path):
        pipeline = AssetPipeline()
        pipeline.set_generator(FakeGenerator(num_images=4))

        controls = AssetControls(
            asset_type=AssetType.CHARACTER,
            view=View.FRONT,
            animation=Animation.WALK,
        )

        result = pipeline.run(controls, output_dir=str(tmp_path))

        assert not any(p.endswith("_tileset.png") for p in result.output_paths)
        assert any(p.endswith(".tres") for p in result.output_paths)

    def test_single_tile_image_no_tileset_sheet(self, tmp_path):
        pipeline = AssetPipeline()
        pipeline.set_generator(FakeGenerator(num_images=1))

        controls = AssetControls(
            asset_type=AssetType.TILESET,
            view=View.TOP,
            animation=Animation.NONE,
        )

        result = pipeline.run(controls, output_dir=str(tmp_path))

        assert len(result.images) == 1
        assert not any(p.endswith("_tileset.png") for p in result.output_paths)
        assert any(p.endswith(".png") for p in result.output_paths)

    def test_tileset_outputs_are_in_zip(self, tmp_path):
        pipeline = AssetPipeline()
        pipeline.set_generator(FakeGenerator(num_images=4))

        controls = AssetControls(
            asset_type=AssetType.TILESET,
            view=View.TOP,
            animation=Animation.NONE,
        )

        result = pipeline.run(controls, output_dir=str(tmp_path))

        assert result.zip_path is not None
        import zipfile
        with zipfile.ZipFile(result.zip_path, "r") as zf:
            names = zf.namelist()
        assert any(n.endswith("_tileset.png") for n in names)
        assert any(n.endswith("_tileset.json") for n in names)


class TestTilesetGenerationHash:
    def test_pack_tileset_changes_generation_hash(self):
        controls = AssetControls(asset_type=AssetType.TILESET, view=View.TOP)
        h1 = compute_generation_hash(controls, PipelineConfig(pack_tileset=False))
        h2 = compute_generation_hash(controls, PipelineConfig(pack_tileset=True))
        assert h1 != h2

    def test_same_pack_tileset_same_hash(self):
        controls = AssetControls(asset_type=AssetType.TILESET, view=View.TOP)
        h1 = compute_generation_hash(controls, PipelineConfig(pack_tileset=True))
        h2 = compute_generation_hash(controls, PipelineConfig(pack_tileset=True))
        assert h1 == h2


class TestTilesetAPI:
    @pytest.fixture(autouse=True)
    def reset(self):
        reset_state()

    @pytest.fixture
    def client(self):
        pipe = AssetPipeline()
        pipe.set_generator(FakeGenerator(num_images=4))
        set_pipeline(pipe)
        set_generator_loaded(True)
        return TestClient(app)

    def test_generate_tileset_asset_packs_as_tileset(self, client):
        resp = client.post("/generate", json={
            "asset_type": "tileset",
            "view": "top",
            "animation": "none",
            "palette": "auto",
            "sprite_size": "32x32",
            "num_frames": 4,
        })
        assert resp.status_code == 202
        data = poll_job(client, resp.json()["job_id"])
        assert data["status"] == "done"
        assert any(p.endswith("_tileset.png") for p in data["output_paths"])
        assert any(p.endswith("_tileset.json") for p in data["output_paths"])

    def test_generate_pack_tileset_flag(self, client):
        resp = client.post("/generate", json={
            "asset_type": "character",
            "view": "front",
            "animation": "walk",
            "num_frames": 4,
            "pack_tileset": True,
        })
        assert resp.status_code == 202
        data = poll_job(client, resp.json()["job_id"])
        assert data["status"] == "done"
        assert any(p.endswith("_tileset.json") for p in data["output_paths"])

    def test_generate_non_tileset_without_flag_unaffected(self, client):
        resp = client.post("/generate", json={
            "asset_type": "character",
            "view": "front",
            "animation": "walk",
            "num_frames": 4,
        })
        assert resp.status_code == 202
        data = poll_job(client, resp.json()["job_id"])
        assert data["status"] == "done"
        assert not any(p.endswith("_tileset.png") for p in data["output_paths"])
