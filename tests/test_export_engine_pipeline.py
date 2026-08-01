"""Tests that the export engine selection is honored end-to-end.

Roadmap (Phase 3 "More exporters (GameMaker, Phaser)" / AI doc Phase 4
"[ ] Add the remaining exporters: Unity, GameMaker, Phaser"): the selected
export engine must produce engine-specific output for ALL multi-frame export
paths (animation strips, static sprite sheets, and tilesets), not just the
animation-strip path.
"""

import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pytest
from PIL import Image

from backend.modules.pipeline.orchestrator import AssetPipeline, PipelineConfig
from backend.modules.prompt_builder.controls import (
    AssetControls, AssetType, View, Palette, Animation, SpriteSize,
)


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


def _static_controls(animation=Animation.NONE, asset_type=AssetType.CHARACTER):
    return AssetControls(
        asset_type=asset_type,
        view=View.FRONT,
        animation=animation,
        palette=Palette.AUTO,
        sprite_size=SpriteSize.S_32,
    )


class TestPipelineHonorsEngineForStaticSheets:
    def test_static_sheet_default_engine_is_godot(self, tmp_path):
        pipeline = AssetPipeline()
        pipeline.set_generator(FakeGenerator(num_images=3))
        result = pipeline.run(_static_controls(), output_dir=str(tmp_path))
        assert any(p.endswith(".tres") for p in result.output_paths)

    def test_static_sheet_phaser_engine(self, tmp_path):
        config = PipelineConfig(export_engine="phaser")
        pipeline = AssetPipeline(config=config)
        pipeline.set_generator(FakeGenerator(num_images=3))
        result = pipeline.run(_static_controls(), output_dir=str(tmp_path))
        atlas_jsons = [p for p in result.output_paths if p.endswith(".json")]
        assert len(atlas_jsons) >= 1
        phaser_json = [p for p in atlas_jsons if "metadata.json" not in p]
        assert len(phaser_json) == 1
        data = json.loads(Path(phaser_json[0]).read_text())
        assert "frames" in data and "meta" in data

    def test_static_sheet_unity_engine(self, tmp_path):
        config = PipelineConfig(export_engine="unity")
        pipeline = AssetPipeline(config=config)
        pipeline.set_generator(FakeGenerator(num_images=3))
        result = pipeline.run(_static_controls(), output_dir=str(tmp_path))
        assert any(p.endswith(".png.meta") for p in result.output_paths)

    def test_static_sheet_gamemaker_engine(self, tmp_path):
        config = PipelineConfig(export_engine="gamemaker")
        pipeline = AssetPipeline(config=config)
        pipeline.set_generator(FakeGenerator(num_images=3))
        result = pipeline.run(_static_controls(), output_dir=str(tmp_path))
        yy = [p for p in result.output_paths if p.endswith(".yy")]
        assert len(yy) == 1
        data = json.loads(Path(yy[0]).read_text())
        assert len(data["$GMSprite"]["frames"]) == 3

    def test_animation_strip_still_honors_engine(self, tmp_path):
        config = PipelineConfig(export_engine="phaser")
        pipeline = AssetPipeline(config=config)
        pipeline.set_generator(FakeGenerator(num_images=4))
        controls = _static_controls(animation=Animation.WALK)
        result = pipeline.run(controls, output_dir=str(tmp_path))
        atlas_jsons = [p for p in result.output_paths if p.endswith(".json")]
        phaser_json = [p for p in atlas_jsons if "metadata.json" not in p]
        assert len(phaser_json) == 1
        data = json.loads(Path(phaser_json[0]).read_text())
        assert "frames" in data and "meta" in data


class TestPipelineHonorsEngineForTilesets:
    def _run_tileset(self, tmp_path, engine):
        config = PipelineConfig(export_engine=engine)
        pipeline = AssetPipeline(config=config)
        pipeline.set_generator(FakeGenerator(num_images=4))
        controls = AssetControls(
            asset_type=AssetType.TILESET,
            view=View.TOP,
            animation=Animation.NONE,
        )
        return pipeline.run(controls, output_dir=str(tmp_path))

    def test_tileset_keeps_canonical_output(self, tmp_path):
        result = self._run_tileset(tmp_path, "gamemaker")
        assert any(p.endswith("_tileset.png") for p in result.output_paths)
        tileset_json = [p for p in result.output_paths if p.endswith("_tileset.json")]
        assert len(tileset_json) == 1
        with open(tileset_json[0]) as f:
            tmeta = json.load(f)
        assert tmeta["type"] == "tileset"
        assert tmeta["frame_count"] == 4

    def test_tileset_gamemaker_engine_descriptor(self, tmp_path):
        result = self._run_tileset(tmp_path, "gamemaker")
        yy = [p for p in result.output_paths if p.endswith(".yy")]
        assert len(yy) == 1
        data = json.loads(Path(yy[0]).read_text())
        assert len(data["$GMSprite"]["frames"]) == 4

    def test_tileset_phaser_engine_descriptor(self, tmp_path):
        result = self._run_tileset(tmp_path, "phaser")
        atlas_json = [p for p in result.output_paths if p.endswith("_atlas.json")]
        assert len(atlas_json) == 1
        data = json.loads(Path(atlas_json[0]).read_text())
        assert "frames" in data and "meta" in data
        assert len(data["frames"]) == 4

    def test_tileset_unity_engine_descriptor(self, tmp_path):
        result = self._run_tileset(tmp_path, "unity")
        assert any(p.endswith("_atlas.png.meta") for p in result.output_paths)

    def test_tileset_godot_engine_descriptor(self, tmp_path):
        result = self._run_tileset(tmp_path, "godot")
        assert any(p.endswith("_atlas.tres") for p in result.output_paths)


class TestAPIHonorsEngineSelection:
    @pytest.fixture(autouse=True)
    def reset_state(self):
        import tempfile
        from backend.modules.storage.file_storage import FileStorage
        from backend.modules.storage.asset_library import AssetLibrary
        from backend.modules.rate_limiter import RateLimiter, set_rate_limiter
        from backend.modules.tasks.queue import TaskQueue, set_task_queue
        from backend.api.routes import (
            set_pipeline, set_generator_loaded, set_storage, set_library, _batch_jobs,
        )
        tmp = tempfile.mkdtemp()
        set_storage(FileStorage(base_dir=tmp))
        set_library(AssetLibrary(base_dir=os.path.join(tmp, "lib")))
        set_rate_limiter(RateLimiter(max_requests=100, window_seconds=60))
        set_task_queue(TaskQueue(max_workers=4))
        _batch_jobs.clear()

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        from backend.api.routes import set_pipeline, set_generator_loaded
        pipe = AssetPipeline()
        pipe.set_generator(FakeGenerator(num_images=3))
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
        raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")

    def test_generate_static_sheet_with_gamemaker_engine(self, client):
        resp = client.post("/generate", json={
            "asset_type": "character",
            "view": "front",
            "animation": "none",
            "num_frames": 3,
            "engine": "gamemaker",
        })
        assert resp.status_code == 202
        result = self._poll(client, resp.json()["job_id"])
        assert result["status"] == "done"
        assert any(p.endswith(".yy") for p in result["output_paths"])

    def test_generate_static_sheet_with_phaser_engine(self, client):
        resp = client.post("/generate", json={
            "asset_type": "character",
            "view": "front",
            "animation": "none",
            "num_frames": 3,
            "engine": "phaser",
        })
        assert resp.status_code == 202
        result = self._poll(client, resp.json()["job_id"])
        assert result["status"] == "done"
        phaser_jsons = [p for p in result["output_paths"]
                        if p.endswith(".json") and "metadata.json" not in p]
        assert len(phaser_jsons) == 1
