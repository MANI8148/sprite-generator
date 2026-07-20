"""Tests for LLM Project Director (roadmap: Explicitly Deferred -> LLM Project Director)."""

import json
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.modules.project_director import ProjectDirector, ProjectPlan, PlanStep
from backend.modules.project_director.director import (
    _detect_asset_type, _detect_view, _detect_animation, _detect_palette,
    _detect_num_frames, _split_request, _extract_theme,
)
from backend.modules.prompt_builder.controls import AssetType, View, Animation, Palette, SpriteSize
from backend.modules.rate_limiter import RateLimiter, set_rate_limiter, get_rate_limiter
from backend.api.routes import (
    set_pipeline, set_generator_loaded, set_storage, set_library,
    set_director, _batch_jobs,
)
from backend.modules.pipeline.orchestrator import AssetPipeline
from backend.modules.storage.file_storage import FileStorage
from backend.modules.storage.asset_library import AssetLibrary


@pytest.fixture(autouse=True)
def test_setup():
    tmp = tempfile.mkdtemp()

    old_limiter = get_rate_limiter()
    limiter = RateLimiter(max_requests=1000, window_seconds=60)
    set_rate_limiter(limiter)

    set_generator_loaded(False)
    set_storage(FileStorage(base_dir=os.path.join(tmp, "storage")))
    set_library(AssetLibrary(base_dir=os.path.join(tmp, "lib")))
    _batch_jobs.clear()
    set_director(ProjectDirector())

    yield

    set_rate_limiter(old_limiter)


class TestParse:
    def test_single_character_default(self):
        director = ProjectDirector()
        plan = director.parse("Create a character")
        assert len(plan.steps) == 1
        assert plan.steps[0].asset_type == AssetType.CHARACTER
        assert plan.steps[0].view == View.FRONT
        assert plan.steps[0].animation == Animation.IDLE

    def test_single_enemy(self):
        plan = ProjectDirector().parse("Generate an enemy sprite")
        assert len(plan.steps) == 1
        assert plan.steps[0].asset_type == AssetType.ENEMY

    def test_character_with_view_and_animation(self):
        plan = ProjectDirector().parse("Create a side-view character running")
        assert len(plan.steps) == 1
        assert plan.steps[0].asset_type == AssetType.CHARACTER
        assert plan.steps[0].view == View.SIDE
        assert plan.steps[0].animation == Animation.RUN

    def test_multi_asset_with_and(self):
        plan = ProjectDirector().parse("Create a character and a tileset")
        assert len(plan.steps) == 2
        assert plan.steps[0].asset_type == AssetType.CHARACTER
        assert plan.steps[1].asset_type == AssetType.TILESET

    def test_multi_asset_with_comma(self):
        plan = ProjectDirector().parse("character, enemy, prop")
        assert len(plan.steps) == 3
        assert plan.steps[0].asset_type == AssetType.CHARACTER
        assert plan.steps[1].asset_type == AssetType.ENEMY
        assert plan.steps[2].asset_type == AssetType.PROP

    def test_palette_detection(self):
        plan = ProjectDirector().parse("Create a character with gameboy palette")
        assert plan.steps[0].palette == Palette.GAMEBOY

    def test_size_detection(self):
        plan = ProjectDirector().parse("Generate a 64x64 character sprite")
        assert plan.steps[0].sprite_size == SpriteSize.S_64

    def test_num_frames_detection(self):
        plan = ProjectDirector().parse("character with 4 frames")
        assert plan.steps[0].num_frames == 4

    def test_fallback_to_default(self):
        plan = ProjectDirector().parse("something completely random without keywords")
        assert len(plan.steps) == 1
        assert plan.steps[0].asset_type == AssetType.CHARACTER

    def test_theme_extraction(self):
        plan = ProjectDirector().parse("Create a forest character")
        assert "forest" in plan.steps[0].theme

    def test_plan_title_single(self):
        plan = ProjectDirector().parse("Create a character")
        assert "character" in plan.title

    def test_plan_title_multi(self):
        plan = ProjectDirector().parse("character and enemy")
        assert "character" in plan.title
        assert "enemy" in plan.title

    def test_to_dict(self):
        step = PlanStep(asset_type=AssetType.CHARACTER, theme="knight")
        d = step.to_dict()
        assert d["asset_type"] == "character"
        assert d["theme"] == "knight"

    def test_plan_to_dict(self):
        plan = ProjectPlan(title="test", steps=[
            PlanStep(asset_type=AssetType.CHARACTER),
            PlanStep(asset_type=AssetType.ENEMY),
        ])
        d = plan.to_dict()
        assert d["title"] == "test"
        assert d["total_steps"] == 2
        assert len(d["steps"]) == 2

    def test_view_detection_isometric(self):
        plan = ProjectDirector().parse("isometric building")
        assert plan.steps[0].view == View.ISOMETRIC

    def test_animation_walking(self):
        plan = ProjectDirector().parse("walking character")
        assert plan.steps[0].animation == Animation.WALK

    def test_palette_retro_16(self):
        plan = ProjectDirector().parse("character with retro_16 palette")
        assert plan.steps[0].palette == Palette.RETRO_16


class TestDirectorHelpers:
    def test_detect_asset_type_character(self):
        assert _detect_asset_type("main character") == AssetType.CHARACTER

    def test_detect_asset_type_enemy(self):
        assert _detect_asset_type("scary monster") == AssetType.ENEMY

    def test_detect_asset_type_default(self):
        assert _detect_asset_type("unknown thing") == AssetType.CHARACTER

    def test_detect_view_side(self):
        assert _detect_view("side view") == View.SIDE

    def test_detect_animation_attack(self):
        assert _detect_animation("attacking") == Animation.ATTACK

    def test_detect_palette_monochrome(self):
        assert _detect_palette("monochrome") == Palette.MONOCHROME

    def test_detect_num_frames_8(self):
        assert _detect_num_frames("8 frames") == 8

    def test_detect_num_frames_default(self):
        assert _detect_num_frames("no number here") == 1

    def test_split_request_and(self):
        parts = _split_request("character and tileset")
        assert len(parts) == 2

    def test_split_request_comma(self):
        parts = _split_request("character, enemy, prop")
        assert len(parts) == 3

    def test_extract_theme_removes_keywords(self):
        theme = _extract_theme("green forest character", AssetType.CHARACTER)
        assert "character" not in theme
        assert "green" in theme
        assert "forest" in theme


class TestProjectDirectorAPI:
    @pytest.fixture
    def loaded_client(self):
        pipe = AssetPipeline()
        from tests.test_api import FakeGenerator
        pipe.set_generator(FakeGenerator(num_images=1))
        set_pipeline(pipe)
        set_generator_loaded(True)
        return TestClient(app)

    def test_plan_endpoint(self):
        resp = TestClient(app).post("/plan", json={
            "request": "Create a character and a tileset",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_steps"] == 2
        assert data["steps"][0]["asset_type"] == "character"
        assert data["steps"][1]["asset_type"] == "tileset"

    def test_plan_requires_generator_for_execute(self):
        resp = TestClient(app).post("/plan/execute", json={
            "request": "Create a character",
        })
        assert resp.status_code == 503

    def test_execute_plan_creates_jobs(self, loaded_client):
        resp = loaded_client.post("/plan/execute", json={
            "request": "Create a character and a tileset",
        })
        assert resp.status_code == 202
        data = resp.json()
        assert data["total"] == 2
        assert len(data["job_ids"]) == 2
        assert data["batch_id"] != ""

    def test_execute_plan_batch_status(self, loaded_client):
        resp = loaded_client.post("/plan/execute", json={
            "request": "Create a character",
        })
        assert resp.status_code == 202
        data = resp.json()
        batch_id = data["batch_id"]

        status_resp = loaded_client.get(f"/batch-status/{batch_id}")
        assert status_resp.status_code == 200
        assert status_resp.json()["total"] == 1

    def test_plan_with_palette(self):
        resp = TestClient(app).post("/plan", json={
            "request": "Create a retro_16 character",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["steps"][0]["palette"] == "retro_16"

    def test_plan_with_frames(self):
        resp = TestClient(app).post("/plan", json={
            "request": "Create a character with 8 frames",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["steps"][0]["num_frames"] == 8
