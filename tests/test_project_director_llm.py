"""Tests for LLM Project Director (roadmap: Explicitly Deferred -> LLM Project Director).

Verifies the LLM-backed director: structured plan generation from an OpenAI-compatible
chat completion response, robust coercion of model output, graceful fallback to the
rule-based parser, and wiring through the FastAPI /plan endpoint.
"""

import json
import os
import tempfile
from typing import Optional

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.modules.project_director import ProjectDirector, LLMProjectDirector
from backend.modules.project_director.llm import (
    _coerce_enum, _coerce_int, _plan_from_llm_dict, SYSTEM_PROMPT,
)
from backend.modules.project_director.director import PlanStep
from backend.modules.prompt_builder.controls import AssetType, View, Animation, Palette, SpriteSize
from backend.modules.rate_limiter import RateLimiter, set_rate_limiter, get_rate_limiter
from backend.api.routes import (
    set_director, set_pipeline, set_generator_loaded, set_storage, set_library,
    _batch_jobs,
)
from backend.modules.pipeline.orchestrator import AssetPipeline
from backend.modules.storage.file_storage import FileStorage
from backend.modules.storage.asset_library import AssetLibrary
from tests.test_api import FakeGenerator


class FakeHTTPClient:
    """Stand-in for httpx.Client returning a canned chat completion response."""

    def __init__(self, content: str = None, status_code: int = 200, error: Optional[Exception] = None):
        self.content = content if content is not None else json.dumps({
            "title": "Forest hero and tileset",
            "steps": [
                {
                    "asset_type": "character",
                    "view": "side",
                    "animation": "run",
                    "palette": "gameboy",
                    "sprite_size": "64x64",
                    "theme": "forest",
                    "num_frames": 4,
                    "seed": -1,
                },
                {
                    "asset_type": "tileset",
                    "view": "top",
                    "animation": "none",
                    "palette": "retro_8",
                    "sprite_size": "32x32",
                    "theme": "forest",
                    "num_frames": 1,
                    "seed": 7,
                },
            ],
        })
        self.status_code = status_code
        self.error = error
        self.last_payload = None
        self.calls = 0

    def post(self, url: str, headers=None, json=None):
        self.calls += 1
        self.last_payload = json
        if self.error is not None:
            raise self.error
        return FakeHTTPResponse(self.content, self.status_code)


class FakeHTTPResponse:
    def __init__(self, content: str, status_code: int):
        self.content = content
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return {
            "choices": [{"message": {"content": self.content}}],
        }


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


def make_director(client: FakeHTTPClient, api_key: str = "test-key") -> LLMProjectDirector:
    return LLMProjectDirector(api_key=api_key, http_client=client)


class TestLLMProjectDirector:
    def test_parse_uses_llm_response(self):
        client = FakeHTTPClient()
        director = make_director(client)
        plan = director.parse("Create a forest hero running and a tileset")

        assert len(plan.steps) == 2
        step = plan.steps[0]
        assert step.asset_type == AssetType.CHARACTER
        assert step.view == View.SIDE
        assert step.animation == Animation.RUN
        assert step.palette == Palette.GAMEBOY
        assert step.sprite_size == SpriteSize.S_64
        assert step.theme == "forest"
        assert step.num_frames == 4
        assert plan.steps[1].asset_type == AssetType.TILESET
        assert plan.steps[1].palette == Palette.RETRO_8
        assert plan.steps[1].seed == 7
        assert client.calls == 1

    def test_parse_posts_chat_completion_payload(self):
        client = FakeHTTPClient()
        director = make_director(client)
        director.parse("Create a character")

        assert client.calls == 1
        payload = client.last_payload
        assert payload["model"] == director.model
        assert payload["temperature"] == 0.0
        assert payload["response_format"] == {"type": "json_object"}
        assert payload["messages"][0]["role"] == "system"
        assert SYSTEM_PROMPT in payload["messages"][0]["content"]
        assert payload["messages"][1]["role"] == "user"
        assert payload["messages"][1]["content"] == "Create a character"

    def test_parse_sends_auth_header(self):
        class CapturingClient(FakeHTTPClient):
            def post(self, url, headers=None, json=None):
                self.headers = headers
                return super().post(url, headers=headers, json=json)

        client = CapturingClient()
        director = make_director(client, api_key="secret-123")
        director.parse("Create a character")

        assert client.headers["Authorization"] == "Bearer secret-123"

    def test_llm_enabled_only_with_api_key(self):
        no_key = LLMProjectDirector(api_key=None, http_client=FakeHTTPClient())
        assert not no_key.llm_enabled
        with_key = LLMProjectDirector(api_key="k", http_client=FakeHTTPClient())
        assert with_key.llm_enabled

    def test_fallback_when_no_api_key(self):
        client = FakeHTTPClient()
        director = LLMProjectDirector(api_key=None, http_client=client)
        plan = director.parse("Create a character and a tileset")

        assert client.calls == 0
        assert plan.steps[0].asset_type == AssetType.CHARACTER
        assert plan.steps[1].asset_type == AssetType.TILESET

    def test_fallback_when_http_error(self):
        client = FakeHTTPClient(content="{}", status_code=500)
        director = make_director(client)
        plan = director.parse("Create an enemy sprite")

        assert plan.steps[0].asset_type == AssetType.ENEMY

    def test_fallback_when_network_error(self):
        client = FakeHTTPClient(error=RuntimeError("connection refused"))
        director = make_director(client)
        plan = director.parse("Create an enemy sprite")

        assert plan.steps[0].asset_type == AssetType.ENEMY

    def test_fallback_when_invalid_json(self):
        client = FakeHTTPClient(content="not json at all")
        director = make_director(client)
        plan = director.parse("Create an enemy sprite")

        assert plan.steps[0].asset_type == AssetType.ENEMY

    def test_fallback_when_no_steps(self):
        client = FakeHTTPClient(content=json.dumps({"title": "x", "steps": []}))
        director = make_director(client)
        plan = director.parse("Create an enemy sprite")

        assert plan.steps[0].asset_type == AssetType.ENEMY

    def test_plan_has_llm_title(self):
        client = FakeHTTPClient()
        director = make_director(client)
        plan = director.parse("Create a forest hero")

        assert plan.title == "Forest hero and tileset"

    def test_title_fallback_when_missing(self):
        client = FakeHTTPClient(content=json.dumps({
            "steps": [{"asset_type": "character"}],
        }))
        director = make_director(client)
        plan = director.parse("Create a character")

        assert "character" in plan.title

    def test_missing_fields_coerced_to_defaults(self):
        client = FakeHTTPClient(content=json.dumps({
            "title": "minimal",
            "steps": [{"asset_type": "character"}],
        }))
        director = make_director(client)
        plan = director.parse("anything")

        step = plan.steps[0]
        assert step.view == View.FRONT
        assert step.animation == Animation.IDLE
        assert step.palette == Palette.AUTO
        assert step.sprite_size == SpriteSize.S_32
        assert step.num_frames == 1
        assert step.theme == ""
        assert step.seed == -1

    def test_invalid_enum_values_fall_back_to_defaults(self):
        client = FakeHTTPClient(content=json.dumps({
            "title": "bad enums",
            "steps": [
                {
                    "asset_type": "not-a-real-type",
                    "view": "diagonal",
                    "palette": "rainbow",
                    "sprite_size": "999x999",
                    "num_frames": "lots",
                }
            ],
        }))
        director = make_director(client)
        plan = director.parse("Create something weird")

        step = plan.steps[0]
        assert step.asset_type == AssetType.CHARACTER
        assert step.view == View.FRONT
        assert step.palette == Palette.AUTO
        assert step.sprite_size == SpriteSize.S_32
        assert step.num_frames == 1

    def test_num_frames_clamped_positive(self):
        plan_dict = {
            "steps": [{"asset_type": "character", "num_frames": 0}],
        }
        plan = _plan_from_llm_dict(plan_dict, "x")
        assert plan.steps[0].num_frames == 1

    def test_seed_default(self):
        plan_dict = {"steps": [{"asset_type": "character"}]}
        plan = _plan_from_llm_dict(plan_dict, "x")
        assert plan.steps[0].seed == -1


class TestCoercionHelpers:
    def test_coerce_enum_value(self):
        assert _coerce_enum(AssetType, "tileset", AssetType.CHARACTER) == AssetType.TILESET

    def test_coerce_enum_member_name(self):
        assert _coerce_enum(View, "SIDE", View.FRONT) == View.SIDE

    def test_coerce_enum_case_insensitive(self):
        assert _coerce_enum(Palette, "GameBoy", Palette.AUTO) == Palette.GAMEBOY

    def test_coerce_enum_invalid(self):
        assert _coerce_enum(Animation, "flying", Animation.IDLE) == Animation.IDLE

    def test_coerce_enum_none(self):
        assert _coerce_enum(AssetType, None, AssetType.CHARACTER) == AssetType.CHARACTER

    def test_coerce_int_valid(self):
        assert _coerce_int("8", 1) == 8

    def test_coerce_int_invalid(self):
        assert _coerce_int("nope", 5) == 5

    def test_coerce_int_min_value(self):
        assert _coerce_int("-3", 1, min_value=1) == 1

    def test_coerce_int_none(self):
        assert _coerce_int(None, -1) == -1


class TestLLMProjectDirectorAPI:
    @pytest.fixture
    def loaded_client(self):
        pipe = AssetPipeline()
        pipe.set_generator(FakeGenerator(num_images=1))
        set_pipeline(pipe)
        set_generator_loaded(True)
        return TestClient(app)

    def test_plan_endpoint_uses_llm_director(self, loaded_client):
        client = FakeHTTPClient()
        set_director(make_director(client))

        resp = loaded_client.post("/plan", json={
            "request": "Create a forest hero running and a tileset",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_steps"] == 2
        assert data["steps"][0]["asset_type"] == "character"
        assert data["steps"][0]["animation"] == "run"
        assert data["steps"][1]["asset_type"] == "tileset"
        assert data["steps"][1]["seed"] == 7

    def test_plan_endpoint_llm_output_normalized(self, loaded_client):
        client = FakeHTTPClient(content=json.dumps({
            "title": "bad values",
            "steps": [{"asset_type": "bogus", "view": "diagonal"}],
        }))
        set_director(make_director(client))

        resp = loaded_client.post("/plan", json={"request": "weird request"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["steps"][0]["asset_type"] == "character"
        assert data["steps"][0]["view"] == "front"

    def test_plan_endpoint_fallback_without_llm(self, loaded_client):
        set_director(ProjectDirector())

        resp = loaded_client.post("/plan", json={"request": "Create a character and a tileset"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["steps"][0]["asset_type"] == "character"
        assert data["steps"][1]["asset_type"] == "tileset"
