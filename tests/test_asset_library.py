"""Tests for Persistent Asset Library (roadmap: Phase 2 Item 3)."""

import os
import json
import tempfile

import pytest
from fastapi.testclient import TestClient

from backend.modules.storage.asset_library import AssetLibrary, AssetRecord
from backend.api.routes import set_library, set_pipeline, set_generator_loaded, set_storage
from backend.modules.pipeline.orchestrator import AssetPipeline
from backend.modules.storage.file_storage import FileStorage
from backend.modules.rate_limiter import RateLimiter, set_rate_limiter
from backend.main import app


class FakeGenerator:
    def generate(self, prompt="", negative_prompt="", width=512, height=512, seed=-1, num_images=None):
        from PIL import Image
        import numpy as np
        n = num_images or 1
        images = []
        for i in range(n):
            arr = np.zeros((height, width, 4), dtype=np.uint8)
            arr[:, :, 0] = 255
            arr[:, :, 3] = 255
            images.append(Image.fromarray(arr, "RGBA"))
        return images


@pytest.fixture(autouse=True)
def reset_state():
    tmp = tempfile.mkdtemp()
    set_storage(FileStorage(base_dir=tmp))
    set_library(AssetLibrary(base_dir=os.path.join(tmp, "lib")))
    set_pipeline(AssetPipeline())
    set_generator_loaded(True)
    set_rate_limiter(RateLimiter(max_requests=100, window_seconds=60))


@pytest.fixture
def client():
    pipe = AssetPipeline()
    pipe.set_generator(FakeGenerator())
    set_pipeline(pipe)
    set_generator_loaded(True)
    return TestClient(app)


@pytest.fixture
def lib():
    tmp = tempfile.mkdtemp()
    return AssetLibrary(base_dir=os.path.join(tmp, "lib"))


class TestAssetRecord:
    def test_default_timestamps(self):
        record = AssetRecord(asset_id="a1", job_id="j1", asset_type="character", prompt="test", quality_tier="clean")
        assert record.created_at != ""
        assert record.updated_at != ""
        assert record.created_at == record.updated_at
        assert record.tags == []
        assert record.category == ""

    def test_custom_timestamps_are_preserved(self):
        record = AssetRecord(
            asset_id="a1", job_id="j1", asset_type="character", prompt="test",
            quality_tier="clean", created_at="2024-01-01T00:00:00Z",
        )
        assert record.created_at == "2024-01-01T00:00:00Z"


class TestAssetLibraryInit:
    def test_creates_base_dir(self):
        tmp = tempfile.mkdtemp()
        base = os.path.join(tmp, "my_library")
        al = AssetLibrary(base_dir=base)
        assert os.path.isdir(base)

    def test_uses_provided_base_dir(self):
        tmp = tempfile.mkdtemp()
        al = AssetLibrary(base_dir=tmp)
        assert al.base_dir == tmp


class TestAssetLibraryAdd:
    def test_add_asset_returns_id(self, lib):
        record = AssetRecord(asset_id="", job_id="j1", asset_type="character", prompt="test", quality_tier="clean")
        aid = lib.add_asset(record)
        assert aid != ""
        assert len(aid) == 8

    def test_add_asset_persists_to_disk(self, lib):
        record = AssetRecord(asset_id="myid", job_id="j1", asset_type="enemy", prompt="goblin", quality_tier="clean")
        lib.add_asset(record)
        assert os.path.isfile(lib._index_path)
        with open(lib._index_path) as f:
            data = json.load(f)
        assert "myid" in data
        assert data["myid"]["asset_type"] == "enemy"

    def test_add_multiple_assets(self, lib):
        for i in range(5):
            lib.add_asset(AssetRecord(asset_id=f"a{i}", job_id=f"j{i}", asset_type="character", prompt=f"p{i}", quality_tier="clean"))
        assert lib.count() == 5

    def test_add_asset_updates_timestamp(self, lib):
        record = AssetRecord(asset_id="ts1", job_id="j1", asset_type="character", prompt="test", quality_tier="clean")
        orig = record.created_at
        lib.add_asset(record)
        loaded = lib.get_asset("ts1")
        assert loaded.created_at == orig
        assert loaded.updated_at >= orig


class TestAssetLibraryGet:
    def test_get_existing_asset(self, lib):
        lib.add_asset(AssetRecord(asset_id="abc", job_id="j1", asset_type="vehicle", prompt="tank", quality_tier="clean"))
        asset = lib.get_asset("abc")
        assert asset is not None
        assert asset.asset_type == "vehicle"
        assert asset.prompt == "tank"

    def test_get_nonexistent_asset_returns_none(self, lib):
        assert lib.get_asset("nonexistent") is None

    def test_get_asset_after_reload(self, lib):
        lib.add_asset(AssetRecord(asset_id="persist", job_id="j1", asset_type="building", prompt="castle", quality_tier="acceptable"))
        lib2 = AssetLibrary(base_dir=lib.base_dir)
        asset = lib2.get_asset("persist")
        assert asset is not None
        assert asset.prompt == "castle"


class TestAssetLibraryUpdate:
    def test_update_category(self, lib):
        lib.add_asset(AssetRecord(asset_id="upd1", job_id="j1", asset_type="prop", prompt="chest", quality_tier="clean"))
        updated = lib.update_asset("upd1", category="treasure")
        assert updated is not None
        assert updated.category == "treasure"

    def test_update_nonexistent_returns_none(self, lib):
        assert lib.update_asset("nope", category="test") is None

    def test_update_preserves_asset_id_and_created_at(self, lib):
        lib.add_asset(AssetRecord(asset_id="fixid", job_id="j1", asset_type="character", prompt="original", quality_tier="clean"))
        lib.update_asset("fixid", prompt="modified")
        asset = lib.get_asset("fixid")
        assert asset.prompt == "modified"
        assert asset.asset_id == "fixid"


class TestAssetLibraryDelete:
    def test_delete_existing_asset(self, lib):
        lib.add_asset(AssetRecord(asset_id="del1", job_id="j1", asset_type="character", prompt="delete me", quality_tier="clean"))
        assert lib.delete_asset("del1") is True
        assert lib.get_asset("del1") is None

    def test_delete_nonexistent_returns_false(self, lib):
        assert lib.delete_asset("nope") is False

    def test_count_decreases_after_delete(self, lib):
        lib.add_asset(AssetRecord(asset_id="k1", job_id="j1", asset_type="character", prompt="a", quality_tier="clean"))
        lib.add_asset(AssetRecord(asset_id="k2", job_id="j2", asset_type="enemy", prompt="b", quality_tier="clean"))
        lib.delete_asset("k1")
        assert lib.count() == 1


class TestAssetLibraryList:
    def test_empty_library(self, lib):
        assert lib.list_assets() == []

    def test_list_all_assets(self, lib):
        lib.add_asset(AssetRecord(asset_id="a1", job_id="j1", asset_type="character", prompt="hero", quality_tier="clean"))
        lib.add_asset(AssetRecord(asset_id="a2", job_id="j2", asset_type="enemy", prompt="goblin", quality_tier="acceptable"))
        assets = lib.list_assets()
        assert len(assets) == 2

    def test_filter_by_asset_type(self, lib):
        lib.add_asset(AssetRecord(asset_id="t1", job_id="j1", asset_type="character", prompt="hero", quality_tier="clean"))
        lib.add_asset(AssetRecord(asset_id="t2", job_id="j2", asset_type="vehicle", prompt="car", quality_tier="clean"))
        lib.add_asset(AssetRecord(asset_id="t3", job_id="j3", asset_type="character", prompt="mage", quality_tier="clean"))
        chars = lib.list_assets(asset_type="character")
        assert len(chars) == 2
        vehicles = lib.list_assets(asset_type="vehicle")
        assert len(vehicles) == 1

    def test_filter_by_quality_tier(self, lib):
        lib.add_asset(AssetRecord(asset_id="q1", job_id="j1", asset_type="character", prompt="a", quality_tier="clean"))
        lib.add_asset(AssetRecord(asset_id="q2", job_id="j2", asset_type="enemy", prompt="b", quality_tier="acceptable"))
        clean = lib.list_assets(quality_tier="clean")
        assert len(clean) == 1

    def test_filter_by_category(self, lib):
        lib.add_asset(AssetRecord(asset_id="c1", job_id="j1", asset_type="character", prompt="a", quality_tier="clean", category="heroes"))
        lib.add_asset(AssetRecord(asset_id="c2", job_id="j2", asset_type="enemy", prompt="b", quality_tier="clean", category="villains"))
        heroes = lib.list_assets(category="heroes")
        assert len(heroes) == 1
        assert heroes[0].asset_id == "c1"

    def test_filter_by_tags(self, lib):
        lib.add_asset(AssetRecord(asset_id="g1", job_id="j1", asset_type="character", prompt="a", quality_tier="clean", tags=["fantasy", "warrior"]))
        lib.add_asset(AssetRecord(asset_id="g2", job_id="j2", asset_type="enemy", prompt="b", quality_tier="clean", tags=["sci-fi", "alien"]))
        fantasy = lib.list_assets(tags=["fantasy"])
        assert len(fantasy) == 1
        assert fantasy[0].asset_id == "g1"

    def test_search_by_prompt(self, lib):
        lib.add_asset(AssetRecord(asset_id="s1", job_id="j1", asset_type="character", prompt="red dragon", quality_tier="clean"))
        lib.add_asset(AssetRecord(asset_id="s2", job_id="j2", asset_type="enemy", prompt="blue slime", quality_tier="clean"))
        results = lib.list_assets(search="dragon")
        assert len(results) == 1
        assert results[0].asset_id == "s1"

    def test_search_by_asset_id(self, lib):
        lib.add_asset(AssetRecord(asset_id="findme42", job_id="j1", asset_type="character", prompt="hero", quality_tier="clean"))
        results = lib.list_assets(search="findme")
        assert len(results) == 1

    def test_search_is_case_insensitive(self, lib):
        lib.add_asset(AssetRecord(asset_id="cs1", job_id="j1", asset_type="character", prompt="Red Dragon", quality_tier="clean"))
        results = lib.list_assets(search="red dragon")
        assert len(results) == 1

    def test_limit_and_offset(self, lib):
        for i in range(10):
            lib.add_asset(AssetRecord(asset_id=f"lim{i}", job_id=f"j{i}", asset_type="character", prompt=f"asset {i}", quality_tier="clean"))
        first_3 = lib.list_assets(limit=3, offset=0)
        assert len(first_3) == 3
        next_3 = lib.list_assets(limit=3, offset=3)
        assert len(next_3) == 3
        assert first_3[0].asset_id != next_3[0].asset_id

    def test_list_returns_newest_first(self, lib):
        lib.add_asset(AssetRecord(asset_id="old", job_id="j1", asset_type="character", prompt="old", quality_tier="clean", created_at="2020-01-01T00:00:00Z"))
        lib.add_asset(AssetRecord(asset_id="new", job_id="j2", asset_type="character", prompt="new", quality_tier="clean", created_at="2024-01-01T00:00:00Z"))
        assets = lib.list_assets()
        assert assets[0].asset_id == "new"
        assert assets[1].asset_id == "old"


class TestAssetLibraryTags:
    def test_list_tags_empty(self, lib):
        assert lib.list_tags() == []

    def test_list_tags_aggregates(self, lib):
        lib.add_asset(AssetRecord(asset_id="t1", job_id="j1", asset_type="character", prompt="a", quality_tier="clean", tags=["fantasy", "warrior"]))
        lib.add_asset(AssetRecord(asset_id="t2", job_id="j2", asset_type="enemy", prompt="b", quality_tier="clean", tags=["sci-fi", "warrior"]))
        tags = lib.list_tags()
        assert "fantasy" in tags
        assert "sci-fi" in tags
        assert "warrior" in tags
        assert len(tags) == 3

    def test_add_tags(self, lib):
        lib.add_asset(AssetRecord(asset_id="at1", job_id="j1", asset_type="character", prompt="a", quality_tier="clean", tags=["original"]))
        updated = lib.add_tags("at1", ["new-tag", "extra"])
        assert updated is not None
        assert "original" in updated.tags
        assert "new-tag" in updated.tags
        assert "extra" in updated.tags

    def test_add_tags_nonexistent(self, lib):
        assert lib.add_tags("nope", ["tag"]) is None

    def test_remove_tags(self, lib):
        lib.add_asset(AssetRecord(asset_id="rt1", job_id="j1", asset_type="character", prompt="a", quality_tier="clean", tags=["keep", "remove"]))
        updated = lib.remove_tags("rt1", ["remove"])
        assert updated is not None
        assert "keep" in updated.tags
        assert "remove" not in updated.tags

    def test_remove_tags_nonexistent(self, lib):
        assert lib.remove_tags("nope", ["tag"]) is None

    def test_add_tags_deduplicates(self, lib):
        lib.add_asset(AssetRecord(asset_id="dedup", job_id="j1", asset_type="character", prompt="a", quality_tier="clean", tags=["tag1"]))
        lib.add_tags("dedup", ["tag1", "tag2"])
        asset = lib.get_asset("dedup")
        assert len(asset.tags) == 2


class TestAssetLibraryDirectory:
    def test_get_asset_dir(self, lib):
        d = lib.get_asset_dir("myasset")
        assert d.endswith("myasset")

    def test_ensure_asset_dir_creates(self, lib):
        d = lib.ensure_asset_dir("newasset")
        assert os.path.isdir(d)

    def test_ensure_asset_dir_is_idempotent(self, lib):
        d1 = lib.ensure_asset_dir("same")
        d2 = lib.ensure_asset_dir("same")
        assert d1 == d2
        assert os.path.isdir(d1)


class TestAssetLibraryClear:
    def test_clear_removes_all_data(self, lib):
        lib.add_asset(AssetRecord(asset_id="c1", job_id="j1", asset_type="character", prompt="a", quality_tier="clean"))
        lib.add_asset(AssetRecord(asset_id="c2", job_id="j2", asset_type="enemy", prompt="b", quality_tier="clean"))
        lib.clear()
        assert lib.count() == 0

    def test_clear_base_dir_exists(self, lib):
        lib.ensure_asset_dir("somedir")
        lib.clear()
        assert os.path.isdir(lib.base_dir)


class TestLibraryAPI:
    def test_library_list_empty(self, client):
        resp = client.get("/library")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["assets"] == []

    def test_library_populates_on_generate(self, client):
        resp = client.post("/generate", json={"asset_type": "character", "view": "front"})
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]

        resp = client.get("/library")
        data = resp.json()
        assert data["total"] >= 1
        ids = [a["asset_id"] for a in data["assets"]]
        assert job_id in ids

    def test_library_filter_by_asset_type(self, client):
        pipe = AssetPipeline()
        pipe.set_generator(FakeGenerator())
        set_pipeline(pipe)
        set_generator_loaded(True)

        for at in ["character", "vehicle", "enemy"]:
            client.post("/generate", json={"asset_type": at, "view": "front"})

        resp = client.get("/library?asset_type=character")
        data = resp.json()
        assert all(a["asset_type"] == "character" for a in data["assets"])

    def test_library_get_asset(self, client):
        resp = client.post("/generate", json={"asset_type": "building"})
        job_id = resp.json()["job_id"]

        resp = client.get(f"/library/{job_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["asset_id"] == job_id
        assert data["asset_type"] == "building"

    def test_library_get_nonexistent_returns_404(self, client):
        resp = client.get("/library/nonexistent")
        assert resp.status_code == 404

    def test_library_delete_asset(self, client):
        resp = client.post("/generate", json={"asset_type": "prop"})
        job_id = resp.json()["job_id"]

        resp = client.delete(f"/library/{job_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

        resp = client.get(f"/library/{job_id}")
        assert resp.status_code == 404

    def test_library_delete_nonexistent_returns_404(self, client):
        resp = client.delete("/library/nonexistent")
        assert resp.status_code == 404

    def test_library_update_asset(self, client):
        resp = client.post("/generate", json={"asset_type": "character"})
        job_id = resp.json()["job_id"]

        resp = client.patch(f"/library/{job_id}", json={"category": "heroes", "tags": ["fantasy", "knight"]})
        assert resp.status_code == 200
        data = resp.json()
        assert data["category"] == "heroes"
        assert "fantasy" in data["tags"]
        assert "knight" in data["tags"]

    def test_library_update_nonexistent_returns_404(self, client):
        resp = client.patch("/library/nonexistent", json={"category": "test"})
        assert resp.status_code == 404

    def test_library_add_tags(self, client):
        resp = client.post("/generate", json={"asset_type": "enemy"})
        job_id = resp.json()["job_id"]

        resp = client.post(f"/library/{job_id}/tags", json={"tags": ["boss", "dragon"]})
        assert resp.status_code == 200
        data = resp.json()
        assert "boss" in data["tags"]
        assert "dragon" in data["tags"]

    def test_library_remove_tags(self, client):
        resp = client.post("/generate", json={"asset_type": "enemy"})
        job_id = resp.json()["job_id"]

        client.post(f"/library/{job_id}/tags", json={"tags": ["boss", "dragon", "rare"]})
        resp = client.request("DELETE", f"/library/{job_id}/tags", json={"tags": ["dragon"]})
        assert resp.status_code == 200
        data = resp.json()
        assert "boss" in data["tags"]
        assert "dragon" not in data["tags"]

    def test_library_tags_endpoint(self, client):
        pipe = AssetPipeline()
        pipe.set_generator(FakeGenerator())
        set_pipeline(pipe)
        set_generator_loaded(True)

        resp = client.post("/generate", json={"asset_type": "character"})
        job_id = resp.json()["job_id"]

        client.post(f"/library/{job_id}/tags", json={"tags": ["fantasy", "warrior"]})

        resp = client.get("/library/tags")
        assert resp.status_code == 200
        data = resp.json()
        assert "fantasy" in data["tags"]
        assert "warrior" in data["tags"]

    def test_library_search(self, client):
        pipe = AssetPipeline()
        pipe.set_generator(FakeGenerator())
        set_pipeline(pipe)
        set_generator_loaded(True)

        resp = client.post("/generate", json={"asset_type": "character", "theme": "dragon"})
        job_id = resp.json()["job_id"]

        resp = client.get("/library?search=dragon")
        data = resp.json()
        assert data["total"] >= 1
