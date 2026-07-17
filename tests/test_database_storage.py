import os
import tempfile

import pytest

from backend.modules.storage.database import DatabaseLibrary
from backend.modules.storage.asset_library import AssetRecord


@pytest.fixture
def db():
    tmp = tempfile.mkdtemp()
    db_path = os.path.join(tmp, "test.db")
    return DatabaseLibrary(db_path=db_path)


class TestDatabaseLibraryInit:
    def test_creates_db_file(self):
        tmp = tempfile.mkdtemp()
        db_path = os.path.join(tmp, "library.db")
        dbl = DatabaseLibrary(db_path=db_path)
        assert os.path.isfile(db_path)

    def test_creates_tables(self):
        tmp = tempfile.mkdtemp()
        db_path = os.path.join(tmp, "test.db")
        dbl = DatabaseLibrary(db_path=db_path)
        import sqlite3
        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = [row[0] for row in cursor.fetchall()]
            assert "assets" in tables
            assert "tags" in tables
        finally:
            conn.close()


class TestDatabaseLibraryAdd:
    def test_add_asset_returns_id(self, db):
        record = AssetRecord(asset_id="", job_id="j1", asset_type="character", prompt="test", quality_tier="clean")
        aid = db.add_asset(record)
        assert aid != ""
        assert len(aid) == 8

    def test_add_asset_persists(self, db):
        record = AssetRecord(asset_id="myid", job_id="j1", asset_type="enemy", prompt="goblin", quality_tier="clean")
        db.add_asset(record)
        loaded = db.get_asset("myid")
        assert loaded is not None
        assert loaded.asset_type == "enemy"
        assert loaded.prompt == "goblin"

    def test_add_multiple_assets(self, db):
        for i in range(5):
            db.add_asset(AssetRecord(asset_id=f"a{i}", job_id=f"j{i}", asset_type="character", prompt=f"p{i}", quality_tier="clean"))
        assert db.count() == 5

    def test_add_asset_updates_timestamp(self, db):
        record = AssetRecord(asset_id="ts1", job_id="j1", asset_type="character", prompt="test", quality_tier="clean")
        orig = record.created_at
        db.add_asset(record)
        loaded = db.get_asset("ts1")
        assert loaded.created_at == orig
        assert loaded.updated_at >= orig


class TestDatabaseLibraryGet:
    def test_get_existing_asset(self, db):
        db.add_asset(AssetRecord(asset_id="abc", job_id="j1", asset_type="vehicle", prompt="tank", quality_tier="clean"))
        asset = db.get_asset("abc")
        assert asset is not None
        assert asset.asset_type == "vehicle"
        assert asset.prompt == "tank"

    def test_get_nonexistent_asset_returns_none(self, db):
        assert db.get_asset("nonexistent") is None

    def test_get_asset_after_reload(self, db):
        db.add_asset(AssetRecord(asset_id="persist", job_id="j1", asset_type="building", prompt="castle", quality_tier="acceptable"))
        db2 = DatabaseLibrary(db_path=db.db_path)
        asset = db2.get_asset("persist")
        assert asset is not None
        assert asset.prompt == "castle"


class TestDatabaseLibraryUpdate:
    def test_update_category(self, db):
        db.add_asset(AssetRecord(asset_id="upd1", job_id="j1", asset_type="prop", prompt="chest", quality_tier="clean"))
        updated = db.update_asset("upd1", category="treasure")
        assert updated is not None
        assert updated.category == "treasure"

    def test_update_nonexistent_returns_none(self, db):
        assert db.update_asset("nope", category="test") is None

    def test_update_preserves_asset_id_and_created_at(self, db):
        db.add_asset(AssetRecord(asset_id="fixid", job_id="j1", asset_type="character", prompt="original", quality_tier="clean"))
        db.update_asset("fixid", prompt="modified")
        asset = db.get_asset("fixid")
        assert asset.prompt == "modified"
        assert asset.asset_id == "fixid"


class TestDatabaseLibraryDelete:
    def test_delete_existing_asset(self, db):
        db.add_asset(AssetRecord(asset_id="del1", job_id="j1", asset_type="character", prompt="delete me", quality_tier="clean"))
        assert db.delete_asset("del1") is True
        assert db.get_asset("del1") is None

    def test_delete_nonexistent_returns_false(self, db):
        assert db.delete_asset("nope") is False

    def test_count_decreases_after_delete(self, db):
        db.add_asset(AssetRecord(asset_id="k1", job_id="j1", asset_type="character", prompt="a", quality_tier="clean"))
        db.add_asset(AssetRecord(asset_id="k2", job_id="j2", asset_type="enemy", prompt="b", quality_tier="clean"))
        db.delete_asset("k1")
        assert db.count() == 1


class TestDatabaseLibraryList:
    def test_empty_library(self, db):
        assert db.list_assets() == []

    def test_list_all_assets(self, db):
        db.add_asset(AssetRecord(asset_id="a1", job_id="j1", asset_type="character", prompt="hero", quality_tier="clean"))
        db.add_asset(AssetRecord(asset_id="a2", job_id="j2", asset_type="enemy", prompt="goblin", quality_tier="acceptable"))
        assets = db.list_assets()
        assert len(assets) == 2

    def test_filter_by_asset_type(self, db):
        db.add_asset(AssetRecord(asset_id="t1", job_id="j1", asset_type="character", prompt="hero", quality_tier="clean"))
        db.add_asset(AssetRecord(asset_id="t2", job_id="j2", asset_type="vehicle", prompt="car", quality_tier="clean"))
        db.add_asset(AssetRecord(asset_id="t3", job_id="j3", asset_type="character", prompt="mage", quality_tier="clean"))
        chars = db.list_assets(asset_type="character")
        assert len(chars) == 2
        vehicles = db.list_assets(asset_type="vehicle")
        assert len(vehicles) == 1

    def test_filter_by_quality_tier(self, db):
        db.add_asset(AssetRecord(asset_id="q1", job_id="j1", asset_type="character", prompt="a", quality_tier="clean"))
        db.add_asset(AssetRecord(asset_id="q2", job_id="j2", asset_type="enemy", prompt="b", quality_tier="acceptable"))
        clean = db.list_assets(quality_tier="clean")
        assert len(clean) == 1

    def test_filter_by_category(self, db):
        db.add_asset(AssetRecord(asset_id="c1", job_id="j1", asset_type="character", prompt="a", quality_tier="clean", category="heroes"))
        db.add_asset(AssetRecord(asset_id="c2", job_id="j2", asset_type="enemy", prompt="b", quality_tier="clean", category="villains"))
        heroes = db.list_assets(category="heroes")
        assert len(heroes) == 1
        assert heroes[0].asset_id == "c1"

    def test_filter_by_tags(self, db):
        db.add_asset(AssetRecord(asset_id="g1", job_id="j1", asset_type="character", prompt="a", quality_tier="clean", tags=["fantasy", "warrior"]))
        db.add_asset(AssetRecord(asset_id="g2", job_id="j2", asset_type="enemy", prompt="b", quality_tier="clean", tags=["sci-fi", "alien"]))
        fantasy = db.list_assets(tags=["fantasy"])
        assert len(fantasy) == 1
        assert fantasy[0].asset_id == "g1"

    def test_search_by_prompt(self, db):
        db.add_asset(AssetRecord(asset_id="s1", job_id="j1", asset_type="character", prompt="red dragon", quality_tier="clean"))
        db.add_asset(AssetRecord(asset_id="s2", job_id="j2", asset_type="enemy", prompt="blue slime", quality_tier="clean"))
        results = db.list_assets(search="dragon")
        assert len(results) == 1
        assert results[0].asset_id == "s1"

    def test_search_by_asset_id(self, db):
        db.add_asset(AssetRecord(asset_id="findme42", job_id="j1", asset_type="character", prompt="hero", quality_tier="clean"))
        results = db.list_assets(search="findme")
        assert len(results) == 1

    def test_search_is_case_insensitive(self, db):
        db.add_asset(AssetRecord(asset_id="cs1", job_id="j1", asset_type="character", prompt="Red Dragon", quality_tier="clean"))
        results = db.list_assets(search="red dragon")
        assert len(results) == 1

    def test_limit_and_offset(self, db):
        for i in range(10):
            db.add_asset(AssetRecord(asset_id=f"lim{i}", job_id=f"j{i}", asset_type="character", prompt=f"asset {i}", quality_tier="clean"))
        first_3 = db.list_assets(limit=3, offset=0)
        assert len(first_3) == 3
        next_3 = db.list_assets(limit=3, offset=3)
        assert len(next_3) == 3
        assert first_3[0].asset_id != next_3[0].asset_id

    def test_list_returns_newest_first(self, db):
        db.add_asset(AssetRecord(asset_id="old", job_id="j1", asset_type="character", prompt="old", quality_tier="clean", created_at="2020-01-01T00:00:00Z"))
        db.add_asset(AssetRecord(asset_id="new", job_id="j2", asset_type="character", prompt="new", quality_tier="clean", created_at="2024-01-01T00:00:00Z"))
        assets = db.list_assets()
        assert assets[0].asset_id == "new"
        assert assets[1].asset_id == "old"


class TestDatabaseLibraryTags:
    def test_list_tags_empty(self, db):
        assert db.list_tags() == []

    def test_list_tags_aggregates(self, db):
        db.add_asset(AssetRecord(asset_id="t1", job_id="j1", asset_type="character", prompt="a", quality_tier="clean", tags=["fantasy", "warrior"]))
        db.add_asset(AssetRecord(asset_id="t2", job_id="j2", asset_type="enemy", prompt="b", quality_tier="clean", tags=["sci-fi", "warrior"]))
        tags = db.list_tags()
        assert "fantasy" in tags
        assert "sci-fi" in tags
        assert "warrior" in tags
        assert len(tags) == 3

    def test_add_tags(self, db):
        db.add_asset(AssetRecord(asset_id="at1", job_id="j1", asset_type="character", prompt="a", quality_tier="clean", tags=["original"]))
        updated = db.add_tags("at1", ["new-tag", "extra"])
        assert updated is not None
        assert "original" in updated.tags
        assert "new-tag" in updated.tags
        assert "extra" in updated.tags

    def test_add_tags_nonexistent(self, db):
        assert db.add_tags("nope", ["tag"]) is None

    def test_remove_tags(self, db):
        db.add_asset(AssetRecord(asset_id="rt1", job_id="j1", asset_type="character", prompt="a", quality_tier="clean", tags=["keep", "remove"]))
        updated = db.remove_tags("rt1", ["remove"])
        assert updated is not None
        assert "keep" in updated.tags
        assert "remove" not in updated.tags

    def test_remove_tags_nonexistent(self, db):
        assert db.remove_tags("nope", ["tag"]) is None

    def test_add_tags_deduplicates(self, db):
        db.add_asset(AssetRecord(asset_id="dedup", job_id="j1", asset_type="character", prompt="a", quality_tier="clean", tags=["tag1"]))
        db.add_tags("dedup", ["tag1", "tag2"])
        asset = db.get_asset("dedup")
        assert len(asset.tags) == 2


class TestDatabaseLibraryDirectory:
    def test_get_asset_dir(self, db):
        d = db.get_asset_dir("myasset")
        assert d.endswith("myasset")

    def test_ensure_asset_dir_creates(self, db):
        d = db.ensure_asset_dir("newasset")
        assert os.path.isdir(d)

    def test_ensure_asset_dir_is_idempotent(self, db):
        d1 = db.ensure_asset_dir("same")
        d2 = db.ensure_asset_dir("same")
        assert d1 == d2
        assert os.path.isdir(d1)


class TestDatabaseLibraryClear:
    def test_clear_removes_all_data(self, db):
        db.add_asset(AssetRecord(asset_id="c1", job_id="j1", asset_type="character", prompt="a", quality_tier="clean"))
        db.add_asset(AssetRecord(asset_id="c2", job_id="j2", asset_type="enemy", prompt="b", quality_tier="clean"))
        db.clear()
        assert db.count() == 0

    def test_clear_tags_also_cleared(self, db):
        db.add_asset(AssetRecord(asset_id="ct1", job_id="j1", asset_type="character", prompt="a", quality_tier="clean", tags=["fantasy"]))
        db.clear()
        assert db.list_tags() == []

    def test_clear_then_add_works(self, db):
        db.add_asset(AssetRecord(asset_id="x", job_id="j1", asset_type="character", prompt="a", quality_tier="clean"))
        db.clear()
        db.add_asset(AssetRecord(asset_id="y", job_id="j2", asset_type="enemy", prompt="b", quality_tier="clean"))
        assert db.count() == 1


class TestDatabaseLibraryConcurrency:
    def test_add_and_read_race(self, db):
        import threading
        results = []

        def worker(i):
            aid = f"con{i}"
            db.add_asset(AssetRecord(asset_id=aid, job_id=f"j{i}", asset_type="character", prompt=f"worker {i}", quality_tier="clean"))
            results.append(db.get_asset(aid))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
        assert all(r is not None for r in results)
        assert db.count() == 10
