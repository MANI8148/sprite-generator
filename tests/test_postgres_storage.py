"""Tests for the PostgreSQL backend of the asset library database.

Covers the ROADMAP Phase 2 item "SQLite -> PostgreSQL (only when needed)":
the SQLite-backed ``DatabaseLibrary`` must expose the same interface against a
PostgreSQL backend, selected via a ``postgres://`` / ``postgresql://`` URL (or
the ``DATABASE_URL`` environment variable) while keeping SQLite the default.

No live PostgreSQL server is required: the driver connection is faked and the
Postgres SQL is executed against an in-memory engine so the query logic is
verified end to end, while assertions confirm the emitted SQL actually uses
Postgres idioms (``%s`` placeholders, ``ON CONFLICT``, ``->>`` JSON lookup,
no ``PRAGMA``).
"""

import os
import re
import sqlite3
import tempfile
from unittest.mock import patch

import pytest

from backend.modules.storage import database as database_module
from backend.modules.storage.database import DatabaseLibrary, create_database_library
from backend.modules.storage.asset_library import AssetRecord

POSTGRES_URL = "postgresql://user:pass@localhost:5432/sprite_gen"


class FakePsycopg2Cursor:
    """Executes PostgreSQL-flavored SQL against a shared in-memory engine."""

    def __init__(self, fake):
        self._fake = fake
        self.rowcount = -1
        self._result = None

    @staticmethod
    def _translate(sql):
        translated = sql.replace("%s", "?")
        translated = translated.replace(
            "metadata->>'generation_hash'",
            "json_extract(metadata, '$.generation_hash')",
        )
        return translated

    def execute(self, sql, params=None):
        params = params or ()
        self._fake.executed_sql.append((sql, tuple(params)))

        if "FROM information_schema.columns" in sql:
            table = re.search(r"table_name = '([^']+)'", sql).group(1)
            column = re.search(r"column_name = '([^']+)'", sql).group(1)
            cols = [
                r[1]
                for r in self._fake.conn.execute(f"PRAGMA table_info({table})").fetchall()
            ]
            self._result = [{"column_name": c} for c in cols if c == column]
            return self

        translated = self._translate(sql)
        try:
            cur = self._fake.conn.cursor()
            cur.execute(translated, tuple(params))
        except Exception as exc:
            raise AssertionError(
                f"FakePostgres rejected SQL: {sql!r}\n"
                f"  translated: {translated!r}\n  error: {exc}"
            ) from exc
        self.rowcount = cur.rowcount
        self._result = cur.fetchall() if cur.description is not None else None
        return self

    def fetchone(self):
        return self._result[0] if self._result else None

    def fetchall(self):
        return list(self._result or [])


class FakeConnection:
    def __init__(self, fake):
        self._fake = fake
        self.closed = False

    def cursor(self):
        return FakePsycopg2Cursor(self._fake)

    def commit(self):
        self._fake.conn.commit()

    def rollback(self):
        self._fake.conn.rollback()

    def close(self):
        self.closed = True


class FakePsycopg2:
    def __init__(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self.executed_sql = []

    def connect(self, dsn, cursor_factory=None):
        return FakeConnection(self)


@pytest.fixture
def fake_pg():
    fake = FakePsycopg2()
    with patch.object(
        database_module, "_connect_postgres", side_effect=lambda url: fake.connect(url)
    ):
        yield fake


@pytest.fixture
def db(fake_pg, tmp_path):
    return DatabaseLibrary(db_path=str(tmp_path / "assets.db"), database_url=POSTGRES_URL)


class TestBackendSelection:
    def test_defaults_to_sqlite(self, tmp_path):
        dbl = create_database_library(db_path=str(tmp_path / "a.db"))
        assert dbl.backend == "sqlite"

    def test_postgresql_url_selects_postgres(self, tmp_path, fake_pg):
        dbl = create_database_library(db_path=str(tmp_path / "b.db"), database_url=POSTGRES_URL)
        assert dbl.backend == "postgres"

    def test_postgres_scheme_also_accepted(self, tmp_path, fake_pg):
        dbl = create_database_library(
            db_path=str(tmp_path / "c.db"), database_url="postgres://u:p@localhost/db"
        )
        assert dbl.backend == "postgres"

    def test_sqlite_scheme_stays_sqlite(self, tmp_path):
        dbl = create_database_library(
            db_path=str(tmp_path / "d.db"), database_url="sqlite:///data/library.db"
        )
        assert dbl.backend == "sqlite"

    def test_env_var_selects_postgres(self, tmp_path, fake_pg, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", POSTGRES_URL)
        dbl = create_database_library(db_path=str(tmp_path / "e.db"))
        assert dbl.backend == "postgres"

    def test_explicit_url_overrides_env(self, tmp_path, fake_pg, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", POSTGRES_URL)
        dbl = create_database_library(db_path=str(tmp_path / "f.db"), database_url="sqlite:///x.db")
        assert dbl.backend == "sqlite"

    def test_constructor_reads_env_var(self, tmp_path, fake_pg, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", POSTGRES_URL)
        dbl = DatabaseLibrary(db_path=str(tmp_path / "g.db"))
        assert dbl.backend == "postgres"


class TestPostgresDialect:
    def test_placeholder_is_postgres_paramstyle(self, db):
        assert db._ph == "%s"

    def test_insert_asset_uses_on_conflict_upsert(self, db):
        sql = db._insert_asset_sql()
        assert sql.startswith("INSERT INTO assets")
        assert "ON CONFLICT (asset_id) DO UPDATE SET" in sql
        assert "EXCLUDED.job_id" in sql
        assert "INSERT OR REPLACE" not in sql

    def test_insert_tag_uses_do_nothing(self, db):
        sql = db._insert_tag_sql()
        assert sql.startswith("INSERT INTO tags")
        assert "ON CONFLICT (tag) DO NOTHING" in sql
        assert "INSERT OR IGNORE" not in sql

    def test_hash_lookup_uses_json_operator(self, db):
        assert "metadata->>'generation_hash'" in db._find_hash_sql()

    def test_sqlite_dialect_is_unaffected(self, tmp_path):
        sdb = DatabaseLibrary(db_path=str(tmp_path / "sqlite.db"))
        assert sdb._ph == "?"
        assert "INSERT OR REPLACE" in sdb._insert_asset_sql()
        assert "INSERT OR IGNORE" in sdb._insert_tag_sql()
        assert "json_extract(metadata, '$.generation_hash')" in sdb._find_hash_sql()


class TestPostgresExecution:
    def test_add_and_get_asset(self, db):
        aid = db.add_asset(
            AssetRecord(asset_id="pg1", job_id="j1", asset_type="character", prompt="hero", quality_tier="clean")
        )
        assert aid == "pg1"
        record = db.get_asset("pg1")
        assert record is not None
        assert record.asset_type == "character"
        assert record.prompt == "hero"
        assert record.job_id == "j1"

    def test_add_generates_id_and_persists(self, db):
        aid = db.add_asset(AssetRecord(asset_id="", job_id="j1", asset_type="enemy", prompt="goblin", quality_tier="clean"))
        assert len(aid) == 8
        assert db.get_asset(aid) is not None

    def test_add_upserts_existing_id(self, db):
        db.add_asset(AssetRecord(asset_id="up", job_id="j1", asset_type="character", prompt="first", quality_tier="clean"))
        db.add_asset(AssetRecord(asset_id="up", job_id="j2", asset_type="character", prompt="second", quality_tier="clean"))
        record = db.get_asset("up")
        assert record.prompt == "second"
        assert record.job_id == "j2"

    def test_update_asset(self, db):
        db.add_asset(AssetRecord(asset_id="m1", job_id="j1", asset_type="prop", prompt="chest", quality_tier="clean"))
        updated = db.update_asset("m1", category="treasure", quality_tier="acceptable")
        assert updated is not None
        assert updated.category == "treasure"
        assert updated.quality_tier == "acceptable"
        assert db.get_asset("m1").category == "treasure"

    def test_update_nonexistent_returns_none(self, db):
        assert db.update_asset("nope", category="x") is None

    def test_delete_asset(self, db):
        db.add_asset(AssetRecord(asset_id="del", job_id="j1", asset_type="character", prompt="a", quality_tier="clean"))
        assert db.delete_asset("del") is True
        assert db.get_asset("del") is None
        assert db.delete_asset("del") is False

    def test_list_and_filter(self, db):
        db.add_asset(AssetRecord(asset_id="a1", job_id="j1", asset_type="character", prompt="hero", quality_tier="clean"))
        db.add_asset(AssetRecord(asset_id="a2", job_id="j2", asset_type="vehicle", prompt="car", quality_tier="clean"))
        db.add_asset(AssetRecord(asset_id="a3", job_id="j3", asset_type="character", prompt="mage", quality_tier="acceptable"))
        assert db.count() == 3
        assert len(db.list_assets(asset_type="character")) == 2
        assert len(db.list_assets(quality_tier="clean")) == 2
        assert len(db.list_assets(search="mage")) == 1
        assert len(db.list_assets(search="HERO")) == 1

    def test_tags(self, db):
        db.add_asset(AssetRecord(asset_id="t1", job_id="j1", asset_type="character", prompt="a", quality_tier="clean", tags=["fantasy", "warrior"]))
        db.add_asset(AssetRecord(asset_id="t2", job_id="j2", asset_type="enemy", prompt="b", quality_tier="clean", tags=["sci-fi"]))
        assert db.list_tags() == ["fantasy", "sci-fi", "warrior"]
        assert len(db.list_assets(tags=["warrior"])) == 1

        updated = db.add_tags("t1", ["new", "fantasy"])
        assert "new" in updated.tags
        assert updated.tags.count("fantasy") == 1

        updated = db.remove_tags("t1", ["fantasy"])
        assert "fantasy" not in updated.tags
        assert db.add_tags("missing", ["x"]) is None
        assert db.remove_tags("missing", ["x"]) is None

    def test_find_by_generation_hash(self, db):
        db.add_asset(AssetRecord(asset_id="h1", job_id="j1", asset_type="character", prompt="a", quality_tier="clean", generation_hash="hash-123"))
        db.add_asset(AssetRecord(asset_id="h2", job_id="j2", asset_type="enemy", prompt="b", quality_tier="clean", metadata={"generation_hash": "meta-456"}))
        assert db.find_by_generation_hash("hash-123").asset_id == "h1"
        assert db.find_by_generation_hash("meta-456").asset_id == "h2"
        assert db.find_by_generation_hash("nope") is None

    def test_clear(self, db):
        db.add_asset(AssetRecord(asset_id="c1", job_id="j1", asset_type="character", prompt="a", quality_tier="clean", tags=["t"]))
        db.clear()
        assert db.count() == 0
        assert db.list_tags() == []

    def test_asset_dirs(self, db):
        d = db.ensure_asset_dir("dir1")
        assert os.path.isdir(d)
        assert db.get_asset_dir("dir1") == d

    def test_emits_postgres_sql_no_pragma(self, db, fake_pg):
        db.add_asset(AssetRecord(asset_id="s1", job_id="j1", asset_type="character", prompt="p", quality_tier="clean"))
        db.find_by_generation_hash("x")
        statements = [sql for sql, _ in fake_pg.executed_sql]
        assert statements, "expected Postgres SQL to be executed"
        assert not any("PRAGMA" in s for s in statements)
        assert all("?" not in s for s in statements)
        assert any("ON CONFLICT" in s for s in statements)


class TestPostgresDriverMissing:
    def test_raises_clear_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(database_module, "_POSTGRES_AVAILABLE", False)
        with pytest.raises(RuntimeError, match="psycopg2-binary"):
            DatabaseLibrary(db_path=str(tmp_path / "x.db"), database_url=POSTGRES_URL)
