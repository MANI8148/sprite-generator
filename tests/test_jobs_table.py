"""Tests for the persistent jobs table and job status polling.

Covers the AI_GAME_STUDIO_ARCHITECTURE_AND_ROADMAP item "Add a real jobs
table / status polling so the frontend can show generation progress": jobs
previously lived only in the in-memory ``TaskQueue`` and vanished on restart.
``DatabaseLibrary`` now keeps a ``jobs`` table (SQLite by default, PostgreSQL
when a ``postgres://`` URL is configured) and the API persists every state
transition (pending -> running -> done/failed) there, falling back to the
table from ``/status`` and ``/batch-status`` when the in-memory queue has no
record for a job (e.g. after a restart).
"""

import sqlite3
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.api.routes import (
    set_pipeline, set_generator_loaded, set_storage, set_library,
    set_job_store, _persist_job, _batch_jobs,
)
from backend.modules.pipeline.orchestrator import AssetPipeline
from backend.modules.storage.database import DatabaseLibrary
from backend.modules.storage import database as database_module
from backend.modules.storage.file_storage import FileStorage
from backend.modules.storage.asset_library import AssetLibrary
from backend.modules.rate_limiter import RateLimiter, set_rate_limiter
from backend.modules.tasks.queue import TaskQueue, set_task_queue
from backend.main import app

from tests.test_api import FakeGenerator, poll_job
from tests.test_postgres_storage import FakePsycopg2, POSTGRES_URL


@pytest.fixture
def db(tmp_path):
    return DatabaseLibrary(db_path=str(tmp_path / "jobs.db"))


@pytest.fixture
def fake_pg():
    fake = FakePsycopg2()
    with patch.object(
        database_module, "_connect_postgres", side_effect=lambda url: fake.connect(url)
    ):
        yield fake


@pytest.fixture
def pg_db(fake_pg, tmp_path):
    return DatabaseLibrary(db_path=str(tmp_path / "assets.db"), database_url=POSTGRES_URL)


@pytest.fixture(autouse=True)
def reset_api_state(tmp_path):
    set_generator_loaded(False)
    set_storage(FileStorage(base_dir=str(tmp_path / "storage")))
    set_library(AssetLibrary(base_dir=str(tmp_path / "lib")))
    set_rate_limiter(RateLimiter(max_requests=100, window_seconds=60))
    set_task_queue(TaskQueue(max_workers=4))
    set_job_store(None)
    _batch_jobs.clear()
    yield
    set_job_store(None)


@pytest.fixture
def client():
    pipe = AssetPipeline()
    pipe.set_generator(FakeGenerator(num_images=1))
    set_pipeline(pipe)
    set_generator_loaded(True)
    return TestClient(app)


class TestJobsTableSqlite:
    def test_creates_jobs_table(self, tmp_path):
        db_path = str(tmp_path / "created.db")
        DatabaseLibrary(db_path=db_path)
        conn = sqlite3.connect(db_path)
        try:
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
        finally:
            conn.close()
        assert "jobs" in tables

    def test_add_job_persists_with_defaults(self, db):
        db.add_job("j1")
        rec = db.get_job("j1")
        assert rec is not None
        assert rec["status"] == "pending"
        assert rec["prompt"] == ""
        assert rec["quality_tier"] == ""
        assert rec["validation"] == {}
        assert rec["output_paths"] == []
        assert rec["error"] == ""
        assert rec["created_at"]
        assert rec["updated_at"]

    def test_add_job_round_trips_fields(self, db):
        db.add_job(
            "j2",
            status="done",
            prompt="a red dragon",
            quality_tier="clean",
            validation={"quality_tier": "clean", "score": 0.9},
            zip_path="/tmp/pkg.zip",
            output_paths=["/tmp/a.png", "/tmp/b.png"],
            batch_id="batch-1",
        )
        rec = db.get_job("j2")
        assert rec["prompt"] == "a red dragon"
        assert rec["quality_tier"] == "clean"
        assert rec["validation"] == {"quality_tier": "clean", "score": 0.9}
        assert rec["zip_path"] == "/tmp/pkg.zip"
        assert rec["output_paths"] == ["/tmp/a.png", "/tmp/b.png"]
        assert rec["batch_id"] == "batch-1"

    def test_add_job_same_id_upserts(self, db):
        db.add_job("j3", status="pending")
        db.add_job("j3", status="done", prompt="hero")
        rec = db.get_job("j3")
        assert rec["status"] == "done"
        assert rec["prompt"] == "hero"

    def test_get_job_nonexistent_returns_none(self, db):
        assert db.get_job("nope") is None

    def test_update_job_transitions_status(self, db):
        db.add_job("j4", status="pending")
        updated = db.update_job("j4", status="running")
        assert updated["status"] == "running"
        updated = db.update_job("j4", status="done", prompt="hero", quality_tier="clean")
        assert updated["status"] == "done"
        assert updated["prompt"] == "hero"
        assert db.get_job("j4")["status"] == "done"

    def test_update_job_partial_preserves_other_fields(self, db):
        db.add_job("j5", status="running", zip_path="/tmp/x.zip", output_paths=["/tmp/a.png"])
        db.update_job("j5", status="done")
        rec = db.get_job("j5")
        assert rec["zip_path"] == "/tmp/x.zip"
        assert rec["output_paths"] == ["/tmp/a.png"]

    def test_update_job_json_fields(self, db):
        db.add_job("j6", status="pending")
        updated = db.update_job(
            "j6",
            status="done",
            validation={"quality_tier": "acceptable"},
            output_paths=["/tmp/a.png"],
        )
        assert updated["validation"] == {"quality_tier": "acceptable"}
        assert updated["output_paths"] == ["/tmp/a.png"]

    def test_update_job_nonexistent_returns_none(self, db):
        assert db.update_job("nope", status="done") is None

    def test_failed_job_persists_error(self, db):
        db.add_job("j7", status="running")
        updated = db.update_job("j7", status="failed", error="boom")
        assert updated["error"] == "boom"
        assert db.get_job("j7")["error"] == "boom"

    def test_list_jobs_orders_newest_first(self, db):
        db.add_job("old", status="done")
        db.add_job("new", status="pending")
        jobs = db.list_jobs()
        assert [j["job_id"] for j in jobs] == ["new", "old"]

    def test_list_jobs_filters_by_status(self, db):
        db.add_job("f1", status="done")
        db.add_job("f2", status="failed")
        db.add_job("f3", status="running")
        done = db.list_jobs(status="done")
        assert [j["job_id"] for j in done] == ["f1"]
        assert db.count_jobs() == 3
        assert db.count_jobs(status="failed") == 1

    def test_list_jobs_limit_and_offset(self, db):
        for i in range(5):
            db.add_job(f"lim{i}")
        assert len(db.list_jobs(limit=2)) == 2
        assert len(db.list_jobs(limit=2, offset=4)) == 1

    def test_persistence_across_reload(self, db):
        db.add_job("persist", status="done", prompt="castle")
        db2 = DatabaseLibrary(db_path=db.db_path)
        rec = db2.get_job("persist")
        assert rec is not None
        assert rec["status"] == "done"
        assert rec["prompt"] == "castle"

    def test_clear_removes_jobs(self, db):
        db.add_job("c1")
        db.clear()
        assert db.count_jobs() == 0
        assert db.get_job("c1") is None


class TestJobsTablePostgres:
    def test_insert_job_uses_on_conflict_upsert(self, pg_db):
        sql = pg_db._insert_job_sql()
        assert sql.startswith("INSERT INTO jobs")
        assert "ON CONFLICT (job_id) DO UPDATE SET" in sql
        assert "INSERT OR REPLACE" not in sql

    def test_add_get_update_job(self, pg_db):
        pg_db.add_job("pg1", status="pending", batch_id="b1")
        assert pg_db.get_job("pg1")["batch_id"] == "b1"
        pg_db.update_job("pg1", status="done", prompt="hero")
        rec = pg_db.get_job("pg1")
        assert rec["status"] == "done"
        assert rec["prompt"] == "hero"

    def test_emits_postgres_sql(self, pg_db, fake_pg):
        pg_db.add_job("pg2", status="running")
        statements = [sql for sql, _ in fake_pg.executed_sql]
        assert statements
        assert not any("PRAGMA" in s for s in statements)
        assert all("?" not in s for s in statements)
        assert any("ON CONFLICT" in s for s in statements)

    def test_sqlite_dialect_unaffected(self, tmp_path):
        sdb = DatabaseLibrary(db_path=str(tmp_path / "sqlite.db"))
        assert "INSERT OR REPLACE" in sdb._insert_job_sql()
        assert "ON CONFLICT" not in sdb._insert_job_sql()


class TestPersistJobHelper:
    def test_creates_then_updates(self, db):
        set_job_store(db)
        _persist_job("h1", "pending")
        assert db.get_job("h1")["status"] == "pending"
        _persist_job("h1", "done", prompt="hero", output_paths=["/tmp/a.png"])
        rec = db.get_job("h1")
        assert rec["status"] == "done"
        assert rec["prompt"] == "hero"

    def test_noop_without_store(self, db):
        set_job_store(None)
        _persist_job("h2", "pending")
        assert db.count_jobs() == 0


class TestStatusFallback:
    def test_status_reads_persisted_job_after_queue_reset(self, client, tmp_path):
        store = DatabaseLibrary(db_path=str(tmp_path / "jobs.db"))
        set_job_store(store)

        resp = client.post("/generate", json={"asset_type": "character"})
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]
        assert poll_job(client, job_id)["status"] == "done"

        rec = store.get_job(job_id)
        assert rec is not None
        assert rec["status"] == "done"
        assert rec["prompt"]
        assert rec["quality_tier"] in ("clean", "acceptable", "noisy", "blurry", "broken_outline", "empty", "extreme_aspect")
        assert rec["output_paths"]

        set_task_queue(TaskQueue(max_workers=4))
        data = client.get(f"/status/{job_id}").json()
        assert data["status"] == "done"
        assert data["prompt"] == rec["prompt"]
        assert data["output_paths"] == rec["output_paths"]
        assert data["zip_path"] == rec["zip_path"]

    def test_status_unknown_job_returns_404(self, client, tmp_path):
        store = DatabaseLibrary(db_path=str(tmp_path / "jobs.db"))
        set_job_store(store)
        resp = client.get("/status/unknown_job")
        assert resp.status_code == 404

    def test_batch_jobs_persist_batch_id(self, client, tmp_path):
        store = DatabaseLibrary(db_path=str(tmp_path / "jobs.db"))
        set_job_store(store)

        resp = client.post("/generate/batch", json={
            "items": [
                {"asset_type": "character"},
                {"asset_type": "enemy"},
            ],
        })
        assert resp.status_code == 202
        data = resp.json()
        batch_id = data["batch_id"]
        for job_id in data["job_ids"]:
            poll_job(client, job_id)

        assert store.count_jobs(status="done") == 2
        for job_id in data["job_ids"]:
            rec = store.get_job(job_id)
            assert rec["batch_id"] == batch_id

    def test_batch_status_falls_back_to_store(self, client, tmp_path):
        store = DatabaseLibrary(db_path=str(tmp_path / "jobs.db"))
        set_job_store(store)

        resp = client.post("/generate/batch", json={
            "items": [{"asset_type": "character"}],
        })
        data = resp.json()
        batch_id = data["batch_id"]
        poll_job(client, data["job_ids"][0])

        set_task_queue(TaskQueue(max_workers=4))
        resp = client.get(f"/batch-status/{batch_id}")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["status"] == "done"
        assert payload["completed"] == 1


class TestStoreFailureIsolation:
    def test_broken_store_does_not_break_generation(self, client):
        class BrokenStore:
            def get_job(self, job_id):
                raise RuntimeError("db down")

            def add_job(self, *args, **kwargs):
                raise RuntimeError("db down")

            def update_job(self, *args, **kwargs):
                raise RuntimeError("db down")

        set_job_store(BrokenStore())
        resp = client.post("/generate", json={"asset_type": "character"})
        assert resp.status_code == 202
        result = poll_job(client, resp.json()["job_id"])
        assert result["status"] == "done"
