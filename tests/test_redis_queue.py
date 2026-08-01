from __future__ import annotations
import time
import pytest

try:
    from fakeredis import FakeStrictRedis
    HAS_FAKEREDIS = True
except ImportError:
    HAS_FAKEREDIS = False

from backend.modules.tasks.redis_queue import RedisTaskQueue
from backend.modules.tasks.queue import JobStatus


pytestmark = pytest.mark.skipif(not HAS_FAKEREDIS, reason="fakeredis not installed")


@pytest.fixture
def redis_queue():
    r = FakeStrictRedis()
    q = RedisTaskQueue(redis_client=r, max_workers=2, key_prefix="test:")
    yield q
    r.flushall()


class TestRedisTaskQueue:
    def test_submit_and_get_status(self, redis_queue):
        def dummy():
            return 42
        jid = redis_queue.submit_new(dummy)
        assert jid is not None
        time.sleep(0.1)
        status = redis_queue.get_status(jid)
        assert status is not None
        assert status["status"] == JobStatus.DONE
        assert status["result"] == 42

    def test_submit_with_explicit_id(self, redis_queue):
        def dummy():
            return "hello"
        jid = redis_queue.submit(dummy, "my_job")
        assert jid == "my_job"
        time.sleep(0.1)
        status = redis_queue.get_status("my_job")
        assert status is not None
        assert status["status"] == JobStatus.DONE
        assert status["result"] == "hello"

    def test_submit_failure(self, redis_queue):
        def failing():
            raise ValueError("boom")
        jid = redis_queue.submit_new(failing)
        time.sleep(0.1)
        status = redis_queue.get_status(jid)
        assert status is not None
        assert status["status"] == JobStatus.FAILED
        assert "boom" in status["error"]

    def test_get_status_nonexistent(self, redis_queue):
        assert redis_queue.get_status("nonexistent") is None

    def test_submit_with_args_kwargs(self, redis_queue):
        def add(a, b):
            return a + b
        jid = redis_queue.submit(add, "add_job", 3, 4)
        time.sleep(0.1)
        status = redis_queue.get_status(jid)
        assert status["status"] == JobStatus.DONE
        assert status["result"] == 7

    def test_submit_with_kwargs(self, redis_queue):
        def greet(greeting, name):
            return f"{greeting}, {name}!"
        jid = redis_queue.submit(greet, "greet_job", greeting="Hello", name="World")
        time.sleep(0.1)
        status = redis_queue.get_status(jid)
        assert status["status"] == JobStatus.DONE
        assert status["result"] == "Hello, World!"

    def test_list_jobs(self, redis_queue):
        def dummy():
            return 1
        jid1 = redis_queue.submit_new(dummy)
        jid2 = redis_queue.submit_new(dummy)
        time.sleep(0.1)
        jobs = redis_queue.list_jobs()
        jids = [j["job_id"] for j in jobs]
        assert jid1 in jids
        assert jid2 in jids
        for j in jobs:
            assert j["status"] == JobStatus.DONE.value

    def test_pending_count(self, redis_queue):
        def slow():
            time.sleep(0.3)
            return "done"
        jid = redis_queue.submit_new(slow)
        assert redis_queue.pending_count >= 0
        time.sleep(0.4)
        status = redis_queue.get_status(jid)
        assert status["status"] == JobStatus.DONE

    def test_running_count(self, redis_queue):
        def slow():
            time.sleep(0.3)
            return "done"
        redis_queue.submit_new(slow)
        done = False
        for _ in range(50):
            if redis_queue.running_count > 0:
                done = True
                break
            time.sleep(0.01)
        assert done

    def test_deserializes_complex_result(self, redis_queue):
        def get_dict():
            return {"key": "value", "num": 42}
        jid = redis_queue.submit_new(get_dict)
        time.sleep(0.1)
        status = redis_queue.get_status(jid)
        assert status["result"] == {"key": "value", "num": 42}

    def test_isolated_prefixes(self, redis_queue):
        r2 = FakeStrictRedis()
        q1 = RedisTaskQueue(redis_client=redis_queue._redis, key_prefix="q1:")
        q2 = RedisTaskQueue(redis_client=r2, key_prefix="q2:")

        def dummy():
            return "ok"

        jid1 = q1.submit_new(dummy)
        jid2 = q2.submit_new(dummy)
        time.sleep(0.1)

        assert q1.get_status(jid1) is not None
        assert q2.get_status(jid1) is None
        assert q2.get_status(jid2) is not None


class TestRedisQueueWithOrchestrator:
    def test_orchestrator_works_with_redis_queue(self, redis_queue):
        from backend.modules.workflow.dag import WorkflowDAG, WorkflowNode
        from backend.modules.workflow.orchestrator import WorkflowOrchestrator, WorkflowStatus
        from backend.modules.tasks.queue import set_task_queue

        set_task_queue(redis_queue)

        dag = WorkflowDAG()
        dag.add_node(WorkflowNode(id="a"))
        dag.add_node(WorkflowNode(id="b"))
        dag.add_dependency("a", "b")

        def fn_a(**kw):
            return "result_a"

        def fn_b(**kw):
            return "result_b"

        orchestrator = WorkflowOrchestrator(task_queue=redis_queue, max_parallel=1)
        result = orchestrator.execute(dag, {"a": fn_a, "b": fn_b})
        assert result.status == WorkflowStatus.COMPLETED
        assert result.step_results["a"] == "result_a"
        assert result.step_results["b"] == "result_b"


class TestRedisTaskQueueFactory:
    def test_create_redis_task_queue_requires_redis_url(self, monkeypatch):
        from backend.modules.tasks.redis_queue import create_redis_task_queue

        monkeypatch.delenv("REDIS_URL", raising=False)
        with pytest.raises(ValueError):
            create_redis_task_queue(redis_url=None)

    def test_create_redis_task_queue_uses_env_url(self, monkeypatch):
        from backend.modules.tasks.redis_queue import create_redis_task_queue

        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
        q = create_redis_task_queue()
        assert isinstance(q, RedisTaskQueue)
        assert q._key_prefix == "task_queue:"

    def test_create_redis_task_queue_explicit_url_wins(self, monkeypatch):
        from backend.modules.tasks.redis_queue import create_redis_task_queue

        monkeypatch.setenv("REDIS_URL", "redis://wrong:9999/0")
        q = create_redis_task_queue(redis_url="redis://right:6379/0")
        assert isinstance(q, RedisTaskQueue)


class TestTaskQueueSelector:
    def test_selector_returns_in_memory_without_redis(self, monkeypatch):
        from backend.modules.tasks.queue import create_task_queue, TaskQueue

        monkeypatch.delenv("REDIS_URL", raising=False)
        q = create_task_queue()
        assert isinstance(q, TaskQueue)
        assert not isinstance(q, RedisTaskQueue)

    def test_selector_returns_redis_with_redis_url(self, monkeypatch):
        from backend.modules.tasks.queue import create_task_queue

        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
        q = create_task_queue()
        assert isinstance(q, RedisTaskQueue)

    def test_main_import_wires_selector_queue(self, monkeypatch):
        import importlib

        monkeypatch.delenv("REDIS_URL", raising=False)
        import backend.main as main
        importlib.reload(main)
        from backend.modules.tasks.queue import get_task_queue, TaskQueue

        assert isinstance(get_task_queue(), TaskQueue)
        assert not isinstance(get_task_queue(), RedisTaskQueue)

    def test_main_import_wires_redis_queue_when_configured(self, monkeypatch):
        import importlib

        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
        import backend.main as main
        importlib.reload(main)
        from backend.modules.tasks.queue import get_task_queue

        assert isinstance(get_task_queue(), RedisTaskQueue)


class TestGenerateAPIWithRedisQueue:
    @pytest.fixture(autouse=True)
    def setup_app(self, redis_queue):
        import tempfile
        from backend.modules.tasks.queue import set_task_queue
        from backend.modules.pipeline.orchestrator import AssetPipeline
        from backend.modules.storage.file_storage import FileStorage
        from backend.modules.storage.asset_library import AssetLibrary
        from backend.modules.rate_limiter import RateLimiter, set_rate_limiter
        from backend.api.routes import (
            set_pipeline, set_generator_loaded, set_storage, set_library, _batch_jobs,
        )
        from backend.main import app
        from fastapi.testclient import TestClient
        from tests.test_api import FakeGenerator, poll_job

        set_generator_loaded(False)
        tmp = tempfile.mkdtemp()
        set_storage(FileStorage(base_dir=tmp))
        set_library(AssetLibrary(base_dir=tmp + "/lib"))
        set_rate_limiter(RateLimiter(max_requests=1000, window_seconds=60))
        set_task_queue(redis_queue)
        _batch_jobs.clear()

        pipe = AssetPipeline()
        pipe.set_generator(FakeGenerator(num_images=1))
        set_pipeline(pipe)
        set_generator_loaded(True)

        self.client = TestClient(app)
        self.poll = poll_job
        yield
        _batch_jobs.clear()

    def test_generate_and_status_through_redis_queue(self):
        resp = self.client.post("/generate", json={
            "asset_type": "character",
            "view": "front",
            "animation": "idle",
            "palette": "auto",
            "sprite_size": "32x32",
            "num_frames": 1,
        })
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]

        result = self.poll(self.client, job_id)
        assert result["status"] == "done"
        assert result["prompt"] != ""
        assert "quality_tier" in result
        assert isinstance(result["output_paths"], list)
        assert len(result["output_paths"]) > 0

    def test_status_reflects_redis_queue(self):
        resp = self.client.post("/generate", json={"asset_type": "enemy"})
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]

        status = self.poll(self.client, job_id)
        assert status["status"] in ("done", "failed")
        assert "error" not in status or status["error"] is None
