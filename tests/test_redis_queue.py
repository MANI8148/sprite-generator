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
