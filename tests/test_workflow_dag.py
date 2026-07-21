from __future__ import annotations
import time
import pytest
from typing import Dict, Any

from backend.modules.workflow.dag import WorkflowDAG, WorkflowNode
from backend.modules.workflow.orchestrator import (
    WorkflowOrchestrator,
    WorkflowResult,
    WorkflowStatus,
)
from backend.modules.tasks.queue import TaskQueue, set_task_queue


class TestWorkflowNode:
    def test_default_values(self):
        node = WorkflowNode(id="step1")
        assert node.id == "step1"
        assert node.description == ""
        assert node.retry_count == 0
        assert node.retry_delay == 0.0
        assert node.timeout is None

    def test_custom_values(self):
        node = WorkflowNode(
            id="gen",
            description="Generate image",
            retry_count=2,
            retry_delay=0.1,
            timeout=30.0,
        )
        assert node.id == "gen"
        assert node.description == "Generate image"
        assert node.retry_count == 2
        assert node.retry_delay == 0.1
        assert node.timeout == 30.0


class TestWorkflowDAG:
    def test_add_node(self):
        dag = WorkflowDAG()
        dag.add_node(WorkflowNode(id="a"))
        assert "a" in dag.nodes
        assert dag.get_node("a") is not None

    def test_add_duplicate_node_raises(self):
        dag = WorkflowDAG()
        dag.add_node(WorkflowNode(id="a"))
        with pytest.raises(ValueError, match="already exists"):
            dag.add_node(WorkflowNode(id="a"))

    def test_add_dependency(self):
        dag = WorkflowDAG()
        dag.add_node(WorkflowNode(id="a"))
        dag.add_node(WorkflowNode(id="b"))
        dag.add_dependency("a", "b")
        assert dag.get_dependencies("b") == ["a"]
        assert dag.get_dependents("a") == ["b"]

    def test_add_dependency_nonexistent_from_raises(self):
        dag = WorkflowDAG()
        dag.add_node(WorkflowNode(id="b"))
        with pytest.raises(ValueError, match="not found"):
            dag.add_dependency("a", "b")

    def test_add_dependency_nonexistent_to_raises(self):
        dag = WorkflowDAG()
        dag.add_node(WorkflowNode(id="a"))
        with pytest.raises(ValueError, match="not found"):
            dag.add_dependency("a", "b")

    def test_has_cycle_no_cycle(self):
        dag = WorkflowDAG()
        dag.add_node(WorkflowNode(id="a"))
        dag.add_node(WorkflowNode(id="b"))
        dag.add_node(WorkflowNode(id="c"))
        dag.add_dependency("a", "b")
        dag.add_dependency("b", "c")
        assert not dag.has_cycle()

    def test_has_cycle_detects_cycle(self):
        dag = WorkflowDAG()
        dag.add_node(WorkflowNode(id="a"))
        dag.add_node(WorkflowNode(id="b"))
        dag.add_node(WorkflowNode(id="c"))
        dag.add_dependency("a", "b")
        dag.add_dependency("b", "c")
        dag.add_dependency("c", "a")
        assert dag.has_cycle()

    def test_has_cycle_self_loop(self):
        dag = WorkflowDAG()
        dag.add_node(WorkflowNode(id="a"))
        dag.add_dependency("a", "a")
        assert dag.has_cycle()

    def test_topological_sort_simple(self):
        dag = WorkflowDAG()
        dag.add_node(WorkflowNode(id="a"))
        dag.add_node(WorkflowNode(id="b"))
        dag.add_dependency("a", "b")
        order = dag.topological_sort()
        assert order.index("a") < order.index("b")

    def test_topological_sort_complex(self):
        dag = WorkflowDAG()
        for nid in ["prompt", "generate", "postprocess", "validate", "export"]:
            dag.add_node(WorkflowNode(id=nid))
        dag.add_dependency("prompt", "generate")
        dag.add_dependency("generate", "postprocess")
        dag.add_dependency("postprocess", "validate")
        dag.add_dependency("validate", "export")
        order = dag.topological_sort()
        idx = {n: i for i, n in enumerate(order)}
        assert idx["prompt"] < idx["generate"]
        assert idx["generate"] < idx["postprocess"]
        assert idx["postprocess"] < idx["validate"]
        assert idx["validate"] < idx["export"]

    def test_topological_sort_with_branching(self):
        dag = WorkflowDAG()
        for nid in ["a", "b", "c", "d", "e"]:
            dag.add_node(WorkflowNode(id=nid))
        dag.add_dependency("a", "b")
        dag.add_dependency("a", "c")
        dag.add_dependency("b", "d")
        dag.add_dependency("c", "d")
        dag.add_dependency("d", "e")
        order = dag.topological_sort()
        idx = {n: i for i, n in enumerate(order)}
        assert idx["a"] < idx["b"]
        assert idx["a"] < idx["c"]
        assert idx["b"] < idx["d"]
        assert idx["c"] < idx["d"]
        assert idx["d"] < idx["e"]

    def test_topological_sort_cycle_raises(self):
        dag = WorkflowDAG()
        dag.add_node(WorkflowNode(id="a"))
        dag.add_node(WorkflowNode(id="b"))
        dag.add_dependency("a", "b")
        dag.add_dependency("b", "a")
        with pytest.raises(ValueError, match="contains a cycle"):
            dag.topological_sort()

    def test_get_ready_nodes_no_deps(self):
        dag = WorkflowDAG()
        dag.add_node(WorkflowNode(id="a"))
        dag.add_node(WorkflowNode(id="b"))
        assert dag.get_ready_nodes(set(), set()) == ["a", "b"]

    def test_get_ready_nodes_with_deps(self):
        dag = WorkflowDAG()
        dag.add_node(WorkflowNode(id="a"))
        dag.add_node(WorkflowNode(id="b"))
        dag.add_dependency("a", "b")
        assert dag.get_ready_nodes(set(), set()) == ["a"]
        assert dag.get_ready_nodes({"a"}, set()) == ["b"]

    def test_get_ready_nodes_excludes_completed_and_failed(self):
        dag = WorkflowDAG()
        dag.add_node(WorkflowNode(id="a"))
        dag.add_node(WorkflowNode(id="b"))
        assert dag.get_ready_nodes({"a"}, set()) == ["b"]
        assert dag.get_ready_nodes(set(), {"a"}) == ["b"]

    def test_get_levels_linear(self):
        dag = WorkflowDAG()
        for nid in ["a", "b", "c"]:
            dag.add_node(WorkflowNode(id=nid))
        dag.add_dependency("a", "b")
        dag.add_dependency("b", "c")
        levels = dag.get_levels()
        assert levels == [["a"], ["b"], ["c"]]

    def test_get_levels_parallel(self):
        dag = WorkflowDAG()
        for nid in ["a", "b", "c", "d"]:
            dag.add_node(WorkflowNode(id=nid))
        dag.add_dependency("a", "b")
        dag.add_dependency("a", "c")
        dag.add_dependency("b", "d")
        dag.add_dependency("c", "d")
        levels = dag.get_levels()
        assert levels[0] == ["a"]
        assert set(levels[1]) == {"b", "c"}
        assert levels[2] == ["d"]

    def test_get_levels_cycle_raises(self):
        dag = WorkflowDAG()
        dag.add_node(WorkflowNode(id="a"))
        dag.add_node(WorkflowNode(id="b"))
        dag.add_dependency("a", "b")
        dag.add_dependency("b", "a")
        with pytest.raises(ValueError, match="contains a cycle"):
            dag.get_levels()

    def test_subgraph(self):
        dag = WorkflowDAG()
        for nid in ["a", "b", "c", "d"]:
            dag.add_node(WorkflowNode(id=nid))
        dag.add_dependency("a", "b")
        dag.add_dependency("b", "c")
        dag.add_dependency("c", "d")

        sub = dag.subgraph({"a", "b", "c"})
        assert set(sub.nodes.keys()) == {"a", "b", "c"}
        assert sub.get_dependencies("b") == ["a"]
        assert sub.get_dependencies("c") == ["b"]
        assert "d" not in sub.nodes

    def test_subgraph_preserves_node_attrs(self):
        dag = WorkflowDAG()
        dag.add_node(WorkflowNode(id="a", description="start", retry_count=2))
        dag.add_node(WorkflowNode(id="b"))
        dag.add_dependency("a", "b")
        sub = dag.subgraph({"a", "b"})
        assert sub.get_node("a").description == "start"
        assert sub.get_node("a").retry_count == 2

    def test_get_node_nonexistent(self):
        dag = WorkflowDAG()
        assert dag.get_node("nonexistent") is None


class TestWorkflowOrchestrator:
    def _make_dag(self, linear: bool = True):
        dag = WorkflowDAG()
        dag.add_node(WorkflowNode(id="step_a"))
        dag.add_node(WorkflowNode(id="step_b"))
        dag.add_node(WorkflowNode(id="step_c"))
        if linear:
            dag.add_dependency("step_a", "step_b")
            dag.add_dependency("step_b", "step_c")
        return dag

    def test_execute_simple(self):
        dag = self._make_dag()
        results: Dict[str, Any] = {}

        def fn_a(**kw):
            results["a"] = "done_a"
            return "result_a"

        def fn_b(**kw):
            results["b"] = f"done_b_{results['a']}"
            return "result_b"

        def fn_c(**kw):
            results["c"] = f"done_c_{results['b']}"
            return "result_c"

        orchestrator = WorkflowOrchestrator(max_parallel=1)
        result = orchestrator.execute(dag, {
            "step_a": fn_a,
            "step_b": fn_b,
            "step_c": fn_c,
        })

        assert result.status == WorkflowStatus.COMPLETED
        assert result.step_results["step_a"] == "result_a"
        assert result.step_results["step_b"] == "result_b"
        assert result.step_results["step_c"] == "result_c"
        assert len(result.step_errors) == 0
        assert len(result.execution_order) == 3

    def test_execute_with_global_kwargs(self):
        dag = self._make_dag()
        seen: Dict[str, Any] = {}

        def fn_a(**kw):
            seen["a"] = kw.get("shared")
            return kw.get("shared")

        def fn_b(**kw):
            seen["b"] = fn_a(**kw) if False else kw.get("shared")
            return "ok"

        orchestrator = WorkflowOrchestrator(max_parallel=1)
        result = orchestrator.execute(dag, {
            "step_a": fn_a,
            "step_b": fn_b,
            "step_c": lambda **kw: "ok",
        }, global_kwargs={"shared": "hello"})

        assert result.status == WorkflowStatus.COMPLETED
        assert result.step_results["step_a"] == "hello"

    def test_execute_step_failure(self):
        dag = self._make_dag()

        def fn_a(**kw):
            return "ok"

        def fn_b(**kw):
            raise ValueError("step b failed")

        def fn_c(**kw):
            return "ok"

        orchestrator = WorkflowOrchestrator(max_parallel=1)
        result = orchestrator.execute(dag, {
            "step_a": fn_a,
            "step_b": fn_b,
            "step_c": fn_c,
        })

        assert result.status in (WorkflowStatus.FAILED, WorkflowStatus.PARTIAL)
        assert "step_b" in result.step_errors
        assert "ValueError" in result.step_errors["step_b"]
        assert result.step_results.get("step_a") == "ok"

    def test_execute_missing_function(self):
        dag = self._make_dag()

        def fn_a(**kw):
            return "ok"

        orchestrator = WorkflowOrchestrator(max_parallel=1)
        result = orchestrator.execute(dag, {"step_a": fn_a})

        assert result.status == WorkflowStatus.PARTIAL
        assert result.step_results.get("step_a") == "ok"
        assert "step_b" in result.step_errors

    def test_execute_retry_success(self):
        dag = WorkflowDAG()
        dag.add_node(WorkflowNode(id="unstable", retry_count=2, retry_delay=0.01))
        dag.add_node(WorkflowNode(id="stable"))
        dag.add_dependency("unstable", "stable")

        call_count = [0]

        def flaky(**kw):
            call_count[0] += 1
            if call_count[0] < 3:
                raise RuntimeError(f"attempt {call_count[0]} failed")
            return "success"

        def stable(**kw):
            return "ok"

        orchestrator = WorkflowOrchestrator(max_parallel=1)
        result = orchestrator.execute(dag, {
            "unstable": flaky,
            "stable": stable,
        })

        assert result.status == WorkflowStatus.COMPLETED
        assert result.step_results["unstable"] == "success"
        assert call_count[0] == 3

    def test_execute_retry_exhausted(self):
        dag = WorkflowDAG()
        dag.add_node(WorkflowNode(id="always_fails", retry_count=1, retry_delay=0.01))

        call_count = [0]

        def always_fails(**kw):
            call_count[0] += 1
            raise RuntimeError("always fails")

        orchestrator = WorkflowOrchestrator(max_parallel=1)
        result = orchestrator.execute(dag, {"always_fails": always_fails})

        assert result.status == WorkflowStatus.FAILED
        assert call_count[0] == 2  # initial + 1 retry

    def test_execute_timeout(self):
        dag = WorkflowDAG()
        dag.add_node(WorkflowNode(id="slow", timeout=0.05))

        def slow(**kw):
            time.sleep(0.5)
            return "done"

        orchestrator = WorkflowOrchestrator(max_parallel=1)
        result = orchestrator.execute(dag, {"slow": slow})

        assert result.status == WorkflowStatus.FAILED
        assert "slow" in result.step_errors

    def test_execute_parallel_levels(self):
        dag = WorkflowDAG()
        for nid in ["a", "b", "c", "d"]:
            dag.add_node(WorkflowNode(id=nid))
        dag.add_dependency("a", "b")
        dag.add_dependency("a", "c")
        dag.add_dependency("b", "d")
        dag.add_dependency("c", "d")

        execution_order: list = []

        def make_fn(nid: str):
            def fn(**kw):
                execution_order.append(nid)
                return nid
            return fn

        orchestrator = WorkflowOrchestrator(max_parallel=2)
        result = orchestrator.execute(dag, {nid: make_fn(nid) for nid in ["a", "b", "c", "d"]})

        assert result.status == WorkflowStatus.COMPLETED
        assert execution_order[0] == "a"
        assert set(execution_order[1:3]) == {"b", "c"}
        assert execution_order[3] == "d"
        assert result.step_results["a"] == "a"
        assert result.step_results["d"] == "d"

    def test_execute_empty_dag(self):
        dag = WorkflowDAG()
        orchestrator = WorkflowOrchestrator(max_parallel=1)
        result = orchestrator.execute(dag, {})
        assert result.status == WorkflowStatus.COMPLETED
        assert len(result.execution_order) == 0

    def test_execute_async(self):
        set_task_queue(TaskQueue(max_workers=2))
        dag = self._make_dag()

        def fn_a(**kw):
            return "a"

        def fn_b(**kw):
            return "b"

        def fn_c(**kw):
            return "c"

        orchestrator = WorkflowOrchestrator(max_parallel=1)
        job_id = orchestrator.execute_async(dag, {
            "step_a": fn_a,
            "step_b": fn_b,
            "step_c": fn_c,
        })
        assert job_id.startswith("wf_")

        status = orchestrator.get_workflow_status(job_id)
        assert status is not None
        assert status["job_id"] == job_id

    def test_get_workflow_status_nonexistent(self):
        orchestrator = WorkflowOrchestrator()
        assert orchestrator.get_workflow_status("nonexistent") is None

    def test_to_dict_serializable(self):
        result = WorkflowResult(
            status=WorkflowStatus.COMPLETED,
            step_results={"a": "ok"},
            step_durations={"a": 0.1},
            execution_order=["a"],
        )
        d = result.to_dict()
        assert d["status"] == "completed"
        assert d["step_results"]["a"] == "ok"
        assert d["step_durations"]["a"] == 0.1
        assert d["execution_order"] == ["a"]
        assert d["error"] is None

    def test_to_dict_with_error(self):
        result = WorkflowResult(
            status=WorkflowStatus.FAILED,
            error="step_a: something broke",
        )
        d = result.to_dict()
        assert d["status"] == "failed"
        assert d["error"] == "step_a: something broke"

    def test_execute_failure_chain_stops(self):
        dag = WorkflowDAG()
        for nid in ["a", "b", "c"]:
            dag.add_node(WorkflowNode(id=nid))
        dag.add_dependency("a", "b")
        dag.add_dependency("b", "c")

        results_tracker: list = []

        def fn_a(**kw):
            results_tracker.append("a")
            return "a"

        def fn_b(**kw):
            results_tracker.append("b")
            raise ValueError("b failed")

        def fn_c(**kw):
            results_tracker.append("c")
            return "c"

        orchestrator = WorkflowOrchestrator(max_parallel=1)
        result = orchestrator.execute(dag, {
            "a": fn_a,
            "b": fn_b,
            "c": fn_c,
        })

        assert result.status in (WorkflowStatus.FAILED, WorkflowStatus.PARTIAL)
        assert "a" in results_tracker
        assert "b" in results_tracker
        assert "c" not in results_tracker

    def test_execute_non_callable_step(self):
        dag = WorkflowDAG()
        dag.add_node(WorkflowNode(id="a"))

        orchestrator = WorkflowOrchestrator(max_parallel=1)
        result = orchestrator.execute(dag, {"a": "not_a_function"})

        assert result.status == WorkflowStatus.FAILED
        assert "a" in result.step_errors
