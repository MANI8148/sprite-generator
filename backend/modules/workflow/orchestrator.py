from __future__ import annotations
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set
from enum import Enum

from .dag import WorkflowDAG, WorkflowNode
from ..tasks.queue import TaskQueue, JobStatus, get_task_queue


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class WorkflowResult:
    status: WorkflowStatus
    step_results: Dict[str, Any] = field(default_factory=dict)
    step_errors: Dict[str, str] = field(default_factory=dict)
    step_durations: Dict[str, float] = field(default_factory=dict)
    execution_order: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "step_results": {k: str(v) if not isinstance(v, (dict, list, int, float, bool, type(None))) else v for k, v in self.step_results.items()},
            "step_errors": dict(self.step_errors),
            "step_durations": dict(self.step_durations),
            "execution_order": list(self.execution_order),
            "error": self.error,
        }


class WorkflowOrchestrator:
    def __init__(self, task_queue: Optional[TaskQueue] = None, max_parallel: int = 2):
        self._task_queue = task_queue or get_task_queue()
        self._max_parallel = max_parallel

    def execute(
        self,
        dag: WorkflowDAG,
        step_funcs: Dict[str, Callable[..., Any]],
        global_kwargs: Optional[Dict[str, Any]] = None,
    ) -> WorkflowResult:
        result = WorkflowResult(status=WorkflowStatus.RUNNING)
        completed: Set[str] = set()
        failed: Set[str] = set()
        skipped: Set[str] = set()
        global_kwargs = global_kwargs or {}

        levels = dag.get_levels()

        for level in levels:
            with ThreadPoolExecutor(max_workers=min(self._max_parallel, len(level))) as executor:
                fut_map = {}
                for nid in level:
                    node = dag.get_node(nid)
                    if node is None:
                        failed.add(nid)
                        result.step_errors[nid] = f"Node '{nid}' not found in DAG"
                        continue

                    deps = dag.get_dependencies(nid)
                    dep_failed = [d for d in deps if d in failed]
                    if dep_failed:
                        skipped.add(nid)
                        result.step_errors[nid] = f"Skipped because dependencies failed: {', '.join(dep_failed)}"
                        continue

                    func = step_funcs.get(nid)
                    if func is None:
                        failed.add(nid)
                        result.step_errors[nid] = f"No function provided for step '{nid}'"
                        continue

                    fut = executor.submit(
                        self._run_step, node, func, global_kwargs, result
                    )
                    fut_map[fut] = nid

                for fut in as_completed(fut_map):
                    nid = fut_map[fut]
                    step_ok, step_result, step_error, step_duration = fut.result()
                    result.step_durations[nid] = step_duration
                    if step_ok:
                        completed.add(nid)
                        result.step_results[nid] = step_result
                    else:
                        failed.add(nid)
                        result.step_errors[nid] = step_error

        if not failed and not skipped:
            result.status = WorkflowStatus.COMPLETED
        elif completed and (failed or skipped):
            result.status = WorkflowStatus.PARTIAL
        else:
            result.status = WorkflowStatus.FAILED
            errors = [f"{nid}: {err}" for nid, err in result.step_errors.items()]
            result.error = "; ".join(errors)

        return result

    def _run_step(
        self,
        node: WorkflowNode,
        func: Callable,
        global_kwargs: Dict[str, Any],
        result: WorkflowResult,
    ) -> tuple:
        last_error: Optional[str] = None
        last_result: Any = None
        attempts = max(1, node.retry_count + 1)

        for attempt in range(attempts):
            try:
                start = time.monotonic()
                if node.timeout:
                    with ThreadPoolExecutor(max_workers=1) as te:
                        fut = te.submit(func, **global_kwargs)
                        last_result = fut.result(timeout=node.timeout)
                else:
                    last_result = func(**global_kwargs)
                duration = time.monotonic() - start
                result.execution_order.append(node.id)
                return True, last_result, None, duration
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                if attempt < attempts - 1 and node.retry_delay > 0:
                    time.sleep(node.retry_delay)

        return False, None, last_error, 0.0

    def execute_async(
        self,
        dag: WorkflowDAG,
        step_funcs: Dict[str, Callable[..., Any]],
        global_kwargs: Optional[Dict[str, Any]] = None,
        job_id: Optional[str] = None,
    ) -> str:
        q_job_id = job_id or f"wf_{int(time.time())}"
        self._task_queue.submit(
            self._run_async_wrapper,
            q_job_id,
            dag, step_funcs, global_kwargs or {},
        )
        return q_job_id

    def _run_async_wrapper(
        self,
        dag: WorkflowDAG,
        step_funcs: Dict[str, Callable],
        global_kwargs: Dict[str, Any],
    ):
        return self.execute(dag, step_funcs, global_kwargs)

    def get_workflow_status(self, job_id: str) -> Optional[dict]:
        job = self._task_queue.get_status(job_id)
        if job is None:
            return None
        return {
            "job_id": job_id,
            "status": job["status"].value if isinstance(job["status"], JobStatus) else job["status"],
            "result": job["result"].to_dict() if isinstance(job.get("result"), WorkflowResult) else job.get("result"),
            "error": job.get("error"),
        }
