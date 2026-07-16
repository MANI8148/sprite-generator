from concurrent.futures import ThreadPoolExecutor
import uuid
from enum import Enum
from typing import Any, Callable, Optional


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class TaskQueue:
    def __init__(self, max_workers: int = 2):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: dict[str, dict] = {}

    def submit(self, fn: Callable, job_id: str, *args: Any, **kwargs: Any) -> str:
        self._jobs[job_id] = {
            "status": JobStatus.PENDING,
            "result": None,
            "error": None,
        }

        def wrapper():
            job = self._jobs[job_id]
            job["status"] = JobStatus.RUNNING
            try:
                result = fn(*args, **kwargs)
                job["result"] = result
                job["status"] = JobStatus.DONE
            except Exception as e:
                job["error"] = str(e)
                job["status"] = JobStatus.FAILED

        self._executor.submit(wrapper)
        return job_id

    def submit_new(self, fn: Callable, *args: Any, **kwargs: Any) -> str:
        job_id = str(uuid.uuid4())[:8]
        return self.submit(fn, job_id, *args, **kwargs)

    def get_status(self, job_id: str) -> Optional[dict]:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list:
        return [{"job_id": k, "status": v["status"].value} for k, v in self._jobs.items()]

    @property
    def pending_count(self) -> int:
        return sum(1 for v in self._jobs.values() if v["status"] == JobStatus.PENDING)

    @property
    def running_count(self) -> int:
        return sum(1 for v in self._jobs.values() if v["status"] == JobStatus.RUNNING)


_default_queue = TaskQueue()


def get_task_queue() -> TaskQueue:
    return _default_queue


def set_task_queue(queue: TaskQueue) -> None:
    global _default_queue
    _default_queue = queue
