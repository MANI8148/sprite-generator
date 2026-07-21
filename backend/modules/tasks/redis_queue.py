from __future__ import annotations
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional
from .queue import JobStatus, TaskQueue


class RedisTaskQueue(TaskQueue):
    def __init__(self, redis_client: Any, max_workers: int = 2, key_prefix: str = "task_queue:"):
        self._redis = redis_client
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._key_prefix = key_prefix

    def _job_key(self, job_id: str, field: str) -> str:
        return f"{self._key_prefix}{job_id}:{field}"

    def submit(self, fn: Callable, job_id: str, *args: Any, **kwargs: Any) -> str:
        self._redis.set(self._job_key(job_id, "status"), JobStatus.PENDING.value)
        self._redis.set(self._job_key(job_id, "result"), "")
        self._redis.set(self._job_key(job_id, "error"), "")
        self._redis.sadd(f"{self._key_prefix}jobs", job_id)

        def wrapper():
            self._redis.set(self._job_key(job_id, "status"), JobStatus.RUNNING.value)
            try:
                result = fn(*args, **kwargs)
                serialized = json.dumps(result, default=str, ensure_ascii=False)
                self._redis.set(self._job_key(job_id, "result"), serialized)
                self._redis.set(self._job_key(job_id, "status"), JobStatus.DONE.value)
            except Exception as e:
                self._redis.set(self._job_key(job_id, "error"), str(e))
                self._redis.set(self._job_key(job_id, "status"), JobStatus.FAILED.value)

        self._executor.submit(wrapper)
        return job_id

    def submit_new(self, fn: Callable, *args: Any, **kwargs: Any) -> str:
        job_id = str(uuid.uuid4())[:8]
        return self.submit(fn, job_id, *args, **kwargs)

    def get_status(self, job_id: str) -> Optional[dict]:
        status_raw = self._redis.get(self._job_key(job_id, "status"))
        if status_raw is None:
            return None
        status_val = status_raw.decode("utf-8") if isinstance(status_raw, bytes) else status_raw
        try:
            status_enum = JobStatus(status_val)
        except ValueError:
            return None
        result_raw = self._redis.get(self._job_key(job_id, "result"))
        result_val = None
        if result_raw:
            decoded = result_raw.decode("utf-8") if isinstance(result_raw, bytes) else result_raw
            if decoded:
                try:
                    result_val = json.loads(decoded)
                except (json.JSONDecodeError, ValueError):
                    result_val = decoded
        error_raw = self._redis.get(self._job_key(job_id, "error"))
        error_val = None
        if error_raw:
            decoded = error_raw.decode("utf-8") if isinstance(error_raw, bytes) else error_raw
            error_val = decoded or None
        return {
            "status": status_enum,
            "result": result_val,
            "error": error_val,
        }

    def list_jobs(self) -> list:
        job_ids = self._redis.smembers(f"{self._key_prefix}jobs")
        result = []
        for raw_id in job_ids:
            jid = raw_id.decode("utf-8") if isinstance(raw_id, bytes) else raw_id
            status_raw = self._redis.get(self._job_key(jid, "status"))
            if status_raw:
                sv = status_raw.decode("utf-8") if isinstance(status_raw, bytes) else status_raw
                result.append({"job_id": jid, "status": sv})
        return result

    @property
    def pending_count(self) -> int:
        return sum(1 for j in self.list_jobs() if j["status"] == JobStatus.PENDING.value)

    @property
    def running_count(self) -> int:
        return sum(1 for j in self.list_jobs() if j["status"] == JobStatus.RUNNING.value)
