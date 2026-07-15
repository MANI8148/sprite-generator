import os
import time
import asyncio
from collections import defaultdict
from fastapi import Request, HTTPException
from starlette.status import HTTP_429_TOO_MANY_REQUESTS


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError, TypeError):
        return default


class RateLimiter:
    def __init__(self, max_requests: int = None, window_seconds: int = None):
        self.max_requests = max_requests if max_requests is not None else _env_int("RATE_LIMIT_MAX_REQUESTS", 10)
        self.window_seconds = window_seconds if window_seconds is not None else _env_int("RATE_LIMIT_WINDOW_SECONDS", 60)
        self._requests: dict[str, list[float]] = defaultdict(list)

    def _cleanup(self, key: str, now: float):
        cutoff = now - self.window_seconds
        self._requests[key] = [t for t in self._requests[key] if t > cutoff]

    def remaining(self, key: str) -> int:
        now = time.time()
        self._cleanup(key, now)
        return max(0, self.max_requests - len(self._requests[key]))

    def reset_time(self, key: str) -> float:
        now = time.time()
        self._cleanup(key, now)
        if self._requests[key]:
            return self._requests[key][0] + self.window_seconds
        return now

    def check(self, key: str) -> bool:
        now = time.time()
        self._cleanup(key, now)
        if len(self._requests[key]) >= self.max_requests:
            return False
        self._requests[key].append(now)
        return True

    async def __call__(self, request: Request):
        client_ip = request.client.host if request.client else "unknown"
        if not self.check(client_ip):
            raise HTTPException(
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {self.max_requests} requests per {self.window_seconds}s. Try again later.",
            )
        return True


_default_limiter = RateLimiter()


def get_rate_limiter() -> RateLimiter:
    return _default_limiter


def set_rate_limiter(limiter: RateLimiter):
    global _default_limiter
    _default_limiter = limiter


EXEMPT_PATHS = {"/health", "/docs", "/openapi.json"}
