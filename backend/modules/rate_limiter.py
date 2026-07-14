import time
import asyncio
from collections import defaultdict
from fastapi import Request, HTTPException
from starlette.status import HTTP_429_TOO_MANY_REQUESTS


class RateLimiter:
    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    def _cleanup(self, key: str, now: float):
        cutoff = now - self.window_seconds
        self._requests[key] = [t for t in self._requests[key] if t > cutoff]

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
