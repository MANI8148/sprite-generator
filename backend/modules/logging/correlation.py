import uuid
from contextvars import ContextVar

_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")


def get_correlation_id() -> str:
    return _correlation_id.get()


def set_correlation_id(value: str) -> None:
    _correlation_id.set(value)


def generate_correlation_id() -> str:
    return uuid.uuid4().hex[:12]


class CorrelationMiddleware:
    def __init__(self, header_name: str = "X-Correlation-ID"):
        self.header_name = header_name

    async def __call__(self, request, call_next):
        corr_id = request.headers.get(self.header_name) or generate_correlation_id()
        set_correlation_id(corr_id)
        response = await call_next(request)
        response.headers[self.header_name] = corr_id
        return response
