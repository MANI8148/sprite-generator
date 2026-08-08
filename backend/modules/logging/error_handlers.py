"""Global exception handling for the FastAPI application.

Implements the "proper error handling" half of the Phase 5 roadmap item
(proper error handling, structured logging, correlation IDs). Every error
response produced by these handlers carries the request's correlation ID
header so a failing request can be traced end to end, and unexpected
exceptions are emitted as structured JSON log records (with exception
type, message, and traceback) instead of a bare ``500 Internal Server
Error`` page.
"""

import traceback

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse

from .correlation import get_correlation_id
from .structured_logger import get_logger


def _error_response(status_code: int, detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"detail": detail},
        headers={"X-Correlation-ID": get_correlation_id()},
    )


def register_error_handlers(app: FastAPI) -> None:
    """Attach global exception handlers to ``app``.

    Handles three classes of failures:

    * uncaught exceptions -> structured ``unhandled_exception`` event and a
      JSON 500 (detail is a generic message so internals are not leaked);
    * ``StarletteHTTPException`` (``HTTPException``, unknown routes, method
      mismatches) -> JSON status with the original detail;
    * ``RequestValidationError`` (Pydantic body/query/path validation) ->
      structured ``validation_error`` event and a JSON 422 listing the
      offending fields.

    All responses include the ``X-Correlation-ID`` header.
    """

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        get_logger("backend.exception_handler").error(
            "unhandled_exception",
            method=request.method,
            path=request.url.path,
            exception_type=type(exc).__name__,
            error=str(exc),
            traceback="\n".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            ),
        )
        return _error_response(500, "Internal server error")

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        logger = get_logger("backend.exception_handler")
        if exc.status_code >= 500:
            logger.error(
                "http_exception",
                method=request.method,
                path=request.url.path,
                status_code=exc.status_code,
                error=str(exc.detail),
            )
        else:
            logger.warning(
                "http_exception",
                method=request.method,
                path=request.url.path,
                status_code=exc.status_code,
                error=str(exc.detail),
            )
        return _error_response(exc.status_code, str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        errors = jsonable_encoder(exc.errors())
        get_logger("backend.exception_handler").warning(
            "validation_error",
            method=request.method,
            path=request.url.path,
            errors=errors,
        )
        return JSONResponse(
            status_code=422,
            content={"detail": errors},
            headers={"X-Correlation-ID": get_correlation_id()},
        )