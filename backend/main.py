import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.status import HTTP_429_TOO_MANY_REQUESTS
from backend.api.routes import router, set_pipeline, set_job_store
from backend.api.auth_routes import router as auth_router
from backend.api.billing_routes import router as billing_router
from backend.api.team_routes import router as team_router
from backend.modules.pipeline.orchestrator import AssetPipeline
from backend.modules.rate_limiter import get_rate_limiter, EXEMPT_PATHS
from backend.modules.tasks.queue import set_task_queue, create_task_queue
from backend.modules.storage.database import create_database_library
from backend.modules.logging.correlation import generate_correlation_id, set_correlation_id, get_correlation_id
from backend.modules.logging.structured_logger import get_logger

def resolve_allowed_origins() -> list:
    """Build the list of origins permitted to call this API cross-origin.

    Configured via the ``ALLOWED_ORIGINS`` environment variable as a
    comma-separated list. Defaults to the local Next.js dev server and the
    backend itself so the Phase 1 frontend can talk to the FastAPI service
    out of the box.
    """
    raw = os.environ.get(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8000,http://127.0.0.1:8000",
    )
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    return origins or ["*"]


app = FastAPI(title="AI Game Asset Pipeline API")

app.include_router(router)
app.include_router(auth_router)
app.include_router(billing_router)
app.include_router(team_router)

logger = get_logger("backend.main")

set_pipeline(AssetPipeline())

set_task_queue(create_task_queue())

database_url = os.environ.get("DATABASE_URL")
if database_url:
    set_job_store(create_database_library(database_url=database_url))


@app.middleware("http")
async def correlation_middleware(request: Request, call_next):
    corr_id = request.headers.get("X-Correlation-ID") or generate_correlation_id()
    set_correlation_id(corr_id)
    response: Response = await call_next(request)
    response.headers["X-Correlation-ID"] = corr_id
    logger.info("request", method=request.method, path=request.url.path, status=response.status_code)
    return response


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path not in EXEMPT_PATHS:
        limiter = get_rate_limiter()
        client_ip = request.client.host if request.client else "unknown"

        remaining = limiter.remaining(client_ip)
        reset_at = limiter.reset_time(client_ip)
        is_limited = remaining <= 0

        if is_limited:
            logger.warning("rate_limit_exceeded", client_ip=client_ip, path=request.url.path)
            return JSONResponse(
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": f"Rate limit exceeded: {limiter.max_requests} requests per {limiter.window_seconds}s. Try again later."},
                headers={
                    "X-RateLimit-Limit": str(limiter.max_requests),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(reset_at)),
                },
            )

        response: Response = await call_next(request)
        limiter.check(client_ip)
        response.headers["X-RateLimit-Limit"] = str(limiter.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(max(0, remaining - 1))
        response.headers["X-RateLimit-Reset"] = str(int(reset_at))
        return response

    return await call_next(request)


# CORS must be the outermost middleware so browser preflight (OPTIONS)
# requests from a different origin are answered with the right CORS headers
# before auth / rate-limit middleware process them. Starlette applies the
# last-added middleware outermost, so this is added last.
app.add_middleware(
    CORSMiddleware,
    allow_origins=resolve_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
