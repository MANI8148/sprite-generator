from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.status import HTTP_429_TOO_MANY_REQUESTS
from backend.api.routes import router, set_pipeline
from backend.api.auth_routes import router as auth_router
from backend.api.billing_routes import router as billing_router
from backend.api.team_routes import router as team_router
from backend.modules.pipeline.orchestrator import AssetPipeline
from backend.modules.rate_limiter import get_rate_limiter, EXEMPT_PATHS
from backend.modules.logging.correlation import generate_correlation_id, set_correlation_id, get_correlation_id
from backend.modules.logging.structured_logger import get_logger

app = FastAPI(title="AI Game Asset Pipeline API")
app.include_router(router)
app.include_router(auth_router)
app.include_router(billing_router)
app.include_router(team_router)

logger = get_logger("backend.main")

set_pipeline(AssetPipeline())


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
