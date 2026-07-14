from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.status import HTTP_429_TOO_MANY_REQUESTS
from backend.api.routes import router, set_pipeline
from backend.modules.pipeline.orchestrator import AssetPipeline
from backend.modules.rate_limiter import get_rate_limiter, EXEMPT_PATHS

app = FastAPI(title="AI Game Asset Pipeline API")
app.include_router(router)

set_pipeline(AssetPipeline())


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path not in EXEMPT_PATHS:
        limiter = get_rate_limiter()
        client_ip = request.client.host if request.client else "unknown"
        if not limiter.check(client_ip):
            return JSONResponse(
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": f"Rate limit exceeded: {limiter.max_requests} requests per {limiter.window_seconds}s. Try again later."},
            )
    return await call_next(request)
