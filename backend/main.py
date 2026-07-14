from fastapi import FastAPI
from backend.api.routes import router, set_pipeline
from backend.modules.pipeline.orchestrator import AssetPipeline

app = FastAPI(title="AI Game Asset Pipeline API")
app.include_router(router)

set_pipeline(AssetPipeline())
