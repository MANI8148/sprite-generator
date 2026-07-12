from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import os
import uuid
import json

from backend.modules.prompt_builder.controls import (
    AssetControls, AssetType, View, Palette, Animation, SpriteSize,
)
from backend.modules.pipeline.orchestrator import AssetPipeline, PipelineConfig

app = FastAPI(title="AI Game Asset Pipeline API")
pipeline = AssetPipeline()
generator_loaded = False
history = []


class GenerateRequest(BaseModel):
    asset_type: str = "character"
    view: str = "front"
    animation: str = "idle"
    palette: str = "auto"
    sprite_size: str = "32x32"
    theme: str = ""
    seed: int = -1
    remove_bg: bool = True
    reduce_palette: bool = True
    max_colors: int = 32
    pixel_cleanup: bool = True
    auto_center: bool = True
    upscale: int = 1
    engine: str = "godot"
    num_frames: int = 1


class GenerateResponse(BaseModel):
    job_id: str
    prompt: str
    quality_tier: str
    validation: dict
    zip_path: Optional[str]
    output_paths: List[str]


class HealthResponse(BaseModel):
    status: str
    generator_loaded: bool


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", generator_loaded=generator_loaded)


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest, background_tasks: BackgroundTasks):
    global generator_loaded
    if not generator_loaded:
        raise HTTPException(status_code=503, detail="Generator not loaded. POST /load-model first.")

    controls = AssetControls(
        asset_type=AssetType(req.asset_type),
        view=View(req.view),
        animation=Animation(req.animation),
        palette=Palette(req.palette),
        sprite_size=SpriteSize(req.sprite_size),
        theme=req.theme,
        seed=req.seed,
    )

    pipeline.config.remove_bg = req.remove_bg
    pipeline.config.reduce_palette = req.reduce_palette
    pipeline.config.max_colors = req.max_colors
    pipeline.config.pixel_cleanup = req.pixel_cleanup
    pipeline.config.auto_center = req.auto_center
    pipeline.config.upscale = req.upscale
    pipeline.config.export_engine = req.engine
    pipeline.config.pack_sheet = req.num_frames > 1

    job_id = str(uuid.uuid4())[:8]
    output_dir = f"/tmp/sprite_gen/{job_id}"

    result = pipeline.run(controls, output_dir=output_dir)

    history.append({
        "job_id": job_id,
        "prompt": result.metadata["prompt"],
        "quality_tier": result.validation[0]["quality_tier"],
        "outputs": result.output_paths,
    })

    return GenerateResponse(
        job_id=job_id,
        prompt=result.metadata["prompt"],
        quality_tier=result.validation[0]["quality_tier"],
        validation=result.validation[0],
        zip_path=result.zip_path,
        output_paths=result.output_paths,
    )


class LoadModelRequest(BaseModel):
    lora_path: Optional[str] = None


@app.post("/load-model")
def load_model(req: LoadModelRequest):
    global generator_loaded
    from backend.modules.generator.sd_generator import SDGenerator
    gen = SDGenerator(lora_path=req.lora_path)
    gen.load()
    pipeline.set_generator(gen)
    generator_loaded = True
    return {"status": "loaded", "lora_path": req.lora_path}


@app.get("/download/{job_id}")
def download(job_id: str):
    for entry in history:
        if entry["job_id"] == job_id and entry.get("zip_path"):
            return FileResponse(entry["zip_path"], media_type="application/zip")
    raise HTTPException(status_code=404, detail="Job not found or ZIP not available")


@app.get("/history")
def get_history():
    return history
