from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
import os
import uuid

from backend.modules.prompt_builder.controls import (
    AssetControls, AssetType, View, Palette, Animation, SpriteSize,
)
from backend.modules.pipeline.orchestrator import AssetPipeline, PipelineConfig
from backend.modules.storage.file_storage import FileStorage

router = APIRouter()

_pipeline: AssetPipeline = None
_generator_loaded: bool = False
_storage = FileStorage()


def get_pipeline() -> AssetPipeline:
    if _pipeline is None:
        raise RuntimeError("Pipeline not initialized")
    return _pipeline


def set_pipeline(pipe: AssetPipeline) -> None:
    global _pipeline
    _pipeline = pipe


def set_generator_loaded(loaded: bool) -> None:
    global _generator_loaded
    _generator_loaded = loaded


def get_storage() -> FileStorage:
    return _storage


def set_storage(st: FileStorage) -> None:
    global _storage
    _storage = st


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


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", generator_loaded=_generator_loaded)


@router.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest, background_tasks: BackgroundTasks, pipe: AssetPipeline = Depends(get_pipeline)):
    global _generator_loaded
    if not _generator_loaded:
        raise HTTPException(status_code=503, detail="Generator not set. POST /load-model first.")

    controls = AssetControls(
        asset_type=AssetType(req.asset_type),
        view=View(req.view),
        animation=Animation(req.animation),
        palette=Palette(req.palette),
        sprite_size=SpriteSize(req.sprite_size),
        theme=req.theme,
        seed=req.seed,
    )

    pipe.config.remove_bg = req.remove_bg
    pipe.config.reduce_palette = req.reduce_palette
    pipe.config.max_colors = req.max_colors
    pipe.config.pixel_cleanup = req.pixel_cleanup
    pipe.config.auto_center = req.auto_center
    pipe.config.upscale = req.upscale
    pipe.config.export_engine = req.engine
    pipe.config.pack_sheet = req.num_frames > 1

    job_id = str(uuid.uuid4())[:8]
    output_dir = _storage.ensure_output_dir(job_id)

    result = pipe.run(controls, output_dir=output_dir)

    _storage.add_job(job_id, {
        "prompt": result.metadata["prompt"],
        "quality_tier": result.validation[0]["quality_tier"],
        "outputs": result.output_paths,
        "zip_path": result.zip_path,
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


@router.post("/load-model")
def load_model(req: LoadModelRequest):
    global _generator_loaded
    from backend.modules.generator.sd_generator import SDGenerator
    gen = SDGenerator(lora_path=req.lora_path)
    gen.load()
    get_pipeline().set_generator(gen)
    _generator_loaded = True
    return {"status": "loaded", "lora_path": req.lora_path}


@router.get("/download/{job_id}")
def download(job_id: str, storage: FileStorage = Depends(get_storage)):
    entry = storage.get_job(job_id)
    if entry and entry.get("zip_path") and os.path.isfile(entry["zip_path"]):
        return FileResponse(entry["zip_path"], media_type="application/zip")
    raise HTTPException(status_code=404, detail="Job not found or ZIP not available")


@router.get("/history")
def list_history(storage: FileStorage = Depends(get_storage)):
    return storage.list_jobs()
