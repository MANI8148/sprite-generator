from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
import os
import uuid

from backend.modules.prompt_builder.controls import (
    AssetControls, AssetType, View, Palette, Animation, SpriteSize,
)
from backend.modules.pipeline.orchestrator import AssetPipeline
from backend.modules.storage.file_storage import FileStorage
from backend.modules.storage.asset_library import AssetLibrary, AssetRecord
from backend.modules.tasks.queue import TaskQueue, get_task_queue, JobStatus

router = APIRouter()

_pipeline: AssetPipeline = None
_generator_loaded: bool = False
_storage = FileStorage()
_library = AssetLibrary()


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


def get_library() -> AssetLibrary:
    return _library


def set_library(lib: AssetLibrary) -> None:
    global _library
    _library = lib


class GenerateResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    prompt: Optional[str] = None
    quality_tier: Optional[str] = None
    validation: Optional[dict] = None
    zip_path: Optional[str] = None
    output_paths: Optional[List[str]] = None
    error: Optional[str] = None


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


class BatchItem(BaseModel):
    asset_type: str = "character"
    view: str = "front"
    animation: str = "idle"
    palette: str = "auto"
    sprite_size: str = "32x32"
    theme: str = ""
    seed: int = -1
    num_frames: int = 1
    remove_bg: bool = True
    reduce_palette: bool = True
    max_colors: int = 32
    pixel_cleanup: bool = True
    auto_center: bool = True
    upscale: int = 1
    engine: str = "godot"


class BatchGenerateRequest(BaseModel):
    items: List[BatchItem]
    batch_id: Optional[str] = None


class BatchResult(BaseModel):
    job_id: str
    prompt: str
    quality_tier: str
    validation: dict
    zip_path: Optional[str]
    output_paths: List[str]


class BatchGenerateResponse(BaseModel):
    batch_id: str
    results: List[BatchResult]
    total: int
    succeeded: int
    failed: int


class HealthResponse(BaseModel):
    status: str
    generator_loaded: bool


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", generator_loaded=_generator_loaded)


def _run_generation_job(pipe, controls, req, output_dir, job_id):
    result = pipe.run(controls, output_dir=output_dir)

    _storage.add_job(job_id, {
        "prompt": result.metadata["prompt"],
        "quality_tier": result.validation[0]["quality_tier"],
        "outputs": result.output_paths,
        "zip_path": result.zip_path,
    })

    _library.add_asset(AssetRecord(
        asset_id=job_id,
        job_id=job_id,
        asset_type=req.asset_type,
        prompt=result.metadata["prompt"],
        quality_tier=result.validation[0]["quality_tier"],
        zip_path=result.zip_path,
        output_paths=result.output_paths,
        metadata={"view": req.view, "animation": req.animation, "palette": req.palette, "sprite_size": req.sprite_size},
    ))

    return {
        "prompt": result.metadata["prompt"],
        "quality_tier": result.validation[0]["quality_tier"],
        "validation": result.validation[0],
        "zip_path": result.zip_path,
        "output_paths": result.output_paths,
    }


@router.post("/generate", response_model=GenerateResponse, status_code=202)
def generate(req: GenerateRequest, pipe: AssetPipeline = Depends(get_pipeline)):
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

    queue = get_task_queue()
    queue.submit(_run_generation_job, job_id, pipe, controls, req, output_dir, job_id)

    return GenerateResponse(job_id=job_id, status=JobStatus.PENDING)


@router.get("/status/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    queue = get_task_queue()
    job = queue.get_status(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    resp = JobStatusResponse(job_id=job_id, status=job["status"].value)
    if job["result"] is not None:
        resp.prompt = job["result"].get("prompt")
        resp.quality_tier = job["result"].get("quality_tier")
        resp.validation = job["result"].get("validation")
        resp.zip_path = job["result"].get("zip_path")
        resp.output_paths = job["result"].get("output_paths")
    if job["error"] is not None:
        resp.error = job["error"]
    return resp


@router.post("/generate/batch", response_model=BatchGenerateResponse)
def generate_batch(req: BatchGenerateRequest, pipe: AssetPipeline = Depends(get_pipeline)):
    global _generator_loaded
    if not _generator_loaded:
        raise HTTPException(status_code=503, detail="Generator not set. POST /load-model first.")

    batch_id = req.batch_id or str(uuid.uuid4())[:8]
    results: List[BatchResult] = []
    succeeded = 0
    failed = 0

    for i, item in enumerate(req.items):
        try:
            controls = AssetControls(
                asset_type=AssetType(item.asset_type),
                view=View(item.view),
                animation=Animation(item.animation),
                palette=Palette(item.palette),
                sprite_size=SpriteSize(item.sprite_size),
                theme=item.theme,
                seed=item.seed,
            )

            pipe.config.remove_bg = item.remove_bg
            pipe.config.reduce_palette = item.reduce_palette
            pipe.config.max_colors = item.max_colors
            pipe.config.pixel_cleanup = item.pixel_cleanup
            pipe.config.auto_center = item.auto_center
            pipe.config.upscale = item.upscale
            pipe.config.export_engine = item.engine
            pipe.config.pack_sheet = item.num_frames > 1

            job_id = f"{batch_id}_{i}"
            output_dir = _storage.ensure_output_dir(job_id)

            result = pipe.run(controls, output_dir=output_dir)

            _storage.add_job(job_id, {
                "prompt": result.metadata["prompt"],
                "quality_tier": result.validation[0]["quality_tier"],
                "outputs": result.output_paths,
                "zip_path": result.zip_path,
                "batch_id": batch_id,
            })

            results.append(BatchResult(
                job_id=job_id,
                prompt=result.metadata["prompt"],
                quality_tier=result.validation[0]["quality_tier"],
                validation=result.validation[0],
                zip_path=result.zip_path,
                output_paths=result.output_paths,
            ))
            succeeded += 1
        except Exception as e:
            failed += 1
            results.append(BatchResult(
                job_id=f"{batch_id}_{i}",
                prompt="",
                quality_tier="error",
                validation={"error": str(e)},
                zip_path=None,
                output_paths=[],
            ))

    return BatchGenerateResponse(
        batch_id=batch_id,
        results=results,
        total=len(req.items),
        succeeded=succeeded,
        failed=failed,
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


class LibraryAssetResponse(BaseModel):
    asset_id: str
    job_id: str
    asset_type: str
    prompt: str
    quality_tier: str
    tags: List[str]
    category: str
    thumbnail_path: Optional[str]
    zip_path: Optional[str]
    output_paths: List[str]
    created_at: str
    updated_at: str


class LibraryListResponse(BaseModel):
    assets: List[LibraryAssetResponse]
    total: int


class AddTagsRequest(BaseModel):
    tags: List[str]


class RemoveTagsRequest(BaseModel):
    tags: List[str]


class UpdateAssetRequest(BaseModel):
    category: Optional[str] = None
    tags: Optional[List[str]] = None


@router.get("/library", response_model=LibraryListResponse)
def list_library(
    asset_type: Optional[str] = None,
    quality_tier: Optional[str] = None,
    category: Optional[str] = None,
    tags: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    library: AssetLibrary = Depends(get_library),
):
    tag_list = tags.split(",") if tags else None
    assets = library.list_assets(
        asset_type=asset_type,
        quality_tier=quality_tier,
        category=category,
        tags=tag_list,
        search=search,
        limit=limit,
        offset=offset,
    )
    total = library.count()
    return LibraryListResponse(
        assets=[LibraryAssetResponse(**r.__dict__) for r in assets],
        total=total,
    )


@router.get("/library/tags")
def list_library_tags(library: AssetLibrary = Depends(get_library)):
    return {"tags": library.list_tags()}


@router.get("/library/{asset_id}", response_model=LibraryAssetResponse)
def get_library_asset(asset_id: str, library: AssetLibrary = Depends(get_library)):
    asset = library.get_asset(asset_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    return LibraryAssetResponse(**asset.__dict__)


@router.delete("/library/{asset_id}")
def delete_library_asset(asset_id: str, library: AssetLibrary = Depends(get_library)):
    if not library.delete_asset(asset_id):
        raise HTTPException(status_code=404, detail="Asset not found")
    return {"status": "deleted", "asset_id": asset_id}


@router.patch("/library/{asset_id}", response_model=LibraryAssetResponse)
def update_library_asset(asset_id: str, req: UpdateAssetRequest, library: AssetLibrary = Depends(get_library)):
    updates = {}
    if req.category is not None:
        updates["category"] = req.category
    if req.tags is not None:
        updates["tags"] = req.tags
    asset = library.update_asset(asset_id, **updates)
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    return LibraryAssetResponse(**asset.__dict__)


@router.post("/library/{asset_id}/tags", response_model=LibraryAssetResponse)
def add_asset_tags(asset_id: str, req: AddTagsRequest, library: AssetLibrary = Depends(get_library)):
    asset = library.add_tags(asset_id, req.tags)
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    return LibraryAssetResponse(**asset.__dict__)


@router.delete("/library/{asset_id}/tags", response_model=LibraryAssetResponse)
def remove_asset_tags(asset_id: str, req: RemoveTagsRequest, library: AssetLibrary = Depends(get_library)):
    asset = library.remove_tags(asset_id, req.tags)
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    return LibraryAssetResponse(**asset.__dict__)
