from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, model_validator
from typing import Optional, List
import os
import uuid

from backend.modules.prompt_builder.controls import (
    AssetControls, AssetType, View, Palette, Animation, SpriteSize,
)
from backend.modules.pipeline.orchestrator import AssetPipeline
from backend.modules.storage.file_storage import FileStorage
from backend.modules.storage.r2_storage import R2Storage
from backend.modules.storage.asset_library import AssetLibrary, AssetRecord
from backend.modules.storage.database import DatabaseLibrary
from backend.modules.tasks.queue import TaskQueue, get_task_queue, JobStatus
from backend.modules.style_engine import StyleEngine, STYLE_PRESETS
from backend.modules.asset_memory import compute_generation_hash
from backend.modules.auth import OptionalAuth, TokenData
from backend.modules.billing import CreditManager, get_credit_manager, UsageTracker, get_usage_tracker
from backend.modules.project_director import ProjectDirector, ProjectPlan, LLMProjectDirector

router = APIRouter()

_pipeline: AssetPipeline = None
_generator_loaded: bool = False
_storage = FileStorage()
_library = AssetLibrary()
_batch_jobs: dict[str, list[str]] = {}
_r2_storage: R2Storage = R2Storage()
_job_store: Optional[DatabaseLibrary] = None


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


_style_engine = StyleEngine()


def get_style_engine() -> StyleEngine:
    return _style_engine


def get_r2_storage() -> R2Storage:
    return _r2_storage


def set_r2_storage(st: R2Storage) -> None:
    global _r2_storage
    _r2_storage = st


def get_job_store() -> Optional[DatabaseLibrary]:
    return _job_store


def set_job_store(store: Optional[DatabaseLibrary]) -> None:
    global _job_store
    _job_store = store


def _persist_job(job_id: str, status_value: str, **fields) -> None:
    store = get_job_store()
    if store is None:
        return
    try:
        if store.get_job(job_id) is None:
            store.add_job(job_id, status=status_value, **fields)
        else:
            store.update_job(job_id, status=status_value, **fields)
    except Exception:
        pass


def _stored_job_as_queue_entry(job_id: str) -> Optional[dict]:
    store = get_job_store()
    if store is None:
        return None
    record = store.get_job(job_id)
    if record is None:
        return None
    entry = {"status": record["status"], "result": None, "error": record["error"] or None}
    if record["status"] == JobStatus.DONE.value and (record["zip_path"] or record["output_paths"]):
        entry["result"] = {
            "prompt": record["prompt"],
            "quality_tier": record["quality_tier"],
            "validation": record["validation"],
            "zip_path": record["zip_path"],
            "output_paths": record["output_paths"],
        }
    return entry


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
    cached: bool = False
    error: Optional[str] = None


class GenerateRequest(BaseModel):
    asset_type: str = "character"
    view: str = "front"
    animation: str = "idle"
    palette: str = "auto"
    sprite_size: str = "32x32"
    theme: str = ""
    style: str = "pixel art"
    seed: int = -1
    remove_bg: bool = True
    reduce_palette: bool = True
    max_colors: int = 32
    pixel_cleanup: bool = True
    auto_center: bool = True
    upscale: int = 1
    use_realesrgan: bool = False
    engine: str = "godot"
    num_frames: int = 1
    palette_lock: bool = False
    palette_name: str = "retro_16"
    style_preset: str = ""
    pack_tileset: bool = False
    ip_adapter: bool = False
    ip_adapter_scale: float = 0.6
    reference_image: Optional[str] = None

    @model_validator(mode="after")
    def _validate_ip_adapter(self):
        if self.ip_adapter and not self.reference_image:
            raise ValueError(
                "reference_image is required when ip_adapter is enabled"
            )
        return self


class BatchItem(BaseModel):
    asset_type: str = "character"
    view: str = "front"
    animation: str = "idle"
    palette: str = "auto"
    sprite_size: str = "32x32"
    theme: str = ""
    style: str = "pixel art"
    seed: int = -1
    num_frames: int = 1
    remove_bg: bool = True
    reduce_palette: bool = True
    max_colors: int = 32
    pixel_cleanup: bool = True
    auto_center: bool = True
    upscale: int = 1
    use_realesrgan: bool = False
    engine: str = "godot"
    palette_lock: bool = False
    palette_name: str = "retro_16"
    style_preset: str = ""
    pack_tileset: bool = False
    ip_adapter: bool = False
    ip_adapter_scale: float = 0.6
    reference_image: Optional[str] = None

    @model_validator(mode="after")
    def _validate_ip_adapter(self):
        if self.ip_adapter and not self.reference_image:
            raise ValueError(
                "reference_image is required when ip_adapter is enabled"
            )
        return self


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
    cached: bool = False


class BatchGenerateResponse(BaseModel):
    batch_id: str
    job_ids: List[str]
    total: int
    status: str = "pending"


class BatchStatusResponse(BaseModel):
    batch_id: str
    total: int
    completed: int
    failed: int
    running: int
    pending: int
    status: str
    results: List[BatchResult] = []


class RegenerateRequest(BaseModel):
    base_id: str
    num_variants: int = 4
    seed_stride: int = 1
    theme_override: str = ""


class RegenerateResponse(BaseModel):
    base_id: str
    batch_id: str
    job_ids: List[str]
    num_variants: int


_director: ProjectDirector = LLMProjectDirector()


def get_director() -> ProjectDirector:
    return _director


def set_director(d: ProjectDirector) -> None:
    global _director
    _director = d


class PlanRequest(BaseModel):
    request: str
    auto_execute: bool = False


class PlanResponse(BaseModel):
    title: str
    steps: List[dict]
    total_steps: int


class ExecutePlanResponse(BaseModel):
    batch_id: str
    job_ids: List[str]
    total: int


@router.post("/plan", response_model=PlanResponse)
def create_plan(req: PlanRequest, director: ProjectDirector = Depends(get_director)):
    plan = director.parse(req.request)
    return PlanResponse(
        title=plan.title,
        steps=[s.to_dict() for s in plan.steps],
        total_steps=len(plan.steps),
    )


@router.post("/plan/execute", response_model=ExecutePlanResponse, status_code=202)
def execute_plan(
    req: PlanRequest,
    pipe: AssetPipeline = Depends(get_pipeline),
    director: ProjectDirector = Depends(get_director),
    current_user: Optional[TokenData] = Depends(OptionalAuth),
):
    global _generator_loaded
    if not _generator_loaded:
        raise HTTPException(status_code=503, detail="Generator not set. POST /load-model first.")

    plan = director.parse(req.request)
    batch_id = str(uuid.uuid4())[:8]
    job_ids: List[str] = []

    user_id = current_user.user_id if current_user else None
    if user_id:
        credits = get_credit_manager()
        credits.ensure_user_exists(user_id)
        total_cost = sum(
            credits.get_generation_cost() * max(1, step.num_frames)
            for step in plan.steps
        )
        if credits.get_balance(user_id) < total_cost:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail=f"Insufficient credits. Need {total_cost}, have {credits.get_balance(user_id)}",
            )

    for i, step in enumerate(plan.steps):
        job_id = f"{batch_id}_{i}"
        job_ids.append(job_id)
        output_dir = _storage.ensure_output_dir(job_id)

        item = GenerateRequest(
            asset_type=step.asset_type.value,
            view=step.view.value,
            animation=step.animation.value,
            palette=step.palette.value,
            sprite_size=step.sprite_size.value,
            theme=step.theme,
            num_frames=step.num_frames,
            seed=step.seed,
        )

        queue = get_task_queue()
        queue.submit(_run_batch_item, job_id, pipe, item, output_dir, batch_id, user_id)
        _persist_job(job_id, JobStatus.PENDING.value, batch_id=batch_id)

    _batch_jobs[batch_id] = job_ids

    return ExecutePlanResponse(
        batch_id=batch_id,
        job_ids=job_ids,
        total=len(plan.steps),
    )


class HealthResponse(BaseModel):
    status: str
    generator_loaded: bool


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", generator_loaded=_generator_loaded)


class StylePresetResponse(BaseModel):
    name: str
    description: str
    palette_name: str
    apply_palette_lock: bool


@router.get("/style-presets")
def list_style_presets():
    return {
        "presets": [
            StylePresetResponse(
                name=p.name,
                description=p.description,
                palette_name=p.palette_name,
                apply_palette_lock=p.apply_palette_lock,
            )
            for p in STYLE_PRESETS.values()
        ]
    }


class PaletteListResponse(BaseModel):
    palettes: List[str]


@router.get("/palettes", response_model=PaletteListResponse)
def list_palettes(engine: StyleEngine = Depends(get_style_engine)):
    return PaletteListResponse(palettes=engine.get_available_palettes())


class R2StatusResponse(BaseModel):
    available: bool
    bucket: str = ""
    endpoint: str = ""


@router.get("/storage/r2-status", response_model=R2StatusResponse)
def r2_status(r2: R2Storage = Depends(get_r2_storage)):
    return R2StatusResponse(
        available=r2.available,
        bucket=r2._bucket_name if hasattr(r2, '_bucket_name') else "",
        endpoint=r2._endpoint if hasattr(r2, '_endpoint') else "",
    )


def _upload_to_r2(storage: R2Storage, job_id: str, output_paths: list, zip_path: Optional[str] = None):
    if not storage.available:
        return
    for path in output_paths:
        if os.path.isfile(path):
            rel = os.path.relpath(path, os.path.dirname(output_paths[0]) if output_paths else ".")
            storage.upload_file(path, f"jobs/{job_id}/{rel}")
    if zip_path and os.path.isfile(zip_path):
        storage.upload_file(zip_path, f"jobs/{job_id}/sprite_package.zip")


def _store_control_snapshot(meta: dict, req) -> None:
    """Persist the generation controls so later regeneration can replicate
    the exact style context (incremental regeneration / asset memory)."""
    meta["seed"] = req.seed
    meta["theme"] = req.theme
    meta["control_snapshot"] = {
        "asset_type": req.asset_type,
        "view": req.view,
        "animation": req.animation,
        "palette": req.palette,
        "sprite_size": req.sprite_size,
        "theme": req.theme,
        "style": req.style,
        "seed": req.seed,
    }


def _deduct_credits_for_job(user_id, num_frames, credits):
    if user_id is None:
        return
    cost = credits.get_generation_cost() * max(1, num_frames)
    credits.deduct_credits(user_id, cost, reason="generation")
    usage = get_usage_tracker()
    usage.record_generation(user_id, cost)


def _run_generation_job(pipe, controls, req, output_dir, job_id, user_id=None):
    credits = get_credit_manager()
    if user_id:
        credits.ensure_user_exists(user_id)

    _persist_job(job_id, JobStatus.RUNNING.value)

    original_config = pipe.config
    from copy import deepcopy
    config = deepcopy(pipe.config)
    config.remove_bg = req.remove_bg
    config.reduce_palette = req.reduce_palette
    config.max_colors = req.max_colors
    config.pixel_cleanup = req.pixel_cleanup
    config.auto_center = req.auto_center
    config.upscale = req.upscale
    config.use_realesrgan = req.use_realesrgan
    config.export_engine = req.engine
    config.pack_sheet = req.num_frames > 1
    config.pack_tileset = req.pack_tileset
    config.palette_lock = req.palette_lock
    config.palette_name = req.palette_name
    config.ip_adapter = req.ip_adapter
    config.ip_adapter_scale = req.ip_adapter_scale
    config.reference_image = req.reference_image
    if req.style_preset:
        preset = STYLE_PRESETS.get(req.style_preset.lower())
        if preset:
            config.palette_lock = preset.apply_palette_lock
            config.palette_name = preset.palette_name
    pipe.config = config

    gen_hash = compute_generation_hash(controls, config)

    cached = _library.find_by_generation_hash(gen_hash)
    if cached is not None:
        result = {
            "prompt": cached.prompt,
            "quality_tier": cached.quality_tier,
            "validation": cached.metadata.get("validation", {}),
            "zip_path": cached.zip_path,
            "output_paths": cached.output_paths,
            "cached": True,
        }
        _storage.add_job(job_id, {
            "prompt": cached.prompt,
            "quality_tier": cached.quality_tier,
            "outputs": cached.output_paths,
            "zip_path": cached.zip_path,
            "cached": True,
        })
        _persist_job(
            job_id,
            JobStatus.DONE.value,
            prompt=cached.prompt,
            quality_tier=cached.quality_tier,
            validation=cached.metadata.get("validation", {}),
            zip_path=cached.zip_path,
            output_paths=cached.output_paths,
        )
        return result

    try:
        result = pipe.run(controls, output_dir=output_dir)

        _deduct_credits_for_job(user_id, req.num_frames, credits)

        _storage.add_job(job_id, {
            "prompt": result.metadata["prompt"],
            "quality_tier": result.validation[0]["quality_tier"],
            "outputs": result.output_paths,
            "zip_path": result.zip_path,
        })

        _upload_to_r2(_r2_storage, job_id, result.output_paths, result.zip_path)

        meta = {"view": req.view, "animation": req.animation, "palette": req.palette, "sprite_size": req.sprite_size, "style": req.style}
        meta["generation_hash"] = gen_hash
        meta["validation"] = result.validation[0] if result.validation else {}
        _store_control_snapshot(meta, req)

        _library.add_asset(AssetRecord(
            asset_id=job_id,
            job_id=job_id,
            asset_type=req.asset_type,
            prompt=result.metadata["prompt"],
            quality_tier=result.validation[0]["quality_tier"],
            zip_path=result.zip_path,
            output_paths=result.output_paths,
            metadata=meta,
            generation_hash=gen_hash,
        ))
    except Exception as exc:
        _persist_job(job_id, JobStatus.FAILED.value, error=str(exc))
        raise
    finally:
        pipe.config = original_config

    _persist_job(
        job_id,
        JobStatus.DONE.value,
        prompt=result.metadata["prompt"],
        quality_tier=result.validation[0]["quality_tier"],
        validation=result.validation[0] if result.validation else {},
        zip_path=result.zip_path,
        output_paths=result.output_paths,
    )

    return {
        "prompt": result.metadata["prompt"],
        "quality_tier": result.validation[0]["quality_tier"],
        "validation": result.validation[0],
        "zip_path": result.zip_path,
        "output_paths": result.output_paths,
    }


@router.post("/generate", response_model=GenerateResponse, status_code=202)
def generate(
    req: GenerateRequest,
    pipe: AssetPipeline = Depends(get_pipeline),
    current_user: Optional[TokenData] = Depends(OptionalAuth),
):
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
        style=req.style,
        seed=req.seed,
    )

    user_id = current_user.user_id if current_user else None
    if user_id:
        credits = get_credit_manager()
        credits.ensure_user_exists(user_id)
        cost = credits.get_generation_cost() * max(1, req.num_frames)
        if credits.get_balance(user_id) < cost:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail=f"Insufficient credits. Need {cost}, have {credits.get_balance(user_id)}",
            )

    job_id = str(uuid.uuid4())[:8]
    output_dir = _storage.ensure_output_dir(job_id)

    queue = get_task_queue()
    queue.submit(_run_generation_job, job_id, pipe, controls, req, output_dir, job_id, user_id)
    _persist_job(job_id, JobStatus.PENDING.value)

    return GenerateResponse(job_id=job_id, status=JobStatus.PENDING)


@router.get("/status/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    queue = get_task_queue()
    job = queue.get_status(job_id)
    if job is None:
        job = _stored_job_as_queue_entry(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    resp = JobStatusResponse(job_id=job_id, status=job["status"].value if isinstance(job["status"], JobStatus) else job["status"])
    if job["result"] is not None:
        resp.prompt = job["result"].get("prompt")
        resp.quality_tier = job["result"].get("quality_tier")
        resp.validation = job["result"].get("validation")
        resp.zip_path = job["result"].get("zip_path")
        resp.output_paths = job["result"].get("output_paths")
        resp.cached = bool(job["result"].get("cached", False))
    if job["error"] is not None:
        resp.error = job["error"]
    return resp


@router.post("/generate/batch", response_model=BatchGenerateResponse, status_code=202)
def generate_batch(
    req: BatchGenerateRequest,
    pipe: AssetPipeline = Depends(get_pipeline),
    current_user: Optional[TokenData] = Depends(OptionalAuth),
):
    global _generator_loaded, _batch_jobs
    if not _generator_loaded:
        raise HTTPException(status_code=503, detail="Generator not set. POST /load-model first.")

    user_id = current_user.user_id if current_user else None
    if user_id:
        credits = get_credit_manager()
        credits.ensure_user_exists(user_id)
        total_cost = sum(
            credits.get_generation_cost() * max(1, item.num_frames)
            for item in req.items
        )
        if credits.get_balance(user_id) < total_cost:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail=f"Insufficient credits. Need {total_cost}, have {credits.get_balance(user_id)}",
            )

    batch_id = req.batch_id or str(uuid.uuid4())[:8]
    job_ids: List[str] = []

    for i, item in enumerate(req.items):
        job_id = f"{batch_id}_{i}"
        job_ids.append(job_id)
        output_dir = _storage.ensure_output_dir(job_id)

        queue = get_task_queue()
        queue.submit(_run_batch_item, job_id, pipe, item, output_dir, batch_id, user_id)
        _persist_job(job_id, JobStatus.PENDING.value, batch_id=batch_id)

    _batch_jobs[batch_id] = job_ids

    return BatchGenerateResponse(
        batch_id=batch_id,
        job_ids=job_ids,
        total=len(req.items),
        status="pending",
    )


def _run_batch_item(pipe, item, output_dir, batch_id, user_id=None):
    credits = get_credit_manager()
    if user_id:
        credits.ensure_user_exists(user_id)

    job_id = os.path.basename(output_dir.rstrip("/\\"))
    _persist_job(job_id, JobStatus.RUNNING.value, batch_id=batch_id)

    controls = AssetControls(
        asset_type=AssetType(item.asset_type),
        view=View(item.view),
        animation=Animation(item.animation),
        palette=Palette(item.palette),
        sprite_size=SpriteSize(item.sprite_size),
        theme=item.theme,
        style=item.style,
        seed=item.seed,
    )

    original_config = pipe.config
    from copy import deepcopy
    config = deepcopy(pipe.config)
    config.remove_bg = item.remove_bg
    config.reduce_palette = item.reduce_palette
    config.max_colors = item.max_colors
    config.pixel_cleanup = item.pixel_cleanup
    config.auto_center = item.auto_center
    config.upscale = item.upscale
    config.use_realesrgan = item.use_realesrgan
    config.export_engine = item.engine
    config.pack_sheet = item.num_frames > 1
    config.pack_tileset = item.pack_tileset
    config.palette_lock = item.palette_lock
    config.palette_name = item.palette_name
    config.ip_adapter = item.ip_adapter
    config.ip_adapter_scale = item.ip_adapter_scale
    config.reference_image = item.reference_image
    if item.style_preset:
        preset = STYLE_PRESETS.get(item.style_preset.lower())
        if preset:
            config.palette_lock = preset.apply_palette_lock
            config.palette_name = preset.palette_name
    pipe.config = config

    gen_hash = compute_generation_hash(controls, config)

    cached = _library.find_by_generation_hash(gen_hash)
    if cached is not None:
        job_id = os.path.basename(output_dir.rstrip("/\\"))
        _storage.add_job(job_id, {
            "prompt": cached.prompt,
            "quality_tier": cached.quality_tier,
            "outputs": cached.output_paths,
            "zip_path": cached.zip_path,
            "batch_id": batch_id,
            "cached": True,
        })
        _persist_job(
            job_id,
            JobStatus.DONE.value,
            batch_id=batch_id,
            prompt=cached.prompt,
            quality_tier=cached.quality_tier,
            validation=cached.metadata.get("validation", {}),
            zip_path=cached.zip_path,
            output_paths=cached.output_paths,
        )
        return {
            "prompt": cached.prompt,
            "quality_tier": cached.quality_tier,
            "validation": cached.metadata.get("validation", {}),
            "zip_path": cached.zip_path,
            "output_paths": cached.output_paths,
            "cached": True,
        }

    job_id = os.path.basename(output_dir.rstrip("/\\"))

    try:
        result = pipe.run(controls, output_dir=output_dir)

        _deduct_credits_for_job(user_id, item.num_frames, credits)

        _upload_to_r2(_r2_storage, job_id, result.output_paths, result.zip_path)

        _storage.add_job(job_id, {
            "prompt": result.metadata["prompt"],
            "quality_tier": result.validation[0]["quality_tier"],
            "outputs": result.output_paths,
            "zip_path": result.zip_path,
            "batch_id": batch_id,
        })

        meta = {"view": item.view, "animation": item.animation, "palette": item.palette, "sprite_size": item.sprite_size, "style": item.style}
        meta["generation_hash"] = gen_hash
        meta["validation"] = result.validation[0] if result.validation else {}
        _store_control_snapshot(meta, item)

        _library.add_asset(AssetRecord(
            asset_id=job_id,
            job_id=job_id,
            asset_type=item.asset_type,
            prompt=result.metadata["prompt"],
            quality_tier=result.validation[0]["quality_tier"],
            zip_path=result.zip_path,
            output_paths=result.output_paths,
            metadata=meta,
            generation_hash=gen_hash,
        ))
    except Exception as exc:
        _persist_job(job_id, JobStatus.FAILED.value, error=str(exc), batch_id=batch_id)
        raise
    finally:
        pipe.config = original_config

    _persist_job(
        job_id,
        JobStatus.DONE.value,
        batch_id=batch_id,
        prompt=result.metadata["prompt"],
        quality_tier=result.validation[0]["quality_tier"],
        validation=result.validation[0] if result.validation else {},
        zip_path=result.zip_path,
        output_paths=result.output_paths,
    )

    return {
        "prompt": result.metadata["prompt"],
        "quality_tier": result.validation[0]["quality_tier"],
        "validation": result.validation[0],
        "zip_path": result.zip_path,
        "output_paths": result.output_paths,
    }


@router.get("/batch-status/{batch_id}", response_model=BatchStatusResponse)
def get_batch_status(batch_id: str):
    global _batch_jobs
    if batch_id not in _batch_jobs:
        raise HTTPException(status_code=404, detail="Batch not found")

    job_ids = _batch_jobs[batch_id]
    queue = get_task_queue()
    results: List[BatchResult] = []
    completed = 0
    failed = 0
    running = 0
    pending = 0

    for job_id in job_ids:
        job = queue.get_status(job_id) or _stored_job_as_queue_entry(job_id)
        if job is None:
            pending += 1
        elif job["status"] == JobStatus.DONE:
            completed += 1
            if job["result"] is not None:
                results.append(BatchResult(
                    job_id=job_id,
                    prompt=job["result"].get("prompt", ""),
                    quality_tier=job["result"].get("quality_tier", ""),
                    validation=job["result"].get("validation", {}),
                    zip_path=job["result"].get("zip_path"),
                    output_paths=job["result"].get("output_paths", []),
                    cached=bool(job["result"].get("cached", False)),
                ))
        elif job["status"] == JobStatus.FAILED:
            failed += 1
            results.append(BatchResult(
                job_id=job_id,
                prompt="",
                quality_tier="error",
                validation={"error": job.get("error", "unknown")},
                zip_path=None,
                output_paths=[],
            ))
        elif job["status"] == JobStatus.RUNNING:
            running += 1
        else:
            pending += 1

    total = len(job_ids)
    done_count = completed + failed
    if done_count == total:
        batch_status = "done"
    elif failed > 0:
        batch_status = "partial_failure"
    else:
        batch_status = "running"

    return BatchStatusResponse(
        batch_id=batch_id,
        total=total,
        completed=completed,
        failed=failed,
        running=running,
        pending=pending,
        status=batch_status,
        results=results,
    )


@router.post("/regenerate", response_model=RegenerateResponse, status_code=202)
def regenerate(
    req: RegenerateRequest,
    pipe: AssetPipeline = Depends(get_pipeline),
    current_user: Optional[TokenData] = Depends(OptionalAuth),
):
    """Incremental regeneration (asset memory): spawn variant jobs for a base asset.

    Each variant keeps the base asset's style context (asset type / view /
    animation / palette / sprite size / style / theme) while varying the seed
    deterministically, so identical regenerate requests are served from cache
    and every distinct variant is stored in the asset library.
    """
    global _generator_loaded, _batch_jobs
    if not _generator_loaded:
        raise HTTPException(status_code=503, detail="Generator not set. POST /load-model first.")

    if req.num_variants < 1 or req.num_variants > 64:
        raise HTTPException(status_code=422, detail="num_variants must be between 1 and 64")

    from backend.modules.asset_memory import AssetMemory
    library = get_library()
    memory = AssetMemory(library=library)
    try:
        _, variant_controls = memory.plan_variants(
            req.base_id,
            num_variants=req.num_variants,
            seed_stride=req.seed_stride,
            theme_override=req.theme_override,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Base asset {req.base_id} not found")

    user_id = current_user.user_id if current_user else None
    credits = get_credit_manager()
    if user_id:
        credits.ensure_user_exists(user_id)
        total_cost = credits.get_generation_cost() * len(variant_controls)
        if credits.get_balance(user_id) < total_cost:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail=f"Insufficient credits. Need {total_cost}, have {credits.get_balance(user_id)}",
            )

    batch_id = str(uuid.uuid4())[:8]
    job_ids: List[str] = []
    queue = get_task_queue()

    for i, ctrl in enumerate(variant_controls):
        job_id = f"{batch_id}_{i}"
        job_ids.append(job_id)
        output_dir = _storage.ensure_output_dir(job_id)

        item = GenerateRequest(
            asset_type=ctrl.asset_type.value,
            view=ctrl.view.value,
            animation=ctrl.animation.value,
            palette=ctrl.palette.value,
            sprite_size=ctrl.sprite_size.value,
            theme=ctrl.theme,
            style=ctrl.style,
            seed=ctrl.seed,
        )
        queue.submit(_run_batch_item, job_id, pipe, item, output_dir, batch_id, user_id)
        _persist_job(job_id, JobStatus.PENDING.value, batch_id=batch_id)

    _batch_jobs[batch_id] = job_ids

    return RegenerateResponse(
        base_id=req.base_id,
        batch_id=batch_id,
        job_ids=job_ids,
        num_variants=len(job_ids),
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
