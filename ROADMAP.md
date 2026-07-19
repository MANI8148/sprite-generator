# AI Game Asset Pipeline — ROADMAP

## Philosophy
The LoRA is NOT the product. The pipeline is the product.
We build an end-to-end asset pipeline, not a better image generator.

## Structure
```
backend/
├── main.py                    # FastAPI entry point
├── api/                       # API route definitions
└── modules/
    ├── generator/             # SD 1.5 LoRA inference
    ├── pipeline/              # Orchestrator (end-to-end flow)
    ├── postprocess/           # rembg, palette, cleanup, upscale
    ├── packing/               # Sprite sheet, tileset, animation strip
    ├── exporters/             # Godot, Unity, generic PNG, ZIP
    ├── validation/            # Quality metrics (palette, sharpness, etc.)
    ├── prompt_builder/        # Structured controls → optimized prompt
    └── storage/               # Local file management

gradio_app/
└── app.py                     # MVP frontend (Gradio)

kaggle/
└── kaggle_complete_train.ipynb  # LoRA training notebook
```

## Pipeline Flow
```
User Controls (AssetType, View, Palette, etc.)
    ↓
Prompt Builder → structured prompt
    ↓
SD 1.5 LoRA Generator → raw images
    ↓
Post-Processing (rembg, palette↓, cleanup, center, upscale)
    ↓
Validation Metrics (quality tier, report)
    ↓
Packing (sprite sheet / animation strip / tileset)
    ↓
Exporters (Godot .tres / Unity .meta / PNG + ZIP)
    ↓
Download
```

## Milestones

### MVP — Gradio Demo (NOW)
- [x] Modular pipeline architecture (all 8 modules)
- [x] Prompt builder with structured controls
- [x] Post-processing (rembg, palette reduction, pixel cleanup, auto-center, upscale)
- [x] Validation metrics (palette, sharpness, centering, transparency, outline)
- [x] Sprite packing (sprite sheet, animation strip)
- [x] Exporters (Godot, Unity, generic PNG + ZIP)
- [x] Pipeline orchestrator
- [x] Deploy Gradio demo on Hugging Face Spaces
- [x] Smoke test with real LoRA weights end-to-end

### Phase 1 — FastAPI + Next.js
- [x] FastAPI backend with /generate, /download, /health, /history
- [x] Next.js frontend with Generate, History, Downloads, Settings pages
- [x] File-based storage (no DB yet)
- [x] Rate limiting (per-IP, no auth)
- [x] Deploy on HF Spaces or cheap VPS

### Phase 2 — Production Hardening
- [x] SQLite → PostgreSQL (only when needed)
- [x] Background task queue (only when concurrent users > 1)
- [x] Asset library with persistent storage
- [x] Multi-asset batch generation
- [x] Style consistency engine (IP-Adapter, palette lock)

### Phase 3 — Scale
- [x] More generator modules (tilesets, environments, UI, props)
- [x] More exporters (GameMaker, Phaser)
- [x] Cloud storage (Cloudflare R2)
- [x] Incremental regeneration ("asset memory")

## Explicitly Deferred
- ❌ Billing / payments
- [x] Authentication / user accounts
- ❌ LLM Project Director
- ❌ Celery / Redis / DAG orchestrator
- ❌ Multi-user team features
- ❌ Real-ESRGAN upscaling

Until the MVP is validated with real users.
