---
title: Sprite Generator API
emoji: 🎮
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Sprite Generator API

FastAPI backend for the AI Game Asset Pipeline, deployed to Hugging Face
Spaces as a Docker SDK space. Exposes the `/generate`, `/download`,
`/health`, `/history`, `/load-model`, `/status/{job_id}`, `/generate/batch`,
`/batch-status/{id}`, `/regenerate`, `/library`, `/style-presets`,
`/palettes`, `/plan` and `/plan/execute` endpoints.

## Run locally

```bash
pip install -r requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

## Endpoints

- `GET  /health` — liveness probe
- `POST /load-model` — load the SD 1.5 LoRA generator
- `POST /generate` — submit a generation job
- `GET  /status/{job_id}` — poll a job
- `GET  /download/{job_id}` — download the ZIP package
- `GET  /history` — list past jobs

See the [OpenAPI docs](https://github.com/MANI8148/sprite-generator) for the
full schema.
