# Deployment Guide

This document covers how to run and deploy the AI Game Asset Pipeline in
development, in Docker, and on Hugging Face Spaces.

## Architecture

- `backend/` — FastAPI service exposing `/generate`, `/download`, `/history`,
  `/health`, `/status/{job_id}`, `/generate/batch`, `/batch-status/{id}`,
  `/regenerate`, `/library`, `/style-presets`, `/palettes`, `/plan`,
  `/plan/execute`, plus auth, billing and team routes.
- `frontend/` — Next.js app (Generate, History, Downloads, Settings pages)
  that talks to the backend through an `/api` rewrite.
- `gradio_app/` — Gradio demo deployed to Hugging Face Spaces.
- `demo/` — older VQ-VAE Gradio demo also deployed to Spaces.

## Local development

### Backend

```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```

The service listens on `http://localhost:8000` and serves the OpenAPI docs at
`/docs`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend proxies `/api/*` to `NEXT_PUBLIC_API_URL` (defaults to
`http://localhost:8000`). Start it alongside the backend.

### Running the tests

```bash
python -m pytest tests/ -v --tb=short
```

The canonical CI pipeline lives at `scripts/ci.yml` and is mirrored by the
`test` service in `docker-compose.yml`.

## Environment variables

All configuration is read from environment variables; no secrets are
hardcoded. Set these in your deployment environment.

| Variable | Purpose | Default |
| --- | --- | --- |
| `ALLOWED_ORIGINS` | Comma-separated CORS origins for the API | `http://localhost:3000,http://127.0.0.1:3000,http://localhost:8000,http://127.0.0.1:8000` |
| `DATABASE_URL` | PostgreSQL URL; when set the job store / asset library uses Postgres instead of SQLite | unset (SQLite) |
| `REDIS_URL` | Redis URL; when set the task queue uses Redis instead of the in-memory queue | unset (in-memory) |
| `AUTH_SECRET_KEY` | Secret used to sign auth tokens | unset |
| `AUTH_TOKEN_EXPIRE_MINUTES` | Auth token lifetime | unset |
| `BILLING_FREE_CREDITS` | Free credits granted to new users | unset |
| `BILLING_GENERATION_COST` | Credits deducted per generation | unset |
| `BILLING_ADMIN_USERNAME` | Username with billing admin rights | unset |
| `STRIPE_API_KEY` | Stripe secret key for payments | unset |
| `STRIPE_WEBHOOK_SECRET` | Stripe webhook signing secret | unset |
| `STRIPE_PRICE_STARTER` / `STRIPE_PRICE_PRO` / `STRIPE_PRICE_STUDIO` | Stripe price IDs | unset |
| `STRIPE_CREDITS_PER_UNIT_CENTS` | Credits granted per cent charged | unset |
| `PROJECT_DIRECTOR_API_KEY` | API key for the LLM Project Director provider | unset |
| `PROJECT_DIRECTOR_BASE_URL` | OpenAI-compatible base URL for the Project Director | unset |
| `PROJECT_DIRECTOR_MODEL` | Model name for the Project Director | unset |
| `OPENAI_API_KEY` | OpenAI key used by the Project Director | unset |

## Docker

A multi-stage `Dockerfile` builds the backend and frontend images; the
`docker-compose.yml` orchestrates the full stack.

### Building and running the full stack

```bash
docker compose up --build
```

This starts:

- `db` — PostgreSQL 16 (port 5432), used when `DATABASE_URL` is set
- `redis` — Redis 7 (port 6379), used when `REDIS_URL` is set
- `backend` — FastAPI on port 8000
- `frontend` — Next.js on port 3000

### Running the test service

```bash
docker compose run --rm test
```

The `test` service builds the `test` target of the Dockerfile and runs the
pytest suite. It mounts `./tests` and the canonical CI definition
(`./scripts/ci.yml`) so the CI that runs locally matches CI in GitHub Actions.

### Environment overrides

Override configuration per environment with a `.env` file or inline variables:

```bash
DATABASE_URL=postgresql://user:pass@db:5432/sprite_gen \
REDIS_URL=redis://redis:6379/0 \
docker compose up --build
```

## Deployment on a VPS

1. Check out the repository on the host.
2. Set the environment variables above (secrets via your secret manager or
   `.env`).
3. Build and start the stack:

```bash
docker compose up -d --build
```

4. Put a reverse proxy (Caddy/nginx) in front of ports 8000 and 3000 with TLS,
   and point `ALLOWED_ORIGINS` at the public origin.

## Deployment on Hugging Face Spaces

### Gradio demo

The current Gradio demo lives in `gradio_app/` and deploys to
`darklord8777/sprite-generator-demo`.

Automated deployment:

- `.github/workflows/deploy_demo.yml` uploads the `demo/` and `models/`
  directories to the Space on push to `main` or via `workflow_dispatch`.
  It needs the `HF_TOKEN` repository secret.

Manual deployment:

```bash
python scripts/deploy_spaces.py \
  --space-repo darklord8777/sprite-generator-demo \
  --token "$HF_TOKEN"
```

Use `--dry-run` to preview the files that would be uploaded:

```bash
python scripts/deploy_spaces.py --space-repo darklord8777/sprite-generator-demo --dry-run
```

The Space README (`gradio_app/README.md`) carries the Spaces configuration
(`sdk: gradio`, `app_file: app.py`).

## CI/CD

The canonical CI workflow is `scripts/ci.yml`. It:

- triggers on push and pull requests to `main`
- runs on Python 3.10
- installs dependencies (CPU torch + `requirements.txt` + pytest)
- verifies no hardcoded credentials (`python -m scripts.hf_token_hygiene`)
- runs the full test suite

It is executed by the `test` service of `docker-compose.yml` and is the single
source of truth for what CI does. GitHub Actions workflows in
`.github/workflows/` cover automation (auto-improve, HF Spaces deploy, Kaggle
training, data pipeline, checkpoint sync, docs update).

## Storage

- **Local / SQLite**: default. Job records and the asset library are stored in
  SQLite (file-backed).
- **PostgreSQL**: set `DATABASE_URL` to a valid connection string. Backend,
  job store, and asset library all support the Postgres dialect.
- **Cloudflare R2**: set R2 credentials/environment and the storage layer
  uploads job outputs to R2. Health is exposed via `/storage/r2-status`.

## Health checks

- API: `GET /health`
- R2: `GET /storage/r2-status`
