# AI Game Studio — Architecture Map & Conversion Roadmap

Generated from the current sprite-generator project (VQ-VAE + Transformer / SD-LoRA pivot)
toward the full "AI Game Studio" vision. This is sequenced realistically for a solo
developer using AI-assisted coding, free/low-cost infra, starting from what already exists.

---

## PART 1 — ARCHITECTURE MAP

### 1.1 Full target architecture (end-state vision)

```
                                   ┌─────────────────────┐
                                   │        USER          │
                                   └──────────┬───────────┘
                                              │
                                   ┌──────────▼───────────┐
                                   │   Next.js Frontend    │
                                   │  (React, TS, Tailwind)│
                                   │  - Prompt builder UI  │
                                   │  - Project dashboard  │
                                   │  - Asset library view │
                                   │  - Export panel       │
                                   └──────────┬───────────┘
                                              │ REST/WebSocket
                                   ┌──────────▼───────────┐
                                   │   FastAPI Backend     │
                                   │  - Auth (JWT)         │
                                   │  - Project CRUD       │
                                   │  - Job submission      │
                                   │  - Billing/usage       │
                                   └──────┬──────────┬─────┘
                                          │          │
                         ┌────────────────▼──┐   ┌──▼─────────────────┐
                         │  AI Project        │   │   PostgreSQL        │
                         │  Director (LLM)    │   │  - users             │
                         │  - Parses request   │   │  - projects          │
                         │  - Builds asset plan│   │  - assets             │
                         │  - Breaks into DAG  │   │  - jobs/status        │
                         └────────────┬────────┘   │  - style profiles     │
                                      │             └───────────────────────┘
                         ┌────────────▼────────────┐
                         │  Workflow Orchestrator    │
                         │  (DAG scheduler)          │
                         │  - Dependency resolution   │
                         │  - Retry/failure handling  │
                         └────────────┬───────────────┘
                                      │
                         ┌────────────▼───────────────┐
                         │   Redis Queue (Celery)       │
                         └────────────┬───────────────┘
                                      │
                ┌─────────────────────┼─────────────────────┐
                │                     │                       │
     ┌──────────▼─────────┐ ┌────────▼─────────┐  ┌──────────▼──────────┐
     │  GPU Worker Pool     │ │  GPU Worker Pool  │  │  GPU Worker Pool     │
     │  Character Generator │ │ Building Generator│  │  Tileset Generator   │
     └──────────┬─────────┘ └────────┬─────────┘  └──────────┬──────────┘
                │                     │                       │
                └──────────┬──────────┴───────────┬───────────┘
                           │                       │
                ┌──────────▼───────────┐ ┌─────────▼────────────┐
                │   Style Engine         │ │  (Environment / UI /  │
                │  - LoRA weights        │ │   Animation / Props    │
                │  - IP-Adapter          │ │   generators — same    │
                │  - Palette lock        │ │   worker pattern)       │
                └──────────┬───────────┘ └─────────┬────────────┘
                           │                        │
                           └───────────┬────────────┘
                                       │
                            ┌──────────▼───────────┐
                            │  Image Processing      │
                            │  - rembg (bg removal)  │
                            │  - Pillow/OpenCV clean  │
                            │  - Palette quantization │
                            └──────────┬───────────┘
                                       │
                            ┌──────────▼───────────┐
                            │  Sprite Sheet Builder   │
                            │  - Packing/atlasing     │
                            │  - Metadata (JSON/XML)  │
                            └──────────┬───────────┘
                                       │
                            ┌──────────▼───────────┐
                            │  Engine Exporters       │
                            │  - Godot (.tres/.tscn)  │
                            │  - Unity (.meta/prefab) │
                            │  - GameMaker (.yy)      │
                            │  - Phaser (JSON atlas)  │
                            └──────────┬───────────┘
                                       │
                            ┌──────────▼───────────┐
                            │  Storage                │
                            │  - Local (dev)          │
                            │  - Cloudflare R2 (prod)  │
                            └──────────┬───────────┘
                                       │
                            ┌──────────▼───────────┐
                            │  Download / Asset Library│
                            └──────────────────────────┘
```

### 1.2 What you have TODAY, mapped onto this diagram

```
[Kaggle Notebook: VQ-VAE + Transformer training]  ──►  maps to: ONE box —
                                                          "Character Generator"
                                                          (GPU Worker Pool, singular)

[Planned: HF Spaces + Gradio]                      ──►  maps to: a placeholder for
                                                          BOTH "Next.js Frontend" AND
                                                          "FastAPI Backend" combined
                                                          (Gradio does both jobs crudely)

[HF Model Hub: darklord8777/sprite-generator-model] ──►  maps to: "Storage" (partial —
                                                          model storage, not asset storage)

Everything else in the diagram: Postgres, Redis, Celery, Orchestrator, LLM Project
Director, Style Engine, other 7 generators, Image Processing, Sprite Sheet Builder,
all 4 Exporters, Asset Library — NONE OF THIS EXISTS YET.
```

This is the honest scope gap: you have roughly **one box out of ~20** built, and that one
box (Character Generator) is itself mid-training and not yet validated for quality.

### 1.3 Realistic MVP architecture (what to actually build first)

Strip the diagram down to the smallest slice that is still a genuinely useful, shippable
product — this is Phase 1 in the to-do list below:

```
        USER
          │
   ┌──────▼───────┐
   │  Simple web UI │   (single page: prompt box, class/action/direction pickers,
   │  (Next.js OR   │    generate button, gallery of results, download button)
   │  plain HTML)   │
   └──────┬───────┘
          │ REST
   ┌──────▼───────┐
   │  FastAPI       │   (single service, no Celery/Redis/DAG yet — just an
   │  - /generate   │    async endpoint that calls the model directly or via
   │  - /export     │    a simple in-process queue if concurrency demands it)
   └──────┬───────┘
          │
   ┌──────▼───────┐
   │  Character     │   (your existing VQ-VAE+Transformer OR SD-LoRA pipeline —
   │  Generator     │    whichever you finalize)
   └──────┬───────┘
          │
   ┌──────▼───────┐
   │  Image         │   (rembg for transparency, basic Pillow cleanup —
   │  Processing    │    skip OpenCV/palette quantization until needed)
   └──────┬───────┘
          │
   ┌──────▼───────┐
   │  Sprite Sheet  │   (pack N generated frames into one atlas + JSON metadata —
   │  Builder       │    a solved problem, don't build custom, use a library)
   └──────┬───────┘
          │
   ┌──────▼───────┐
   │  ONE Exporter  │   (Godot only — smallest, most indie-friendly format,
   │  (Godot)       │    skip Unity/GameMaker/Phaser until Godot version ships)
   └──────┬───────┘
          │
   ┌──────▼───────┐
   │  Download /    │   (direct file download — skip R2/cloud storage until
   │  local storage │    you have real users generating real volume)
   └───────────────┘
```

No LLM Project Director, no DAG orchestrator, no Postgres (SQLite is enough at this
scale), no Redis/Celery (unless a single FastAPI process genuinely can't keep up — measure
before adding), no multi-module generation, no style engine beyond your LoRA/model weights
themselves. Everything skipped here is still on the roadmap — just sequenced for later.

---

## PART 2 — CONVERSION TO-DO LIST

Ordered in the sequence you should actually attempt them. Each phase should be a genuinely
usable, demoable checkpoint — not a pile of half-built infrastructure.

### PHASE 0 — Finish and validate what's already running (do this first, no shortcuts)

- [ ] Let the current VQ-VAE training run complete (or reach a quality plateau) —
      don't abandon it mid-run for the LoRA pivot without a clean stopping point
- [ ] Visually validate reconstructions AND, once Step 6 runs, generated samples —
      loss numbers looking good is not the same as sprites looking good
- [ ] Decide: continue with VQ-VAE+Transformer, OR pivot fully to SD1.5+LoRA/DreamBooth,
      OR run both and compare — don't build a website around an unvalidated model
- [ ] Rotate/confirm the HF token issue from earlier is resolved (if not already done)
- [ ] Caption/tag your existing sprite dataset if going the LoRA route (this is on the
      critical path for LoRA — can be done in parallel with Phase 0's other items)

### PHASE 1 — Ship the smallest real product (Character Generator + Godot export)

This phase = the "Realistic MVP architecture" diagram above. Goal: a working website
a stranger could use to generate a sprite and download a Godot-ready file.

- [ ] Build a minimal FastAPI service wrapping your chosen generator (`/generate` endpoint)
- [ ] Deploy the model + API — start with **Hugging Face Spaces** (matches your existing
      HF usage, has a free tier, avoids new infra to learn)
- [ ] Build a minimal frontend — a single page is enough: prompt/label inputs, generate
      button, image preview, download button. Next.js is fine, but plain HTML+JS is
      equally valid for v1 and faster to ship
- [ ] Add background removal (`rembg`) so sprites have proper transparency
- [ ] Add a sprite-sheet packer — use an existing library/tool rather than writing packing
      logic from scratch (e.g. a texture-atlas packer); output PNG + JSON metadata
- [ ] Write ONE exporter: Godot. Research Godot's `.tres`/`.tscn`/`SpriteFrames` format,
      generate a minimal valid resource file pointing at the packed sheet
- [ ] Add basic usage limiting (even just a simple per-IP rate limit) before making it public
- [ ] Ship it. Get real feedback before building module #2.

### PHASE 2 — Second generator module + real job handling

Only start this once Phase 1 is live and you've used it enough to trust the pipeline shape.

- [ ] Pick the SECOND generator module based on what users actually ask for in Phase 1
      feedback — don't default to "Building Generator" just because it's next on the
      original roadmap list if user demand points elsewhere
- [ ] Introduce Redis + a task queue (Celery, or a lighter alternative like `arq`) —
      only now does concurrent multi-module generation justify this complexity
- [ ] Migrate SQLite → PostgreSQL once you have actual concurrent users, not before
- [ ] Add a real jobs table / status polling so the frontend can show generation progress
      for longer-running or queued requests
- [ ] Add a basic Asset Library page (list of a user's past generations, re-downloadable)

### PHASE 3 — Style consistency + orchestration

- [ ] Build the Style Engine properly: consistent LoRA application, optional IP-Adapter
      for "match this reference image," palette-lock post-processing
- [ ] Introduce the Workflow Orchestrator / DAG only once you have 3+ generator modules
      that sometimes need to run in sequence or share style context (e.g. "generate a
      character AND matching enemy in the same art style") — this is the point where a
      DAG actually earns its complexity, not before
- [ ] Add the AI Project Director (LLM planner) — this turns "generate a forest tileset
      with 3 enemy types" into a structured multi-job plan. Build this AFTER the modules
      it's meant to orchestrate exist, not before — it has nothing to orchestrate yet
      if built earlier

### PHASE 4 — Scale out generator coverage

- [ ] Add remaining generator modules one at a time, each validated for quality before
      moving to the next: Tileset, Environment, UI, Animation, Props/Icons, Portraits
- [ ] Add the remaining exporters: Unity, GameMaker, Phaser — prioritize based on which
      engine your actual users request most, not the roadmap's listed order
- [ ] Move storage from local/dev to Cloudflare R2 once asset volume justifies it

### PHASE 5 — Platform hardening (production readiness, same shape as your Eridian work)

- [ ] Auth, rate limiting, Pydantic validation on all endpoints (you've done this exact
      pattern before on Eridian — same playbook applies here)
- [ ] Billing/usage metering if this becomes a paid product
- [ ] Proper error handling, structured logging, correlation IDs
- [ ] Docker + deployment docs, CI/CD
- [ ] Incremental regeneration ("asset memory" — re-generate one variant of an existing
      asset while keeping style/seed context) — this is a genuine differentiator per the
      original roadmap's "competitive moat" section, worth prioritizing once core
      generation is solid

---

## PART 3 — WHAT TO EXPLICITLY DEFER OR CUT FOR NOW

Being honest about what's on the original roadmap but shouldn't be touched until much
later, if ever, given solo/free-tier constraints:

- **LLM Project Director** — genuinely last, not early. It orchestrates modules that
  don't exist yet if built now.
- **Full DAG workflow engine** — a real dependency-graph scheduler is overkill until
  you have multiple interdependent generation steps happening regularly.
- **4 engine exporters simultaneously** — pick one, prove it works end-to-end
  (a real Godot project that actually imports and runs), then add more.
- **Billing** — irrelevant until you have users who'd pay; premature billing
  infrastructure is pure sunk cost early on.
- **Cloudflare R2 / production storage** — local storage is fine until real traffic
  volume makes it not fine.

---

## PART 4 — TECH STACK DECISIONS (confirmed vs. still open)

| Layer | Roadmap's choice | Recommendation for you | Why |
|---|---|---|---|
| Frontend | Next.js | Start with plain HTML/JS or minimal Next.js | Don't let frontend tooling slow down Phase 1 shipping |
| Backend | FastAPI | FastAPI | Matches your Eridian experience directly |
| Database | PostgreSQL | SQLite until Phase 2 | Postgres is right long-term, premature now |
| Queue | Redis + Celery | Skip until Phase 2 | Adds ops complexity with no payoff at MVP scale |
| Image model | FLUX + LoRA | SD 1.5 + LoRA/DreamBooth first | FLUX is heavier; prove the pipeline cheaply first, upgrade the base model later — LoRA weights aren't perfectly portable across base models, but the *pipeline shape* is, so switching later isn't a rebuild |
| Hosting (inference) | — | Hugging Face Spaces | Free tier, matches your existing HF usage |
| Storage | Local → R2 | Local only until Phase 4 | No reason to pay for storage with no users yet |

---

## PART 5 — HOW TO USE THIS WITH AI-ASSISTED CODING

Since you mentioned rewriting everything with AI as needed: use this document as the
spec you hand to coding agents, phase by phase — not all at once. Concretely:

1. Give an agent **Phase 1 only** as its task, with this file as context, and a hard
   instruction: "do not add Redis, Celery, Postgres, or any exporter other than Godot,
   even if it seems easy — those are explicitly deferred."
2. Only after Phase 1 is verified working do you hand an agent Phase 2, and so on.
3. This stops an agent from "helpfully" scaffolding the entire target architecture in
   one pass, which is the exact trap discussed earlier — fast code generation makes
   over-scoping easy, not harder, so the phase gating has to be an explicit instruction
   each time, not assumed.
