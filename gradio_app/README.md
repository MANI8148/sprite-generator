---
title: AI Game Asset Pipeline
emoji: 🎮
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: gradio_app/app.py
pinned: false
---

# AI Game Asset Pipeline

Generate game-ready 2D sprites from structured controls using SD 1.5 + LoRA.

## Usage

1. Set a LoRA path (or leave empty for base SD 1.5)
2. Click **Load Model**
3. Configure asset type, view, animation, palette, and post-processing
4. Click **Generate Asset**

## Pipeline

```
Controls → Prompt Builder → SD 1.5 LoRA → Post-Process → Validation → Export
```
