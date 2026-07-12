import gradio as gr
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.modules.prompt_builder.controls import (
    AssetControls, AssetType, View, Palette, Animation, SpriteSize,
)
from backend.modules.prompt_builder.builder import build_prompt
from backend.modules.pipeline.orchestrator import AssetPipeline, PipelineConfig
from backend.modules.generator.sd_generator import SDGenerator

generator = None
pipeline = AssetPipeline()


def load_model(lora_path: str):
    global generator
    gen = SDGenerator(lora_path=lora_path if lora_path else None)
    gen.load()
    pipeline.set_generator(gen)
    generator = gen
    return f"Loaded: {lora_path or 'base SD 1.5'}"


def generate(
    asset_type,
    view,
    animation,
    palette,
    sprite_size,
    theme,
    seed,
    remove_bg,
    reduce_palette,
    max_colors,
    pixel_cleanup,
    auto_center,
    upscale_factor,
):
    global generator, pipeline
    if generator is None:
        return None, "No model loaded. Set LoRA path and click Load Model first.", ""

    controls = AssetControls(
        asset_type=AssetType(asset_type.lower()),
        view=View(view.lower()),
        animation=Animation(animation.lower()),
        palette=Palette(palette.lower()),
        sprite_size=SpriteSize(sprite_size),
        theme=theme.lower() if theme else "",
        seed=int(seed) if seed else -1,
    )

    prompt = build_prompt(controls)

    pipeline.config.remove_bg = remove_bg
    pipeline.config.reduce_palette = reduce_palette
    pipeline.config.max_colors = int(max_colors)
    pipeline.config.pixel_cleanup = pixel_cleanup
    pipeline.config.auto_center = auto_center
    pipeline.config.upscale = int(upscale_factor)
    pipeline.config.pack_sheet = False

    result = pipeline.run(controls)

    img = result.images[0]
    val = result.validation[0]
    report = (
        f"Prompt: {prompt}\n\n"
        f"Quality: {val['quality_tier']}\n"
        f"Palette: {val['palette_size']} colors\n"
        f"Sharpness: {val['sharpness']}\n"
        f"Center: ({val['center_x']}, {val['center_y']})\n"
        f"Transparency: {val['transparency_ratio']*100:.0f}%\n"
        f"Outline: {val.get('outline_continuity', 'N/A')}\n"
    )

    return img, report, prompt


css = """
.section { margin-bottom: 1.5rem; }
.generated-img { border: 2px solid #333; border-radius: 8px; }
"""

with gr.Blocks(title="AI Game Asset Pipeline", css=css) as demo:
    gr.Markdown("# AI Game Asset Pipeline")
    gr.Markdown("Generate game-ready 2D sprites from structured controls.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Model Setup")
            lora_path = gr.Textbox(
                label="LoRA Path",
                value="/kaggle/working/output/sprite_smoke_test/sprite_smoke_test.safetensors",
                placeholder="Path to .safetensors file",
            )
            load_btn = gr.Button("Load Model")
            load_status = gr.Textbox(label="Status", interactive=False)

            gr.Markdown("### Asset Controls")
            asset_type = gr.Dropdown(
                label="Asset Type", choices=[t.value for t in AssetType],
                value="character",
            )
            view = gr.Dropdown(
                label="View", choices=[v.value for v in View],
                value="front",
            )
            animation = gr.Dropdown(
                label="Animation", choices=[a.value for a in Animation],
                value="idle",
            )
            palette = gr.Dropdown(
                label="Palette", choices=[p.value for p in Palette],
                value="auto",
            )
            sprite_size = gr.Dropdown(
                label="Sprite Size", choices=[s.value for s in SpriteSize],
                value="32x32",
            )
            theme = gr.Textbox(label="Theme", placeholder="fantasy, sci-fi, etc.")
            seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)

            gr.Markdown("### Post-Processing")
            remove_bg = gr.Checkbox(label="Remove Background", value=True)
            reduce_palette = gr.Checkbox(label="Reduce Palette", value=True)
            max_colors = gr.Slider(label="Max Colors", minimum=2, maximum=128, value=32, step=1)
            pixel_cleanup = gr.Checkbox(label="Pixel Cleanup", value=True)
            auto_center = gr.Checkbox(label="Auto Center", value=True)
            upscale_factor = gr.Slider(label="Upscale Factor", minimum=1, maximum=8, value=1, step=1)

            generate_btn = gr.Button("Generate Asset", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### Output")
            output_img = gr.Image(label="Generated Sprite", elem_classes="generated-img")
            output_prompt = gr.Textbox(label="Generated Prompt", lines=3)
            output_report = gr.Textbox(label="Quality Report", lines=10)

    load_btn.click(
        fn=load_model,
        inputs=[lora_path],
        outputs=[load_status],
    )

    generate_btn.click(
        fn=generate,
        inputs=[
            asset_type, view, animation, palette, sprite_size,
            theme, seed, remove_bg, reduce_palette, max_colors,
            pixel_cleanup, auto_center, upscale_factor,
        ],
        outputs=[output_img, output_report, output_prompt],
    )

if __name__ == "__main__":
    demo.launch(debug=True)
