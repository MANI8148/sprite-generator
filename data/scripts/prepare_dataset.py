"""
Unified dataset preparation pipeline: raw sprites -> cleaned -> captioned -> HF Datasets.

Chains clean_normalize, caption_ai, and push_to_hf into a single command.

Usage:
    python -m data.scripts.prepare_dataset \\
        --input-dir data/raw --output-dir data/processed \\
        --hf-repo username/sprites --hf-token hf_xxx

Roadmap Phase 0 Item 5: "Caption/tag your existing sprite dataset if going the LoRA route"
"""

import argparse
import sys
from pathlib import Path


def run_pipeline(
    input_dir: str,
    output_dir: str,
    canvas_size: int = 32,
    palette_size: int = 24,
    dedup_threshold: int = 5,
    hf_repo: str = "",
    hf_token: str = "",
    private: bool = False,
    skip_push: bool = False,
    augment: bool = False,
) -> int:
    from data.scripts.clean_normalize import main as clean_main

    clean_argv = [
        "clean_normalize",
        "--input", input_dir,
        "--output", output_dir,
        "--canvas-size", str(canvas_size),
        "--palette-size", str(palette_size),
        "--dedup-threshold", str(dedup_threshold),
    ]
    old = sys.argv
    sys.argv = clean_argv
    try:
        rc = clean_main()
    finally:
        sys.argv = old
    if rc != 0:
        print("ERROR: clean_normalize failed", file=sys.stderr)
        return rc

    from data.scripts.caption_ai import main as caption_main

    processed_meta = Path(output_dir) / "metadata.json"
    labeled_meta = Path(output_dir) / "metadata_labeled.json"
    if not processed_meta.exists():
        print(f"ERROR: {processed_meta} not found after clean step", file=sys.stderr)
        return 1

    caption_argv = [
        "caption_ai",
        "--input", output_dir,
        "--output", str(labeled_meta),
    ]
    sys.argv = caption_argv
    try:
        rc = caption_main()
    finally:
        sys.argv = old
    if rc != 0:
        print("ERROR: caption_ai failed", file=sys.stderr)
        return rc

    if augment:
        from data.scripts.augment_dataset import main as augment_main
        augment_argv = [
            "augment_dataset",
            "--input", output_dir,
            "--label-file", "metadata_labeled.json",
        ]
        sys.argv = augment_argv
        try:
            rc = augment_main()
        finally:
            sys.argv = old
        if rc != 0:
            print("ERROR: augment_dataset failed", file=sys.stderr)
            return rc

    if skip_push or not hf_repo or not hf_token:
        print("Skipping HF push (--skip-push or missing --hf-repo/--hf-token)")
        return 0

    from data.scripts.push_to_hf import main as push_main

    push_argv = [
        "push_to_hf",
        "--input", output_dir,
        "--label-file", "metadata_labeled.json",
        "--repo", hf_repo,
        "--token", hf_token,
    ]
    if private:
        push_argv.append("--private")
    sys.argv = push_argv
    try:
        rc = push_main()
    finally:
        sys.argv = old
    return rc


def main():
    parser = argparse.ArgumentParser(
        description="Unified dataset preparation: raw sprites -> cleaned -> captioned -> HF",
    )
    parser.add_argument("--input-dir", default="data/raw",
                        help="Input directory with raw sprite images")
    parser.add_argument("--output-dir", default="data/processed",
                        help="Output directory for processed dataset")
    parser.add_argument("--canvas-size", type=int, default=32,
                        help="Target canvas size (default: 32)")
    parser.add_argument("--palette-size", type=int, default=24,
                        help="Global palette color count (default: 24)")
    parser.add_argument("--dedup-threshold", type=int, default=5,
                        help="Perceptual hash distance for dedup (default: 5)")
    parser.add_argument("--hf-repo", default="",
                        help="HuggingFace Dataset repo (e.g. username/sprites)")
    parser.add_argument("--hf-token", default="",
                        help="HuggingFace write token")
    parser.add_argument("--private", action="store_true",
                        help="Create private HF dataset")
    parser.add_argument("--skip-push", action="store_true",
                        help="Skip pushing to HF (just prepare locally)")
    parser.add_argument("--augment", action="store_true",
                        help="Run augmentation step after captioning")

    args = parser.parse_args()
    return run_pipeline(**vars(args))


if __name__ == "__main__":
    sys.exit(main())
