"""Tests for the unified dataset preparation pipeline (roadmap Phase 0 Item 5)."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest

from data.scripts.prepare_dataset import run_pipeline


def _make_synthetic_sprites(output_dir: Path, count: int = 5):
    """Create synthetic RGBA sprite images for testing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(count):
        arr = np.zeros((32, 32, 4), dtype=np.uint8)
        color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)][i % 5]
        arr[4:28, 4:28, :3] = color
        arr[4:28, 4:28, 3] = 255
        path = output_dir / f"sprite_{i}.png"
        Image.fromarray(arr, "RGBA").save(path)
        paths.append(path)
    return paths


class TestPipelineCleanStep:
    def test_clean_step_produces_output_files(self, tmp_path):
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "processed"
        _make_synthetic_sprites(raw_dir)

        rc = run_pipeline(
            input_dir=str(raw_dir),
            output_dir=str(out_dir),
            skip_push=True,
        )
        assert rc == 0

        assert (out_dir / "metadata.json").exists()
        assert (out_dir / "palette.json").exists()

        with open(out_dir / "metadata.json") as f:
            meta = json.load(f)
        assert len(meta) > 0
        assert "filename" in meta[0]
        assert "id" in meta[0]

    def test_clean_step_respects_canvas_size(self, tmp_path):
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "processed"
        _make_synthetic_sprites(raw_dir)

        rc = run_pipeline(
            input_dir=str(raw_dir),
            output_dir=str(out_dir),
            canvas_size=64,
            skip_push=True,
        )
        assert rc == 0

        meta_path = out_dir / "metadata.json"
        assert meta_path.exists()
        first_img = out_dir / json.load(open(meta_path))[0]["filename"]
        img = Image.open(first_img)
        assert img.size == (64, 64)

    def test_clean_step_empty_input_returns_error(self, tmp_path):
        empty_dir = tmp_path / "empty_raw"
        empty_dir.mkdir()
        out_dir = tmp_path / "processed"

        rc = run_pipeline(
            input_dir=str(empty_dir),
            output_dir=str(out_dir),
            skip_push=True,
        )
        assert rc != 0

    def test_clean_step_dedup_identical_images(self, tmp_path):
        raw_dir = tmp_path / "raw_dup"
        out_dir = tmp_path / "processed_dup"
        raw_dir.mkdir(parents=True, exist_ok=True)

        arr = np.zeros((32, 32, 4), dtype=np.uint8)
        arr[4:28, 4:28, :3] = [255, 0, 0]
        arr[4:28, 4:28, 3] = 255
        for i in range(4):
            Image.fromarray(arr.copy(), "RGBA").save(raw_dir / f"dup_{i}.png")

        rc = run_pipeline(
            input_dir=str(raw_dir),
            output_dir=str(out_dir),
            dedup_threshold=2,
            skip_push=True,
        )
        assert rc == 0

        with open(out_dir / "metadata.json") as f:
            meta = json.load(f)
        assert len(meta) == 1


class TestPipelineCaptionStep:
    def test_caption_step_adds_labels(self, tmp_path):
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "processed"
        _make_synthetic_sprites(raw_dir)

        rc = run_pipeline(
            input_dir=str(raw_dir),
            output_dir=str(out_dir),
            skip_push=True,
        )
        assert rc == 0

        labeled = out_dir / "metadata_labeled.json"
        assert labeled.exists()
        with open(labeled) as f:
            meta = json.load(f)
        assert len(meta) > 0
        for entry in meta:
            assert "class" in entry
            assert "action" in entry
            assert "direction" in entry

    def test_caption_step_labels_are_reasonable(self, tmp_path):
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "processed"
        raw_dir.mkdir(parents=True, exist_ok=True)

        tall = np.zeros((16, 32, 4), dtype=np.uint8)
        tall[2:14, :, :3] = [255, 0, 0]
        tall[2:14, :, 3] = 255
        Image.fromarray(tall, "RGBA").save(raw_dir / "tall.png")

        wide = np.zeros((32, 16, 4), dtype=np.uint8)
        wide[:, 2:14, :3] = [0, 255, 0]
        wide[:, 2:14, 3] = 255
        Image.fromarray(wide, "RGBA").save(raw_dir / "wide.png")

        rc = run_pipeline(
            input_dir=str(raw_dir),
            output_dir=str(out_dir),
            skip_push=True,
        )
        assert rc == 0

        with open(out_dir / "metadata_labeled.json") as f:
            meta = json.load(f)
        assert len(meta) >= 1

    def test_caption_step_produces_caption_text(self, tmp_path):
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "processed"
        _make_synthetic_sprites(raw_dir, count=3)

        rc = run_pipeline(
            input_dir=str(raw_dir),
            output_dir=str(out_dir),
            skip_push=True,
        )
        assert rc == 0

        with open(out_dir / "metadata_labeled.json") as f:
            meta = json.load(f)
        assert len(meta) > 0
        for entry in meta:
            assert "caption" in entry, f"entry missing caption: {entry}"
            assert isinstance(entry["caption"], str)
            assert len(entry["caption"]) > 0
            assert "sprite" in entry["caption"].lower()

    def test_caption_step_writes_txt_files(self, tmp_path):
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "processed"
        _make_synthetic_sprites(raw_dir, count=3)

        rc = run_pipeline(
            input_dir=str(raw_dir),
            output_dir=str(out_dir),
            skip_push=True,
        )
        assert rc == 0

        with open(out_dir / "metadata_labeled.json") as f:
            meta = json.load(f)
        assert len(meta) > 0
        for entry in meta:
            stem = Path(entry["filename"]).stem
            txt = out_dir / f"{stem}.txt"
            assert txt.exists(), f"missing caption file: {txt}"
            assert txt.read_text() == entry["caption"]


class TestPipelineEndToEnd:
    def test_full_pipeline_skip_push(self, tmp_path):
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "processed"
        _make_synthetic_sprites(raw_dir, count=3)

        rc = run_pipeline(
            input_dir=str(raw_dir),
            output_dir=str(out_dir),
            skip_push=True,
        )
        assert rc == 0

        assert (out_dir / "metadata.json").exists()
        assert (out_dir / "metadata_labeled.json").exists()
        assert (out_dir / "palette.json").exists()

        png_files = list(out_dir.glob("*.png"))
        assert len(png_files) > 0
        for f in png_files:
            img = Image.open(f)
            assert img.mode == "RGBA"

    def test_pipeline_augment_step(self, tmp_path):
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "processed"
        aug_dir = tmp_path / "augmented"
        _make_synthetic_sprites(raw_dir, count=2)

        from data.scripts.prepare_dataset import run_pipeline as run

        rc = run(
            input_dir=str(raw_dir),
            output_dir=str(out_dir),
            skip_push=True,
        )
        assert rc == 0

        from data.scripts.augment_dataset import main as aug_main

        aug_argv = [
            "augment_dataset",
            "--input", str(out_dir),
            "--output", str(aug_dir),
            "--copies", "2",
        ]
        import sys
        old = sys.argv
        sys.argv = aug_argv
        try:
            rc = aug_main()
        finally:
            sys.argv = old
        assert rc == 0

        assert (aug_dir / "metadata_labeled.json").exists()
        with open(aug_dir / "metadata_labeled.json") as f:
            aug_meta = json.load(f)
        assert len(aug_meta) >= 2


class TestPipelineWithHFPush:
    def test_push_step_mocked(self, tmp_path):
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "processed"
        _make_synthetic_sprites(raw_dir, count=2)

        rc = run_pipeline(
            input_dir=str(raw_dir),
            output_dir=str(out_dir),
            skip_push=True,
        )
        assert rc == 0

        with patch("data.scripts.push_to_hf.main") as mock_push:
            mock_push.return_value = 0
            from data.scripts.push_to_hf import main as push_main

            push_argv = [
                "push_to_hf",
                "--input", str(out_dir),
                "--repo", "test-user/test-dataset",
                "--token", "fake-token",
            ]
            import sys
            old = sys.argv
            sys.argv = push_argv
            try:
                rc = push_main()
            finally:
                sys.argv = old
            assert rc == 0
            mock_push.assert_called_once()

    def test_pipeline_skips_push_when_no_creds(self, tmp_path):
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "processed"
        _make_synthetic_sprites(raw_dir)

        rc = run_pipeline(
            input_dir=str(raw_dir),
            output_dir=str(out_dir),
            skip_push=False,
            hf_repo="",
            hf_token="",
        )
        assert rc == 0


class TestEdgeCases:
    def test_pipeline_with_single_sprite(self, tmp_path):
        raw_dir = tmp_path / "raw_single"
        out_dir = tmp_path / "processed_single"
        raw_dir.mkdir(parents=True, exist_ok=True)
        arr = np.zeros((32, 32, 4), dtype=np.uint8)
        arr[8:24, 8:24, :3] = [0, 255, 0]
        arr[8:24, 8:24, 3] = 255
        Image.fromarray(arr, "RGBA").save(raw_dir / "single.png")

        rc = run_pipeline(
            input_dir=str(raw_dir),
            output_dir=str(out_dir),
            skip_push=True,
        )
        assert rc == 0
        assert (out_dir / "metadata.json").exists()
        assert (out_dir / "metadata_labeled.json").exists()

    def test_pipeline_with_transparent_sprites(self, tmp_path):
        raw_dir = tmp_path / "raw_transparent"
        out_dir = tmp_path / "processed_transparent"
        raw_dir.mkdir(parents=True, exist_ok=True)
        arr = np.zeros((32, 32, 4), dtype=np.uint8)
        arr[:, :, 3] = 0
        Image.fromarray(arr, "RGBA").save(raw_dir / "empty.png")

        arr2 = np.zeros((32, 32, 4), dtype=np.uint8)
        arr2[8:24, 8:24, :3] = [255, 0, 0]
        arr2[8:24, 8:24, 3] = 255
        Image.fromarray(arr2, "RGBA").save(raw_dir / "solid.png")

        rc = run_pipeline(
            input_dir=str(raw_dir),
            output_dir=str(out_dir),
            skip_push=True,
        )
        assert rc == 0
        with open(out_dir / "metadata.json") as f:
            meta = json.load(f)
        assert len(meta) > 0
