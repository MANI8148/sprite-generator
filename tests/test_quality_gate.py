"""Tests for the automated visual quality gate (roadmap Phase 0 Item 2:
"Visually validate reconstructions AND generated samples -- loss numbers
looking good is not the same as sprites looking good")."""

import json

import numpy as np
import pytest
from PIL import Image

from eval.quality_gate import (
    assess_reconstruction,
    assess_generated_samples,
    run_quality_gate,
    write_report,
    main,
    QualityGateResult,
)


def _clean_sprite(size=32):
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    arr[6:26, 6:26] = [120, 60, 200, 255]
    arr[2:12, 11:21] = [220, 180, 60, 255]
    arr[4:7, 13:15] = [20, 20, 20, 255]
    arr[4:7, 18:20] = [20, 20, 20, 255]
    arr[24:28, 8:14] = [40, 40, 90, 255]
    arr[24:28, 18:24] = [40, 40, 90, 255]
    return Image.fromarray(arr, "RGBA")


def _empty_sprite(size=32):
    return Image.new("RGBA", (size, size), (0, 0, 0, 0))


def _noisy_sprite(size=32, seed=0):
    rng = np.random.default_rng(seed)
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    arr[:, :, :3] = rng.integers(0, 256, (size, size, 3))
    arr[:, :, 3] = 255
    return Image.fromarray(arr, "RGBA")


def _blurry_sprite(size=32):
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    for row in range(size):
        val = int(255 * row / (size - 1))
        arr[row, :, :3] = val
        arr[row, :, 3] = 255
    return Image.fromarray(arr, "RGBA")


def _good_metric(index=0):
    return {"index": index, "mse": 12.5, "psnr": 30.0, "pixel_count": 1024}


class TestAssessReconstruction:
    def test_all_good_metrics_pass(self):
        metrics = [_good_metric(i) for i in range(3)]
        result = assess_reconstruction(metrics)
        assert result["total"] == 3
        assert result["passed"] == 3
        assert result["failed"] == 0

    def test_psnr_below_threshold_flags_sample(self):
        metrics = [{"mse": 10.0, "psnr": 15.0}]
        result = assess_reconstruction(metrics, min_psnr=20.0)
        assert result["failed"] == 1
        sample = result["samples"][0]
        assert sample["ok"] is False
        assert any("psnr" in r for r in sample["reasons"])

    def test_mse_above_threshold_flags_sample(self):
        metrics = [{"mse": 500.0, "psnr": 25.0}]
        result = assess_reconstruction(metrics, max_mse=150.0)
        assert result["failed"] == 1
        assert any("mse" in r for r in result["samples"][0]["reasons"])

    def test_infinite_psnr_counts_as_pass(self):
        metrics = [{"mse": 0.0, "psnr": float("inf")}]
        result = assess_reconstruction(metrics)
        assert result["passed"] == 1
        assert result["failed"] == 0
        assert result["samples"][0]["psnr"] is None

    def test_avg_psnr_excludes_infinite(self):
        metrics = [
            {"mse": 0.0, "psnr": float("inf")},
            {"mse": 10.0, "psnr": 20.0},
            {"mse": 10.0, "psnr": 40.0},
        ]
        result = assess_reconstruction(metrics)
        assert result["avg_psnr"] == 30.0

    def test_empty_list(self):
        result = assess_reconstruction([])
        assert result["total"] == 0
        assert result["passed"] == 0
        assert result["failed"] == 0
        assert result["avg_psnr"] is None


class TestAssessGeneratedSamples:
    def test_clean_sprite_passes(self):
        result = assess_generated_samples([_clean_sprite()])
        assert result["passed"] == 1
        assert result["failed"] == 0
        assert result["samples"][0]["ok"] is True

    def test_empty_transparent_sprite_fails(self):
        result = assess_generated_samples([_empty_sprite()])
        assert result["failed"] == 1
        sample = result["samples"][0]
        assert sample["ok"] is False
        assert any("empty" in r or "palette" in r or "transparency" in r for r in sample["reasons"])

    def test_noisy_sprite_fails(self):
        result = assess_generated_samples([_noisy_sprite()])
        assert result["failed"] == 1
        assert any("quality_tier" in r for r in result["samples"][0]["reasons"])

    def test_blurry_sprite_fails(self):
        result = assess_generated_samples([_blurry_sprite()])
        assert result["failed"] == 1

    def test_mixed_batch_counts(self):
        images = [_clean_sprite(), _empty_sprite(), _noisy_sprite(), _clean_sprite()]
        result = assess_generated_samples(images)
        assert result["total"] == 4
        assert result["passed"] == 2
        assert result["failed"] == 2

    def test_tier_counts_populated(self):
        result = assess_generated_samples([_clean_sprite(), _empty_sprite()])
        assert result["quality_tier_counts"].get("clean") == 1
        assert result["quality_tier_counts"].get("empty") == 1

    def test_empty_list(self):
        result = assess_generated_samples([])
        assert result["total"] == 0
        assert result["passed"] == 0
        assert result["failed"] == 0


class TestRunQualityGate:
    def test_passes_when_both_signals_are_good(self):
        result = run_quality_gate([_good_metric()], [_clean_sprite()])
        assert result.passed is True
        assert result.issues == []

    def test_fails_when_reconstruction_is_poor(self):
        result = run_quality_gate([{"mse": 900.0, "psnr": 8.0}], [_clean_sprite()])
        assert result.passed is False
        assert any("reconstruction" in issue for issue in result.issues)

    def test_fails_when_generated_samples_are_poor(self):
        result = run_quality_gate([_good_metric()], [_empty_sprite()])
        assert result.passed is False
        assert any("generated samples" in issue for issue in result.issues)

    def test_fails_with_no_data(self):
        result = run_quality_gate([], [])
        assert result.passed is False
        assert len(result.issues) == 2

    def test_returns_quality_gate_result(self):
        result = run_quality_gate([_good_metric()], [_clean_sprite()])
        assert isinstance(result, QualityGateResult)
        summary = result.summary
        assert summary["verdict"] == "PASS"
        assert summary["reconstruction"]["total"] == 1
        assert summary["samples"]["total"] == 1

    def test_to_dict_serializable(self):
        result = run_quality_gate([_good_metric()], [_clean_sprite()])
        json.dumps(result.to_dict())


class TestWriteReport:
    def test_writes_json_and_markdown(self, tmp_path):
        result = run_quality_gate([_good_metric()], [_clean_sprite()])
        paths = write_report(result, tmp_path)
        assert paths["json"].exists()
        assert paths["markdown"].exists()

    def test_json_round_trips(self, tmp_path):
        result = run_quality_gate([_good_metric()], [_clean_sprite()])
        paths = write_report(result, tmp_path)
        with open(paths["json"]) as f:
            data = json.load(f)
        assert data["passed"] is True
        assert data["summary"]["verdict"] == "PASS"

    def test_markdown_contains_verdict(self, tmp_path):
        result = run_quality_gate([_good_metric()], [_empty_sprite()])
        paths = write_report(result, tmp_path)
        text = paths["markdown"].read_text()
        assert "FAIL" in text
        assert "Issues" in text

    def test_creates_output_dir(self, tmp_path):
        result = run_quality_gate([_good_metric()], [_clean_sprite()])
        out = tmp_path / "nested" / "reports"
        paths = write_report(result, out)
        assert paths["json"].exists()


class TestMain:
    def test_main_returns_zero_on_pass(self, tmp_path, monkeypatch):
        recon = tmp_path / "recon.json"
        recon.write_text(json.dumps({"per_sample": [_good_metric()]}))
        samples = tmp_path / "samples"
        samples.mkdir()
        _clean_sprite().save(str(samples / "a.png"))
        _clean_sprite().save(str(samples / "b.png"))

        rc = main([
            "--recon-metrics", str(recon),
            "--samples-dir", str(samples),
            "--output", str(tmp_path / "out"),
        ])
        assert rc == 0
        assert (tmp_path / "out" / "quality_report.json").exists()
        assert (tmp_path / "out" / "quality_report.md").exists()

    def test_main_returns_one_on_fail(self, tmp_path):
        recon = tmp_path / "recon.json"
        recon.write_text(json.dumps([_good_metric()]))
        samples = tmp_path / "samples"
        samples.mkdir()
        _empty_sprite().save(str(samples / "empty.png"))

        rc = main([
            "--recon-metrics", str(recon),
            "--samples-dir", str(samples),
            "--output", str(tmp_path / "out"),
        ])
        assert rc == 1

    def test_main_accepts_bare_list_recon_metrics(self, tmp_path):
        recon = tmp_path / "recon.json"
        recon.write_text(json.dumps([_good_metric()]))
        samples = tmp_path / "samples"
        samples.mkdir()
        _clean_sprite().save(str(samples / "a.png"))

        rc = main([
            "--recon-metrics", str(recon),
            "--samples-dir", str(samples),
            "--output", str(tmp_path / "out"),
        ])
        assert rc == 0

    def test_main_missing_metrics_file_returns_two(self, tmp_path):
        rc = main([
            "--recon-metrics", str(tmp_path / "missing.json"),
            "--samples-dir", str(tmp_path),
        ])
        assert rc == 2

    def test_main_missing_samples_dir_returns_two(self, tmp_path):
        rc = main([
            "--samples-dir", str(tmp_path / "missing"),
        ])
        assert rc == 2

    def test_main_requires_at_least_one_input(self):
        with pytest.raises(SystemExit):
            main([])
