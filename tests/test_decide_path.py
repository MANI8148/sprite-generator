"""Tests for the automated model-path decision tool (eval/decide_path.py).

Covers the Phase 0 roadmap item "Decide: continue with VQ-VAE+Transformer, OR
pivot fully to SD1.5+LoRA/DreamBooth, OR run both and compare -- don't build a
website around an unvalidated model."
"""
import json
from pathlib import Path

import pytest

from eval.decide_path import (
    PATH_KEYS,
    VALID_RECOMMENDATIONS,
    DecisionThresholds,
    PathDecision,
    decide_path,
    write_decision_report,
    main,
)


def _comparison(vq=None, lora=None):
    return {
        "comparison": {
            "vqvae_transformer": vq if vq is not None else {},
            "lora": lora if lora is not None else {},
        }
    }


def _path_metrics(score):
    return {
        "palette_adherence_mean": score,
        "grid_alignment_mean": score,
    }


class TestDecidePath:
    def test_vqvae_validated_lora_not(self):
        comparison = _comparison(
            vq=_path_metrics(0.9),
            lora=_path_metrics(0.4),
        )
        decision = decide_path(comparison)
        assert decision.recommendation == "vqvae_transformer"
        assert decision.validated["vqvae_transformer"] is True
        assert decision.validated["lora"] is False

    def test_lora_validated_vqvae_not(self):
        comparison = _comparison(
            vq=_path_metrics(0.4),
            lora=_path_metrics(0.9),
        )
        decision = decide_path(comparison)
        assert decision.recommendation == "lora"
        assert decision.validated["vqvae_transformer"] is False
        assert decision.validated["lora"] is True

    def test_both_validated_clear_winner(self):
        comparison = _comparison(
            vq=_path_metrics(0.95),
            lora=_path_metrics(0.8),
        )
        decision = decide_path(comparison)
        assert decision.recommendation == "vqvae_transformer"
        assert all(decision.validated.values())

    def test_both_validated_lora_wins(self):
        comparison = _comparison(
            vq=_path_metrics(0.8),
            lora=_path_metrics(0.95),
        )
        decision = decide_path(comparison)
        assert decision.recommendation == "lora"

    def test_both_validated_within_tie_margin(self):
        comparison = _comparison(
            vq=_path_metrics(0.9),
            lora=_path_metrics(0.88),
        )
        decision = decide_path(comparison)
        assert decision.recommendation == "both"

    def test_neither_validated(self):
        comparison = _comparison(
            vq=_path_metrics(0.3),
            lora=_path_metrics(0.2),
        )
        decision = decide_path(comparison)
        assert decision.recommendation == "none"
        assert not any(decision.validated.values())

    def test_missing_metrics_are_not_validated(self):
        decision = decide_path(_comparison())
        assert decision.recommendation == "none"
        assert decision.scores["vqvae_transformer"] == 0.0
        assert decision.scores["lora"] == 0.0

    def test_accepts_bare_comparison_without_wrapper(self):
        bare = {
            "vqvae_transformer": _path_metrics(0.9),
            "lora": _path_metrics(0.4),
        }
        decision = decide_path(bare)
        assert decision.recommendation == "vqvae_transformer"

    def test_custom_thresholds(self):
        comparison = _comparison(
            vq=_path_metrics(0.6),
            lora=_path_metrics(0.5),
        )
        strict = decide_path(
            comparison,
            DecisionThresholds(min_composite_score=0.8),
        )
        assert strict.recommendation == "none"

        lenient = decide_path(
            comparison,
            DecisionThresholds(min_composite_score=0.5, tie_margin=0.2),
        )
        assert lenient.recommendation == "both"

    def test_recommendation_always_valid(self):
        for vq in (0.1, 0.9):
            for lora in (0.1, 0.9):
                decision = decide_path(_comparison(_path_metrics(vq), _path_metrics(lora)))
                assert decision.recommendation in VALID_RECOMMENDATIONS

    def test_to_dict_shape(self):
        decision = decide_path(_comparison(vq=_path_metrics(0.9), lora=_path_metrics(0.4)))
        data = decision.to_dict()
        assert data["recommendation"] == "vqvae_transformer"
        assert set(data["scores"]) == set(PATH_KEYS)
        assert data["summary"]["validated_paths"] == ["vqvae_transformer"]
        assert data["summary"]["unvalidated_paths"] == ["lora"]

    def test_decision_is_dataclass(self):
        decision = decide_path(_comparison(vq=_path_metrics(0.9), lora=_path_metrics(0.4)))
        assert isinstance(decision, PathDecision)
        assert decision.thresholds is not None


class TestWriteDecisionReport:
    def test_creates_json_and_markdown(self, tmp_path):
        decision = decide_path(_comparison(vq=_path_metrics(0.9), lora=_path_metrics(0.4)))
        paths = write_decision_report(decision, tmp_path)

        json_path = paths["json"]
        md_path = paths["markdown"]
        assert json_path.exists()
        assert md_path.exists()

        with open(json_path) as f:
            data = json.load(f)
        assert data["recommendation"] == "vqvae_transformer"

        md_text = md_path.read_text()
        assert "Model Path Decision" in md_text
        assert "vqvae_transformer" in md_text

    def test_creates_parent_directories(self, tmp_path):
        decision = decide_path(_comparison(vq=_path_metrics(0.9), lora=_path_metrics(0.4)))
        nested = tmp_path / "a" / "b" / "c"
        paths = write_decision_report(decision, nested)
        assert paths["json"].exists()
        assert paths["markdown"].exists()


class TestMain:
    def test_main_end_to_end(self, tmp_path):
        comparison_path = tmp_path / "comparison.json"
        with open(comparison_path, "w") as f:
            json.dump(_comparison(vq=_path_metrics(0.9), lora=_path_metrics(0.4)), f)

        output_dir = tmp_path / "report"
        code = main(
            [
                "--comparison", str(comparison_path),
                "--output", str(output_dir),
            ]
        )
        assert code == 0
        assert (output_dir / "path_decision.json").exists()
        assert (output_dir / "path_decision.md").exists()

    def test_main_custom_thresholds(self, tmp_path, capsys):
        comparison_path = tmp_path / "comparison.json"
        with open(comparison_path, "w") as f:
            json.dump(_comparison(vq=_path_metrics(0.6), lora=_path_metrics(0.5)), f)

        code = main(
            [
                "--comparison", str(comparison_path),
                "--output", str(tmp_path / "report"),
                "--min-score", "0.5",
                "--tie-margin", "0.2",
            ]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "Recommendation: both" in out

    def test_main_missing_file_returns_2(self, tmp_path, capsys):
        code = main(
            [
                "--comparison", str(tmp_path / "missing.json"),
                "--output", str(tmp_path / "report"),
            ]
        )
        assert code == 2
        assert "ERROR" in capsys.readouterr().err

    def test_main_invalid_json_returns_2(self, tmp_path, capsys):
        bad = tmp_path / "bad.json"
        bad.write_text("not json")
        code = main(["--comparison", str(bad), "--output", str(tmp_path / "report")])
        assert code == 2
        assert "ERROR" in capsys.readouterr().err
