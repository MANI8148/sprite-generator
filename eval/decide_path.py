"""Automated model-path decision for the sprite generator.

Phase 0 Item 3: "Decide: continue with VQ-VAE+Transformer, OR pivot fully to
SD1.5+LoRA/DreamBooth, OR run both and compare -- don't build a website around
an unvalidated model."

This module turns that decision into a reproducible, testable check. It
consumes the comparison output produced by ``eval/compare_paths.py`` (per-path
palette-adherence and grid-alignment metrics) and applies explicit criteria:

  * each path gets a composite score = weighted mean of palette adherence and
    grid alignment,
  * a path is only considered "validated" if that score clears a minimum bar,
  * the recommendation is one of:

    - ``vqvae_transformer`` -- only that path clears the bar,
    - ``lora`` -- only that path clears the bar,
    - ``both`` -- both paths clear the bar and are within the tie margin, so
      keep running and comparing before committing,
    - ``none`` -- neither path clears the bar; do not build a website around
      an unvalidated model.

The verdict is persisted as a JSON + Markdown report so it can be gated in CI
or before shipping, matching the pattern of ``eval/quality_gate.py``.
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

PATH_KEYS = ("vqvae_transformer", "lora")

VALID_RECOMMENDATIONS = ("vqvae_transformer", "lora", "both", "none")


@dataclass(frozen=True)
class DecisionThresholds:
    min_composite_score: float = 0.75
    tie_margin: float = 0.05
    palette_weight: float = 0.5
    grid_weight: float = 0.5


@dataclass
class PathDecision:
    recommendation: str
    scores: Dict[str, float]
    validated: Dict[str, bool]
    reasons: List[str] = field(default_factory=list)
    thresholds: Optional[DecisionThresholds] = None

    def to_dict(self) -> Dict:
        return {
            "recommendation": self.recommendation,
            "scores": dict(self.scores),
            "validated": dict(self.validated),
            "reasons": list(self.reasons),
            "thresholds": {
                "min_composite_score": self.thresholds.min_composite_score,
                "tie_margin": self.thresholds.tie_margin,
                "palette_weight": self.thresholds.palette_weight,
                "grid_weight": self.thresholds.grid_weight,
            },
            "summary": {
                "recommendation": self.recommendation,
                "validated_paths": [k for k, v in self.validated.items() if v],
                "unvalidated_paths": [k for k, v in self.validated.items() if not v],
            },
        }


def _path_metrics(comparison: Dict) -> Dict[str, Dict[str, float]]:
    """Normalize a comparison dict into per-path metric dicts.

    Accepts either the full ``compare_paths.py`` output (with a ``comparison``
    key) or a bare ``{path: metrics}`` mapping. Missing metrics default to 0.0
    so an absent path is treated as unvalidated rather than crashing.
    """
    raw = comparison.get("comparison", comparison)
    out: Dict[str, Dict[str, float]] = {}
    for key in PATH_KEYS:
        metrics = raw.get(key, {})
        out[key] = {
            "palette_adherence_mean": float(metrics.get("palette_adherence_mean", 0.0) or 0.0),
            "grid_alignment_mean": float(metrics.get("grid_alignment_mean", 0.0) or 0.0),
        }
    return out


def _composite_score(metrics: Dict[str, float], thresholds: DecisionThresholds) -> float:
    return (
        thresholds.palette_weight * metrics["palette_adherence_mean"]
        + thresholds.grid_weight * metrics["grid_alignment_mean"]
    )


def decide_path(
    comparison: Dict,
    thresholds: Optional[DecisionThresholds] = None,
) -> PathDecision:
    thresholds = thresholds or DecisionThresholds()
    path_metrics = _path_metrics(comparison)
    scores = {key: _composite_score(m, thresholds) for key, m in path_metrics.items()}
    validated = {
        key: score >= thresholds.min_composite_score for key, score in scores.items()
    }

    reasons: List[str] = []
    for key in PATH_KEYS:
        metrics = path_metrics[key]
        reasons.append(
            f"{key}: composite={scores[key]:.3f} "
            f"(palette_adherence={metrics['palette_adherence_mean']:.3f}, "
            f"grid_alignment={metrics['grid_alignment_mean']:.3f}), "
            f"{'validated' if validated[key] else 'not validated'}"
        )

    vq = validated["vqvae_transformer"]
    lr = validated["lora"]

    if not vq and not lr:
        recommendation = "none"
        reasons.append(
            "neither path clears the validation bar; do not build a website "
            "around an unvalidated model"
        )
    elif vq and not lr:
        recommendation = "vqvae_transformer"
    elif lr and not vq:
        recommendation = "lora"
    else:
        diff = scores["vqvae_transformer"] - scores["lora"]
        if abs(diff) <= thresholds.tie_margin:
            recommendation = "both"
            reasons.append(
                "both paths are validated and within the tie margin; keep "
                "running and comparing before committing"
            )
        else:
            recommendation = "vqvae_transformer" if diff > 0 else "lora"

    return PathDecision(
        recommendation=recommendation,
        scores=scores,
        validated=validated,
        reasons=reasons,
        thresholds=thresholds,
    )


def write_decision_report(decision: PathDecision, output_dir: Path) -> Dict[str, Path]:
    """Persist the decision as JSON plus a human-readable Markdown summary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "path_decision.json"
    with open(json_path, "w") as f:
        json.dump(decision.to_dict(), f, indent=2)

    md_path = output_dir / "path_decision.md"
    lines = [
        "# Model Path Decision",
        "",
        f"**Recommendation:** {decision.recommendation}",
        "",
        "## Validated paths",
        "",
    ]
    for key in PATH_KEYS:
        lines.append(
            f"- {key}: {decision.validated.get(key, False)} "
            f"(score {decision.scores.get(key, 0.0):.3f})"
        )
    lines += ["", "## Reasons", ""]
    for reason in decision.reasons:
        lines.append(f"- {reason}")
    md_path.write_text("\n".join(lines))

    return {"json": json_path, "markdown": md_path}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Decide between the VQ-VAE+Transformer and SD-LoRA paths from "
            "comparison metrics (eval/compare_paths.py output)"
        )
    )
    parser.add_argument(
        "--comparison",
        required=True,
        help="JSON comparison results from eval/compare_paths.py",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="path_decision",
        help="Output directory for the decision report",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.75,
        help="Minimum composite score for a path to count as validated",
    )
    parser.add_argument(
        "--tie-margin",
        type=float,
        default=0.05,
        help="Score margin below which validated paths are too close to call",
    )
    args = parser.parse_args(argv)

    try:
        with open(args.comparison) as f:
            comparison = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"ERROR: could not read comparison file: {exc}", file=sys.stderr)
        return 2

    decision = decide_path(
        comparison,
        DecisionThresholds(
            min_composite_score=args.min_score,
            tie_margin=args.tie_margin,
        ),
    )
    paths = write_decision_report(decision, Path(args.output))
    print(f"Recommendation: {decision.recommendation}")
    print(f"Report written to {paths['json']}")
    print(f"Report written to {paths['markdown']}")

    if decision.recommendation not in VALID_RECOMMENDATIONS:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
