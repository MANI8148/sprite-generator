"""Automated visual quality gate for the sprite model.

Phase 0 Item 2: "Visually validate reconstructions AND, once Step 6 runs,
generated samples -- loss numbers looking good is not the same as sprites
looking good."

This module turns that roadmap item into an automated check instead of a
one-off eyeball test. It combines two independent signals:

  1. Reconstruction fidelity (PSNR / MSE) against a configurable bar, using
     the same per-sample metrics written by ``reconstruction_validation.py``.
  2. The production pipeline's sprite-quality validation
     (``backend.modules.validation.metrics.assess_all``: palette size,
     transparency coverage, centering, sharpness, outline continuity) applied
     to generated samples.

Both are consolidated into a single pass/fail report (JSON + Markdown) so the
model can be gated in CI or before shipping, not only inspected by hand.
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

from backend.modules.validation.metrics import assess_all

ACCEPTABLE_TIERS = ("clean", "acceptable")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}


def _round_or_none(value: Optional[float], ndigits: int = 2) -> Optional[float]:
    """Round a finite float, mapping infinities/None to None (perfect PSNR)."""
    if value is None:
        return None
    try:
        if value == float("inf"):
            return None
        return round(value, ndigits)
    except (TypeError, ValueError):
        return None


def assess_reconstruction(
    metrics_list: List[Dict],
    min_psnr: float = 20.0,
    max_mse: float = 150.0,
) -> Dict:
    """Summarize per-sample reconstruction metrics and flag low-quality samples.

    ``metrics_list`` follows the shape produced by
    ``eval.reconstruction_validation.compute_metrics``: each item is a dict
    with ``mse`` and ``psnr`` keys.
    """
    total = len(metrics_list)
    passed = 0
    psnr_vals: List[float] = []
    samples: List[Dict] = []

    for i, m in enumerate(metrics_list):
        psnr = m.get("psnr")
        mse = float(m.get("mse", 0.0))
        reasons: List[str] = []
        ok = True

        if psnr is not None and psnr != float("inf") and psnr < min_psnr:
            ok = False
            reasons.append(f"psnr {psnr:.1f} < {min_psnr}")
        if mse > max_mse:
            ok = False
            reasons.append(f"mse {mse:.1f} > {max_mse}")

        if ok:
            passed += 1
        if psnr is not None and psnr != float("inf"):
            psnr_vals.append(float(psnr))

        samples.append(
            {
                "index": i,
                "psnr": _round_or_none(psnr),
                "mse": round(mse, 4),
                "ok": ok,
                "reasons": reasons,
            }
        )

    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "avg_psnr": round(sum(psnr_vals) / len(psnr_vals), 2) if psnr_vals else None,
        "min_psnr": min_psnr,
        "max_mse": max_mse,
        "samples": samples,
    }


def assess_generated_samples(
    images: List[Image.Image],
    min_palette: int = 4,
    max_transparency: float = 0.99,
    acceptable_tiers: tuple = ACCEPTABLE_TIERS,
) -> Dict:
    """Run the production sprite-quality validation over generated samples."""
    total = len(images)
    passed = 0
    tier_counts: Dict[str, int] = {}
    samples: List[Dict] = []

    for i, img in enumerate(images):
        metrics = assess_all(img)
        tier = metrics.get("quality_tier", "unknown")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

        reasons: List[str] = []
        ok = True
        if tier not in acceptable_tiers:
            ok = False
            reasons.append(f"quality_tier '{tier}' not acceptable")
        if metrics.get("palette_size", 0) < min_palette:
            ok = False
            reasons.append(f"palette_size {metrics.get('palette_size')} < {min_palette}")
        if metrics.get("transparency_ratio", 1.0) > max_transparency:
            ok = False
            reasons.append(
                f"transparency_ratio {metrics.get('transparency_ratio')} > {max_transparency}"
            )

        if ok:
            passed += 1
        samples.append({"index": i, "ok": ok, "reasons": reasons, "metrics": metrics})

    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "min_palette": min_palette,
        "max_transparency": max_transparency,
        "quality_tier_counts": tier_counts,
        "samples": samples,
    }


@dataclass
class QualityGateResult:
    passed: bool
    reconstruction: Dict = field(default_factory=dict)
    samples: Dict = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    summary: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "passed": self.passed,
            "reconstruction": self.reconstruction,
            "samples": self.samples,
            "issues": list(self.issues),
            "summary": dict(self.summary),
        }


def run_quality_gate(
    recon_metrics_list: List[Dict],
    sample_images: List[Image.Image],
    min_psnr: float = 20.0,
    max_mse: float = 150.0,
    min_palette: int = 4,
    max_transparency: float = 0.99,
) -> QualityGateResult:
    """Combine reconstruction and generated-sample checks into one verdict."""
    recon = assess_reconstruction(recon_metrics_list, min_psnr=min_psnr, max_mse=max_mse)
    samples = assess_generated_samples(
        sample_images,
        min_palette=min_palette,
        max_transparency=max_transparency,
    )

    issues: List[str] = []
    if recon["total"] and recon["failed"]:
        issues.append(
            f"reconstruction: {recon['failed']}/{recon['total']} samples below "
            f"min_psnr={min_psnr} / max_mse={max_mse}"
        )
    if samples["total"] and samples["failed"]:
        issues.append(
            f"generated samples: {samples['failed']}/{samples['total']} failed "
            f"sprite-quality validation"
        )

    # An empty signal on either side is not a pass -- the gate needs real data.
    if recon["total"] == 0:
        issues.append("no reconstruction metrics supplied")
    if samples["total"] == 0:
        issues.append("no generated sample images supplied")

    passed = not issues

    return QualityGateResult(
        passed=passed,
        reconstruction=recon,
        samples=samples,
        issues=issues,
        summary={
            "reconstruction": {
                "total": recon["total"],
                "passed": recon["passed"],
                "failed": recon["failed"],
                "avg_psnr": recon["avg_psnr"],
            },
            "samples": {
                "total": samples["total"],
                "passed": samples["passed"],
                "failed": samples["failed"],
                "quality_tier_counts": samples["quality_tier_counts"],
            },
            "verdict": "PASS" if passed else "FAIL",
        },
    )


def write_report(result: QualityGateResult, output_dir: Path) -> Dict[str, Path]:
    """Persist the gate verdict as JSON plus a human-readable Markdown summary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "quality_report.json"
    with open(json_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    md_path = output_dir / "quality_report.md"
    lines = [
        "# Sprite Model Quality Gate",
        "",
        f"**Verdict:** {result.summary.get('verdict', 'PASS' if result.passed else 'FAIL')}",
        "",
    ]
    if result.issues:
        lines.append("## Issues")
        lines.append("")
        for issue in result.issues:
            lines.append(f"- {issue}")
        lines.append("")
    lines += [
        "## Reconstruction",
        "",
        f"- Total: {result.reconstruction.get('total', 0)}",
        f"- Passed: {result.reconstruction.get('passed', 0)}",
        f"- Failed: {result.reconstruction.get('failed', 0)}",
        f"- Average PSNR: {result.reconstruction.get('avg_psnr')}",
        "",
        "## Generated Samples",
        "",
        f"- Total: {result.samples.get('total', 0)}",
        f"- Passed: {result.samples.get('passed', 0)}",
        f"- Failed: {result.samples.get('failed', 0)}",
        f"- Quality tiers: {result.samples.get('quality_tier_counts', {})}",
        "",
    ]
    md_path.write_text("\n".join(lines))

    return {"json": json_path, "markdown": md_path}


def _load_recon_metrics(path: Path) -> List[Dict]:
    """Load per-sample reconstruction metrics from a JSON file.

    Accepts either the full summary written by ``reconstruction_validation.py``
    (a dict with a ``per_sample`` key) or a bare list of sample dicts.
    """
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        per_sample = data.get("per_sample")
        if per_sample is None:
            raise ValueError(f"{path}: expected 'per_sample' key in summary dict")
        return list(per_sample)
    if isinstance(data, list):
        return data
    raise ValueError(f"{path}: expected a list or summary dict of metrics")


def _load_sample_images(samples_dir: Path) -> List[Image.Image]:
    if not samples_dir.is_dir():
        raise ValueError(f"samples dir not found: {samples_dir}")
    paths = sorted(
        p for p in samples_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    return [Image.open(p).convert("RGBA") for p in paths]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Automated visual quality gate for reconstructions and generated samples"
    )
    parser.add_argument(
        "--recon-metrics",
        default=None,
        help="JSON of per-sample reconstruction metrics "
             "(reconstruction_validation.py summary or list)",
    )
    parser.add_argument(
        "--samples-dir",
        default=None,
        help="Directory of generated sample images to validate",
    )
    parser.add_argument("--output", "-o", default="quality_report",
                        help="Output directory for the report")
    parser.add_argument("--min-psnr", type=float, default=20.0)
    parser.add_argument("--max-mse", type=float, default=150.0)
    parser.add_argument("--min-palette", type=int, default=4)
    parser.add_argument("--max-transparency", type=float, default=0.99)
    args = parser.parse_args(argv)

    if not args.recon_metrics and not args.samples_dir:
        parser.error("at least one of --recon-metrics / --samples-dir is required")

    recon_metrics: List[Dict] = []
    if args.recon_metrics:
        try:
            recon_metrics = _load_recon_metrics(Path(args.recon_metrics))
        except (ValueError, OSError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2

    sample_images: List[Image.Image] = []
    if args.samples_dir:
        try:
            sample_images = _load_sample_images(Path(args.samples_dir))
        except (ValueError, OSError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2

    result = run_quality_gate(
        recon_metrics,
        sample_images,
        min_psnr=args.min_psnr,
        max_mse=args.max_mse,
        min_palette=args.min_palette,
        max_transparency=args.max_transparency,
    )

    paths = write_report(result, Path(args.output))
    print(f"Report written to {paths['json']}")
    print(f"Report written to {paths['markdown']}")
    print(f"Verdict: {result.summary.get('verdict')}")
    if result.issues:
        for issue in result.issues:
            print(f"  - {issue}")

    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
