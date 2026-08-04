"""Tests for the VQ-VAE quality-plateau clean stopping point.

Roadmap Phase 0 item: "Let the current VQ-VAE training run complete (or reach a
quality plateau) — don't abandon it mid-run for the LoRA pivot without a clean
stopping point."
"""

import pytest

from models.vqvae.plateau import PlateauStopper
from models.vqvae.train import build_parser


class TestPlateauStopperMinMode:
    def test_improving_series_never_stops(self):
        stopper = PlateauStopper(patience=3, min_delta=1e-4)
        losses = [10.0, 8.0, 6.0, 4.0, 2.0, 1.0, 0.5]
        for epoch, loss in enumerate(losses, start=1):
            assert not stopper.step(epoch, loss)
        assert not stopper.stopped
        assert stopper.best_value == 0.5
        assert stopper.best_epoch == len(losses)

    def test_plateau_detected_after_patience_epochs(self):
        stopper = PlateauStopper(patience=3, min_delta=0.0)
        assert not stopper.step(1, 1.0)
        assert not stopper.step(2, 0.5)
        assert not stopper.step(3, 0.5)
        assert not stopper.step(4, 0.5)
        # Third consecutive non-improving epoch (patience=3) stops the run.
        assert stopper.step(5, 0.5)
        assert stopper.stopped
        assert stopper.stop_epoch == 5

    def test_improvement_resets_counter(self):
        stopper = PlateauStopper(patience=3, min_delta=0.0)
        assert not stopper.step(1, 1.0)
        assert not stopper.step(2, 0.5)
        assert not stopper.step(3, 0.5)
        assert not stopper.step(4, 0.5)
        # A real improvement resets the no-improvement counter.
        assert not stopper.step(5, 0.3)
        assert not stopper.step(6, 0.3)
        assert not stopper.step(7, 0.3)
        assert stopper.step(8, 0.3)
        assert stopper.epochs_no_improve == 3

    def test_min_delta_ignores_small_improvements(self):
        stopper = PlateauStopper(patience=2, min_delta=0.1)
        assert not stopper.step(1, 1.0)
        # Each subsequent value is an improvement, but below min_delta (0.1),
        # so it does not count as an improvement and the counter accumulates.
        assert not stopper.step(2, 0.95)
        assert stopper.step(3, 0.90)
        assert stopper.stopped

    def test_min_epochs_gate_prevents_early_stop(self):
        stopper = PlateauStopper(patience=2, min_delta=0.0, min_epochs=10)
        for epoch in range(1, 10):
            assert not stopper.step(epoch, 0.5)
        assert not stopper.stopped
        assert stopper.step(10, 0.5)
        assert stopper.stopped

    def test_stop_reason_describes_plateau(self):
        stopper = PlateauStopper(patience=2, min_delta=0.0)
        for epoch in range(1, 4):
            stopper.step(epoch, 0.5)
        assert stopper.stopped
        assert "plateau" in stopper.stop_reason
        assert "epoch 3" in stopper.stop_reason
        assert "best=0.5" in stopper.stop_reason

    def test_verdict_is_sticky(self):
        stopper = PlateauStopper(patience=2, min_delta=0.0)
        for epoch in range(1, 4):
            stopper.step(epoch, 0.5)
        assert stopper.stopped
        assert stopper.step(10, 0.0)
        assert stopper.step(11, 0.0)
        assert stopper.stop_epoch == 3


class TestPlateauStopperMaxMode:
    def test_max_mode_tracks_improvements(self):
        stopper = PlateauStopper(patience=3, min_delta=0.0, mode="max")
        psnr = [10.0, 15.0, 18.0, 20.0, 20.0, 20.0, 20.0]
        for epoch, value in enumerate(psnr, start=1):
            stopped = stopper.step(epoch, value)
            if epoch < 7:
                assert not stopped
        assert stopper.best_value == 20.0
        assert stopper.best_epoch == 4
        # With patience=3, the 4th consecutive flat epoch (epoch 7) stops.
        assert stopper.step(7, 20.0)
        assert stopper.stopped

    def test_min_delta_in_max_mode(self):
        stopper = PlateauStopper(patience=1, min_delta=2.0, mode="max")
        assert not stopper.step(1, 10.0)
        # +1 is an improvement but below min_delta of 2.0, so the single
        # allowed patience epoch is consumed and the run stops.
        assert stopper.step(2, 11.0)
        assert stopper.stopped


class TestPlateauStopperValidation:
    def test_invalid_patience(self):
        with pytest.raises(ValueError):
            PlateauStopper(patience=0)

    def test_invalid_min_delta(self):
        with pytest.raises(ValueError):
            PlateauStopper(min_delta=-1.0)

    def test_invalid_min_epochs(self):
        with pytest.raises(ValueError):
            PlateauStopper(min_epochs=-1)

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            PlateauStopper(mode="sideways")

    def test_reset_clears_state(self):
        stopper = PlateauStopper(patience=2, min_delta=0.0)
        for epoch in range(1, 4):
            stopper.step(epoch, 0.5)
        assert stopper.stopped
        stopper.reset()
        assert not stopper.stopped
        assert stopper.best_value is None
        assert stopper.stop_epoch is None
        assert stopper.stop_reason == ""
        assert stopper.epochs_no_improve == 0


class TestTrainCLIIntegration:
    def test_parser_has_early_stop_flags(self):
        parser = build_parser()
        args = parser.parse_args([
            "--dataset", "fake/ds",
            "--early-stop",
            "--patience", "12",
            "--min-delta", "0.001",
            "--min-epochs", "30",
        ])
        assert args.early_stop is True
        assert args.patience == 12
        assert args.min_delta == 0.001
        assert args.min_epochs == 30

    def test_parser_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["--dataset", "fake/ds"])
        assert args.early_stop is False
        assert args.patience == 15
        assert args.min_delta == 1e-4
        assert args.min_epochs == 20

    def test_defaults_match_stopper_semantics(self):
        parser = build_parser()
        args = parser.parse_args(["--dataset", "fake/ds"])
        stopper = PlateauStopper(
            patience=args.patience,
            min_delta=args.min_delta,
            min_epochs=args.min_epochs,
        )
        assert stopper.patience == 15
        assert stopper.min_delta == 1e-4
        assert stopper.min_epochs == 20
