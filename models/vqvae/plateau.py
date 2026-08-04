"""Quality-plateau detection for the VQ-VAE training run.

Phase 0 roadmap item: "Let the current VQ-VAE training run complete (or reach a
quality plateau) — don't abandon it mid-run for the LoRA pivot without a clean
stopping point."

Training is currently driven purely by a fixed epoch count. This module adds the
missing "clean stopping point": a plateau tracker that observes a per-epoch
metric (by default the training loss) and signals when quality has stopped
improving, so the run ends at a genuine plateau instead of being abandoned
mid-run or grinding to the max epoch count pointlessly.

The tracker is pure Python (no torch dependency) so it can be unit-tested in
isolation and reused by any training loop (VQ-VAE, transformer prior, or LoRA).
"""

from typing import Optional


class PlateauStopper:
    """Detect when a monitored metric has stopped improving (quality plateau).

    Tracks a single scalar ``value`` per epoch. The metric improves when it
    moves in the configured direction by more than ``min_delta`` relative to the
    best value seen so far. If the metric fails to improve for ``patience``
    consecutive epochs — and at least ``min_epochs`` epochs have run — the
    stopper reports that training should end cleanly.

    Args:
        patience: Consecutive epochs without improvement before stopping.
        min_delta: Minimum absolute change required to count as improvement.
        min_epochs: Do not signal a stop before this many epochs have run.
        mode: ``"min"`` when lower is better (e.g. loss), ``"max"`` when higher
            is better (e.g. PSNR).
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        min_epochs: int = 0,
        mode: str = "min",
    ) -> None:
        if patience < 1:
            raise ValueError("patience must be >= 1")
        if min_delta < 0:
            raise ValueError("min_delta must be >= 0")
        if min_epochs < 0:
            raise ValueError("min_epochs must be >= 0")
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")

        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.mode = mode

        self.best_value: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.epochs_no_improve: int = 0
        self.stopped: bool = False
        self.stop_epoch: Optional[int] = None
        self.stop_reason: str = ""

    def _is_improvement(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "min":
            delta = self.best_value - value
        else:
            delta = value - self.best_value
        return delta > self.min_delta

    def step(self, epoch: int, value: float) -> bool:
        """Record the metric for ``epoch`` and return whether training should stop.

        ``epoch`` is expected to be 1-indexed (the human-facing epoch number).
        Calling ``step`` after the stopper has already signalled a stop keeps
        returning ``True`` (the verdict is sticky).
        """
        if self.stopped:
            return True

        if self._is_improvement(value):
            self.best_value = value
            self.best_epoch = epoch
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1

        if self.epochs_no_improve >= self.patience and epoch >= self.min_epochs:
            self.stopped = True
            self.stop_epoch = epoch
            self.stop_reason = (
                f"quality plateau: no improvement in the monitored metric for "
                f"{self.epochs_no_improve} epochs (best={self.best_value}, "
                f"mode={self.mode}); stopped cleanly at epoch {epoch}"
            )
        return self.stopped

    def reset(self) -> None:
        """Reset all internal state so the tracker can be reused."""
        self.best_value = None
        self.best_epoch = None
        self.epochs_no_improve = 0
        self.stopped = False
        self.stop_epoch = None
        self.stop_reason = ""
