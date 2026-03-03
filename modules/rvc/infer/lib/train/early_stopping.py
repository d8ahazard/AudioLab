"""
Early stopping monitor for RVC V2 and V3 training.

Detects plateau (no improvement over N epochs) and uptrend (loss increasing for M consecutive epochs)
and signals when training should stop. Shared by both V2 and V3 train loops.
"""

from __future__ import annotations

from collections import deque
from typing import Optional


class EarlyStoppingMonitor:
    """
    Tracks EMA of mel/fm and signals when training should stop based on:
    - Plateau: mel improvement < min_improvement_ratio over plateau_patience epochs
    - Uptrend: mel increases for uptrend_patience consecutive epochs

    Ignores gen/disc losses for stop decisions (they oscillate in GAN training).
    """

    def __init__(
        self,
        ema_alpha: float = 0.05,
        plateau_patience: int = 20,
        uptrend_patience: int = 10,
        min_improvement_ratio: float = 0.01,
        min_epochs: int = 8,
        composite_weight_fm: float = 0.3,
    ):
        self.ema_alpha = float(ema_alpha)
        self.plateau_patience = int(plateau_patience)
        self.uptrend_patience = int(uptrend_patience)
        self.min_improvement_ratio = float(min_improvement_ratio)
        self.min_epochs = int(min_epochs)
        self.composite_weight_fm = float(composite_weight_fm)

        self.ema_mel: Optional[float] = None
        self.ema_fm: Optional[float] = None

        self.mel_history: deque = deque(maxlen=self.plateau_patience)
        self.composite_history: deque = deque(maxlen=self.plateau_patience)
        self.mel_uptrend_epochs: int = 0
        self._stop_reason: str = ""
        self._stopped: bool = False

    def _ema(self, prev: Optional[float], val: float) -> float:
        if prev is None:
            return float(val)
        return (1.0 - self.ema_alpha) * float(prev) + self.ema_alpha * float(val)

    def _compute_composite(self) -> Optional[float]:
        if self.ema_mel is None or self.ema_fm is None:
            return None
        return self.ema_mel + self.composite_weight_fm * self.ema_fm

    def update(
        self,
        loss_gen_all: float,
        loss_disc: float,
        loss_mel: float,
        loss_kl: float,
        loss_fm: float,
    ) -> None:
        """Update EMA for mel and fm (called each step)."""
        self.ema_mel = self._ema(self.ema_mel, loss_mel)
        self.ema_fm = self._ema(self.ema_fm, loss_fm)

    def on_epoch_end(self, epoch: int) -> None:
        """Called at end of each epoch. Updates history for plateau/uptrend detection."""
        if self.ema_mel is not None:
            self.mel_history.append(self.ema_mel)
        composite = self._compute_composite()
        if composite is not None:
            self.composite_history.append(composite)

        if len(self.mel_history) >= 2:
            if self.mel_history[-1] > self.mel_history[-2]:
                self.mel_uptrend_epochs += 1
            else:
                self.mel_uptrend_epochs = 0

    def should_stop(self) -> bool:
        """
        Returns True when plateau or uptrend detected.
        Does not stop before min_epochs have completed.
        """
        if self._stopped:
            return True
        if len(self.mel_history) < self.min_epochs:
            return False
        if len(self.mel_history) < self.plateau_patience:
            # Not enough history for plateau; only check uptrend
            if self.mel_uptrend_epochs >= self.uptrend_patience:
                self._stopped = True
                self._stop_reason = (
                    f"Early stop: mel increasing for {self.mel_uptrend_epochs} consecutive epochs "
                    f"(uptrend_patience={self.uptrend_patience})"
                )
                return True
            return False

        oldest_mel = self.mel_history[0]
        current_mel = self.mel_history[-1]
        improvement_ratio = (oldest_mel - current_mel) / oldest_mel if oldest_mel > 0 else 0.0

        if improvement_ratio < self.min_improvement_ratio:
            self._stopped = True
            self._stop_reason = (
                f"Early stop: mel plateau - improvement {improvement_ratio:.2%} "
                f"< {self.min_improvement_ratio:.2%} over {len(self.mel_history)} epochs "
                f"(plateau_patience={self.plateau_patience})"
            )
            return True

        if self.mel_uptrend_epochs >= self.uptrend_patience:
            self._stopped = True
            self._stop_reason = (
                f"Early stop: mel increasing for {self.mel_uptrend_epochs} consecutive epochs "
                f"(uptrend_patience={self.uptrend_patience})"
            )
            return True

        return False

    def reason(self) -> str:
        """Human-readable reason for stopping (empty if not stopped)."""
        return self._stop_reason
