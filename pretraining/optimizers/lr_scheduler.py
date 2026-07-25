from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class WarmupCosineSchedule(LambdaLR):
    """Linear warmup followed by cosine learning-rate decay."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        cycles: float = 0.5,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.cycles = cycles
        super().__init__(optimizer, self._lr_multiplier, last_epoch)

    def _lr_multiplier(self, step: int) -> float:
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))

        progress = float(step - self.warmup_steps) / float(
            max(1, self.total_steps - self.warmup_steps)
        )
        return max(
            0.0,
            0.5 * (1.0 + math.cos(math.pi * self.cycles * 2.0 * progress)),
        )
