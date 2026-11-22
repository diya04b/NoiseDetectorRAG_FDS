"""Learning-rate schedule and optimizer-step counts for detector training."""

from __future__ import annotations

import math


def optimizer_step_count(n_batches: int, accumulation: int, epochs: int) -> int:
    """How many optimizer updates a full run would take, before early stopping."""
    if n_batches <= 0 or accumulation <= 0 or epochs <= 0:
        raise ValueError("n_batches, accumulation, and epochs must be positive")
    return epochs * math.ceil(n_batches / accumulation)


def learning_rate_multiplier(step: int, total_steps: int, warmup_ratio: float) -> float:
    """Linear warmup, then linear decay to zero.

    `step` is the number of optimizer updates already finished. The value is the
    fraction of the configured learning rate to use on the next update.
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if step < 0:
        raise ValueError("step must be >= 0")
    if not 0 <= warmup_ratio < 1:
        raise ValueError("warmup_ratio must be at least 0 and less than 1")
    warmup_steps = int(total_steps * warmup_ratio)
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    decay_steps = max(1, total_steps - warmup_steps)
    return max(0.0, (total_steps - step) / decay_steps)
