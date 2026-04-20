"""Learning rate scheduling utilities."""

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR


def build_scheduler(
    optimizer: Optimizer,
    n_epochs: int,
    warmup_epochs: int = 5,
    eta_min_factor: float = 0.01,
):
    """Cosine annealing with optional linear warmup.

    Args:
        optimizer:       AdamW or similar.
        n_epochs:        Total number of training epochs.
        warmup_epochs:   Number of warm-up epochs at the start.  0 = no warmup
                         (use for warm-start fine-tuning).
        eta_min_factor:  Minimum lr as fraction of base lr (default 0.01 = 1%).
    Returns:
        scheduler compatible with scheduler.step() per epoch.
    """
    # Extract base lr from the first param group
    base_lr = optimizer.param_groups[0]["lr"]
    eta_min = base_lr * eta_min_factor

    if warmup_epochs <= 0:
        # Pure cosine — no warmup
        return CosineAnnealingLR(
            optimizer,
            T_max=max(1, n_epochs),
            eta_min=eta_min,
        )

    warmup_epochs = min(warmup_epochs, max(1, n_epochs - 1))

    warmup = LinearLR(
        optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(1, n_epochs - warmup_epochs),
        eta_min=eta_min,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )
