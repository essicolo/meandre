"""Heteroscedastic noise head for probabilistic discharge predictions.

Replaces the previous ensemble-based UQ stack (ParamNoise + Concrete Dropout)
with a direct probabilistic output: log σ_Q as a learned function of Q.

The model still produces a deterministic Q_mean per timestep per reach; the
noise head produces a per-timestep per-reach σ_Q used by the Gaussian NLL
loss. One forward pass replaces N members.

Parameterisation
----------------
    log σ_Q(t, n) = a + b · log(|Q(t, n)| + ε)

With ``a, b`` learnable scalars (global). Defaults give σ ≈ 10% × |Q|.

Hetero through ``b > 0`` captures the well-known scaling of hydrological
error with discharge magnitude (Box-Cox / log-normal behaviour).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class HeteroscedasticNoiseHead(nn.Module):
    """Predicts log σ_Q from Q via a 2-parameter heteroscedastic model.

    Parameters
    ----------
    a_init : float
        Initial value of the additive log-scale (intercept). Default -2.3 →
        σ ≈ 0.1 × Q for ``b = 1`` (10% relative noise).
    b_init : float
        Initial value of the multiplicative log-scale (slope on log Q).
        Default 1.0 (purely proportional noise).
    eps : float
        Stabiliser inside ``log(|Q| + ε)``. ε=1 gives σ → exp(a) for Q → 0
        (noise floor of exp(a) m³/s in the limit).
    log_sigma_min, log_sigma_max : float
        Clamp range for the output log σ to prevent gradient blowup or
        deterministic collapse during training.
    """

    def __init__(
        self,
        a_init: float = -2.3,
        b_init: float = 1.0,
        eps: float = 1.0,
        log_sigma_min: float = -8.0,
        log_sigma_max: float = 10.0,
    ) -> None:
        super().__init__()
        self.log_sigma_a = nn.Parameter(torch.tensor(a_init, dtype=torch.float32))
        self.log_sigma_b = nn.Parameter(torch.tensor(b_init, dtype=torch.float32))
        self.eps = eps
        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max

    def forward(self, Q: Tensor) -> Tensor:
        """Returns log σ matching the shape of Q."""
        log_sigma = self.log_sigma_a + self.log_sigma_b * torch.log(Q.abs() + self.eps)
        return log_sigma.clamp(min=self.log_sigma_min, max=self.log_sigma_max)

    def anchor_loss(
        self, target_a: float = -3.0, target_b: float | None = None,
    ) -> Tensor:
        """L2 pull on (log_sigma_a, log_sigma_b) toward an anchor.

        Counters the well-known Gaussian NLL degeneracy where σ inflates to
        paper over a bad μ. Pulls ``log_sigma_a`` toward ``target_a``
        (default -3.0 ≈ 5% baseline relative noise when b=1).

        If ``target_b`` is None (default), b is left free — it will find its
        own value through data. Otherwise b is also pulled toward target_b.

        Returns a scalar tensor (mean of squared deviations). Multiply by
        ``w_sigma_anchor`` in the trainer.
        """
        loss = (self.log_sigma_a - target_a) ** 2
        if target_b is not None:
            loss = loss + (self.log_sigma_b - target_b) ** 2
        return loss
