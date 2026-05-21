"""Heteroscedastic noise heads for probabilistic discharge predictions.

Two variants:

``HeteroscedasticNoiseHead``
    2-scalar global model: log σ(t, n) = a + b · log(|Q| + ε).
    One (a, b) pair shared across all nodes. Cheap but unexpressive.

``SpatialNoiseHead``
    Per-node MLP that takes spatial parameters as input and outputs
    (a_n, b_n) per node: log σ(t, n) = a(n) + b(n) · log(|Q(t,n)| + ε).
    The MLP sees hydrological parameters (K_sat, f_vert, etc.) so nodes
    with different catchment characteristics get different uncertainty.
    Replaces the previous 2-scalar head for the probabilistic phase.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class HeteroscedasticNoiseHead(nn.Module):
    """Predicts log σ_Q from Q via a 2-parameter global model.

    One (a, b) pair shared across all nodes. Kept for backward
    compatibility and for deterministic runs (w_nll=0).

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
        Clamp range for the output log σ.
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
        """L2 pull on (log_sigma_a, log_sigma_b) toward an anchor."""
        loss = (self.log_sigma_a - target_a) ** 2
        if target_b is not None:
            loss = loss + (self.log_sigma_b - target_b) ** 2
        return loss


class SpatialNoiseHead(nn.Module):
    """Per-node heteroscedastic noise head conditioned on spatial parameters.

    Takes the NeRF's constrained spatial parameters (K_sat, f_vert, etc.)
    as input and outputs per-node (a_n, b_n) uncertainty coefficients.
    Each node gets its own uncertainty profile based on catchment
    characteristics, unlike the global 2-scalar model.

    Formula:
        log σ(t, n) = a(n) + b(n) · log(|Q(t,n)| + ε)

    Architecture:
        spatial_params (n_nodes, N_PARAMS) → Linear(N_PARAMS, hidden)
        → SiLU → Linear(hidden, 2) → (a_n, b_n)

    Parameters
    ----------
    n_spatial_params : int
        Number of spatial parameters per node (SpatialParams.N_PARAMS = 36).
    hidden : int
        Hidden layer width. Default 32 (small: the mapping from params to
        uncertainty is simple — only 2 outputs per node).
    eps : float
        Stabiliser inside ``log(|Q| + ε)``.
    log_sigma_min, log_sigma_max : float
        Clamp range for log σ.
    """

    def __init__(
        self,
        n_spatial_params: int = 36,
        hidden: int = 32,
        eps: float = 1.0,
        log_sigma_min: float = -8.0,
        log_sigma_max: float = 10.0,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max
        self.net = nn.Sequential(
            nn.Linear(n_spatial_params, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )
        # Initialise so the output starts near (a=-2.3, b=1.0),
        # matching the global head defaults (≈10% relative noise).
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, -2.3)  # a_init
        # b_init=1.0: set the second output bias to 1.0
        # Since the network outputs 2 values (a, b), we set both.
        # But we want a≈-2.3 and b≈1.0 at init.
        # With zero weights, the output is just the bias.
        # So we set bias[0]=-2.3 and bias[1]=1.0.
        with torch.no_grad():
            self.net[-1].bias[0] = -2.3  # a_init
            self.net[-1].bias[1] = 1.0    # b_init

    def forward(self, spatial_params: Tensor, Q: Tensor) -> Tensor:
        """Compute per-node log σ.

        Parameters
        ----------
        spatial_params : Tensor, shape (n_nodes, N_PARAMS)
            Constrained spatial parameters from the NeRF.
        Q : Tensor, shape (n_timesteps, n_nodes)
            Simulated discharge (should be detached before calling).

        Returns
        -------
        log_sigma : Tensor, shape (n_timesteps, n_nodes)
        """
        ab = self.net(spatial_params)  # (n_nodes, 2)
        a = ab[:, 0]  # (n_nodes,)
        b = ab[:, 1]  # (n_nodes,)
        log_sigma = a.unsqueeze(0) + b.unsqueeze(0) * torch.log(Q.abs() + self.eps)
        return log_sigma.clamp(min=self.log_sigma_min, max=self.log_sigma_max)

    def anchor_loss(
        self, spatial_params: Tensor,
        target_a: float = -3.0,
        target_b: float | None = None,
    ) -> Tensor:
        """L2 pull on the *mean* (a, b) across nodes toward an anchor.

        Regularises the average uncertainty, not per-node values, so
        individual nodes can still deviate from the target. This prevents
        the NLL degeneracy (σ inflates to mask a bad μ) while allowing
        spatially-varying uncertainty.

        Parameters
        ----------
        spatial_params : Tensor, shape (n_nodes, N_PARAMS)
            Needed to compute the forward pass through the MLP.
        target_a : float
            Target for the mean of a(n). Default -3.0 (≈5% relative noise).
        target_b : float or None
            Target for the mean of b(n). If None, b is left free.
        """
        ab = self.net(spatial_params)  # (n_nodes, 2)
        loss = (ab[:, 0].mean() - target_a) ** 2
        if target_b is not None:
            loss = loss + (ab[:, 1].mean() - target_b) ** 2
        return loss
