"""Differentiable Muskingum-Cunge kinematic wave routing.

Discretised form of the kinematic wave equation:
    dQ/dt + c * dQ/dx = q_lateral

The Muskingum method approximates storage as a weighted sum:
    S = K * [x*I + (1-x)*Q]

All operations are differentiable w.r.t. Q_in, q_lateral, and the
routing parameters K (travel time) and x (weight factor).
"""

import torch
import torch.nn as nn
from torch import Tensor


class MuskingumCunge(nn.Module):
    """Differentiable Muskingum-Cunge reach routing.

    Parameters are per-reach and come from the spatial field network
    (via Manning's n and reach geometry).

    Supports sub-timestep routing (Hydrotel-style): an inner loop divides
    the daily timestep into n_substeps smaller steps for better peak timing
    and reduced numerical diffusion.

    Reference: Chow, Maidment, Mays (1988) Applied Hydrology, ch. 9.
    """

    def __init__(self, dt: float = 86400.0, n_substeps: int = 1) -> None:
        """
        Args:
            dt: outer timestep in seconds (default 86400 = 1 day)
            n_substeps: number of inner routing sub-steps per outer step.
                        1 = original behaviour.  4 = 6-hour sub-steps.
        """
        super().__init__()
        self.dt = dt
        self.n_substeps = n_substeps

    def _muskingum_step(
        self,
        Q_in: Tensor,
        Q_out_prev: Tensor,
        q_lateral: Tensor,
        K: Tensor,
        x: Tensor,
        dt_sub: float,
    ) -> Tensor:
        """Single Muskingum sub-step with timestep dt_sub."""
        denom = 2.0 * K * (1.0 - x) + dt_sub + 1e-6
        c0 = (dt_sub - 2.0 * K * x) / denom
        c1 = (dt_sub + 2.0 * K * x) / denom
        c2 = (2.0 * K * (1.0 - x) - dt_sub) / denom

        # Clamp c2 ≥ 0 (dispersive regime) and rescale for mass conservation.
        c2 = torch.clamp(c2, min=0.0)
        c01_sum = c0 + c1
        scale = (1.0 - c2) / (c01_sum + 1e-8)
        c0 = c0 * scale
        c1 = c1 * scale

        Q_out = (c0 + c1) * Q_in + c2 * Q_out_prev + q_lateral
        return torch.clamp(Q_out, min=0.0)

    def forward(
        self,
        Q_in: Tensor,
        Q_out_prev: Tensor,
        q_lateral: Tensor,
        K: Tensor,
        x: Tensor,
        Q_in_prev: Tensor | None = None,
    ) -> Tensor:
        """One Muskingum routing step with sub-timestep inner loop.

        Args:
            Q_in:       (n_reaches,) inflow at upstream end at time t (m3/s)
            Q_out_prev: (n_reaches,) outflow at previous timestep (m3/s)
            q_lateral:  (n_reaches,) total lateral inflow (m3/s)
            K:          (n_reaches,) Muskingum K parameter (travel time, s)
            x:          (n_reaches,) Muskingum x weighting factor [0, 0.5]
            Q_in_prev:  (n_reaches,) ignored (kept for API compat)
        Returns:
            Q_out: (n_reaches,) routed outflow (m3/s)
        """
        n = self.n_substeps
        dt_sub = self.dt / n

        Q_out = Q_out_prev
        for _ in range(n):
            Q_out = self._muskingum_step(Q_in, Q_out, q_lateral, K, x, dt_sub)
        return Q_out
