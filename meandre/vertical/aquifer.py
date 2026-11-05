"""Lumped aquifer module — groundwater storage and baseflow generation.

Intercepts recharge from soil layer 3 and delays it through a linear
reservoir, producing a smoothed baseflow signal.

    dS_gw/dt = recharge - k_gw * S_gw

Analytical solution (exact for constant recharge over one day):

    S_gw(t+1) = S_gw(t) * exp(-k_gw) + (recharge / k_gw) * (1 - exp(-k_gw))
    Q_baseflow = k_gw * S_gw(t+1)

State: S_gw (mm) — groundwater storage per node.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class AquiferModule(nn.Module):
    """Differentiable lumped linear-reservoir aquifer.

    Receives recharge (mm/day) from soil layer 3 and returns delayed
    baseflow (mm/day).  The recession constant k_gw (1/day) is supplied
    per-node from the SpatialFieldNetwork.
    """

    def __init__(self, k_gw_min: float = 1e-6) -> None:
        super().__init__()
        self.k_gw_min = k_gw_min

    def forward(
        self,
        recharge: Tensor,
        S_gw: Tensor,
        k_gw: Tensor,
        gw_withdrawal: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """One-day aquifer update.

        Args:
            recharge:      (n_nodes,) recharge from soil L3 (mm/day), >= 0.
            S_gw:          (n_nodes,) current groundwater storage (mm).
            k_gw:          (n_nodes,) recession coefficient (1/day).
            gw_withdrawal: (n_nodes,) net withdrawal from the aquifer (mm/day).
                Positive = water added (artificial recharge), negative =
                water removed (pumping).  Included as a constant forcing
                alongside recharge in the linear-reservoir ODE.

        Returns:
            Q_baseflow: (n_nodes,) baseflow discharge (mm/day).
            S_gw_new:   (n_nodes,) updated groundwater storage (mm).
        """
        # Clamp k_gw away from zero for numerical safety.
        # For very small k_gw, use Taylor expansion to avoid (1-exp(-k))/k loss.
        k = torch.clamp(k_gw, min=self.k_gw_min)

        # Treat pumping / artificial recharge as a constant flux over dt.
        # Sign: withdrawal tensor is positive=add, negative=remove.
        net_input = recharge
        if gw_withdrawal is not None:
            net_input = net_input + gw_withdrawal

        # Analytical linear reservoir solution (dt = 1 day implicit)
        decay = torch.exp(-k)
        # (1 - exp(-k)) / k  — Taylor-safe: for k < 1e-4, ≈ 1 - k/2
        one_minus_decay_over_k = torch.where(
            k > 1e-4,
            (1.0 - decay) / k,
            1.0 - k * 0.5 + k * k / 6.0,
        )

        S_gw_new = S_gw * decay + net_input * one_minus_decay_over_k
        # Clamp to 0: if pumping exceeds storage + recharge, the aquifer
        # simply empties (physically the well would go dry).
        S_gw_new = torch.clamp(S_gw_new, min=0.0)

        Q_baseflow = k * S_gw_new

        return Q_baseflow, S_gw_new
