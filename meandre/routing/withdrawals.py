"""Net water withdrawal per reach.

Withdrawals are explicit modular tensor inputs, NOT baked into calibrated
parameters.  Setting the tensors to zero produces a structurally clean
counterfactual "naturalized" flow regime.

Two sources of withdrawal are distinguished:

* ``net_surface``  — applied directly to the stream discharge at the
  snapped reach (pumping from rivers and lakes, effluent return flow).
* ``net_gw``       — applied to the groundwater reservoir ``S_gw`` at
  the snapped node (wells, infiltration galleries).  The effect on
  stream Q emerges naturally through the aquifer recession ``k_gw``:
  a sustained well pumping depletes S_gw and progressively reduces
  baseflow — there is no instantaneous effect on river discharge.

Convention (both tensors):
    Positive = water added (effluent, artificial recharge).
    Negative = water removed (pumping).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class WithdrawalData:
    """Net withdrawal tensors (n_timesteps, n_reaches) in m³/s.

    Parameters
    ----------
    net : Tensor
        Surface withdrawals applied to stream discharge.
        Positive = water added (effluent, return flow).
        Negative = water removed (pumping, irrigation).
    net_gw : Tensor, optional
        Groundwater withdrawals applied to the aquifer reservoir ``S_gw``.
        Same sign convention.  Defaults to zeros_like(net).
    """

    net: Tensor  # (n_timesteps, n_reaches) — surface
    net_gw: Tensor | None = None  # (n_timesteps, n_reaches) — aquifer

    def __post_init__(self) -> None:
        if self.net_gw is None:
            self.net_gw = torch.zeros_like(self.net)

    def net_withdrawal(self, t: int) -> Tensor:
        """Surface net withdrawal at timestep *t* (m³/s)."""
        return self.net[t]

    def gw_withdrawal(self, t: int) -> Tensor:
        """Groundwater net withdrawal at timestep *t* (m³/s)."""
        return self.net_gw[t]

    @classmethod
    def zeros_like(cls, template: "WithdrawalData") -> "WithdrawalData":
        """Naturalized scenario: all withdrawals set to zero."""
        return cls(
            net=torch.zeros_like(template.net),
            net_gw=torch.zeros_like(template.net_gw),
        )

    @classmethod
    def zeros(
        cls,
        n_timesteps: int,
        n_reaches: int,
        device: torch.device | None = None,
    ) -> "WithdrawalData":
        """Create a zero-withdrawal instance (naturalized baseline)."""
        z = torch.zeros(n_timesteps, n_reaches, device=device)
        return cls(net=z, net_gw=torch.zeros_like(z))
