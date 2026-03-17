"""Net water withdrawal per reach.

Withdrawals are explicit modular tensor inputs, NOT baked into calibrated
parameters.  Setting the tensor to zero produces a structurally clean
counterfactual "naturalized" flow regime.

Convention (follows natural flow sign):
    Positive = water added to the reach (effluent, return flow, …).
    Negative = water removed from the reach (pumping, irrigation, …).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class WithdrawalData:
    """Net withdrawal tensor (n_timesteps, n_reaches) in m³/s.

    Positive = water added (effluent, return flow).
    Negative = water removed (pumping, irrigation).
    """

    net: Tensor  # (n_timesteps, n_reaches)

    def net_withdrawal(self, t: int) -> Tensor:
        """Net withdrawal at timestep *t* (m³/s). Positive = addition, negative = removal."""
        return self.net[t]

    @classmethod
    def zeros_like(cls, template: "WithdrawalData") -> "WithdrawalData":
        """Naturalized scenario: all withdrawals set to zero."""
        return cls(net=torch.zeros_like(template.net))

    @classmethod
    def zeros(
        cls,
        n_timesteps: int,
        n_reaches: int,
        device: torch.device | None = None,
    ) -> "WithdrawalData":
        """Create a zero-withdrawal instance (naturalized baseline)."""
        return cls(net=torch.zeros(n_timesteps, n_reaches, device=device))
