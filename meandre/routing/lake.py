"""Lake and reservoir node routing.

Replaces kinematic wave routing at lake/reservoir nodes with a
storage-discharge relationship:

    dS/dt = Q_in - Q_out - E_lake + P_lake
    Q_out = f(S)  [power law or weir equation]

For regulated reservoirs with known historical releases, Q_out can be
provided as a forcing tensor instead of being modelled.
"""

import torch
import torch.nn as nn
from torch import Tensor

from meandre.utils.differentiable import soft_relu


class LakeModule(nn.Module):
    """Differentiable lake/reservoir storage-discharge routing.

    Storage-discharge: Q_out = k_lake * max(S - S_dead, 0)^beta

    where S_dead is dead storage (water that can't be released).
    """

    def __init__(
        self,
        k_lake_init: float = 1e-4,
        beta_init: float = 1.5,
        sharpness: float = 10.0,
    ) -> None:
        super().__init__()
        self.sharpness = sharpness
        # exp parameterisation: param = exp(log_param), so log_param = log(init)
        # gives exact init value (unlike softplus(log(x)) = log(1+x) ≠ x).
        self.log_k_lake = nn.Parameter(torch.tensor(k_lake_init).log())
        self.log_beta = nn.Parameter(torch.tensor(beta_init).log())

    @property
    def k_lake(self) -> Tensor:
        return torch.exp(self.log_k_lake)

    @property
    def beta(self) -> Tensor:
        return torch.exp(self.log_beta)

    def forward(
        self,
        Q_in: Tensor,
        S: Tensor,
        area_km2: Tensor,
        E_lake: Tensor,
        P_lake: Tensor,
        S_dead: Tensor,
        Q_release_forced: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            Q_in:             (n_lakes,) total inflow (m3/s)
            S:                (n_lakes,) current storage (m3)
            area_km2:         (n_lakes,) lake surface area (km2)
            E_lake:           (n_lakes,) lake evaporation (mm/day)
            P_lake:           (n_lakes,) lake precipitation (mm/day)
            S_dead:           (n_lakes,) dead storage (m3)
            Q_release_forced: (n_lakes,) or None — if given, overrides model Q_out
        Returns:
            Q_out:  (n_lakes,) outflow (m3/s)
            S_new:  (n_lakes,) updated storage (m3)
        """
        area_m2 = area_km2 * 1e6
        dt = 86400.0  # seconds per day

        # Net surface flux (mm/day -> m3/s)
        Q_surface = (P_lake - E_lake) * 1e-3 * area_m2 / dt

        S_available = soft_relu(S - S_dead, self.sharpness)

        if Q_release_forced is not None:
            Q_out = Q_release_forced
        else:
            # Normalize storage to mean depth (m) before the power law.
            # Raw S is in m³ and can reach 1e9+ m³ for large lakes; computing
            # S**beta directly produces gradients O(S^beta * log S) ≈ 10^13 per
            # timestep, which accumulate across all lake nodes and TBPTT steps to
            # overflow float32 → NaN.  Using depth = S/area keeps the argument
            # in [0, O(100)] m and gradients numerically bounded.
            # clamp(min=1e-6): prevents NaN gradient from depth^beta * log(depth)
            # when depth → 0 (i.e. 0 * log(0) = NaN in autograd).
            depth = (S_available / area_m2.clamp(min=1.0)).clamp(min=1e-6)  # mean depth (m)
            Q_out = self.k_lake * depth**self.beta * area_m2  # (m/s-ish) * m² → m³/s

        # Mass balance
        S_new = S + (Q_in + Q_surface - Q_out) * dt
        S_new = torch.clamp(S_new, min=0.0)

        return Q_out, S_new
