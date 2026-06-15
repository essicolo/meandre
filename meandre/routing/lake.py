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
        k_lake: Tensor | None = None,
        beta: Tensor | None = None,
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
            k_lake, beta:     (n_lakes,) per-node storage-discharge params from
                              the NeRF; None → global scalars (rétrocompat).
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
            S_new = S + (Q_in + Q_surface - Q_out) * dt
            S_new = torch.clamp(S_new, min=0.0)
        else:
            # Implicit Euler with Newton-Raphson iteration.
            # Solve: S_new = S + dt × (Q_in + Q_surface - Q_out(S_new))
            # where Q_out(S) = k × ((S - S_dead)/A)^β × A
            #
            # Explicit Euler (S_new = S + dt × (Q_in - Q_out(S))) is unstable
            # when Q_out(S) × dt > S — the lake can drain to negative,
            # creating mass conservation violations and Q_out oscillations.
            # NR converges in 3-5 iterations and is unconditionally stable
            # for monotone Q_out(S).
            A_safe = area_m2.clamp(min=1.0)
            beta = self.beta if beta is None else beta
            k = self.k_lake if k_lake is None else k_lake

            # Initial guess: explicit Euler (good when stable)
            depth0 = (S_available / A_safe).clamp(min=1e-6)
            Q_out0 = k * depth0**beta * A_safe
            S_new = S + (Q_in + Q_surface - Q_out0) * dt
            S_new = torch.clamp(S_new, min=0.0)

            # Newton iterations
            for _ in range(5):
                depth_n = ((S_new - S_dead).clamp(min=0.0) / A_safe).clamp(min=1e-6)
                Q_out_n = k * depth_n**beta * A_safe
                f = S_new - S - dt * (Q_in + Q_surface - Q_out_n)
                # df/dS = 1 + dt × dQ_out/dS = 1 + dt × k × β × depth^(β-1)
                df = 1.0 + dt * k * beta * depth_n**(beta - 1.0)
                S_new = S_new - f / df
                S_new = torch.clamp(S_new, min=0.0)

            # Final Q_out from converged storage
            depth_final = ((S_new - S_dead).clamp(min=0.0) / A_safe).clamp(min=1e-6)
            Q_out = k * depth_final**beta * A_safe

        return Q_out, S_new
