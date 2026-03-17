"""Snow accumulation and melt module (degree-day method).

Replaces HYDROTEL's hard rain/snow threshold and step-function melt with
differentiable smooth approximations so gradients flow through temperature.

State:  SWE  (snow water equivalent, mm)
Fluxes: snowfall, melt  (mm/day)
"""

import torch
import torch.nn as nn
from torch import Tensor

from meandre.utils.differentiable import soft_threshold


class SnowModule(nn.Module):
    """Differentiable degree-day snow module.

    Accumulation: P_snow = P * soft_threshold(T_snow - T_air)
    Melt:         M      = C_f * max(T_air - T_melt, 0)  [only when SWE > 0]
    Effective P:  P_eff  = P_rain + M

    Parameters are supplied per-node from the spatial field network (SpatialParams),
    not stored as nn.Parameters here.
    """

    def __init__(self, sharpness: float = 10.0) -> None:
        super().__init__()
        self.sharpness = sharpness

    def forward(
        self,
        P: Tensor,
        T_air: Tensor,
        SWE: Tensor,
        C_f: Tensor,
        T_melt: Tensor,
        T_snow: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            P:      (n_nodes,) total precipitation (mm/day)
            T_air:  (n_nodes,) mean air temperature (C)
            SWE:    (n_nodes,) current snow water equivalent (mm)
            C_f:    (n_nodes,) degree-day melt factor (mm/C/day)
            T_melt: (n_nodes,) melting threshold temperature (C)
            T_snow: (n_nodes,) rain/snow partition threshold (C)
        Returns:
            P_eff:   (n_nodes,) effective liquid input reaching the soil (mm/day)
            SWE_new: (n_nodes,) updated SWE (mm)
        """
        # Smooth rain/snow partition
        snow_frac = 1.0 - soft_threshold(T_air, T_snow, self.sharpness)
        rain_frac = 1.0 - snow_frac

        P_snow = snow_frac * P
        P_rain = rain_frac * P

        # Degree-day melt (smooth ReLU so melt -> 0 when T_air < T_melt)
        melt_potential = C_f * torch.nn.functional.softplus(
            (T_air - T_melt) * self.sharpness
        ) / self.sharpness

        # Melt limited by available SWE (already 0 when SWE=0)
        melt = torch.minimum(melt_potential, SWE)

        SWE_new = torch.clamp(SWE + P_snow - melt, min=0.0)
        P_eff = P_rain + melt

        return P_eff, SWE_new
