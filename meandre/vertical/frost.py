"""Frost module — soil temperature profile and hydraulic conductivity reduction.

Tracks a surface soil temperature T_soil using a simple energy balance.
When T_soil < 0, K_sat is smoothly reduced to represent frozen soil.

State: T_soil (surface soil temperature, C)
"""

import torch
import torch.nn as nn
from torch import Tensor


from meandre.utils.differentiable import soft_threshold


class FrostModule(nn.Module):
    """Soil temperature tracking and frost K_sat reduction.

    Temperature update is an exponential relaxation toward air temperature:
        T_soil_new = T_soil + alpha_T * (T_air - T_soil)

    where alpha_T is a learned (or fixed) thermal damping coefficient.

    K_sat reduction factor:
        frost_factor = 1 - frost_alpha * soft_threshold(-T_soil, 0)
    so K_sat_eff = K_sat * frost_factor (approaches 0 when deeply frozen).
    """

    def __init__(self, sharpness: float = 5.0) -> None:
        super().__init__()
        self.sharpness = sharpness

    def forward(
        self,
        T_air: Tensor,
        T_soil: Tensor,
        K_sat: Tensor,
        frost_alpha: Tensor,
        alpha_T: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            T_air:       (n_nodes,) air temperature (C)
            T_soil:      (n_nodes,) current soil temperature (C)
            K_sat:       (n_nodes,) baseline saturated hydraulic conductivity
            frost_alpha: (n_nodes,) frost reduction coefficient [0,1]
            alpha_T:     (n_nodes,) thermal damping coefficient (1/day),
                         fitted per node via spatial network.
                         Small values → slow response (deep soil lag).
        Returns:
            K_sat_eff:  (n_nodes,) effective K_sat after frost reduction
            T_soil_new: (n_nodes,) updated soil temperature
        """
        # Relaxation with per-node thermal lag
        T_soil_new = T_soil + alpha_T * (T_air - T_soil)

        # Frost factor: smoothly reduces K_sat when T_soil < 0
        frozen_frac = soft_threshold(-T_soil_new, threshold=0.0, sharpness=self.sharpness)
        frost_factor = 1.0 - frost_alpha * frozen_frac
        K_sat_eff = K_sat * frost_factor

        return K_sat_eff, T_soil_new
