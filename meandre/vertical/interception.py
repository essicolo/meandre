"""Canopy interception module.

Rainfall is first intercepted by the canopy up to a maximum storage capacity.
Excess (throughfall) reaches the soil. Intercepted water evaporates at the
potential rate before transpiration from the root zone begins.

State: canopy_storage (mm)
"""

import torch
import torch.nn as nn
from torch import Tensor

from meandre.utils.differentiable import soft_relu


class InterceptionModule(nn.Module):
    """Differentiable canopy interception and throughfall.

    Interception fill:  delta_S = min(P_rain, capacity - storage)
    Throughfall:        P_thru  = P_rain - delta_S
    Evaporation:        E_canopy = min(ETP, storage + delta_S)
    """

    def __init__(self, sharpness: float = 10.0) -> None:
        super().__init__()
        self.sharpness = sharpness

    def forward(
        self,
        P_rain: Tensor,
        ETP: Tensor,
        canopy_storage: Tensor,
        capacity: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            P_rain:         (n_nodes,) liquid precipitation (mm/day)
            ETP:            (n_nodes,) potential evapotranspiration (mm/day)
            canopy_storage: (n_nodes,) current intercepted water (mm)
            capacity:       (n_nodes,) max canopy storage (mm), from SpatialParams
        Returns:
            P_throughfall:      (n_nodes,) water reaching soil (mm/day)
            E_canopy:           (n_nodes,) canopy evaporation (mm/day)
            canopy_storage_new: (n_nodes,) updated storage (mm)
        """
        # Available storage space
        available = soft_relu(capacity - canopy_storage, self.sharpness)
        # Fill intercepted water (differentiable min via softplus)
        delta_S = available - soft_relu(available - P_rain, self.sharpness)
        P_throughfall = P_rain - delta_S

        # Evaporate intercepted water first
        storage_after_fill = canopy_storage + delta_S
        E_canopy = storage_after_fill - soft_relu(
            storage_after_fill - ETP, self.sharpness
        )
        canopy_storage_new = storage_after_fill - E_canopy

        return P_throughfall, E_canopy, canopy_storage_new
