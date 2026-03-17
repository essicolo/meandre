"""Wetland storage and release dynamics.

Isolated wetlands act as slow-release reservoirs. A fraction of the node area
is wetland; the remainder drains normally. The wetland contribution is
fraction-weighted.

State: wetland_storage (mm)
"""

import torch
import torch.nn as nn
from torch import Tensor

from meandre.utils.differentiable import soft_relu


class WetlandModule(nn.Module):
    """Differentiable isolated wetland storage-discharge model.

    Wetland storage fills with excess runoff and drains via a nonlinear
    storage-discharge relationship:

        Q_wet = k_wet * S_wet^beta

    where k_wet and beta are learnable parameters shared across all nodes
    (or supplied per-node from the spatial field network in future versions).
    """

    def __init__(
        self,
        k_wet_init: float = 0.1,
        beta_init: float = 1.5,
        sharpness: float = 10.0,
    ) -> None:
        super().__init__()
        self.sharpness = sharpness
        # Positive parameters via softplus
        self.log_k_wet = nn.Parameter(torch.tensor(k_wet_init).log())
        self.log_beta = nn.Parameter(torch.tensor(beta_init).log())

    @property
    def k_wet(self) -> Tensor:
        return torch.nn.functional.softplus(self.log_k_wet)

    @property
    def beta(self) -> Tensor:
        return torch.nn.functional.softplus(self.log_beta)

    def forward(
        self,
        R_surface: Tensor,
        wetland_storage: Tensor,
        f_wetland: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            R_surface:       (n_nodes,) surface runoff entering wetland (mm/day)
            wetland_storage: (n_nodes,) current wetland storage (mm)
            f_wetland:       (n_nodes,) fraction of node area that is wetland [0,1]
        Returns:
            Q_wetland:           (n_nodes,) outflow from wetland (mm/day)
            Q_direct:            (n_nodes,) direct runoff from non-wetland area
            wetland_storage_new: (n_nodes,) updated wetland storage
        """
        # Wetland fills with its fraction of surface runoff
        R_wet = R_surface * f_wetland
        R_direct = R_surface * (1.0 - f_wetland)

        storage_after_fill = wetland_storage + R_wet

        # Nonlinear storage-discharge
        S_pos = soft_relu(storage_after_fill, self.sharpness)
        Q_wet = self.k_wet * S_pos ** self.beta

        # Discharge limited by available storage
        Q_wetland = torch.minimum(Q_wet, storage_after_fill)
        wetland_storage_new = storage_after_fill - Q_wetland

        return Q_wetland, R_direct, wetland_storage_new
