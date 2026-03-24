"""Stream temperature module — heat load advection + atmospheric exchange.

Temperature is tracked as a heat load H = Q * T (m³·°C/s) which is a
conservative quantity that can be scatter-added like discharge through
the topological sweep.  After aggregation at each node, temperature is
recovered as T = H_total / Q_total, then atmospheric exchange is applied.

Key thermal sources per node:
    - Surface runoff + interflow: at air temperature T_air
    - Snowmelt: at 0°C
    - Groundwater baseflow: at T_gw (learned per node, ~8°C)
    - Upstream advection: H_upstream from topological sweep
    - Atmospheric exchange: relaxation toward equilibrium temperature
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class StreamTemperatureModule(nn.Module):
    """Compute stream temperature via heat load advection + atmospheric exchange."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def lateral_heat_load(
        self,
        T_air: Tensor,
        snowmelt: Tensor,
        Q_baseflow: Tensor,
        lateral_inflow: Tensor,
        T_gw: Tensor,
        area_km2_local: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute heat load and temperature of local lateral inflow.

        Args:
            T_air:           (n_nodes,) air temperature (°C)
            snowmelt:        (n_nodes,) snowmelt flux (mm/day)
            Q_baseflow:      (n_nodes,) groundwater baseflow (mm/day)
            lateral_inflow:  (n_nodes,) total lateral inflow (mm/day)
            T_gw:            (n_nodes,) groundwater temperature (°C)
            area_km2_local:  (n_nodes,) local sub-watershed area (km²)

        Returns:
            H_lateral: (n_nodes,) heat load (m³·°C/s)
            T_lateral: (n_nodes,) temperature of lateral inflow (°C)
        """
        lat_safe = lateral_inflow + self.eps

        # Fraction of each source in total lateral inflow
        gw_frac = torch.clamp(Q_baseflow / lat_safe, 0.0, 1.0)
        melt_frac = torch.clamp(snowmelt / lat_safe, 0.0, 1.0)
        surface_frac = torch.clamp(1.0 - gw_frac - melt_frac, 0.0, 1.0)

        # Weighted average temperature of lateral inflow
        # snowmelt at 0°C, surface/interflow at T_air, GW at T_gw
        T_lateral = surface_frac * torch.clamp(T_air, min=-0.05) + gw_frac * T_gw
        # melt_frac * 0.0 is implicit (adds nothing)

        # Convert lateral_inflow mm/day → m³/s, then multiply by temperature
        q_lat_m3s = lateral_inflow * 1e-3 * area_km2_local * 1e6 / 86400.0
        H_lateral = q_lat_m3s * T_lateral

        return H_lateral, T_lateral

    def atmospheric_exchange(
        self,
        T_water: Tensor,
        T_air: Tensor,
        R_n: Tensor,
        K_atm: Tensor,
    ) -> Tensor:
        """Apply linearized atmospheric heat exchange.

        Simple exponential relaxation toward an equilibrium temperature
        approximated from air temperature and net radiation.

        Args:
            T_water: (n_nodes,) current stream temperature (°C)
            T_air:   (n_nodes,) air temperature (°C)
            R_n:     (n_nodes,) net radiation (MJ/m²/day)
            K_atm:   (n_nodes,) exchange coefficient (1/day), [0.05, 0.55]

        Returns:
            T_new: (n_nodes,) updated stream temperature (°C)
        """
        # Equilibrium temperature: T_air + small radiation boost
        # R_n in MJ/m²/day; ~1 MJ/m²/day ≈ 0.3°C warming for shallow streams
        T_eq = T_air + R_n * 0.3

        # Exponential relaxation (dt = 1 day)
        T_new = T_water + K_atm * (T_eq - T_water)
        return torch.clamp(T_new, min=-0.05, max=40.0)
