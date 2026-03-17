"""Vertical column orchestrator — chains all vertical balance modules.

Processes one timestep for all nodes simultaneously (vectorised).
Input:  enriched forcing (raw + temporal context), current HydroState, SpatialParams
Output: ColumnOutput with lateral inflow, updated state, and component fluxes
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from meandre.spatial.field_network import SpatialParams
from meandre.utils.state import HydroState
from meandre.vertical.aquifer import AquiferModule
from meandre.vertical.evapotranspiration import ETModule
from meandre.vertical.frost import FrostModule
from meandre.vertical.interception import InterceptionModule
from meandre.vertical.snow import SnowModule
from meandre.vertical.soil import SoilModule
from meandre.vertical.wetland import WetlandModule


@dataclass
class ColumnOutput:
    """Output from VerticalColumn.forward().

    Always returned (no conditional tuple unpacking).
    """

    lateral_inflow: Tensor   # (n_nodes,) mm/day — total input to routing
    state: HydroState        # updated state after one day
    snowmelt: Tensor         # (n_nodes,) mm/day — for temperature module
    recharge: Tensor         # (n_nodes,) mm/day — soil L3 drainage (pre-aquifer)
    Q_baseflow: Tensor       # (n_nodes,) mm/day — GW contribution (isolated)
    diag: dict | None        # per-node diagnostic tensors, or None


class VerticalColumn(nn.Module):
    """Chain: Snow -> Frost -> Interception -> ET -> Soil -> Wetland -> Aquifer.

    All modules are applied per-node and vectorised over the full graph.
    """

    def __init__(self) -> None:
        super().__init__()
        self.snow = SnowModule()
        self.frost = FrostModule()
        self.interception = InterceptionModule()
        self.et = ETModule()
        self.soil = SoilModule()
        self.wetland = WetlandModule()
        self.aquifer = AquiferModule()

    def forward(
        self,
        forcing: Tensor,
        state: HydroState,
        params: SpatialParams,
        return_diagnostics: bool = False,
    ) -> ColumnOutput:
        """
        Args:
            forcing: (n_nodes, n_forcing_vars)
                Columns (in order): P, T_min, T_max, R_n, u2, e_a
                Optional context appended after first 6 columns.
            state:   Current HydroState (n_nodes,) per variable.
            params:  SpatialParams per node from spatial field network.
            return_diagnostics: If True, populate diag dict with
                intermediate fluxes: etp, etr, snowmelt, lateral_mm.
        Returns:
            ColumnOutput with lateral_inflow, new_state, snowmelt,
            Q_baseflow, and optional diagnostics.
        """
        P      = forcing[:, 0]
        T_min  = forcing[:, 1]
        T_max  = forcing[:, 2]
        R_n    = forcing[:, 3]
        u2     = forcing[:, 4]
        e_a    = forcing[:, 5]
        T_air  = 0.5 * (T_min + T_max)

        # 1. Snow
        P_eff, swe_new = self.snow(
            P, T_air, state.swe,
            params.C_f, params.T_melt, params.T_snow,
        )

        # 2. Frost: update soil temperature and reduce K_sat for all 3 layers
        K_sat_1_eff, t_soil_new = self.frost(
            T_air, state.t_soil, params.K_sat_1, params.frost_alpha, params.alpha_T
        )
        # Apply same frost factor to layers 2 and 3
        frost_factor = K_sat_1_eff / (params.K_sat_1 + 1e-8)
        K_sat_2_eff = params.K_sat_2 * frost_factor
        K_sat_3_eff = params.K_sat_3 * frost_factor

        # 3. Interception: compute ETP first for canopy evap
        ETP_approx = self.et.penman_monteith(T_min, T_max, R_n, u2, e_a)
        P_thru, E_canopy, canopy_new = self.interception(
            P_eff, ETP_approx, state.canopy_storage, params.interception_capacity
        )

        # 4. ET per layer (updated after canopy)
        ET1, ET2, ET3, _ = self.et(
            T_min, T_max, R_n, u2, e_a,
            state.theta1, state.theta2, state.theta3,
            params.theta_wp_1, params.theta_wp_2, params.theta_wp_3,
            params.theta_fc_1, params.theta_fc_2, params.theta_fc_3,
            params.f_root_1,  params.f_root_2,   params.f_root_3,
            E_canopy,
        )

        # 5. Soil balance (frost-modified K_sat for layer 1)
        theta1_new, theta2_new, theta3_new, R_surface, interflow, recharge = self.soil(
            P_thru, ET1, ET2, ET3,
            state.theta1, state.theta2, state.theta3,
            K_sat_1_eff, K_sat_2_eff, K_sat_3_eff,
            params.porosity_1, params.porosity_2, params.porosity_3,
            params.theta_fc_1, params.theta_fc_2, params.theta_fc_3,
            params.theta_wp_1, params.theta_wp_2, params.theta_wp_3,
            slope_factor=params.slope_factor,
            krec=params.krec,
        )

        # 6. Wetland
        Q_wetland, R_direct, wetland_new = self.wetland(
            R_surface, state.wetland_storage, params.f_wetland,
        )

        # 7. Aquifer: intercept soil recharge, delay through GW storage
        Q_baseflow, S_gw_new = self.aquifer(recharge, state.S_gw, params.k_gw)

        lateral_inflow = R_direct + Q_wetland + interflow + Q_baseflow  # mm/day

        # Snowmelt (always computed — needed for temperature module)
        snowmelt = torch.clamp(state.swe - swe_new, min=0.0)

        new_state = HydroState(
            theta1=theta1_new,
            theta2=theta2_new,
            theta3=theta3_new,
            swe=swe_new,
            t_soil=t_soil_new,
            canopy_storage=canopy_new,
            wetland_storage=wetland_new,
            S_gw=S_gw_new,
            T_water=state.T_water,  # updated later by routing temperature
        )

        diag = None
        if return_diagnostics:
            diag = {
                "etp":        ETP_approx,
                "etr":        E_canopy + ET1 + ET2 + ET3,
                "snowmelt":   snowmelt,
                "lateral_mm": lateral_inflow,
                "recharge":   recharge,
                "q_baseflow": Q_baseflow,
            }

        return ColumnOutput(
            lateral_inflow=lateral_inflow,
            state=new_state,
            snowmelt=snowmelt,
            recharge=recharge,
            Q_baseflow=Q_baseflow,
            diag=diag,
        )
