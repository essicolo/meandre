"""Snow accumulation and melt module (degree-day with cold content).

Replaces HYDROTEL's hard rain/snow threshold and step-function melt with
differentiable smooth approximations so gradients flow through temperature.

Cold content: tracks the energy deficit of the snowpack (mm équiv. eau).
Available melt energy first depletes the cold content before producing
liquid melt. Prevents redoux mid-winter from instantly liquidating the
snowpack — physically, the pack needs to warm to 0°C first.

State:  SWE  (snow water equivalent, mm)
        CC   (cold content, mm équiv. eau)
Fluxes: snowfall, melt  (mm/day)
"""

import torch
import torch.nn as nn
from torch import Tensor

from meandre.utils.differentiable import soft_threshold


# Ratio of ice heat capacity to water latent heat of fusion
# c_p_ice = 2.108 kJ/kg/°C, latent_heat_fusion = 334 kJ/kg
# → 1 mm SWE × 1°C cooling = 0.00631 mm cold content
ICE_HEAT_RATIO = 0.00631
# Maximum pack temperature deficit (°C). Pack rarely below -20°C in QC.
MAX_PACK_TEMP_DEFICIT = 15.0


class SnowModule(nn.Module):
    """Differentiable degree-day snow module with cold content.

    Accumulation:  P_snow = P * soft_threshold(T_snow - T_air)
    Cold intake:   new snow at T<0 brings CC ∝ snowfall × |T_air|
    Air cooling:   T_air<T_melt cools the pack (slowed by SWE thermal mass)
    Melt energy:   E = C_f × max(T_air - T_melt, 0)
    Net melt:      first deplete CC, then E - CC_consumed → actual melt

    Parameters are supplied per-node from the spatial field network (SpatialParams).
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
        cold_content: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            P:            (n_nodes,) total precipitation (mm/day)
            T_air:        (n_nodes,) mean air temperature (C)
            SWE:          (n_nodes,) current snow water equivalent (mm)
            C_f:          (n_nodes,) degree-day melt factor (mm/C/day)
            T_melt:       (n_nodes,) melting threshold temperature (C)
            T_snow:       (n_nodes,) rain/snow partition threshold (C)
            cold_content: (n_nodes,) current pack cold content (mm équiv eau).
                          If None, treated as zeros (legacy mode without cold content).
        Returns:
            P_eff:    (n_nodes,) effective liquid input reaching the soil (mm/day)
            SWE_new:  (n_nodes,) updated SWE (mm)
            CC_new:   (n_nodes,) updated cold content (mm équiv eau)
        """
        if cold_content is None:
            cold_content = torch.zeros_like(SWE)

        # Smooth rain/snow partition
        snow_frac = 1.0 - soft_threshold(T_air, T_snow, self.sharpness)
        rain_frac = 1.0 - snow_frac

        P_snow = snow_frac * P
        P_rain = rain_frac * P

        # ── Cold content accumulation ───────────────────────────────────
        # 1. New snow at T_air<0 brings cold (proportional to snowfall × |T|)
        cold_below_freezing = torch.clamp(-T_air, min=0.0)  # max(0, -T_air)
        snow_cold = P_snow * cold_below_freezing * ICE_HEAT_RATIO

        # 2. Air cooling of existing pack (slowed when pack is deep — heat
        # transfer surface-only). Use tanh(SWE/50) as thermal-mass damping.
        air_below_melt = torch.clamp(T_melt - T_air, min=0.0)
        # 0.3 mm/°C/day base cooling rate, attenuated for deep packs
        thermal_factor = torch.tanh(SWE / 50.0)
        air_cool = 0.3 * air_below_melt * thermal_factor

        CC = cold_content + snow_cold + air_cool

        # Cap CC at physical maximum (~15°C deficit × ICE_HEAT_RATIO × SWE)
        CC_max = SWE * MAX_PACK_TEMP_DEFICIT * ICE_HEAT_RATIO
        CC = torch.minimum(CC, CC_max)

        # ── Melt with cold content depletion ───────────────────────────
        melt_potential = C_f * torch.nn.functional.softplus(
            (T_air - T_melt) * self.sharpness
        ) / self.sharpness

        # Energy first depletes CC, only excess goes to melt
        CC_consumed = torch.minimum(CC, melt_potential)
        CC = CC - CC_consumed
        melt_energy_left = melt_potential - CC_consumed

        # Actual melt limited by available SWE (after snowfall added)
        melt = torch.minimum(melt_energy_left, SWE + P_snow)

        SWE_new = torch.clamp(SWE + P_snow - melt, min=0.0)
        # CC capped to what new SWE allows
        CC_new = torch.minimum(CC, SWE_new * MAX_PACK_TEMP_DEFICIT * ICE_HEAT_RATIO)

        P_eff = P_rain + melt

        return P_eff, SWE_new, CC_new
