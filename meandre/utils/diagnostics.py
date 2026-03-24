"""Diagnostic outputs from HydroModel.simulate(return_diagnostics=True).

All tensors have shape (n_timesteps, n_nodes) and are on the same device
as the forcing tensor.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class SimDiagnostics:
    """Per-timestep, per-node diagnostic fluxes from a simulation.

    Vertical (mm/day)
    -----------------
    etp        Potential evapotranspiration (Penman-Monteith).
    etr        Actual evapotranspiration: canopy + soil layers 1-3.
    snowmelt   Snow melt flux (mm/day).  Zero when no snow.
    lateral_mm Effective lateral runoff reaching the river network (mm/day).
               = surface runoff + wetland drainage.  Same signal as Q_sim
               but before the Muskingum routing delay, in mm/day units.

    Groundwater (mm/day)
    --------------------
    recharge   Deep drainage from soil layer 3 into the aquifer (mm/day).
               Raw flux before aquifer delay.
    q_baseflow Groundwater baseflow from the lumped aquifer (mm/day).
               Delayed recharge through linear reservoir (k_gw recession).

    Routing (m³/s)
    --------------
    q_lateral  Lateral inflow converted to m³/s (lateral_mm × local area).
               The actual volume added to each reach each day.
    q_upstream Aggregated upstream inflow at each node (m³/s).
               Sum of Q_out from all immediate upstream neighbours.
               For headwater nodes: 0.

    Temperature (°C)
    ----------------
    T_water    Stream water temperature at each node (°C).
               Computed via heat load advection + atmospheric exchange.
               None/absent when temperature module is disabled.
    """

    # Vertical fluxes
    etp: Tensor        # (T, N) mm/day
    etr: Tensor        # (T, N) mm/day
    snowmelt: Tensor   # (T, N) mm/day
    lateral_mm: Tensor # (T, N) mm/day

    # Groundwater
    recharge: Tensor   # (T, N) mm/day — flux from soil L3 into aquifer
    q_baseflow: Tensor # (T, N) mm/day — aquifer output (delayed recharge)

    # Routing fluxes
    q_lateral: Tensor  # (T, N) m³/s
    q_upstream: Tensor # (T, N) m³/s

    # Temperature
    T_water: Tensor | None = None  # (T, N) °C, None if temperature disabled

    @property
    def n_timesteps(self) -> int:
        return self.etp.shape[0]

    @property
    def n_nodes(self) -> int:
        return self.etp.shape[1]

    def to_dict(self) -> dict[str, Tensor]:
        """Return {name: tensor} for easy NetCDF export."""
        d = {
            "etp":        self.etp,
            "etr":        self.etr,
            "snowmelt":   self.snowmelt,
            "lateral_mm": self.lateral_mm,
            "recharge":   self.recharge,
            "q_baseflow": self.q_baseflow,
            "q_lateral":  self.q_lateral,
            "q_upstream": self.q_upstream,
        }
        if self.T_water is not None:
            d["T_water"] = self.T_water
        return d

    @property
    def units(self) -> dict[str, str]:
        d = {
            "etp":        "mm/day",
            "etr":        "mm/day",
            "snowmelt":   "mm/day",
            "lateral_mm": "mm/day",
            "recharge":   "mm/day",
            "q_baseflow": "mm/day",
            "q_lateral":  "m3/s",
            "q_upstream": "m3/s",
        }
        if self.T_water is not None:
            d["T_water"] = "degC"
        return d
