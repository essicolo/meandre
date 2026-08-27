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
    snowmelt   MAL NOMME : c'est l'APPORT TOTAL au sol (mm/day), pas la fonte.
               La colonne y met diag['apport'], qui vaut la PLUIE en l'absence
               de couvert nival (snow.py l.255) et la sortie du manteau sinon.
               Piege verifie le 2026-08-20 : un cumul hivernal lu comme une
               fonte donnait 1741 % du pic de neige. Pour une vraie perte de
               manteau, sommer les BAISSES de swe.
    lateral_mm Effective lateral runoff reaching the river network (mm/day).
               = R_direct + Q_wetland + interflow + Q_baseflow.
               Same signal as Q_sim but before the Muskingum routing delay,
               in mm/day units.  NOTE: q_baseflow below is already INCLUDED
               in lateral_mm — do not add them when computing total local
               outflow (this would double-count the aquifer contribution).

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
    snowmelt: Tensor   # (T, N) mm/day -- APPORT TOTAL au sol, PAS la fonte
    lateral_mm: Tensor # (T, N) mm/day

    # Groundwater
    recharge: Tensor   # (T, N) mm/day — flux from soil L3 into aquifer
    q_baseflow: Tensor # (T, N) mm/day — aquifer output (delayed recharge)

    # Routing fluxes
    q_lateral: Tensor  # (T, N) m³/s
    q_upstream: Tensor # (T, N) m³/s

    # State snapshots (per timestep, captured after the vertical update).
    # Used by multi-objective NLL against remote-sensed observations.
    swe: Tensor | None = None      # (T, N) mm — snow water equivalent
    theta1: Tensor | None = None   # (T, N) m³/m³ — soil moisture L1 (0-30 cm)
    theta2: Tensor | None = None   # (T, N) m³/m³ — soil moisture L2 (30-100 cm)
    theta3: Tensor | None = None   # (T, N) m³/m³ — soil moisture L3 (100-200 cm)
    s_gw: Tensor | None = None     # (T, N) mm — groundwater storage (aquifer)
    canopy: Tensor | None = None   # (T, N) mm — canopy interception storage
    prof_gel_cm: Tensor | None = None  # (T, N) cm — profondeur du front de gel
    wetland: Tensor | None = None  # (T, N) mm — wetland storage
    # Décomposition de lateral_mm (mm/jour) : surface (ruissellement + débordement de
    # saturation), hypodermique (drainage latéral couche 2), base (recharge q3). Sert à
    # localiser la bifurcation avec Hydrotel étage par étage plutôt qu'au total.
    # Milieu humide : deux termes qui existaient dans la physique mais n'etaient
    # exposes NULLE PART, si bien qu'un bilan d'eau ne pouvait pas fermer et lisait
    # une fuite de 1.97 % de la precipitation (2026-08-20).
    etr_mh: Tensor | None = None    # (T, N) mm/day -- evaporation du milieu humide
    wet_vol: Tensor | None = None   # (T, N) mm -- STOCK du reservoir de milieu humide
    # Sublimation du manteau (Kuzmin, opt-in R32) : SORTIE ATMOSPHERIQUE, comptee par
    # l'audit de fermeture au meme titre que l'ETR -- sans elle le bilan lirait une fuite.
    sublimation: Tensor | None = None  # (T, N) mm/day
    prod_surf: Tensor | None = None
    prod_hypo: Tensor | None = None
    prod_base: Tensor | None = None

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
            "etp": self.etp,
            "etr": self.etr,
            "snowmelt": self.snowmelt,
            "lateral_mm": self.lateral_mm,
            **({"prod_surf": self.prod_surf} if self.prod_surf is not None else {}),
            **({"prod_hypo": self.prod_hypo} if self.prod_hypo is not None else {}),
            **({"prod_base": self.prod_base} if self.prod_base is not None else {}),
            "recharge": self.recharge,
            "q_baseflow": self.q_baseflow,
            "q_lateral": self.q_lateral,
            "q_upstream": self.q_upstream,
        }
        if self.swe is not None:
            d["swe"] = self.swe
        for name in ("theta1", "theta2", "theta3", "s_gw", "canopy", "wetland"):
            if getattr(self, name) is not None:
                d[name] = getattr(self, name)
        if self.T_water is not None:
            d["T_water"] = self.T_water
        return d

    @property
    def units(self) -> dict[str, str]:
        d = {
            "etp": "mm/day",
            "etr": "mm/day",
            "snowmelt": "mm/day",
            "lateral_mm": "mm/day",
            "recharge": "mm/day",
            "q_baseflow": "mm/day",
            "q_lateral": "m3/s",
            "q_upstream": "m3/s",
        }
        if self.swe is not None:
            d["swe"] = "mm"
        for name in ("theta1", "theta2", "theta3"):
            if getattr(self, name) is not None:
                d[name] = "m3/m3"
        for name in ("s_gw", "canopy", "wetland"):
            if getattr(self, name) is not None:
                d[name] = "mm"
        if self.T_water is not None:
            d["T_water"] = "degC"
        return d
