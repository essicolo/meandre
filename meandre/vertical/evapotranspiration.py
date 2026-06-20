"""Evapotranspiration module — Penman-Monteith ETP, root-zone ETR.

ETP (potential):  Full Penman-Monteith equation, all terms differentiable.
ETR (actual):     ETP * f_root_i * f_stress(theta_i) per soil layer.
f_stress:         Smooth linear ramp between theta_wp and theta_fc.

Forcing variables: T_min, T_max, R_n (net radiation), u2 (wind at 2m), e_a (actual vapour pressure).
"""

import math

import torch
import torch.nn as nn
from torch import Tensor

from meandre.utils.physics import (
    LATENT_HEAT_VAPORIZATION,
    PSYCHROMETRIC_CONSTANT,
)


class ETModule(nn.Module):
    """Differentiable ETP (Penman-Monteith ou McGuinness) et ETR réelle par couche.

    et_mode :
      "penman"     — FAO-56 Penman-Monteith (défaut), utilise R_n, u2, e_a.
      "mcguinness" — ETP-MC-GUINESS d'Hydrotel SLSO (clone fidèle, ne dépend que
                     de T, lat, jour julien). Suspect n°1 du déficit de volume :
                     le PM sur-évapore l'eau retenue par le sol clone fidèle.
                     Requiert lat (n_nodes) et doy passés au forward.
    """

    def __init__(self, et_mode: str = "penman") -> None:
        super().__init__()
        self.et_mode = str(et_mode)

    def etp(
        self,
        T_min: Tensor,
        T_max: Tensor,
        R_n: Tensor,
        u2: Tensor,
        e_a: Tensor,
        lat: Tensor | None = None,
        doy: Tensor | int | None = None,
    ) -> Tensor:
        """ETP de référence (mm/jour) selon et_mode. Dispatcher commun aux deux
        sites d'appel (évaporation canopée + ETR par couche)."""
        if self.et_mode == "mcguinness":
            if lat is None or doy is None:
                raise ValueError(
                    "et_mode='mcguinness' requiert lat (n_nodes) et doy ; "
                    "vérifier que node_coords/day_of_year sont threadés."
                )
            from hydrotel_clone.mcguinness import mcguinness_etp
            jd = doy if isinstance(doy, Tensor) else torch.tensor(
                float(doy), device=T_min.device, dtype=T_min.dtype
            )
            return mcguinness_etp(T_min, T_max, lat, jd)
        return self.penman_monteith(T_min, T_max, R_n, u2, e_a)

    def penman_monteith(
        self,
        T_min: Tensor,
        T_max: Tensor,
        R_n: Tensor,
        u2: Tensor,
        e_a: Tensor,
    ) -> Tensor:
        """FAO-56 Penman-Monteith reference ET (mm/day).

        All operations are differentiable with respect to all inputs.

        Args:
            T_min, T_max: (n_nodes,) min/max daily temperature (C)
            R_n:          (n_nodes,) net radiation (MJ/m2/day)
            u2:           (n_nodes,) wind speed at 2m (m/s)
            e_a:          (n_nodes,) actual vapour pressure (kPa)
        Returns:
            ETP: (n_nodes,) potential ET (mm/day)
        """
        T_mean = 0.5 * (T_min + T_max)

        # Saturation vapour pressure (kPa)
        e_s = 0.5 * (
            0.6108 * torch.exp(17.27 * T_max / (T_max + 237.3))
            + 0.6108 * torch.exp(17.27 * T_min / (T_min + 237.3))
        )
        VPD = torch.clamp(e_s - e_a, min=0.0)

        # Slope of saturation vapour pressure curve (kPa/C)
        # FAO-56 Eq. 13: Delta uses e_s(T_mean), not average of e_s(T_min/T_max)
        e_s_Tmean = 0.6108 * torch.exp(17.27 * T_mean / (T_mean + 237.3))
        Delta = 4098.0 * e_s_Tmean / (T_mean + 237.3) ** 2

        gamma = PSYCHROMETRIC_CONSTANT  # kPa/C

        # Soil heat flux G ~ 0 for daily timestep
        G = torch.zeros_like(R_n)

        # Penman-Monteith numerator / denominator
        lam = LATENT_HEAT_VAPORIZATION / 1e6  # MJ/kg
        num = 0.408 * Delta * (R_n - G) + gamma * (900.0 / (T_mean + 273.0)) * u2 * VPD
        denom = Delta + gamma * (1.0 + 0.34 * u2)

        ETP = num / (denom + 1e-6)
        return torch.clamp(ETP, min=0.0)

    def water_stress(
        self, theta: Tensor, theta_wp: Tensor, theta_fc: Tensor
    ) -> Tensor:
        """Smooth linear stress factor in [0, 1].

        0 at wilting point, 1 at field capacity and above.
        """
        span = theta_fc - theta_wp + 1e-6
        return torch.clamp((theta - theta_wp) / span, 0.0, 1.0)

    def forward(
        self,
        T_min: Tensor,
        T_max: Tensor,
        R_n: Tensor,
        u2: Tensor,
        e_a: Tensor,
        theta1: Tensor,
        theta2: Tensor,
        theta3: Tensor,
        theta_wp_1: Tensor,
        theta_wp_2: Tensor,
        theta_wp_3: Tensor,
        theta_fc_1: Tensor,
        theta_fc_2: Tensor,
        theta_fc_3: Tensor,
        f_root_1: Tensor,
        f_root_2: Tensor,
        f_root_3: Tensor,
        E_canopy: Tensor,
        K_c: Tensor | None = None,
        lat: Tensor | None = None,
        doy: Tensor | int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            K_c: optional crop/calibration coefficient applied to ETP.
                 Hydrotel's "coefficient multiplicatif d'optimisation".
                 Defaults to 1.0 (FAO-56 reference, no scaling).
            lat, doy: required when et_mode == "mcguinness".
        Returns:
            ET1, ET2, ET3: actual ET per layer (mm/day)
            ETP:           potential ET (mm/day) — already K_c-scaled
        """
        ETP = self.etp(T_min, T_max, R_n, u2, e_a, lat=lat, doy=doy)
        if K_c is not None:
            ETP = ETP * K_c

        # Remaining ETP after canopy evaporation
        ETP_residual = torch.clamp(ETP - E_canopy, min=0.0)

        stress1 = self.water_stress(theta1, theta_wp_1, theta_fc_1)
        stress2 = self.water_stress(theta2, theta_wp_2, theta_fc_2)
        stress3 = self.water_stress(theta3, theta_wp_3, theta_fc_3)

        ET1 = ETP_residual * f_root_1 * stress1
        ET2 = ETP_residual * f_root_2 * stress2
        ET3 = ETP_residual * f_root_3 * stress3

        return ET1, ET2, ET3, ETP
