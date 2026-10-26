"""Clone FIDÈLE de l'ETP McGuinness-Bordne d'Hydrotel, porté ligne-à-ligne du
C++ (source/etp_mc_guiness.cpp) vers PyTorch, vectorisé sur les nœuds,
différentiable. Sous-projet PROPRE, indépendant du reste de méandre.

Formule (etp_mc_guiness.cpp l.161) :
    ETP = Re / (lambda * rho) * max(Tmoy + 5, 0) / 68    [mm/jour]
avec Tmoy = (Tmin + Tmax) / 2 [degC], lambda = 2.264 MJ/kg, rho = 1.0 kg/L,
et Re = rayonnement extraterrestre [MJ.m-2.j-1] calculé astronomiquement depuis
le jour julien et la latitude (RayonnementExtraterrestre, l.191-216, dév. de
Spencer pour la déclinaison et la correction d'excentricité E0).

McGuinness ne dépend QUE de la température, de la latitude et du jour de l'année
— aucun vent, VPD ni rayonnement mesuré (contrairement à Penman-Monteith). C'est
l'ETP d'Hydrotel SLSO (simulation.csv : EVAPOTRANSPIRATION;ETP-MC-GUINESS).

Invalide au-delà du cercle polaire (lat > 66.5 dd) comme le C++.
"""
from __future__ import annotations
import math
import torch
from torch import Tensor

_PI = math.pi
_I_SC = 118.1        # constante solaire [MJ.m-2.j-1]  (= FAO Gsc 0.0820 MJ/m2/min * 1440)
_OMEGA = 0.2618      # vitesse angulaire de la terre [rad/h]
_LAMBDA = 2.264      # chaleur latente de vaporisation [MJ/kg]
_RHO = 1.0           # masse volumique de l'eau [kg/L]
_MCG_C = 68.0        # constante empirique McGuinness-Bordne


def rayonnement_extraterrestre(julian_day: Tensor, lat_dd: Tensor) -> Tensor:
    """Re [MJ.m-2.j-1] (etp_mc_guiness.cpp l.191-216). Spencer 1971 pour delta/E0.
    julian_day: 1..365 (gérer le 366 -> 365 en amont). lat_dd: latitude [degrés]."""
    Lambda = lat_dd * (2.0 * _PI / 360.0)                      # latitude [rad]
    Gamma = 2.0 * _PI * (julian_day - 1.0) / 365.0             # moment de l'année [rad]
    E0 = (1.00011 + 0.034221 * torch.cos(Gamma) + 0.00128 * torch.sin(Gamma)
          + 0.000719 * torch.cos(2.0 * Gamma) + 0.000077 * torch.sin(2.0 * Gamma))
    delta = (0.006918 - 0.399912 * torch.cos(Gamma) + 0.070257 * torch.sin(Gamma)
             - 0.006758 * torch.cos(2.0 * Gamma) + 0.000907 * torch.sin(2.0 * Gamma)
             - 0.002697 * torch.cos(3.0 * Gamma) + 0.00148 * torch.sin(3.0 * Gamma))   # déclinaison [rad]
    # T_hr : demi-longueur du jour [h]. C++ : acos(min(-tan(delta)*tan(Lambda), 1.0))/omega.
    # On borne [-1, 1] pour la validité de acos (le C++ ne borne que le haut ; en
    # deçà du cercle polaire l'argument reste dans [-1, 1]). 1e-7 évite le gradient
    # infini de acos aux bornes.
    cos_arg = torch.clamp(-torch.tan(delta) * torch.tan(Lambda), -1.0 + 1e-7, 1.0 - 1e-7)
    T_hr = torch.acos(cos_arg) / _OMEGA
    Re = (2.0 * _I_SC * E0 * (torch.cos(delta) * torch.cos(Lambda) * torch.sin(_OMEGA * T_hr) / _OMEGA
                              + torch.sin(delta) * torch.sin(Lambda) * T_hr)) / 24.0
    return Re


def mcguinness_etp(tmin: Tensor, tmax: Tensor, lat_dd: Tensor, julian_day: Tensor,
                   coef: Tensor | float = 1.0) -> Tensor:
    """ETP McGuinness-Bordne [mm/jour], pas de temps JOURNALIER (poids = 1).
    tmin, tmax [degC] ; lat_dd [degrés] ; julian_day 1..365 ; coef = coefficient
    multiplicatif d'optimisation par zone (défaut 1.0)."""
    jd = torch.clamp(julian_day, max=365.0)               # année bissextile : 366 -> 365
    Re = rayonnement_extraterrestre(jd, lat_dd)
    tmoy = (tmin + tmax) / 2.0
    etp = Re / (_LAMBDA * _RHO) * torch.clamp(tmoy + 5.0, min=0.0) / _MCG_C
    return torch.clamp(etp * coef, min=0.0)
