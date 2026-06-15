"""BV3C2 fidèle à Hydrotel — référence d'équivalence différentiable.

Reproduit EXACTEMENT la mathématique du bilan vertical d'Hydrotel
(`source/bv3c2.cpp`, plateforme SLSO_LN24HA_2020), pas la version « inspirée »
de `soil.py`. Objectif : injecter les paramètres calibrés d'Hydrotel et vérifier
que méandre calcule la même chose, AVANT de laisser le NeRF apprendre par-dessus.

Divergences corrigées vs soil.py (inventaire source 2026-06-15) :
  - Rétention/conductivité : Campbell/Clapp-Hornberger, PAS van Genuchten.
        K(omega)   = Ks · omega^(2b+3)              (b = 1/lambda)
        psi(omega) = psis · omega^(-b)   si omega < omegpi
                     spline quadratique -> 0         sinon
  - Ruissellement : excès d'infiltration HORTONIEN plafonné au Ks de la couche 1,
        plus porte GEL (sol gelé + neige<10mm => pinf=0, tout ruisselle) et
        porte SATURATION (theta1=thetas => pinf=0). PAS de VSA lisse.
        pinf = min(prec, ks1) ; ruis = prec - pinf.
  - Flux verticaux (TriCoucheOct97) : Darcy face = max(k_haut, k_bas).
        qq12 = max(k1,k2)·(2·(psi2-psi1)/(z1+z2) + 1)
        qq23 = max(k2,k3)·(2·(psi3-psi2)/(z2+z3) + 1)
  - Interflow : q2 = k2·sin(atan(pente))·z2  (couche 2, piloté pente).
  - Baseflow : q3 = krec·z3·theta3            (réservoir linéaire couche 3).

Tout en mètres et m/h en interne (comme Hydrotel), converti en mm/jour en sortie.
Portes dures (torch.where) : fidélité d'abord, on adoucira pour l'entraînement.
Différentiable (max et where sont sous-différentiables, suffisant pour autograd).
"""
from __future__ import annotations

import math
import torch
from torch import Tensor

# Table proprietehydrolique.sol (Hydrotel SLSO) — par texture.
# thetas, thetacc (capacité champ), thetapf (point flétrissement),
# ks (m/h), psis (m), lambda. b = 1/lambda.
SOIL_TEXTURES = {
    "sand": dict(thetas=0.417, thetacc=0.091, thetapf=0.033, ks=0.2100, psis=0.1598, lam=0.694),
    "sandy_loam": dict(thetas=0.412, thetacc=0.207, thetapf=0.095, ks=0.0259, psis=0.302, lam=0.378),
    "loam": dict(thetas=0.434, thetacc=0.270, thetapf=0.117, ks=0.0132, psis=0.4012, lam=0.252),
    "silt_loam": dict(thetas=0.486, thetacc=0.330, thetapf=0.133, ks=0.0068, psis=0.5087, lam=0.234),
    "clay": dict(thetas=0.385, thetacc=0.396, thetapf=0.272, ks=0.0006, psis=0.856, lam=0.165),
    "peat": dict(thetas=0.930, thetacc=0.275, thetapf=0.050, ks=1.008, psis=0.0103, lam=0.370),
}

# Épaisseurs calibrées bv3c.csv (m) et récession baseflow.
EPAISSEUR = (0.21941, 0.15725, 2.65)   # z1, z2, z3
KREC_DEFAULT = 1.2869e-7               # COEFFICIENT DE RECESSION (m/h)
DT_HOURS = 24.0                        # pas de temps journalier


def campbell_K(omega: Tensor, ks: Tensor, b: Tensor) -> Tensor:
    """Conductivité Campbell : K = Ks · omega^(2b+3). omega in (0,1]."""
    return ks * omega.clamp(min=0.01).pow(2.0 * b + 3.0)


def campbell_psi(omega: Tensor, psis: Tensor, b: Tensor, psi_floor: float = -100.0) -> Tensor:
    """Potentiel matriciel Campbell avec raccord quadratique près de la
    saturation (bv3c2.cpp lignes 1819-1839). psi < 0 (succion), en mètres.

    omegpi = (1+2b)/(2+2b) = saturation d'inflexion. Sous omegpi : loi
    puissance psis·omega^(-b). Au-dessus : spline quadratique psi = -mm·
    (omega-nn)·(omega-1) qui rejoint 0 à omega=1 et se raccorde en omegpi.

    psi borné à psi_floor (m) : à faible humidité psi -> -1e4 m et fait
    exploser le gradient Darcy. Même garde que soil.py (clamp -100 m).
    """
    omega = omega.clamp(min=0.01, max=1.0)
    omegpi = (1.0 + 2.0 * b) / (2.0 + 2.0 * b)
    psi_pow = -psis * omega.pow(-b)                       # branche puissance (succion < 0)
    # Raccord C1 en omegpi : psi(omegpi)=psi_pow, dpsi/domega continu.
    psi_i = -psis * omegpi.pow(-b)
    dpsi_i = psis * b * omegpi.pow(-b - 1.0)              # d(psi_pow)/domega en omegpi
    # spline psi = a2*(omega-1)^2 + a1*(omega-1) telle que psi(1)=0,
    # psi(omegpi)=psi_i, psi'(omegpi)=dpsi_i. Résolu :
    d = omegpi - 1.0
    a1 = (2.0 * psi_i - dpsi_i * d) / d                   # depuis psi_i = a2 d^2 + a1 d et dpsi_i = 2 a2 d + a1
    a2 = (dpsi_i - a1) / (2.0 * d)
    om1 = omega - 1.0
    psi_spline = a2 * om1 * om1 + a1 * om1
    psi = torch.where(omega < omegpi, psi_pow, psi_spline)
    return psi.clamp(min=psi_floor, max=0.0)


class BV3CHydrotel(torch.nn.Module):
    """Bilan vertical 3 couches fidèle à Hydrotel BV3C2.

    forward(theta, apport, et_demand, frozen, swe_mm, params) ->
        (runoff_mm, interflow_mm, baseflow_mm, recharge_mm, theta_new, diag)

    Unités d'entrée : theta en m3/m3 (3 tenseurs), apport (pluie+fonte
    atteignant le sol) en mm/jour, et_demand (ETP) en mm/jour, frozen booléen
    (sol gelé), swe_mm équivalent neige en mm. params : dict de tenseurs par
    couche (ks, b, psis, thetas, thetacc, thetapf, z) + pente (rad implicite),
    krec, coef_recharge.
    """

    def __init__(self, n_substeps_max: int = 48):
        super().__init__()
        self.n_substeps_max = n_substeps_max

    def forward(self, theta1, theta2, theta3, apport_mm, et_mm, frozen, swe_mm, p):
        eps = 1e-9
        z1, z2, z3 = p["z1"], p["z2"], p["z3"]
        ks1, ks2, ks3 = p["ks1"], p["ks2"], p["ks3"]
        b1, b2, b3 = p["b1"], p["b2"], p["b3"]
        psis1, psis2, psis3 = p["psis1"], p["psis2"], p["psis3"]
        ths1, ths2, ths3 = p["thetas1"], p["thetas2"], p["thetas3"]
        slope = p["slope"]                                # pente (m/m)
        krec = p["krec"]
        coef_rech = p.get("coef_recharge", torch.zeros_like(theta1))

        # ── Ruissellement hortonien (CalculeRuisselement, bv3c2.cpp 2175) ──
        # prec = apport en m/h. ks1 en m/h. Portes gel et saturation.
        prec = (apport_mm / 1000.0) / DT_HOURS            # mm/j -> m/h
        omega1 = (theta1 / (ths1 + eps)).clamp(min=0.01, max=1.0)
        saturated = theta1 >= (ths1 - 1e-4)
        frost_gate = frozen & (swe_mm < 10.0)
        # pinf = min(prec, ks1), sauf gel/saturation -> 0.
        pinf = torch.minimum(prec, ks1)
        pinf = torch.where(saturated | frost_gate, torch.zeros_like(pinf), pinf)
        ruis_mh = torch.clamp(prec - pinf, min=0.0)        # m/h
        runoff_mm = ruis_mh * DT_HOURS * 1000.0            # -> mm/j
        infil_mm = pinf * DT_HOURS * 1000.0                # eau qui entre couche 1

        # ── Flux verticaux Darcy + interflow + baseflow (TriCoucheOct97) ──
        # Tout en m/h, intégré sur DT_HOURS avec sous-pas adaptatif (Courant
        # via VARIATION MAXIMALE). On reste simple : sous-pas fixe suffisant.
        nt = self.n_substeps_max
        dtc = DT_HOURS / nt                                 # h
        et_mh = (et_mm / 1000.0) / DT_HOURS                # demande ET m/h (couche 1)
        infil_mh = pinf                                     # m/h entrant couche 1

        t1, t2, t3 = theta1, theta2, theta3
        sin_slope = torch.sin(torch.atan(slope))
        q2_acc = torch.zeros_like(t1); q3_acc = torch.zeros_like(t1)

        def limit_inter(q, tU, zU, thsU, tL, zL, thsL):
            """Borne un flux entre couches (q>0 = vers le bas) à l'eau
            disponible : ne vide pas la couche source < 0 ni ne sature la
            cible > thetas. Garantit stabilité + conservation (idiome sf)."""
            down = torch.minimum(torch.minimum(q, tU * zU / dtc), (thsL - tL) * zL / dtc)
            up = torch.maximum(torch.maximum(q, -tL * zL / dtc), -(thsU - tU) * zU / dtc)
            return torch.where(q > 0, torch.clamp(down, min=0.0), torch.clamp(up, max=0.0))

        for _ in range(nt):
            o1 = (t1 / (ths1 + eps)).clamp(0.01, 1.0)
            o2 = (t2 / (ths2 + eps)).clamp(0.01, 1.0)
            o3 = (t3 / (ths3 + eps)).clamp(0.01, 1.0)
            k1 = campbell_K(o1, ks1, b1); k2 = campbell_K(o2, ks2, b2); k3 = campbell_K(o3, ks3, b3)
            ps1 = campbell_psi(o1, psis1, b1); ps2 = campbell_psi(o2, psis2, b2); ps3 = campbell_psi(o3, psis3, b3)
            # Darcy face = max des deux conductivités (gravité + matriciel).
            qq12 = torch.maximum(k1, k2) * (2.0 * (ps2 - ps1) / (z1 + z2) + 1.0)
            qq23 = torch.maximum(k2, k3) * (2.0 * (ps3 - ps2) / (z2 + z3) + 1.0)
            q2 = k2 * sin_slope * z2                        # interflow couche 2 (m/h)
            q3 = krec * z3 * t3                             # baseflow couche 3 (m/h)
            # Gel : étrangle qq12/qq23/q2, halve q3 (bv3c2.cpp 1888-1914).
            throttle = torch.where(frozen, torch.full_like(t1, 0.1), torch.ones_like(t1))
            qq12 = qq12 * throttle; qq23 = qq23 * throttle; q2 = q2 * throttle
            q3 = torch.where(frozen, q3 * 0.5, q3)
            # Limitation des flux à l'eau disponible (stabilité + masse).
            qq12 = limit_inter(qq12, t1, z1, ths1, t2, z2, ths2)
            qq23 = limit_inter(qq23, t2, z2, ths2, t3, z3, ths3)
            q2 = torch.clamp(torch.minimum(q2, t2 * z2 / dtc), min=0.0)   # sortie L2 <= dispo
            q3 = torch.clamp(torch.minimum(q3, t3 * z3 / dtc), min=0.0)   # sortie L3 <= dispo
            # Mise à jour theta (m), avec ET puisée couche 1 (limitée à dispo).
            et_eff = torch.clamp(torch.minimum(et_mh, t1 * z1 / dtc), min=0.0)
            t1 = t1 + dtc * (infil_mh - qq12 - et_eff) / z1
            t2 = t2 + dtc * (qq12 - qq23 - q2) / z2
            t3 = t3 + dtc * (qq23 - q3) / z3
            # Débordement résiduel (numérique) -> cascade conservative.
            ov1 = torch.clamp(t1 - ths1, min=0.0); t1 = t1 - ov1; t2 = t2 + ov1 * z1 / z2
            ov2 = torch.clamp(t2 - ths2, min=0.0); t2 = t2 - ov2; t3 = t3 + ov2 * z2 / z3
            ov3 = torch.clamp(t3 - ths3, min=0.0); t3 = t3 - ov3
            t1 = t1.clamp(min=0.0); t2 = t2.clamp(min=0.0); t3 = t3.clamp(min=0.0)
            q2_acc = q2_acc + q2 * dtc                      # m intégré
            q3_acc = q3_acc + q3 * dtc + ov3 * z3           # baseflow + débordement profond

        interflow_mm = q2_acc * 1000.0                     # m -> mm
        base_total_mm = q3_acc * 1000.0
        # Recharge : skim coef_recharge·(interflow+baseflow) vers l'aquifère.
        recharge_mm = coef_rech * (interflow_mm + base_total_mm)
        baseflow_mm = base_total_mm - recharge_mm

        diag = dict(pinf_mm=infil_mm, omega1=omega1, saturated=saturated, frost_gate=frost_gate)
        return runoff_mm, interflow_mm, baseflow_mm, recharge_mm, (t1, t2, t3), diag


def make_params(texture1="silt_loam", texture2="loam", texture3="loam",
                slope=0.04, krec=KREC_DEFAULT, coef_recharge=0.0, device="cpu"):
    """Construit le dict de paramètres calibrés Hydrotel pour un nœud scalaire
    (broadcastable). b = 1/lambda. Épaisseurs calibrées bv3c.csv."""
    def T(x): return torch.tensor(float(x), device=device)
    tx = [SOIL_TEXTURES[texture1], SOIL_TEXTURES[texture2], SOIL_TEXTURES[texture3]]
    z = EPAISSEUR
    p = {}
    for i, t in enumerate(tx, start=1):
        p[f"z{i}"] = T(z[i-1]); p[f"ks{i}"] = T(t["ks"]); p[f"b{i}"] = T(1.0 / t["lam"])
        p[f"psis{i}"] = T(t["psis"]); p[f"thetas{i}"] = T(t["thetas"])
        p[f"thetacc{i}"] = T(t["thetacc"]); p[f"thetapf{i}"] = T(t["thetapf"])
    p["slope"] = T(slope); p["krec"] = T(krec); p["coef_recharge"] = T(coef_recharge)
    return p
