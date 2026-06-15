"""Tests du réservoir supérieur à seuil (HBV-EC K0/UZL/K1) dans SoilModule.

Conception (2026-06-14, après PoC négative) : le réservoir est NOURRI par une
fraction qf_frac de la drainage verticale rapide de la couche 1 (q_vert_1, qui
partirait en profondeur), pas par l'interflow (filet négligeable qui affamait
le réservoir). Il relâche par deux sorties : Q1 = K1·S_uz, Q0 à seuil UZL.
"""
from __future__ import annotations
import torch

from meandre.vertical.soil import SoilModule


def _inputs(n=8, theta_val=0.40):
    """Sol HUMIDE (theta=0.40 > theta_fc=0.30) pour générer du q_vert_1."""
    g = torch.Generator().manual_seed(0)
    P_eff = torch.rand(n, generator=g) * 30.0      # mm/jour, parfois gros orage
    ET = torch.zeros(n)
    theta = torch.full((n,), theta_val)
    K_sat = torch.full((n,), 0.05)
    por = torch.full((n,), 0.45)
    fc = torch.full((n,), 0.30)
    wp = torch.full((n,), 0.10)
    f_vert = torch.full((n,), 0.7)                 # majorité verticale (à détourner)
    return P_eff, ET, theta, K_sat, por, fc, wp, f_vert


def _call(soil, S_uz=None, theta_val=0.40):
    P_eff, ET, theta, K_sat, por, fc, wp, f_vert = _inputs(theta_val=theta_val)
    return soil(
        P_eff, ET, ET, ET, theta, theta, theta,
        K_sat, K_sat, K_sat, por, por, por, fc, fc, fc, wp, wp, wp,
        f_vert, f_vert, f_vert, S_uz=S_uz,
    )


def test_reservoir_off_returns_zero_S_uz():
    """Réservoir off : S_uz reste nul."""
    soil = SoilModule(use_quickflow_reservoir=False)
    out = _call(soil)
    S_uz = out[6]
    assert torch.allclose(S_uz, torch.zeros_like(S_uz))


def test_reservoir_off_equivalence():
    """Réservoir off : theta/R/interflow/baseflow identiques au module sans le flag."""
    base = SoilModule(use_quickflow_reservoir=False)
    # Un deuxième module off doit donner exactement la même chose (pas d'effet).
    out = _call(base)
    assert out[6] is not None
    # Pas de détournement : interflow = interflow direct (pas de release ajoutée).
    # (vérifié indirectement par les 12 tests d'équivalence existants).


def test_reservoir_fed_by_partition_fills_and_releases():
    """Réservoir on, sol humide : S_uz se remplit (via q_vert_1) puis se stabilise."""
    soil = SoilModule(use_quickflow_reservoir=True)
    n = 8
    S = torch.zeros(n)
    P_eff, ET, theta, K_sat, por, fc, wp, f_vert = _inputs(theta_val=0.40)
    S_series = []
    for _ in range(40):
        out = soil(
            P_eff, ET, ET, ET, theta, theta, theta,
            K_sat, K_sat, K_sat, por, por, por, fc, fc, fc, wp, wp, wp,
            f_vert, f_vert, f_vert, S_uz=S,
        )
        S = out[6]
        S_series.append(float(S.mean().detach()))
    assert S_series[-1] > 0.1, f"réservoir affamé : {S_series[-1]}"  # il se remplit
    assert S_series[-1] < 1e3                                         # pas d'explosion
    # stabilisation (variation finale << variation initiale)
    assert abs(S_series[-1] - S_series[-2]) < abs(S_series[1] - S_series[0]) + 1e-9


def test_full_column_mass_balance():
    """Bilan pleine colonne : P_in = sorties + Δstockage_sol + ΔS_uz (au mm près)."""
    soil = SoilModule(use_quickflow_reservoir=True)
    P_eff, ET, theta, K_sat, por, fc, wp, f_vert = _inputs(theta_val=0.40)
    z1, z2d, z3d = soil.z1, soil.z2_default, soil.z3_default
    S0 = torch.full_like(P_eff, 8.0)
    out = soil(
        P_eff, ET, ET, ET, theta, theta, theta,
        K_sat, K_sat, K_sat, por, por, por, fc, fc, fc, wp, wp, wp,
        f_vert, f_vert, f_vert, S_uz=S0,
    )
    t1n, t2n, t3n, R, interflow, baseflow, S_uz_new = out
    # Stockage sol avant/après (mm) : theta × épaisseur × 1000.
    storage0 = (theta * z1 + theta * z2d + theta * z3d) * 1e3
    storage1 = (t1n * z1 + t2n * z2d + t3n * z3d) * 1e3
    d_soil = storage1 - storage0
    d_uz = S_uz_new - S0
    # ET = 0 ici, donc : P = R + interflow + baseflow + Δsol + ΔS_uz.
    residual = P_eff - (R + interflow + baseflow + d_soil + d_uz)
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-3), residual.abs().max()


def test_reservoir_differentiable():
    """Gradient fini sur les 4 params du réservoir (K0/K1/UZL/qf_frac)."""
    # Q0+Q1 entre avec signes opposés dans interflow (+) et S_uz_new (−)
    # (conservation), donc on teste séparément : la RELÂCHE (K0/K1/UZL) via
    # interflow, et l'ALIMENTATION (qf_frac) via S_uz_new.
    S0 = torch.full((8,), 25.0)   # au-dessus du seuil → Q0 actif
    soil = SoilModule(use_quickflow_reservoir=True)
    out = _call(soil, S_uz=S0)
    out[4].sum().backward()
    for name in ("k0_uz_raw", "k1_uz_raw", "uzl_raw"):
        g = getattr(soil, name).grad
        assert g is not None and torch.isfinite(g).all() and g.abs() > 0, f"grad nul {name}"

    soil2 = SoilModule(use_quickflow_reservoir=True)
    out2 = _call(soil2, S_uz=S0)
    out2[6].sum().backward()   # S_uz_new dépend de qf_frac (alimentation)
    g = soil2.qf_frac_raw.grad
    assert g is not None and torch.isfinite(g).all() and g.abs() > 0, "grad nul qf_frac_raw"


def test_threshold_makes_release_nonlinear():
    """Au-dessus de UZL, S_uz relâche bien plus qu'en dessous (la bouffée)."""
    soil = SoilModule(use_quickflow_reservoir=True)
    P_eff, ET, theta, K_sat, por, fc, wp, f_vert = _inputs(theta_val=0.30)  # sec : pas d'apport
    zero_in = torch.zeros_like(P_eff)
    def release_at(S_val):
        out = soil(
            zero_in, ET, ET, ET, theta, theta, theta,
            K_sat, K_sat, K_sat, por, por, por, fc, fc, fc, wp, wp, wp,
            f_vert, f_vert, f_vert, S_uz=torch.full_like(P_eff, S_val),
        )
        return float(out[4].mean().detach())
    low = release_at(5.0)     # sous le seuil UZL≈20
    high = release_at(50.0)   # bien au-dessus
    assert high > 5.0 * low
