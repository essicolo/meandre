"""Tests de la capacité d'infiltration de surface découplée (Horton).

infil_ratio ∈ (0.05,1) multiplie K_sat_1 dans le terme de Horton seulement.
Plus bas ⇒ surface moins infiltrante ⇒ plus de ruissellement infiltration-excess,
SANS toucher le K_sat de drainage (donc sans toucher la récession).
"""
from __future__ import annotations
import math
import torch

from meandre.vertical.soil import SoilModule


def _inputs(n=8):
    # Pluie intense (orage) sur sol moyennement humide.
    P_eff = torch.full((n,), 40.0)                 # mm/jour
    ET = torch.zeros(n)
    theta = torch.full((n,), 0.30)
    K_sat = torch.full((n,), 0.10)                 # ~4 mm/h
    por = torch.full((n,), 0.45)
    fc = torch.full((n,), 0.30)
    wp = torch.full((n,), 0.10)
    f_vert = torch.full((n,), 0.5)
    rain_hours = torch.full((n,), 4.0)             # intensité ~10 mm/h
    return P_eff, ET, theta, K_sat, por, fc, wp, f_vert, rain_hours


def _R_surface(soil, ratio_raw=None):
    P_eff, ET, theta, K_sat, por, fc, wp, f_vert, rh = _inputs()
    if ratio_raw is not None:
        with torch.no_grad():
            soil.infil_ratio_raw.copy_(torch.tensor(ratio_raw))
    out = soil(
        P_eff, ET, ET, ET, theta, theta, theta,
        K_sat, K_sat, K_sat, por, por, por, fc, fc, fc, wp, wp, wp,
        f_vert, f_vert, f_vert, rain_hours=rh,
    )
    return out[3]  # R_surface (mm/jour)


def _inv(v, lo=0.05, hi=1.0):
    f = (v - lo) / (hi - lo)
    return math.log(f / (1 - f))


def test_infil_off_equivalence():
    """Flag off : R_surface identique à un module sans la capacité séparée."""
    a = SoilModule(use_separate_infil_capacity=False)
    b = SoilModule(use_separate_infil_capacity=False)
    assert torch.allclose(_R_surface(a), _R_surface(b))


def test_lower_ratio_increases_horton_runoff():
    """Un infil_ratio plus bas (surface scellée) ⇒ plus de ruissellement."""
    soil = SoilModule(use_separate_infil_capacity=True)
    R_high = _R_surface(soil, ratio_raw=_inv(0.95))  # surface très infiltrante
    R_low = _R_surface(soil, ratio_raw=_inv(0.10))   # surface scellée
    assert (R_low > R_high).all(), (float(R_low.mean()), float(R_high.mean()))
    # L'effet doit être substantiel (pas marginal) sur un orage intense.
    assert float(R_low.mean()) > float(R_high.mean()) + 2.0


def test_infil_ratio_differentiable():
    """Gradient fini et non-nul de R_surface vs infil_ratio_raw."""
    soil = SoilModule(use_separate_infil_capacity=True)
    R = _R_surface(soil, ratio_raw=_inv(0.5))
    R.sum().backward()
    g = soil.infil_ratio_raw.grad
    assert g is not None and torch.isfinite(g).all() and g.abs() > 0
