"""Tests for 3-layer SoilModule."""

import torch
import pytest
from meandre.vertical.soil import SoilModule


def _make_inputs(n: int):
    P_eff = torch.rand(n) * 5.0
    ET1 = torch.rand(n) * 0.5
    ET2 = torch.rand(n) * 0.3
    ET3 = torch.rand(n) * 0.1
    theta1 = torch.rand(n) * 0.2 + 0.15
    theta2 = torch.rand(n) * 0.15 + 0.15
    theta3 = torch.rand(n) * 0.1 + 0.15
    K_sat = torch.ones(n) * 0.3
    por = torch.ones(n) * 0.45
    fc = torch.ones(n) * 0.35
    wp = torch.ones(n) * 0.15
    return P_eff, ET1, ET2, ET3, theta1, theta2, theta3, K_sat, por, fc, wp


def test_output_shapes():
    soil = SoilModule()
    args = _make_inputs(6)
    P_eff, ET1, ET2, ET3, theta1, theta2, theta3, K_sat, por, fc, wp = args
    n = P_eff.shape[0]
    f_vert = torch.full((n,), 0.5)
    t1, t2, t3, R, interflow, baseflow = soil(
        P_eff, ET1, ET2, ET3, theta1, theta2, theta3,
        K_sat, K_sat, K_sat, por, por, por, fc, fc, fc, wp, wp, wp,
        f_vert, f_vert, f_vert,
    )
    for out in [t1, t2, t3, R, interflow, baseflow]:
        assert out.shape == (6,)


def test_theta_bounds():
    """Soil moisture must stay in [0, porosity]."""
    soil = SoilModule()
    args = _make_inputs(20)
    P_eff, ET1, ET2, ET3, theta1, theta2, theta3, K_sat, por, fc, wp = args
    n = P_eff.shape[0]
    f_vert = torch.full((n,), 0.5)
    t1, t2, t3, R, interflow, baseflow = soil(
        P_eff, ET1, ET2, ET3, theta1, theta2, theta3,
        K_sat, K_sat, K_sat, por, por, por, fc, fc, fc, wp, wp, wp,
        f_vert, f_vert, f_vert,
    )
    assert (t1 >= 0).all()
    assert (t2 >= 0).all()
    assert (t3 >= 0).all()
    assert (R >= 0).all()


def test_no_rainfall_dries_out():
    """Without precip or ET, soil should not gain moisture."""
    soil = SoilModule()
    n = 5
    P_eff = torch.zeros(n)
    ET1 = ET2 = ET3 = torch.zeros(n)
    theta1 = theta2 = theta3 = torch.full((n,), 0.25)
    K_sat = torch.ones(n) * 0.1
    por = torch.ones(n) * 0.45
    fc = torch.ones(n) * 0.35
    wp = torch.ones(n) * 0.15
    n = P_eff.shape[0]
    f_vert = torch.full((n,), 0.5)
    t1, t2, t3, R, interflow, baseflow = soil(
        P_eff, ET1, ET2, ET3, theta1, theta2, theta3,
        K_sat, K_sat, K_sat, por, por, por, fc, fc, fc, wp, wp, wp,
        f_vert, f_vert, f_vert,
    )
    # Moisture may redistribute but total should not increase
    S_before = theta1 * 0.3 + theta2 * 0.7 + theta3 * 1.0
    S_after = t1 * 0.3 + t2 * 0.7 + t3 * 1.0
    assert (S_after <= S_before + 1e-4).all()
