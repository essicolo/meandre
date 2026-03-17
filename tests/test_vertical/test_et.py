"""Tests for ETModule (Penman-Monteith)."""

import torch
import pytest
from meandre.vertical.evapotranspiration import ETModule


def test_etp_positive():
    """ETP must always be non-negative."""
    et = ETModule()
    n = 10
    T_min = torch.randn(n) * 5 + 10
    T_max = T_min + torch.rand(n) * 10
    R_n = torch.rand(n) * 20
    u2 = torch.rand(n) * 5 + 0.5
    e_a = torch.rand(n) * 2 + 0.5
    ETP = et.penman_monteith(T_min, T_max, R_n, u2, e_a)
    assert (ETP >= 0).all()


def test_etp_increases_with_vpd():
    """Higher VPD (lower e_a) should give higher ETP."""
    et = ETModule()
    n = 5
    T_min = torch.full((n,), 10.0)
    T_max = torch.full((n,), 20.0)
    R_n = torch.full((n,), 15.0)
    u2 = torch.full((n,), 2.0)

    ETP_high_vpd = et.penman_monteith(T_min, T_max, R_n, u2, e_a=torch.full((n,), 0.5))
    ETP_low_vpd  = et.penman_monteith(T_min, T_max, R_n, u2, e_a=torch.full((n,), 1.8))
    assert (ETP_high_vpd > ETP_low_vpd).all()


def test_water_stress_bounds():
    et = ETModule()
    theta = torch.linspace(0.1, 0.5, 20)
    theta_wp = torch.full((20,), 0.15)
    theta_fc = torch.full((20,), 0.35)
    stress = et.water_stress(theta, theta_wp, theta_fc)
    assert (stress >= 0).all()
    assert (stress <= 1).all()


def test_forward_output_shapes():
    et = ETModule()
    n = 8
    T_min = torch.randn(n)
    T_max = T_min + 10
    R_n = torch.rand(n) * 20
    u2 = torch.rand(n) * 4 + 0.5
    e_a = torch.rand(n) + 0.5
    theta = torch.rand(n) * 0.2 + 0.15
    ones = torch.ones(n) * 0.3
    f = torch.full((n,), 1/3)
    E_canopy = torch.zeros(n)
    wp = torch.full((n,), 0.15)
    fc = torch.full((n,), 0.35)

    ET1, ET2, ET3, ETP = et(
        T_min, T_max, R_n, u2, e_a,
        theta, theta, theta,
        wp, wp, wp, fc, fc, fc,
        f, f, f, E_canopy,
    )
    for x in [ET1, ET2, ET3, ETP]:
        assert x.shape == (n,)
        assert (x >= 0).all()
