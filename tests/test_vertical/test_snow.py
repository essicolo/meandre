"""Tests for SnowModule."""

import torch
import pytest
from meandre.vertical.snow import SnowModule


def test_cold_day_accumulates_snow():
    """All precip should become snow on a cold day."""
    snow = SnowModule()
    n = 5
    P = torch.ones(n) * 5.0
    T_air = torch.full((n,), -10.0)  # well below T_snow
    SWE = torch.zeros(n)
    C_f = torch.ones(n) * 3.0
    T_melt = torch.zeros(n)
    T_snow = torch.zeros(n)

    P_eff, SWE_new, _ = snow(P, T_air, SWE, C_f, T_melt, T_snow)

    assert (SWE_new > 4.9).all(), "SWE should increase on cold day"
    assert (P_eff < 0.1).all(), "Effective precip should be ~0 on cold day"


def test_warm_day_melts_snow():
    """A warm day should melt existing SWE."""
    snow = SnowModule()
    n = 5
    P = torch.zeros(n)
    T_air = torch.full((n,), 10.0)
    SWE = torch.full((n,), 20.0)
    C_f = torch.ones(n) * 3.0
    T_melt = torch.zeros(n)
    T_snow = torch.zeros(n)

    P_eff, SWE_new, _ = snow(P, T_air, SWE, C_f, T_melt, T_snow)

    assert (SWE_new < SWE).all(), "SWE should decrease on warm day"
    assert (P_eff > 0).all(), "Meltwater should become effective precip"


def test_mass_conservation():
    snow = SnowModule()
    n = 10
    P = torch.rand(n) * 8.0
    T_air = torch.randn(n)
    SWE = torch.rand(n) * 30.0
    C_f = torch.ones(n) * 3.0
    T_melt = torch.zeros(n)
    T_snow = torch.zeros(n)

    P_eff, SWE_new, _ = snow(P, T_air, SWE, C_f, T_melt, T_snow)
    assert torch.allclose(P + SWE, P_eff + SWE_new, atol=1e-4)
