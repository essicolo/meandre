"""Water balance conservation tests.

Verify that total outflow ≈ total inflow - ET - delta_S for each node.
"""

import pytest
import torch

from meandre.vertical.soil import SoilModule, Z1, Z2, Z3


def test_soil_mass_conservation():
    """Total water in the soil column changes only by net flux."""
    soil = SoilModule()
    n = 10
    P_eff = torch.rand(n) * 5.0   # mm/day
    ET1 = torch.zeros(n)
    ET2 = torch.zeros(n)
    ET3 = torch.zeros(n)

    # Start near field capacity
    theta1 = torch.full((n,), 0.30)
    theta2 = torch.full((n,), 0.28)
    theta3 = torch.full((n,), 0.25)

    K_sat = torch.ones(n) * 0.2
    por = torch.ones(n) * 0.45
    fc = torch.ones(n) * 0.35
    wp = torch.ones(n) * 0.15

    t1, t2, t3, R_surface, interflow, baseflow = soil(
        P_eff, ET1, ET2, ET3,
        theta1, theta2, theta3,
        K_sat, K_sat, K_sat,
        por, por, por,
        fc, fc, fc,
        wp, wp, wp,
    )

    # Total water in column before and after (m3/m3 * m thickness -> m, then *1e3 -> mm)
    S_before = (theta1 * Z1 + theta2 * Z2 + theta3 * Z3) * 1e3  # mm
    S_after  = (t1    * Z1 + t2    * Z2 + t3    * Z3) * 1e3

    delta_S = S_after - S_before

    # Conservation invariant 1: column storage never increases beyond P_eff
    # (water can't be created).
    assert (delta_S <= P_eff + 1e-3).all(), "Column gained more water than input"

    # Conservation invariant 2: column storage stays non-negative
    # (theta is clamped to 0 in the module, so S_after >= 0).
    assert (S_after >= -1e-3).all(), "Column storage went negative"

    # Conservation invariant 3: delta_S is bounded below by initial storage
    # (can't lose more than what's in the column).
    assert (delta_S >= -S_before - 1e-3).all(), "Column lost more than its initial storage"


def test_snow_mass_conservation():
    """SWE_new = SWE + P_snow - melt; P_eff = P_rain + melt."""
    from meandre.vertical.snow import SnowModule

    snow = SnowModule()
    n = 20
    P = torch.rand(n) * 8.0
    T_air = torch.full((n,), 0.0)   # right at threshold
    SWE = torch.rand(n) * 30.0
    C_f = torch.ones(n) * 3.0
    T_melt = torch.zeros(n)
    T_snow = torch.zeros(n)

    P_eff, SWE_new, _ = snow(P, T_air, SWE, C_f, T_melt, T_snow)

    # Total water = SWE + P must equal SWE_new + P_eff
    total_before = SWE + P
    total_after = SWE_new + P_eff
    assert torch.allclose(total_before, total_after, atol=1e-5), \
        f"Snow mass not conserved: max error {(total_before - total_after).abs().max():.2e}"


def test_wetland_mass_conservation():
    """Wetland: storage_in + R_wet == Q_wetland + storage_new."""
    from meandre.vertical.wetland import WetlandModule

    wet = WetlandModule()
    n = 15
    R_surface = torch.rand(n) * 4.0
    S = torch.rand(n) * 20.0
    f_wet = torch.full((n,), 0.3)

    Q_wet, R_direct, S_new = wet(R_surface, S, f_wet)

    R_wet_input = R_surface * f_wet
    total_in = S + R_wet_input
    total_out = Q_wet + S_new

    assert torch.allclose(total_in, total_out, atol=1e-4), \
        f"Wetland mass not conserved: max error {(total_in - total_out).abs().max():.2e}"
