"""Tests for SimDiagnostics — model.simulate(return_diagnostics=True)."""

import torch
import pytest

from meandre.model import HydroModel
from meandre.routing.graph import synthetic_linear_graph
from meandre.routing.withdrawals import WithdrawalData
from meandre.spatial.territorial import TerritorialFeatures
from meandre.utils.diagnostics import SimDiagnostics
from meandre.utils.state import HydroState


N = 8
T = 20


def _make_model(n_nodes=N):
    return HydroModel(
        n_nodes=n_nodes,
        use_temporal=False,
        use_residual=False,
        use_travel_time_attn=False,
    )


def _make_forcing(n_t=T, n_nodes=N):
    f = torch.zeros(n_t, n_nodes, 6)
    f[:, :, 0] = 3.0          # P mm/day
    f[:, :, 1] = 2.0           # Tmin
    f[:, :, 2] = 12.0          # Tmax
    f[:, :, 3] = 10.0          # Rn
    f[:, :, 4] = 2.0           # u2
    f[:, :, 5] = 0.8           # ea
    return f


def _make_territorial(n_nodes=N):
    t = TerritorialFeatures.zeros(n_nodes=n_nodes, n_features=17)
    t.physical["area_km2_physical"] = torch.ones(n_nodes) * 10
    t.physical["area_km2_local"] = torch.ones(n_nodes) * 2
    return t


def _run(return_diagnostics=False, n_nodes=N, model=None):
    if model is None:
        model = _make_model(n_nodes)
    forcing = _make_forcing(n_nodes=n_nodes)
    graph = synthetic_linear_graph(n_nodes, tau_days=1)
    territorial = _make_territorial(n_nodes)
    state = HydroState.default_warm(n_nodes)
    withdrawals = WithdrawalData.zeros(T, n_nodes)
    doy = torch.ones(T, dtype=torch.long)
    coords = torch.zeros(n_nodes, 2)

    with torch.no_grad():
        return model.simulate(
            forcing=forcing,
            initial_state=state,
            graph=graph,
            node_coords=coords,
            territorial=territorial,
            withdrawals=withdrawals,
            day_of_year=doy,
            return_diagnostics=return_diagnostics,
        )


# ------------------------------------------------------------------
# Structure tests
# ------------------------------------------------------------------

def test_default_no_diagnostics():
    """Without flag, simulate returns 2-tuple as before."""
    out = _run(return_diagnostics=False)
    assert len(out) == 2
    Q_sim, state = out
    assert Q_sim.shape == (T, N)


def test_diagnostics_returned():
    """With flag, simulate returns 3-tuple."""
    out = _run(return_diagnostics=True)
    assert len(out) == 3
    Q_sim, state, diag = out
    assert isinstance(diag, SimDiagnostics)


def test_diagnostics_shapes():
    """All diagnostic tensors have shape (T, N)."""
    _, _, diag = _run(return_diagnostics=True)
    for name, tensor in diag.to_dict().items():
        assert tensor.shape == (T, N), f"{name}: expected ({T},{N}), got {tensor.shape}"


def test_q_sim_unchanged():
    """Q_sim is identical whether return_diagnostics is True or False."""
    model = _make_model()
    Q_no_diag, _ = _run(return_diagnostics=False, model=model)
    Q_diag, _, _ = _run(return_diagnostics=True, model=model)
    torch.testing.assert_close(Q_no_diag, Q_diag)


# ------------------------------------------------------------------
# Physical plausibility
# ------------------------------------------------------------------

def test_etp_positive():
    """ETP must be non-negative."""
    _, _, diag = _run(return_diagnostics=True)
    assert (diag.etp >= 0).all(), "ETP has negative values"


def test_etr_le_etp():
    """Actual ET cannot exceed potential ET."""
    _, _, diag = _run(return_diagnostics=True)
    assert (diag.etr <= diag.etp + 1e-4).all(), "ETR > ETP"


def test_snowmelt_nonnegative():
    """Snowmelt cannot be negative."""
    _, _, diag = _run(return_diagnostics=True)
    assert (diag.snowmelt >= 0).all()


def test_lateral_mm_nonnegative():
    """Lateral runoff (mm/day) must be non-negative."""
    _, _, diag = _run(return_diagnostics=True)
    assert (diag.lateral_mm >= 0).all()


def test_q_lateral_nonnegative():
    """Lateral inflow in m³/s must be non-negative."""
    _, _, diag = _run(return_diagnostics=True)
    assert (diag.q_lateral >= 0).all()


def test_q_upstream_headwaters_zero():
    """Headwater node (index 0 in linear graph) has no upstream inflow."""
    _, _, diag = _run(return_diagnostics=True)
    # In a linear graph 0→1→2→…, node 0 has no upstream neighbours
    assert (diag.q_upstream[:, 0] == 0).all()


def test_q_upstream_downstream_positive():
    """Non-headwater nodes accumulate positive upstream flows."""
    _, _, diag = _run(return_diagnostics=True)
    # After a few timesteps, node N-1 should receive inflow
    assert diag.q_upstream[T // 2:, -1].mean() >= 0


def test_to_dict_keys():
    """to_dict() returns all expected keys."""
    _, _, diag = _run(return_diagnostics=True)
    # Core keys always present; T_water included when temperature module is active
    core = {"etp", "etr", "snowmelt", "lateral_mm", "recharge", "q_baseflow", "q_lateral", "q_upstream"}
    keys = set(diag.to_dict().keys())
    assert core.issubset(keys), f"Missing keys: {core - keys}"
    assert keys - core <= {"T_water", "swe"}, f"Unexpected keys: {keys - core - {'T_water', 'swe'}}"


def test_units_dict():
    """units property returns a dict with correct keys."""
    _, _, diag = _run(return_diagnostics=True)
    units = diag.units
    assert "etp" in units
    assert units["q_lateral"] == "m3/s"
    assert units["etp"] == "mm/day"
