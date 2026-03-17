"""Tests for BasinCache (DuckDB-backed basin cache)."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from meandre.data.basin_cache import BasinCache
from meandre.routing.graph import synthetic_linear_graph
from meandre.spatial.territorial import TerritorialFeatures
from meandre.utils.state import HydroState


# ---------------------------------------------------------------------------
# Helpers: build a minimal hydro dict without touching the filesystem
# ---------------------------------------------------------------------------

def _make_hydro(n: int = 8) -> dict:
    graph = synthetic_linear_graph(n, tau_days=1)
    # Mark the middle node as a lake (safe for any n >= 2)
    graph.is_lake[n // 2] = True

    t = TerritorialFeatures.zeros(n_nodes=n, n_features=17)
    t.physical["drainage_area_km2"] = torch.arange(n, dtype=torch.float32) + 1.0
    t.physical["mean_slope_pct"] = torch.full((n,), 2.5)
    t.physical["area_km2_physical"] = torch.ones(n) * 10.0
    t.physical["area_km2_local"] = torch.ones(n) * 2.0

    coords = torch.stack([
        torch.linspace(-74, -72, n),
        torch.linspace(45, 47, n),
    ], dim=1)

    state = HydroState(
        theta1=torch.full((n,), 0.3),
        theta2=torch.full((n,), 0.25),
        theta3=torch.full((n,), 0.2),
        swe=torch.zeros(n),
        t_soil=torch.full((n,), 5.0),
        canopy_storage=torch.zeros(n),
        wetland_storage=torch.zeros(n),
    )

    return {
        "graph": graph,
        "territorial": t,
        "node_coords": coords,
        "initial_state": state,
        "node_ids": list(range(100, 100 + n)),
        "n_nodes": n,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_roundtrip_static(tmp_path):
    """Write hydro dict → DuckDB → load → same values."""
    hydro = _make_hydro(8)
    db_path = tmp_path / "test.duckdb"

    cache = BasinCache(db_path)
    cache.write(hydro, source="/fake/dir")

    loaded = cache.load()

    assert loaded["n_nodes"] == 8
    assert loaded["node_ids"] == list(range(100, 108))

    # Graph topology
    g_orig = hydro["graph"]
    g_load = loaded["graph"]
    assert g_load.n_nodes == g_orig.n_nodes
    assert g_load.n_edges == g_orig.n_edges
    assert torch.equal(g_load.edge_index, g_orig.edge_index)
    assert bool(g_load.is_lake[4]) is True   # n=8, middle=4
    assert bool(g_load.is_lake[0]) is False

    # Topo order preserved
    assert g_load.topo_order.shape == g_orig.topo_order.shape

    # Territorial
    t_load = loaded["territorial"]
    torch.testing.assert_close(
        t_load.drainage_area_km2,
        hydro["territorial"].drainage_area_km2,
    )
    torch.testing.assert_close(
        t_load.area_km2_physical,
        hydro["territorial"].area_km2_physical,
    )
    torch.testing.assert_close(
        t_load.area_km2_local,
        hydro["territorial"].area_km2_local,
    )

    # Coordinates
    torch.testing.assert_close(loaded["node_coords"], hydro["node_coords"])

    # Initial state
    s = loaded["initial_state"]
    torch.testing.assert_close(s.theta1, hydro["initial_state"].theta1)
    torch.testing.assert_close(s.swe, hydro["initial_state"].swe)


def test_save_load_state(tmp_path):
    """save_state → load_state round-trip."""
    hydro = _make_hydro(6)
    db_path = tmp_path / "warm.duckdb"
    cache = BasinCache(db_path)
    cache.write(hydro, source="/fake")

    n = 6
    state = HydroState(
        theta1=torch.rand(n),
        theta2=torch.rand(n),
        theta3=torch.rand(n),
        swe=torch.rand(n),
        t_soil=torch.randn(n),
        canopy_storage=torch.rand(n),
        wetland_storage=torch.rand(n),
    )
    lake_storage = torch.rand(n) * 1e6
    q_out_prev = torch.rand(n) * 500.0
    h_context = torch.randn(1, n, 16)

    cache.save_state("2005-12-31", state, lake_storage, q_out_prev, h_context)

    ws = cache.load_state("2005-12-31")

    torch.testing.assert_close(ws["state"].theta1, state.theta1)
    torch.testing.assert_close(ws["state"].wetland_storage, state.wetland_storage)
    torch.testing.assert_close(ws["lake_storage"], lake_storage)
    torch.testing.assert_close(ws["q_out_prev"], q_out_prev)
    assert ws["h_context"] is not None
    torch.testing.assert_close(ws["h_context"], h_context)


def test_save_state_no_h_context(tmp_path):
    """save_state without h_context → load_state returns None."""
    hydro = _make_hydro(4)
    cache = BasinCache(tmp_path / "s.duckdb")
    cache.write(hydro, source="/fake")

    state = HydroState.zeros(4)
    cache.save_state("2001-01-01", state)

    ws = cache.load_state("2001-01-01")
    assert ws["h_context"] is None


def test_list_states(tmp_path):
    """list_states returns all saved dates in sorted order."""
    hydro = _make_hydro(4)
    cache = BasinCache(tmp_path / "ls.duckdb")
    cache.write(hydro, source="/fake")

    state = HydroState.zeros(4)
    cache.save_state("2003-06-30", state)
    cache.save_state("2001-12-31", state)
    cache.save_state("2005-09-15", state)

    dates = cache.list_states()
    assert dates == ["2001-12-31", "2003-06-30", "2005-09-15"]


def test_overwrite_state(tmp_path):
    """Saving twice for the same date replaces the entry."""
    hydro = _make_hydro(4)
    cache = BasinCache(tmp_path / "ow.duckdb")
    cache.write(hydro, source="/fake")

    state_a = HydroState.zeros(4)
    state_b = HydroState.default_warm(4)

    cache.save_state("2002-12-31", state_a)
    cache.save_state("2002-12-31", state_b)  # should replace

    ws = cache.load_state("2002-12-31")
    torch.testing.assert_close(ws["state"].theta1, state_b.theta1)
    assert len(cache.list_states()) == 1


def test_missing_state_raises(tmp_path):
    """load_state for a missing date raises KeyError."""
    hydro = _make_hydro(4)
    cache = BasinCache(tmp_path / "miss.duckdb")
    cache.write(hydro, source="/fake")

    with pytest.raises(KeyError, match="2099-01-01"):
        cache.load_state("2099-01-01")


def test_from_hydrotel_not_real_dir(tmp_path):
    """from_hydrotel raises on a non-existent directory."""
    with pytest.raises(Exception):
        BasinCache.from_hydrotel(
            project_dir=tmp_path / "nonexistent",
            path=tmp_path / "out.duckdb",
        )


def test_file_created(tmp_path):
    """DuckDB file is created on disk after _write."""
    hydro = _make_hydro(4)
    db_path = tmp_path / "sub" / "basin.duckdb"
    cache = BasinCache(db_path)
    cache.write(hydro, source="/fake")
    assert db_path.exists()
