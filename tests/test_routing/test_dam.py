"""Tests for DamData and regulated reservoir routing."""

import torch
import pytest

from meandre.routing.dam import DamData
from meandre.routing.graph import RiverGraph
from meandre.routing.message_passing import RoutingLayer
from meandre.routing.withdrawals import WithdrawalData
from meandre.temporal.ring_buffer import OutflowRingBuffer


# ---------------------------------------------------------------------------
# DamData unit tests
# ---------------------------------------------------------------------------

def test_unregulated_all_nan():
    dam = DamData.unregulated(n_timesteps=10, n_nodes=5)
    assert dam.releases.shape == (10, 5)
    assert torch.isnan(dam.releases).all()


def test_release_at_unregulated_returns_none():
    dam = DamData.unregulated(n_timesteps=5, n_nodes=3)
    assert dam.release_at(0, 0) is None
    assert dam.release_at(4, 2) is None


def test_from_node_series_correct_shape():
    series = torch.linspace(1.0, 10.0, 10)
    dam = DamData.from_node_series({2: series}, n_timesteps=10, n_nodes=5)
    assert dam.releases.shape == (10, 5)
    # Node 2 should have the series; others should be nan
    assert torch.allclose(dam.releases[:, 2], series)
    assert torch.isnan(dam.releases[:, 0]).all()
    assert torch.isnan(dam.releases[:, 4]).all()


def test_from_node_series_wrong_length_raises():
    with pytest.raises(ValueError, match="length"):
        DamData.from_node_series({0: torch.zeros(5)}, n_timesteps=10, n_nodes=3)


def test_release_at_regulated_returns_tensor():
    series = torch.arange(10, dtype=torch.float32)
    dam = DamData.from_node_series({1: series}, n_timesteps=10, n_nodes=4)
    for t in range(10):
        val = dam.release_at(t, 1)
        assert val is not None
        assert val.shape == (1,)
        assert torch.allclose(val, series[t:t+1])


def test_release_at_mixed_nodes():
    dam = DamData.from_node_series(
        {0: torch.ones(5) * 100.0, 2: torch.ones(5) * 50.0},
        n_timesteps=5, n_nodes=4,
    )
    assert dam.release_at(0, 0) is not None   # regulated
    assert dam.release_at(0, 1) is None        # unregulated
    assert dam.release_at(0, 2) is not None   # regulated
    assert dam.release_at(0, 3) is None        # unregulated


def test_to_device():
    dam = DamData.from_node_series({0: torch.ones(4)}, n_timesteps=4, n_nodes=3)
    dam_cpu = dam.to(torch.device("cpu"))
    assert dam_cpu.releases.device.type == "cpu"


# ---------------------------------------------------------------------------
# RoutingLayer integration tests with DamData
# ---------------------------------------------------------------------------

def _make_lake_graph(n_nodes: int, lake_node: int) -> RiverGraph:
    src = torch.arange(n_nodes - 1, dtype=torch.long)
    dst = torch.arange(1, n_nodes, dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)
    edge_attr = torch.ones(n_nodes - 1, 3)
    topo_order = torch.arange(n_nodes, dtype=torch.long)
    is_lake = torch.zeros(n_nodes, dtype=torch.bool)
    is_lake[lake_node] = True
    travel_time_days = torch.ones(n_nodes - 1, dtype=torch.long)
    return RiverGraph(edge_index, edge_attr, topo_order, is_lake, travel_time_days)


def _routing_step(n_nodes, lake_node, dam_data):
    graph = _make_lake_graph(n_nodes, lake_node)
    layer = RoutingLayer(use_travel_time_attention=False)

    lateral_inflow = torch.ones(n_nodes) * 10.0
    Q_out_prev = torch.zeros(n_nodes)
    buffer = OutflowRingBuffer(n_nodes, depth=5)
    withdrawals = WithdrawalData.zeros(10, n_nodes)
    lake_storage = torch.ones(n_nodes) * 1e6  # 1 million m3 initial storage
    area_km2 = torch.ones(n_nodes) * 20.0

    K_musk = torch.ones(n_nodes) * 86400.0
    x_musk = torch.full((n_nodes,), 0.2)
    dx = torch.ones(n_nodes) * 5000.0

    Q_out, lake_storage_new, _ = layer(
        lateral_inflow, graph, Q_out_prev, buffer, withdrawals, t=0,
        K_musk=K_musk, x_musk=x_musk, dx=dx,
        lake_storage=lake_storage,
        area_km2=area_km2,
        dam_data=dam_data,
    )
    return Q_out, lake_storage_new


def test_forced_release_respected():
    """Q_out at the lake node must equal the forced release."""
    n_nodes, lake_node = 4, 2
    forced_q = 42.0
    dam = DamData.from_node_series(
        {lake_node: torch.full((10,), forced_q)},
        n_timesteps=10, n_nodes=n_nodes,
    )
    Q_out, _ = _routing_step(n_nodes, lake_node, dam)
    assert torch.isclose(Q_out[lake_node], torch.tensor(forced_q), atol=1e-4), (
        f"Expected Q_out[lake]={forced_q}, got {float(Q_out[lake_node])}"
    )


def test_unregulated_dam_data_same_as_none():
    """DamData.unregulated must give identical results to dam_data=None."""
    n_nodes, lake_node = 4, 2
    dam_unreg = DamData.unregulated(n_timesteps=10, n_nodes=n_nodes)

    Q_none, S_none = _routing_step(n_nodes, lake_node, dam_data=None)
    Q_unreg, S_unreg = _routing_step(n_nodes, lake_node, dam_data=dam_unreg)

    assert torch.allclose(Q_none, Q_unreg, atol=1e-6)
    assert torch.allclose(S_none, S_unreg, atol=1e-6)


def test_forced_release_mass_balance():
    """Storage update must follow dS = (Q_in - Q_forced) * dt."""
    n_nodes, lake_node = 3, 1
    forced_q = 10.0
    dam = DamData.from_node_series(
        {lake_node: torch.full((10,), forced_q)},
        n_timesteps=10, n_nodes=n_nodes,
    )
    graph = _make_lake_graph(n_nodes, lake_node)
    layer = RoutingLayer(use_travel_time_attention=False)

    S0 = 5e5
    lateral_inflow = torch.zeros(n_nodes)
    lateral_inflow[lake_node] = 0.0   # no lateral
    Q_out_prev = torch.zeros(n_nodes)
    buffer = OutflowRingBuffer(n_nodes, depth=5)
    withdrawals = WithdrawalData.zeros(10, n_nodes)
    lake_storage = torch.full((n_nodes,), S0)
    area_km2 = torch.ones(n_nodes) * 5.0

    K_musk = torch.ones(n_nodes) * 86400.0
    x_musk = torch.full((n_nodes,), 0.2)
    dx = torch.ones(n_nodes) * 1000.0

    Q_out, S_new, _ = layer(
        lateral_inflow, graph, Q_out_prev, buffer, withdrawals, t=0,
        K_musk=K_musk, x_musk=x_musk, dx=dx,
        lake_storage=lake_storage,
        area_km2=area_km2,
        dam_data=dam,
    )

    # Expected: S_new = S0 + (Q_in_node - forced_q) * 86400
    # Q_in_node at lake_node = upstream Q_out[0] + lateral; both ~0 at t=0
    Q_in_node = float(Q_out[lake_node - 1])  # upstream outflow
    expected_S = max(0.0, S0 + (Q_in_node - forced_q) * 86400.0)
    assert abs(float(S_new[lake_node]) - expected_S) < 1.0, (
        f"Mass balance error: got {float(S_new[lake_node]):.1f}, "
        f"expected {expected_S:.1f}"
    )


def test_non_lake_nodes_unaffected_by_dam_data():
    """DamData must not affect routing at non-lake (river) nodes."""
    n_nodes, lake_node = 4, 2
    # Regulated lake
    dam = DamData.from_node_series(
        {lake_node: torch.full((10,), 5.0)},
        n_timesteps=10, n_nodes=n_nodes,
    )
    Q_dam, _ = _routing_step(n_nodes, lake_node, dam_data=dam)
    Q_none, _ = _routing_step(n_nodes, lake_node, dam_data=None)

    # River nodes (not the lake) should be identical
    for node in range(n_nodes):
        if node != lake_node:
            # Nodes downstream of the lake will differ because Q_out at lake
            # differs; only check the upstream river node (node < lake_node)
            if node < lake_node:
                assert torch.isclose(Q_dam[node], Q_none[node], atol=1e-5), (
                    f"Upstream river node {node} changed unexpectedly"
                )
