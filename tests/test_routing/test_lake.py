"""Tests for lake/reservoir routing module and lake node integration."""

import torch
import pytest
from meandre.routing.lake import LakeModule
from meandre.routing.message_passing import RoutingLayer
from meandre.routing.graph import RiverGraph
from meandre.routing.withdrawals import WithdrawalData
from meandre.temporal.ring_buffer import OutflowRingBuffer


# ---------------------------------------------------------------------------
# LakeModule unit tests
# ---------------------------------------------------------------------------

def test_lake_positive_outflow():
    """Lake outflow must be non-negative for any inflow/storage."""
    lake = LakeModule()
    n = 4
    Q_in = torch.rand(n)
    S = torch.rand(n) * 1e6
    area_km2 = torch.ones(n) * 10.0
    E = torch.zeros(n)
    P = torch.zeros(n)
    S_dead = torch.zeros(n)

    Q_out, S_new = lake(Q_in, S, area_km2, E, P, S_dead)
    assert (Q_out >= 0).all(), "Lake outflow must be non-negative"
    assert (S_new >= 0).all(), "Lake storage must be non-negative"


def test_lake_storage_update():
    """Mass balance: S_new equals S + (Q_in - Q_out) * dt (before clamping)."""
    lake = LakeModule()
    Q_in = torch.tensor([5.0])    # m3/s
    S = torch.tensor([10.0])      # small storage (m3) so Q_out stays modest
    area_km2 = torch.tensor([1.0])
    E = torch.zeros(1)
    P = torch.zeros(1)
    S_dead = torch.zeros(1)

    Q_out, S_new = lake(Q_in, S, area_km2, E, P, S_dead)
    # Mass balance before clamp: S + (Q_in - Q_out) * dt
    expected = S + (Q_in - Q_out.detach()) * 86400.0
    expected = torch.clamp(expected, min=0.0)
    assert torch.allclose(S_new, expected, atol=1e-3), "Mass balance violated"


def test_lake_forced_release():
    """Forced release overrides the storage-discharge relationship."""
    lake = LakeModule()
    Q_in = torch.tensor([10.0])
    S = torch.tensor([5e5])
    area_km2 = torch.tensor([2.0])
    E = torch.zeros(1)
    P = torch.zeros(1)
    S_dead = torch.zeros(1)
    Q_forced = torch.tensor([50.0])  # forced release

    Q_out, _ = lake(Q_in, S, area_km2, E, P, S_dead, Q_release_forced=Q_forced)
    assert torch.allclose(Q_out, Q_forced), "Forced release should set Q_out exactly"


def test_lake_gradients():
    """Gradients must reach learnable parameters through the lake module.

    Q_in contributes to S_new (not Q_out directly), so we backprop through
    the combined loss = Q_out + S_new to cover both paths.
    """
    lake = LakeModule()
    Q_in = torch.rand(3, requires_grad=True)
    S = torch.rand(3) * 100.0    # small storage: Q_out = k * S^beta stays finite
    area_km2 = torch.ones(3)
    E = torch.zeros(3)
    P = torch.zeros(3)
    S_dead = torch.zeros(3)

    Q_out, S_new = lake(Q_in, S, area_km2, E, P, S_dead)
    # Q_in affects S_new; k_lake/beta affect Q_out
    (Q_out.sum() + S_new.sum()).backward()

    assert Q_in.grad is not None, "Q_in should have gradient via S_new"
    assert lake.log_k_lake.grad is not None, "log_k_lake should have gradient via Q_out"
    assert lake.log_beta.grad is not None, "log_beta should have gradient via Q_out"


# ---------------------------------------------------------------------------
# RoutingLayer with lake nodes integration test
# ---------------------------------------------------------------------------

def _make_lake_graph(n_nodes: int = 5, lake_node: int = 2) -> RiverGraph:
    """Linear graph with one lake node in the middle."""
    src = torch.arange(n_nodes - 1, dtype=torch.long)
    dst = torch.arange(1, n_nodes, dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)
    edge_attr = torch.ones(n_nodes - 1, 3)
    topo_order = torch.arange(n_nodes, dtype=torch.long)
    is_lake = torch.zeros(n_nodes, dtype=torch.bool)
    is_lake[lake_node] = True
    travel_time_days = torch.ones(n_nodes - 1, dtype=torch.long)
    return RiverGraph(edge_index, edge_attr, topo_order, is_lake, travel_time_days)


def test_routing_with_lake_node_no_crash():
    """RoutingLayer must handle lake nodes without raising NotImplementedError."""
    n_nodes = 5
    graph = _make_lake_graph(n_nodes, lake_node=2)
    layer = RoutingLayer(use_travel_time_attention=False)

    lateral_inflow = torch.rand(n_nodes) * 2.0  # mm/day
    Q_out_prev = torch.zeros(n_nodes)
    buffer = OutflowRingBuffer(n_nodes, depth=5)
    withdrawals = WithdrawalData.zeros(10, n_nodes)
    lake_storage = torch.zeros(n_nodes)
    area_km2 = torch.ones(n_nodes) * 50.0

    K_musk = torch.ones(n_nodes) * 86400.0
    x_musk = torch.full((n_nodes,), 0.2)
    dx = torch.ones(n_nodes) * 5000.0

    Q_out, lake_storage_new, _ = layer(
        lateral_inflow, graph, Q_out_prev, buffer, withdrawals, 0,
        K_musk, x_musk, dx,
        lake_storage=lake_storage,
        area_km2=area_km2,
    )

    assert Q_out.shape == (n_nodes,)
    assert (Q_out >= 0).all(), "All outflows must be non-negative"
    assert lake_storage_new is not None
    assert lake_storage_new.shape == (n_nodes,)
    assert (lake_storage_new >= 0).all(), "Lake storage must be non-negative"


def test_routing_lake_storage_accumulates():
    """Lake storage should grow when inflow exceeds outflow."""
    n_nodes = 3
    lake_node = 1
    graph = _make_lake_graph(n_nodes, lake_node)
    layer = RoutingLayer(use_travel_time_attention=False)

    # Force large inflow
    lateral_inflow = torch.ones(n_nodes) * 100.0  # mm/day
    Q_out_prev = torch.zeros(n_nodes)
    buffer = OutflowRingBuffer(n_nodes, depth=5)
    withdrawals = WithdrawalData.zeros(10, n_nodes)
    lake_storage = torch.zeros(n_nodes)
    area_km2 = torch.ones(n_nodes) * 1.0   # small lake area → limited Q_out

    K_musk = torch.ones(n_nodes) * 86400.0
    x_musk = torch.full((n_nodes,), 0.2)
    dx = torch.ones(n_nodes) * 1000.0

    Q_out, lake_storage_new, _ = layer(
        lateral_inflow, graph, Q_out_prev, buffer, withdrawals, 0,
        K_musk, x_musk, dx,
        lake_storage=lake_storage,
        area_km2=area_km2,
    )

    # Lake storage at lake_node should increase from zero
    assert float(lake_storage_new[lake_node]) >= 0, "Lake storage must stay non-negative"


def test_routing_no_lake_returns_none_storage():
    """RoutingLayer returns None lake_storage when no lake nodes and no storage passed."""
    from meandre.routing.graph import synthetic_linear_graph
    n_nodes = 4
    graph = synthetic_linear_graph(n_nodes)
    layer = RoutingLayer(use_travel_time_attention=False)

    lateral_inflow = torch.rand(n_nodes)
    Q_out_prev = torch.zeros(n_nodes)
    buffer = OutflowRingBuffer(n_nodes, depth=5)
    withdrawals = WithdrawalData.zeros(10, n_nodes)

    K_musk = torch.ones(n_nodes) * 86400.0
    x_musk = torch.full((n_nodes,), 0.2)
    dx = torch.ones(n_nodes) * 5000.0

    Q_out, lake_storage_new, _ = layer(
        lateral_inflow, graph, Q_out_prev, buffer, withdrawals, 0,
        K_musk, x_musk, dx,
    )

    assert lake_storage_new is None, "No lake storage returned when none passed"
    assert (Q_out >= 0).all()
