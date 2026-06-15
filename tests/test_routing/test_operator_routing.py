"""Validation du routage par opérateur contre le balayage par niveau.

Le mode "operator" doit reproduire le balayage à la précision flottante
(mêmes équations, ordre de calcul différent), gradients compris. Le mode
"operator-lagged" est une approximation contrôlée (lacs sur stockage de la
veille) : on vérifie la stabilité, la positivité et la proximité.
"""
import torch
import pytest

from meandre.routing.graph import RiverGraph, _topological_sort
from meandre.routing.message_passing import RoutingLayer
from meandre.routing.withdrawals import WithdrawalData
from meandre.temporal.ring_buffer import OutflowRingBuffer

N = 10
T = 30


def _branched_graph_with_lakes() -> RiverGraph:
    """0,1 -> 2 -> 3(lac) -> 4 -> 6 <- 5 ; 6 -> 7(lac) -> 8 -> 9 (3 étages)."""
    src = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.long)
    dst = torch.tensor([2, 2, 3, 4, 6, 6, 7, 8, 9], dtype=torch.long)
    edge_index = torch.stack([src, dst])
    edge_attr = torch.ones(9, 3)
    is_lake = torch.zeros(N, dtype=torch.bool)
    is_lake[3] = True
    is_lake[7] = True
    travel = torch.ones(9, dtype=torch.long)
    topo_order = _topological_sort(edge_index, N)
    return RiverGraph(edge_index, edge_attr, topo_order, is_lake, travel)


def _inputs(seed: int = 0, with_withdrawals: bool = False):
    g = torch.Generator().manual_seed(seed)
    K_musk = (4.0 + 44.0 * torch.rand(N, generator=g)) * 3600.0   # s
    x_musk = 0.01 + 0.48 * torch.rand(N, generator=g)
    q_lat = 2.0 * torch.rand(T, N, generator=g)                   # mm/day
    area = 20.0 + 80.0 * torch.rand(N, generator=g)               # km2
    storage0 = 1e5 + 1e6 * torch.rand(N, generator=g)             # m3
    wd = WithdrawalData.zeros(T, N)
    if with_withdrawals:
        # Prélèvement net négatif marqué sur un nœud de tête (active le clamp)
        wd.net[:, 0] = -5.0
    return K_musk, x_musk, q_lat, area, storage0, wd


def _run(mode: str, K_musk, x_musk, q_lat, area, storage0, wd):
    graph = _branched_graph_with_lakes()
    layer = RoutingLayer(use_travel_time_attention=False, routing_mode=mode)
    buf = OutflowRingBuffer(N, depth=5)
    Q_prev = torch.zeros(N)
    storage = storage0.clone()
    out = []
    for t in range(T):
        Q, storage, _ = layer(
            q_lat[t], graph, Q_prev, buf, wd, t, K_musk, x_musk,
            lake_storage=storage, area_km2=area,
        )
        out.append(Q)
        Q_prev = Q
    return torch.stack(out), storage


def test_operator_matches_level_forward():
    args = _inputs(seed=1)
    Q_lvl, S_lvl = _run("level", *args)
    Q_op, S_op = _run("operator", *args)
    assert torch.allclose(Q_op, Q_lvl, rtol=1e-4, atol=1e-5), (
        f"écart max {(Q_op - Q_lvl).abs().max().item():.2e}"
    )
    assert torch.allclose(S_op, S_lvl, rtol=1e-4, atol=1.0)


def test_operator_matches_level_with_withdrawals():
    # Prélèvements négatifs : le clamp par sous-pas (level) vs post-solve
    # (operator) peuvent diverger localement ; l'écart doit rester minime.
    args = _inputs(seed=2, with_withdrawals=True)
    Q_lvl, _ = _run("level", *args)
    Q_op, _ = _run("operator", *args)
    assert torch.isfinite(Q_op).all()
    assert (Q_op >= 0).all()
    rel = (Q_op - Q_lvl).abs().max() / Q_lvl.abs().max().clamp(min=1e-6)
    assert rel < 5e-3, f"écart relatif {rel.item():.2e}"


def test_operator_gradients_match_level():
    base = _inputs(seed=3)
    grads = {}
    for mode in ("level", "operator"):
        K = base[0].clone().requires_grad_(True)
        x = base[1].clone().requires_grad_(True)
        q = base[2].clone().requires_grad_(True)
        Q, _ = _run(mode, K, x, q, base[3], base[4], base[5])
        Q.sum().backward()
        grads[mode] = (K.grad.clone(), x.grad.clone(), q.grad.clone())
    for g_lvl, g_op, name in zip(grads["level"], grads["operator"],
                                 ("K_musk", "x_musk", "q_lat")):
        assert torch.allclose(g_op, g_lvl, rtol=1e-3, atol=1e-6), (
            f"gradient {name}: écart max {(g_op - g_lvl).abs().max().item():.2e}"
        )


def test_lagged_is_stable_and_close():
    args = _inputs(seed=4)
    Q_lvl, _ = _run("level", *args)
    Q_lag, S_lag = _run("operator-lagged", *args)
    assert torch.isfinite(Q_lag).all()
    assert (Q_lag >= 0).all()
    assert (S_lag >= 0).all()
    # Même volume total à ~15 % près (le lag déplace, ne crée pas d'eau)
    v_lvl, v_lag = Q_lvl.sum(), Q_lag.sum()
    assert (v_lag - v_lvl).abs() / v_lvl < 0.15
    # Corrélation temporelle élevée à l'exutoire
    a, b = Q_lvl[:, 9], Q_lag[:, 9]
    r = torch.corrcoef(torch.stack([a, b]))[0, 1]
    assert r > 0.95, f"corrélation exutoire {r.item():.3f}"


def test_operator_no_lakes_single_stage():
    # Sans lacs : operator et operator-lagged sont identiques au level.
    src = torch.arange(N - 1, dtype=torch.long)
    dst = torch.arange(1, N, dtype=torch.long)
    ei = torch.stack([src, dst])
    graph = RiverGraph(ei, torch.ones(N - 1, 3),
                       _topological_sort(ei, N), torch.zeros(N, dtype=torch.bool),
                       torch.ones(N - 1, dtype=torch.long))
    K = (4.0 + 40.0 * torch.rand(N)) * 3600.0
    x = torch.full((N,), 0.2)
    wd = WithdrawalData.zeros(T, N)
    q_lat = torch.rand(T, N)
    res = {}
    for mode in ("level", "operator", "operator-lagged"):
        layer = RoutingLayer(use_travel_time_attention=False, routing_mode=mode)
        buf = OutflowRingBuffer(N, depth=5)
        Q_prev = torch.zeros(N)
        out = []
        for t in range(T):
            Q, _, _ = layer(q_lat[t], graph, Q_prev, buf, wd, t, K, x,
                            lake_storage=None, area_km2=torch.ones(N) * 30)
            out.append(Q)
            Q_prev = Q
        res[mode] = torch.stack(out)
    assert torch.allclose(res["operator"], res["level"], rtol=1e-4, atol=1e-5)
    assert torch.allclose(res["operator-lagged"], res["level"], rtol=1e-4, atol=1e-5)
