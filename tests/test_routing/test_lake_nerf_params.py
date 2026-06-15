"""Tête NeRF de paramètres de lac (k_lake, beta par nœud).

Vérifie : (1) rétrocompat — predict_lake_params=False reproduit exactement le
comportement scalaire global ; (2) la tête produit des params bornés et variant
par nœud ; (3) les gradients de la perte remontent jusqu'à fc_lake.
"""
import torch
import pytest

from meandre.routing.graph import RiverGraph, _topological_sort
from meandre.routing.message_passing import RoutingLayer
from meandre.routing.lake import LakeModule
from meandre.routing.withdrawals import WithdrawalData
from meandre.temporal.ring_buffer import OutflowRingBuffer
from meandre.spatial.field_network import SpatialFieldNetwork

N = 10
T = 20


def _graph_with_lakes() -> RiverGraph:
    src = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.long)
    dst = torch.tensor([2, 2, 3, 4, 6, 6, 7, 8, 9], dtype=torch.long)
    ei = torch.stack([src, dst])
    is_lake = torch.zeros(N, dtype=torch.bool); is_lake[3] = True; is_lake[7] = True
    return RiverGraph(ei, torch.ones(9, 3), _topological_sort(ei, N), is_lake,
                      torch.ones(9, dtype=torch.long))


def test_lake_module_per_node_matches_scalar_when_equal():
    # LakeModule : passer k/beta par nœud égaux aux scalaires globaux = identique
    lake = LakeModule()
    n = 5
    Q_in = torch.rand(n); S = torch.rand(n) * 1e5; A = torch.ones(n) * 10
    z = torch.zeros(n)
    Q_glob, S_glob = lake(Q_in, S, A, z, z, z)
    k = lake.k_lake.detach().expand(n).clone()
    b = lake.beta.detach().expand(n).clone()
    Q_node, S_node = lake(Q_in, S, A, z, z, z, k_lake=k, beta=b)
    assert torch.allclose(Q_glob, Q_node, atol=1e-6)
    assert torch.allclose(S_glob, S_node, atol=1e-6)


def test_nerf_lake_head_bounds_and_variation():
    net = SpatialFieldNetwork(n_territorial=4, predict_lake_params=True)
    coords = torch.rand(N, 2)
    terr = torch.randn(N, 4)
    k, b = net.lake_params(coords, terr)
    assert k.shape == (N,) and b.shape == (N,)
    assert (k >= 1e-6).all() and (k <= 1e-2).all()
    assert (b >= 1.0).all() and (b <= 2.5).all()
    # init : tous proches des défauts globaux (k=1e-4, beta=1.5)
    assert torch.allclose(k, torch.full((N,), 1e-4), rtol=0.05)
    assert torch.allclose(b, torch.full((N,), 1.5), atol=0.05)


def test_nerf_lake_head_disabled_raises():
    net = SpatialFieldNetwork(n_territorial=4, predict_lake_params=False)
    assert not hasattr(net, "fc_lake")
    with pytest.raises(RuntimeError):
        net.lake_params(torch.rand(N, 2), torch.randn(N, 4))


def test_routing_lake_params_backward_compat():
    # _lake_k/_lake_beta = None → comportement scalaire global inchangé
    K = (4.0 + 40.0 * torch.rand(N)) * 3600.0
    x = torch.full((N,), 0.2)
    wd = WithdrawalData.zeros(T, N)
    q_lat = torch.rand(T, N)
    graph = _graph_with_lakes()
    layer = RoutingLayer(use_travel_time_attention=False)
    assert layer._lake_k is None and layer._lake_beta is None
    buf = OutflowRingBuffer(N, depth=5); Q_prev = torch.zeros(N); S = torch.zeros(N)
    for t in range(T):
        Q, S, _ = layer(q_lat[t], graph, Q_prev, buf, wd, t, K, x,
                        lake_storage=S, area_km2=torch.ones(N) * 30)
        Q_prev = Q
    assert torch.isfinite(Q).all()


def test_gradients_reach_lake_head():
    # La perte sur Q doit propager un gradient non nul vers fc_lake
    net = SpatialFieldNetwork(n_territorial=4, predict_lake_params=True)
    coords = torch.rand(N, 2); terr = torch.randn(N, 4)
    k, b = net.lake_params(coords, terr)
    graph = _graph_with_lakes()
    layer = RoutingLayer(use_travel_time_attention=False)
    layer._lake_k = k; layer._lake_beta = b
    wd = WithdrawalData.zeros(T, N)
    K = (4.0 + 40.0 * torch.rand(N)) * 3600.0
    x = torch.full((N,), 0.2)
    q_lat = torch.rand(T, N)
    buf = OutflowRingBuffer(N, depth=5); Q_prev = torch.zeros(N); S = torch.zeros(N) + 1e5
    out = []
    for t in range(T):
        Q, S, _ = layer(q_lat[t], graph, Q_prev, buf, wd, t, K, x,
                        lake_storage=S, area_km2=torch.ones(N) * 30)
        out.append(Q); Q_prev = Q
    torch.stack(out).sum().backward()
    assert net.fc_lake.weight.grad is not None
    assert net.fc_lake.weight.grad.abs().sum() > 0, "gradient nul vers fc_lake"
