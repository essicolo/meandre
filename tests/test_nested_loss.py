"""Terme d'atténuation entre stations emboîtées (R467)."""
import torch

from meandre.training.loss import HydroLoss, differentiable_nested_loss

PAIRS = torch.tensor([[0, 1]])
AREAS = torch.tensor([[10.0, 50.0]])


def _series(seed, n=200):
    g = torch.Generator().manual_seed(seed)
    return torch.relu(torch.randn(n, generator=g)).cumsum(0) % 7.0 + 1.0


def test_zero_when_simulation_matches_observation():
    q = torch.stack([_series(1), _series(2)], dim=1)
    assert float(differentiable_nested_loss(q, q.clone(), PAIRS, AREAS)) < 1e-6


def test_penalises_a_downstream_hydrograph_that_is_too_damped():
    up, down = _series(1), _series(2)
    smooth = torch.nn.functional.avg_pool1d(down[None, None], 9, stride=1, padding=4)[0, 0]
    q_obs = torch.stack([up, down], dim=1)
    q_sim = torch.stack([up, smooth], dim=1)
    assert float(differentiable_nested_loss(q_obs, q_sim, PAIRS, AREAS)) > 0.3


def test_gradient_is_finite_with_missing_observations():
    q_obs = torch.stack([_series(1), _series(2)], dim=1)
    q_obs[20:40, 1] = float("nan")
    q_sim = (q_obs.nan_to_num(1.0) * 1.1).requires_grad_(True)
    differentiable_nested_loss(q_obs, q_sim, PAIRS, AREAS).backward()
    assert torch.isfinite(q_sim.grad).all()


def test_hydroloss_adds_the_term_only_when_weighted_and_paired():
    up, down = _series(1), _series(2)
    q_obs = torch.stack([up, down], dim=1)
    q_sim = torch.stack([up, down * 0.5 + 1.0], dim=1)
    mask = torch.tensor([True, True])
    off = HydroLoss(w_kge=1.0, per_station=True)
    on = HydroLoss(w_kge=1.0, w_nested=1.0, per_station=True)
    on.nested_pairs, on.nested_areas = PAIRS, AREAS
    l_off, c_off = off(q_obs, q_sim, mask)
    l_on, c_on = on(q_obs, q_sim, mask)
    assert float(c_off["nested_loss"]) == 0.0
    assert abs(float(l_on - l_off) - float(c_on["nested_loss"])) < 1e-6
