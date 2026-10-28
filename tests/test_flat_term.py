"""Terme de part de jours plats (R382) : nul sur la vérité, positif sur un plateau, branché sur la perte."""
import math

import torch

from meandre.training.loss import HydroLoss, differentiable_flat_days_loss


def _series(n=400):
    t = torch.arange(n, dtype=torch.float64)
    return 10.0 + 3.0 * torch.sin(2 * math.pi * t / 365.0) + 1.5 * torch.sin(t * 0.9) ** 2


def test_zero_on_truth():
    q = _series()
    assert float(differentiable_flat_days_loss(q, q)) == 0.0


def test_penalizes_plateau_and_keeps_gradient():
    q = _series()
    flat = torch.full_like(q, float(q.mean())).requires_grad_(True)
    loss = differentiable_flat_days_loss(q, flat * (1.0 + 1e-4 * torch.arange(len(q))))
    assert float(loss) > 0.5
    loss.backward()
    assert flat.grad is not None and torch.isfinite(flat.grad).all()


def test_wired_into_the_loss():
    q = _series().reshape(-1, 1).float()
    sim = q.mean().expand_as(q).clone()
    mk = torch.ones(1, dtype=torch.bool)
    base = dict(w_kge=0.0, w_pbias=0.0, w_mse=0.0, w_log_mse=0.0, w_peak=0.0, per_station=True)
    without = HydroLoss(**base)(q_obs=q, q_sim=sim, station_mask=mk)
    f = HydroLoss(**{**base, "w_flat": 1.0})
    f.doy_block = torch.tensor([335 + (i % 120) if 335 + (i % 120) <= 365 else (i % 120) - 30 for i in range(len(q))])
    avec = f(q_obs=q, q_sim=sim, station_mask=mk)
    t = lambda r: float(r[0] if isinstance(r, tuple) else r)
    assert t(avec) > t(without)


def test_term_is_zero_outside_target_months():
    q = _series().reshape(-1, 1).float()
    sim = q.mean().expand_as(q).clone()
    mk = torch.ones(1, dtype=torch.bool)
    base = dict(w_kge=0.0, w_pbias=0.0, w_mse=0.0, w_log_mse=0.0, w_peak=0.0, per_station=True)
    f = HydroLoss(**{**base, "w_flat": 1.0})
    f.doy_block = torch.full((len(q),), 200)
    _, comps = f(q_obs=q, q_sim=sim, station_mask=mk)
    assert float(comps["plat_loss"]) == 0.0
