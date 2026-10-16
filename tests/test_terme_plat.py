"""Terme de part de jours plats (R382) : nul sur la vérité, positif sur un plateau, branché sur la perte."""
import math

import torch

from meandre.training.loss import HydroLoss, differentiable_flat_days_loss


def _serie(n=400):
    t = torch.arange(n, dtype=torch.float64)
    return 10.0 + 3.0 * torch.sin(2 * math.pi * t / 365.0) + 1.5 * torch.sin(t * 0.9) ** 2


def test_nul_sur_la_verite():
    q = _serie()
    assert float(differentiable_flat_days_loss(q, q)) == 0.0


def test_punit_un_plateau_et_garde_un_gradient():
    q = _serie()
    plat = torch.full_like(q, float(q.mean())).requires_grad_(True)
    perte = differentiable_flat_days_loss(q, plat * (1.0 + 1e-4 * torch.arange(len(q))))
    assert float(perte) > 0.5
    perte.backward()
    assert plat.grad is not None and torch.isfinite(plat.grad).all()


def test_branche_sur_la_perte():
    q = _serie().reshape(-1, 1).float()
    sim = q.mean().expand_as(q).clone()
    mk = torch.ones(1, dtype=torch.bool)
    base = dict(w_kge=0.0, w_pbias=0.0, w_mse=0.0, w_log_mse=0.0, w_peak=0.0, per_station=True)
    sans = HydroLoss(**base)(q_obs=q, q_sim=sim, station_mask=mk)
    f = HydroLoss(**{**base, "w_plat": 1.0})
    f.doy_bloc = torch.tensor([335 + (i % 120) if 335 + (i % 120) <= 365 else (i % 120) - 30 for i in range(len(q))])
    avec = f(q_obs=q, q_sim=sim, station_mask=mk)
    t = lambda r: float(r[0] if isinstance(r, tuple) else r)
    assert t(avec) > t(sans)


def test_hors_des_mois_cibles_le_terme_est_nul():
    q = _serie().reshape(-1, 1).float()
    sim = q.mean().expand_as(q).clone()
    mk = torch.ones(1, dtype=torch.bool)
    base = dict(w_kge=0.0, w_pbias=0.0, w_mse=0.0, w_log_mse=0.0, w_peak=0.0, per_station=True)
    f = HydroLoss(**{**base, "w_plat": 1.0})
    f.doy_bloc = torch.full((len(q),), 200)
    _, comps = f(q_obs=q, q_sim=sim, station_mask=mk)
    assert float(comps["plat_loss"]) == 0.0
