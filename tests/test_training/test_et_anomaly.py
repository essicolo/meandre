"""Tests du mode ET en TENDANCE (et_mode="anomaly") : centrage longue durée."""
import numpy as np
import torch
import pytest

from meandre.training.loss import HydroLoss
from meandre.training.trainer import Trainer


class _FakeData:
    def __init__(self, et_obs):
        self.et_obs = et_obs


def _mk_trainer_stub(et_mode):
    """Trainer sans __init__ : on ne teste que _center_et (méthode pure + état)."""
    tr = Trainer.__new__(Trainer)
    tr.loss_fn = HydroLoss(et_mode=et_mode)
    return tr


def test_et_mode_validation():
    HydroLoss(et_mode="level")
    HydroLoss(et_mode="anomaly")
    with pytest.raises(ValueError):
        HydroLoss(et_mode="tendance")


def test_center_et_level_is_identity():
    tr = _mk_trainer_stub("level")
    sim = torch.rand(30, 4)
    obs = torch.rand(30, 4)
    s, o = tr._center_et(sim, obs, _FakeData(obs))
    assert s is sim and o is obs


def test_center_et_anomaly_removes_level_bias():
    torch.manual_seed(0)
    T, N = 400, 3
    saison = torch.sin(torch.arange(T, dtype=torch.float32) * 2 * np.pi / 365.25)[:, None]
    obs = 1.2 + 0.8 * saison + 0.01 * torch.randn(T, N)
    sim = 1.6 + 0.8 * saison + 0.01 * torch.randn(T, N)   # biais de NIVEAU +0.4, même saisonnalité
    obs_nan = obs.clone(); obs_nan[::7] = float("nan")     # trous MODIS
    tr = _mk_trainer_stub("anomaly")
    s, o = tr._center_et(sim, obs_nan, _FakeData(obs_nan))
    v = ~torch.isnan(o)
    # le biais de niveau disparaît : résidu moyen ~0 malgré le +0.4 d'origine
    assert abs(float((s[v] - o[v]).mean())) < 0.05
    # la saisonnalité est PRÉSERVÉE (corrélation quasi parfaite des anomalies)
    r = np.corrcoef(s[v].numpy(), o[v].numpy())[0, 1]
    assert r > 0.99
    # les lignes de base sont bien longue durée : obs = moyenne série, sim = EMA détachée
    assert torch.allclose(tr._et_obs_base,
                          torch.nan_to_num(obs_nan, nan=0.0).sum(0) / (~torch.isnan(obs_nan)).sum(0))
    assert not tr._et_sim_base.requires_grad


def test_center_et_anomaly_ema_converges():
    tr = _mk_trainer_stub("anomaly")
    obs = torch.zeros(10, 2)
    data = _FakeData(obs)
    for _ in range(300):
        tr._center_et(torch.full((10, 2), 5.0), obs, data)
    assert torch.allclose(tr._et_sim_base, torch.full((2,), 5.0), atol=0.05)


def test_center_et_no_grad_leak():
    tr = _mk_trainer_stub("anomaly")
    sim = torch.rand(20, 2, requires_grad=True)
    obs = torch.rand(20, 2)
    s, _ = tr._center_et(sim, obs, _FakeData(obs))
    s.sum().backward()   # la base détachée ne casse pas le graphe
    assert sim.grad is not None
    assert torch.allclose(sim.grad, torch.ones_like(sim))  # d(s)/d(sim) = 1 (base constante)
