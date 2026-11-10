"""Reference simulee de l'ET centree : moyenne exacte de l'epoque, pas du premier bloc."""
import math
import types

import torch

from meandre.training.trainer import Trainer


def _trainer_factice():
    tr = Trainer.__new__(Trainer)
    tr.loss_fn = types.SimpleNamespace(et_mode="anomaly", w_et=0.4)
    return tr


def test_reference_is_epoch_mean():
    tr = _trainer_factice()
    day_count = torch.arange(730, dtype=torch.float32)
    et = (1.0 + math.sqrt(2.0) * torch.sin(2 * math.pi * day_count / 365.0)).clamp(min=0.0).unsqueeze(1)
    data = types.SimpleNamespace(et_obs=et.clone())
    for d in range(0, 730, 45):
        tr._center_et(et[d:d + 45], et[d:d + 45], data)
    tr._et_reference_epoch_end()
    expected = et[:720].mean().item()
    assert abs(float(tr._et_sim_base) - et.mean().item()) < 0.02
    assert abs(float(tr._et_sim_base) - expected) < 0.05


def test_first_winter_block_no_longer_sets_reference():
    tr = _trainer_factice()
    winter = torch.full((45, 1), 0.05)
    summer = torch.full((45, 1), 3.0)
    data = types.SimpleNamespace(et_obs=torch.cat([winter, summer]))
    tr._center_et(winter, winter, data)
    tr._center_et(summer, summer, data)
    tr._et_reference_epoch_end()
    assert abs(float(tr._et_sim_base) - 1.525) < 1e-4


def test_snow_free_weight_excludes_winter_from_means():
    tr = _trainer_factice()
    winter = torch.full((45, 1), 0.05)
    summer = torch.full((45, 1), 3.0)
    obs = torch.cat([torch.full((45, 1), 0.8), torch.full((45, 1), 3.2)])
    poids = torch.cat([torch.zeros(45, 1), torch.ones(45, 1)])
    data = types.SimpleNamespace(et_obs=obs, et_weight=poids, train_slice=slice(0, 90))
    tr._center_et(winter, obs[:45], data, t0=0)
    s_summer, o_summer = tr._center_et(summer, obs[45:], data, t0=45)
    tr._et_reference_epoch_end()
    assert abs(float(tr._et_obs_base) - 3.2) < 1e-5
    assert abs(float(tr._et_sim_base) - 3.0) < 1e-5


def test_observed_reference_uses_training_period_only():
    """La moyenne de MOD16 ne lit pas les jours apres la periode d'entrainement (2026-11-10)."""
    tr = _trainer_factice()
    obs = torch.cat([torch.full((90, 2), 1.0), torch.full((30, 2), 5.0)])
    data = types.SimpleNamespace(et_obs=obs, train_slice=slice(0, 90))
    tr._center_et(torch.ones(90, 2), obs[:90], data, t0=0)
    assert torch.allclose(tr._et_obs_base, torch.full((2,), 1.0))
