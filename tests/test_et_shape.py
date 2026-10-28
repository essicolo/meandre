"""Le terme MOD16 en forme est aveugle à un facteur multiplicatif sur l'ET simulée.

K_c multiplie l'ET. En mode centré, le terme voit l'amplitude du cycle saisonnier, donc le
facteur, et il baissait K_c pour réduire l'amplitude simulée, entraînant le niveau (R310).
En forme, chaque côté est divisé par son propre écart-type : le facteur disparaît.
"""
from types import SimpleNamespace

import torch

from meandre.training.trainer import Trainer


def _center(mode, sim, obs):
    t = object.__new__(Trainer)
    t.loss_fn = SimpleNamespace(et_mode=mode)
    t._et_sim_base = sim.mean(dim=0)
    t._et_sim_std = sim.std(dim=0, unbiased=False)
    data = SimpleNamespace(et_obs=obs, et_weight=None, train_slice=slice(0, obs.shape[0]))
    zs, zo = t._center_et(sim, obs, data, t0=0)
    return zs, zo


def test_shape_blind_to_factor():
    torch.manual_seed(0)
    day_count = torch.arange(730, dtype=torch.float32)
    cycle = 1.5 + torch.sin(2 * torch.pi * day_count / 365.25)
    sim = cycle[:, None] * torch.tensor([0.8, 1.0, 1.2]) + 0.05 * torch.randn(730, 3)
    obs = cycle[:, None].repeat(1, 3) + 0.05 * torch.randn(730, 3)
    zs1, zo1 = _center("forme", sim, obs)
    zs2, zo2 = _center("forme", 0.6 * sim, obs)
    assert torch.allclose(zs1, zs2, atol=1e-5)
    assert torch.allclose(zo1, zo2)


def test_centered_mode_sees_the_factor():
    day_count = torch.arange(730, dtype=torch.float32)
    sim = (1.5 + torch.sin(2 * torch.pi * day_count / 365.25))[:, None].repeat(1, 2)
    zs1, _ = _center("anomaly", sim, sim.clone())
    zs2, _ = _center("anomaly", 0.6 * sim, sim.clone())
    assert not torch.allclose(zs1, zs2)
