"""Reference simulee de l'ET centree : moyenne exacte de l'epoque, pas du premier bloc."""
import math
import types

import torch

from meandre.training.trainer import Trainer


def _trainer_factice():
    tr = Trainer.__new__(Trainer)
    tr.loss_fn = types.SimpleNamespace(et_mode="anomaly", w_et=0.4)
    return tr


def test_reference_est_la_moyenne_de_l_epoque():
    tr = _trainer_factice()
    jours = torch.arange(730, dtype=torch.float32)
    et = (1.0 + math.sqrt(2.0) * torch.sin(2 * math.pi * jours / 365.0)).clamp(min=0.0).unsqueeze(1)
    data = types.SimpleNamespace(et_obs=et.clone())
    for d in range(0, 730, 45):
        tr._center_et(et[d:d + 45], et[d:d + 45], data)
    tr._et_reference_fin_epoque()
    attendu = et[:720].mean().item()
    assert abs(float(tr._et_sim_base) - et.mean().item()) < 0.02
    assert abs(float(tr._et_sim_base) - attendu) < 0.05


def test_premier_bloc_d_hiver_ne_fixe_plus_la_reference():
    tr = _trainer_factice()
    hiver = torch.full((45, 1), 0.05)
    ete = torch.full((45, 1), 3.0)
    data = types.SimpleNamespace(et_obs=torch.cat([hiver, ete]))
    tr._center_et(hiver, hiver, data)
    tr._center_et(ete, ete, data)
    tr._et_reference_fin_epoque()
    assert abs(float(tr._et_sim_base) - 1.525) < 1e-4
