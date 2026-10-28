"""Le terme de volume porte sur la série continue (historique détaché plus bloc), pas sur le bloc seul."""
import torch

from meandre.training.loss import HydroLoss


def _loss():
    return HydroLoss(w_mse=0.0, w_pbias=1.0, w_kge=0.0, w_log_mse=0.0, per_station=True)


def test_volume_on_continuous_series(monkeypatch):
    monkeypatch.setenv("MEANDRE_PBIAS_CONTINU", "1")
    torch.manual_seed(0)
    qo_h, qs_h = torch.rand(300, 2) + 1.0, torch.rand(300, 2) + 1.5
    qo, qs = torch.rand(45, 2) + 1.0, (torch.rand(45, 2) + 0.5).requires_grad_(True)
    mask = torch.tensor([True, True])
    _, comps = _loss()(q_obs=qo, q_sim=qs, station_mask=mask, q_obs_hist=qo_h, q_sim_hist=qs_h)
    expected = (((torch.cat([qs_h, qs]) - torch.cat([qo_h, qo])).sum(0) / torch.cat([qo_h, qo]).sum(0)).abs().mean())
    assert abs(float(comps["pbias_loss"]) - float(expected)) < 1e-5


def test_per_block_volume_restored(monkeypatch):
    monkeypatch.setenv("MEANDRE_PBIAS_CONTINU", "0")
    torch.manual_seed(0)
    qo_h, qs_h = torch.rand(300, 2) + 1.0, torch.rand(300, 2) + 1.5
    qo, qs = torch.rand(45, 2) + 1.0, torch.rand(45, 2) + 0.5
    mask = torch.tensor([True, True])
    _, comps = _loss()(q_obs=qo, q_sim=qs, station_mask=mask, q_obs_hist=qo_h, q_sim_hist=qs_h)
    expected = ((qs - qo).sum(0) / qo.sum(0)).abs().mean()
    assert abs(float(comps["pbias_loss"]) - float(expected)) < 1e-5
