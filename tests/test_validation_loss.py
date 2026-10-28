"""Perte de debit sur la validation, pour choisir le point de reprise sur l'etiage (2026-10-01).

`best_metric = "val_loss"` lisait une cle « loss » que la validation ne remplissait pas.
La perte de validation doit reprendre les poids de la perte d'entrainement, terme d'etiage
compris, et preferer un hydrogramme dont l'etiage est juste.
"""
import types

import torch

from meandre.training.trainer import Trainer


def _fake_trainer(**poids):
    tr = Trainer.__new__(Trainer)
    defaut = dict(w_kge=1.0, w_pbias=0.5, w_log_mse=0.0, w_low_flow=0.0, w_fdc_low=0.0)
    defaut.update(poids)
    tr.loss_fn = types.SimpleNamespace(**defaut)
    return tr


def _series():
    t = torch.arange(730, dtype=torch.float32)
    q = 2.0 + 1.5 * torch.sin(2 * torch.pi * t / 365.0) + 0.3 * torch.sin(2 * torch.pi * t / 17.0)
    return q.unsqueeze(1)


def test_low_flow_too_high_costs_with_low_flow_term():
    q = _series()
    haut = torch.where(q < 1.5, q * 1.4, q)
    without = _fake_trainer()
    avec = _fake_trainer(w_low_flow=1.0)
    assert avec._validation_discharge_loss(q, haut) > without._validation_discharge_loss(q, haut)
    assert avec._validation_discharge_loss(q, q) < 1e-3


def test_too_short_station_ignored():
    q = _series()
    q_court = q.clone()
    q_court[20:] = float("nan")
    assert _fake_trainer()._validation_discharge_loss(q_court, q) == float("inf")
