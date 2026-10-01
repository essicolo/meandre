"""Perte de debit sur la validation, pour choisir le point de reprise sur l'etiage (2026-10-01).

`best_metric = "val_loss"` lisait une cle « loss » que la validation ne remplissait pas.
La perte de validation doit reprendre les poids de la perte d'entrainement, terme d'etiage
compris, et preferer un hydrogramme dont l'etiage est juste.
"""
import types

import torch

from meandre.training.trainer import Trainer


def _faux_trainer(**poids):
    tr = Trainer.__new__(Trainer)
    defaut = dict(w_kge=1.0, w_pbias=0.5, w_log_mse=0.0, w_etiage=0.0, w_fdc_bas=0.0)
    defaut.update(poids)
    tr.loss_fn = types.SimpleNamespace(**defaut)
    return tr


def _serie():
    t = torch.arange(730, dtype=torch.float32)
    q = 2.0 + 1.5 * torch.sin(2 * torch.pi * t / 365.0) + 0.3 * torch.sin(2 * torch.pi * t / 17.0)
    return q.unsqueeze(1)


def test_etiage_trop_haut_coute_avec_le_terme_d_etiage():
    q = _serie()
    haut = torch.where(q < 1.5, q * 1.4, q)
    sans = _faux_trainer()
    avec = _faux_trainer(w_etiage=1.0)
    assert avec._perte_debit_validation(q, haut) > sans._perte_debit_validation(q, haut)
    assert avec._perte_debit_validation(q, q) < 1e-3


def test_station_trop_courte_ignoree():
    q = _serie()
    q_court = q.clone()
    q_court[20:] = float("nan")
    assert _faux_trainer()._perte_debit_validation(q_court, q) == float("inf")
