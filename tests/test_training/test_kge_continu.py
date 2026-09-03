"""Le KGE de la perte porte-t-il sur la séquence continue ou sur le bloc courant ?

Le KGE est une statistique de séquence : moyennes, écarts-types, corrélation. Calculé sur
un bloc d'entraînement de 45 jours, il ne mesure pas la même chose que sur la série
continue qui sert à choisir le modèle. Mesure du 2026-09-03 sur les huit régions du
rapport : le KGE médian par fenêtre de 45 jours vaut -0,01 à 0,27 quand le KGE continu
vaut 0,55 à 0,78, et sur le Saguenay nord-ouest la corrélation entre les deux tombe à
0,05. La perte optimisait un objet sans rapport avec la métrique de sélection.

La perte accepte donc un historique DÉTACHÉ des débits déjà simulés dans l'époque. Ces
tests vérifient que l'historique change bien la statistique, que le gradient ne remonte
que par le bloc courant, et qu'une perte calculée en deux blocs avec historique égale
celle calculée d'un coup sur la séquence entière.
"""
import torch

from meandre.training.loss import HydroLoss, differentiable_kge_loss


def _perte():
    return HydroLoss(w_nse=0.0, w_pbias=0.0, w_kge=1.0, w_mse=0.0, w_nrmse=0.0,
                     w_log_nse=0.0, per_station=True)


def _series(T=80, S=2, graine=0):
    g = torch.Generator().manual_seed(graine)
    obs = 10.0 + 5.0 * torch.rand(T, S, generator=g)
    sim = obs * 0.8 + torch.rand(T, S, generator=g)
    return obs, sim


def test_l_historique_change_la_statistique():
    """Sans historique la perte ne voit que le bloc ; avec, elle voit toute la série."""
    obs, sim = _series()
    masque = torch.ones(2, dtype=torch.bool)
    perte = _perte()
    coupe = 40

    sans, _ = perte(q_obs=obs[coupe:], q_sim=sim[coupe:], station_mask=masque)
    avec, _ = perte(q_obs=obs[coupe:], q_sim=sim[coupe:], station_mask=masque,
                    q_obs_hist=obs[:coupe], q_sim_hist=sim[:coupe])

    assert not torch.isclose(sans, avec, atol=1e-6), \
        "l'historique doit changer la statistique de sequence"


def test_deux_blocs_avec_historique_egalent_la_sequence_entiere():
    """Le second bloc, muni de son historique, retrouve la perte de la serie complete."""
    obs, sim = _series()
    masque = torch.ones(2, dtype=torch.bool)
    perte = _perte()
    coupe = 40

    entiere, _ = perte(q_obs=obs, q_sim=sim, station_mask=masque)
    par_blocs, _ = perte(q_obs=obs[coupe:], q_sim=sim[coupe:], station_mask=masque,
                         q_obs_hist=obs[:coupe], q_sim_hist=sim[:coupe])

    assert torch.isclose(entiere, par_blocs, atol=1e-5)


def test_le_gradient_ne_remonte_que_par_le_bloc_courant():
    """L'historique est detache : les blocs passes ont ete simules avec d'autres poids."""
    obs, sim = _series()
    masque = torch.ones(2, dtype=torch.bool)
    coupe = 40
    hist = sim[:coupe].clone().requires_grad_(True)
    courant = sim[coupe:].clone().requires_grad_(True)

    perte, _ = _perte()(q_obs=obs[coupe:], q_sim=courant, station_mask=masque,
                        q_obs_hist=obs[:coupe], q_sim_hist=hist)
    perte.backward()

    assert courant.grad is not None and float(courant.grad.abs().sum()) > 0
    assert hist.grad is None or float(hist.grad.abs().sum()) == 0.0


def test_la_perte_reste_un_kge_sur_une_station():
    """Controle de coherence : sur une station unique et sans historique, la perte
    par station vaut bien 1 moins le KGE."""
    obs, sim = _series(S=1)
    masque = torch.ones(1, dtype=torch.bool)
    perte, _ = _perte()(q_obs=obs, q_sim=sim, station_mask=masque)
    attendu = differentiable_kge_loss(obs[:, 0], sim[:, 0])
    assert torch.isclose(perte, attendu, atol=1e-5)
