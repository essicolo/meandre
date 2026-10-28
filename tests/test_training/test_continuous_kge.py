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


def _loss():
    return HydroLoss(w_nse=0.0, w_pbias=0.0, w_kge=1.0, w_mse=0.0, w_nrmse=0.0,
                     w_log_nse=0.0, per_station=True)


def _series(T=80, S=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    obs = 10.0 + 5.0 * torch.rand(T, S, generator=g)
    sim = obs * 0.8 + torch.rand(T, S, generator=g)
    return obs, sim


def test_history_changes_the_statistic():
    """Sans historique la perte ne voit que le bloc ; avec, elle voit toute la série."""
    obs, sim = _series()
    mask = torch.ones(2, dtype=torch.bool)
    loss = _loss()
    cut = 40

    without, _ = loss(q_obs=obs[cut:], q_sim=sim[cut:], station_mask=mask)
    avec, _ = loss(q_obs=obs[cut:], q_sim=sim[cut:], station_mask=mask,
                    q_obs_hist=obs[:cut], q_sim_hist=sim[:cut])

    assert not torch.isclose(without, avec, atol=1e-6), \
        "l'historique doit changer la statistique de sequence"


def test_two_blocks_with_history_equal_whole_sequence():
    """Le second bloc, muni de son historique, retrouve la perte de la serie complete."""
    obs, sim = _series()
    mask = torch.ones(2, dtype=torch.bool)
    loss = _loss()
    cut = 40

    whole, _ = loss(q_obs=obs, q_sim=sim, station_mask=mask)
    per_block, _ = loss(q_obs=obs[cut:], q_sim=sim[cut:], station_mask=mask,
                         q_obs_hist=obs[:cut], q_sim_hist=sim[:cut])

    assert torch.isclose(whole, per_block, atol=1e-5)


def test_gradient_flows_only_through_current_block():
    """L'historique est detache : les blocs passes ont ete simules avec d'autres poids."""
    obs, sim = _series()
    mask = torch.ones(2, dtype=torch.bool)
    cut = 40
    hist = sim[:cut].clone().requires_grad_(True)
    current_value = sim[cut:].clone().requires_grad_(True)

    loss, _ = _loss()(q_obs=obs[cut:], q_sim=current_value, station_mask=mask,
                        q_obs_hist=obs[:cut], q_sim_hist=hist)
    loss.backward()

    assert current_value.grad is not None and float(current_value.grad.abs().sum()) > 0
    assert hist.grad is None or float(hist.grad.abs().sum()) == 0.0


def test_loss_remains_a_kge_on_one_station():
    """Controle de coherence : sur une station unique et sans historique, la perte
    par station vaut bien 1 moins le KGE."""
    obs, sim = _series(S=1)
    mask = torch.ones(1, dtype=torch.bool)
    loss, _ = _loss()(q_obs=obs, q_sim=sim, station_mask=mask)
    expected = differentiable_kge_loss(obs[:, 0], sim[:, 0])
    assert torch.isclose(loss, expected, atol=1e-5)
