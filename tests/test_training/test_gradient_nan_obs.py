"""Le gradient reste fini quand des observations manquent.

R92 (2026-09-08) : la perte posait des NaN dans le tenseur SIMULE pour s'appuyer sur
nanmean et nansum. Ces reductions ignorent les NaN a l'aller mais leur retropropagation
rend un gradient NaN, si bien que tout bloc contenant une observation manquante etait
jete par le filet du trainer : 2000 blocs perdus sur le Saint-Laurent sud, pres de la
moitie des pas d'optimisation, alors que la perte affichee restait finie.
"""
import torch

from meandre.training.loss import HydroLoss


def _perte(w):
    return HydroLoss(per_station=True, w_kge=0.0, w_nse=0.0, w_nrmse=0.0, w_log_nse=0.0, **w)


def test_gradient_fini_avec_observations_manquantes():
    torch.manual_seed(0)
    T, N, S = 60, 4, 2
    q_sim = torch.rand(T, N, requires_grad=True)
    masque = torch.zeros(N, dtype=torch.bool)
    masque[:S] = True
    q_obs = torch.rand(T, S) + 0.5
    q_obs[10:20, 0] = float("nan")          # une lacune, cas courant en hiver
    q_obs[:, 1] = float("nan")              # une station entierement absente
    q_obs[30:, 1] = torch.rand(T - 30) + 0.5
    for nom, poids in (("mse", dict(w_mse=1.0, w_pbias=0.0, w_log_mse=0.0)),
                       ("pbias", dict(w_mse=0.0, w_pbias=1.0, w_log_mse=0.0)),
                       ("log_mse", dict(w_mse=0.0, w_pbias=0.0, w_log_mse=1.0))):
        if q_sim.grad is not None:
            q_sim.grad = None
        perte, _ = _perte(poids)(q_obs=q_obs, q_sim=q_sim, station_mask=masque)
        assert torch.isfinite(perte), f"{nom} : perte non finie"
        perte.backward()
        assert torch.isfinite(q_sim.grad).all(), f"{nom} : gradient non fini"


def test_valeur_identique_a_la_moyenne_sans_lacune():
    """Le masque multiplicatif doit rendre la MEME valeur que la moyenne sur les jours
    observes : le correctif ne change pas ce qui est mesure, seulement le gradient."""
    torch.manual_seed(1)
    T, S = 40, 1
    q_sim = torch.rand(T, S)
    q_obs = torch.rand(T, S) + 0.5
    masque = torch.ones(S, dtype=torch.bool)
    p_plein, _ = _perte(dict(w_mse=1.0, w_pbias=0.0, w_log_mse=0.0))(
        q_obs=q_obs, q_sim=q_sim, station_mask=masque)
    q_trou = q_obs.clone()
    q_trou[5:15] = float("nan")
    p_trou, _ = _perte(dict(w_mse=1.0, w_pbias=0.0, w_log_mse=0.0))(
        q_obs=q_trou, q_sim=q_sim, station_mask=masque)
    ref = ((q_obs - q_sim) ** 2)
    garde = torch.ones(T, dtype=torch.bool); garde[5:15] = False
    assert torch.allclose(p_plein, ref.mean(), atol=1e-5)
    assert torch.allclose(p_trou, ref[garde].mean(), atol=1e-5)
