"""Les trois facteurs du KGE, portés séparément dans la perte.

Proposition d'Essi, 2026-09-20 : le KGE entier mélange une erreur de calendrier, une erreur de
volume et une erreur d'amplitude en un seul nombre, et l'optimiseur ne peut plus choisir
laquelle réduire. Le banc de redondance chiffre le gain : à cinq termes contre cinq, la
recette en vigueur laisse 14,2 % de manque moyen et un angle mort de 48,2 % sur le soutien
d'étiage, quand les trois facteurs séparés plus l'étiage plus les variations tombent à 2,8 %
et 5,4 %.

Ce fichier vérifie trois choses : chaque terme voit le défaut qui lui revient et lui seul, les
poids nuls par défaut ne changent rien, et la somme pondérée reste dérivable.
"""
import pytest
import torch

from meandre.training.loss import (HydroLoss, differentiable_beta_loss,
                                   differentiable_gamma_loss, differentiable_kge_loss,
                                   differentiable_r_loss)


def _serie(n=400, graine=3):
    g = torch.Generator().manual_seed(graine)
    t = torch.arange(n, dtype=torch.float32)
    return 10.0 + 5.0 * torch.sin(t / 30.0) + torch.rand(n, generator=g)


def test_chaque_facteur_voit_son_propre_defaut():
    """Un décalage touche le calendrier, un facteur d'échelle le volume, un lissage l'amplitude."""
    q = _serie()
    decale = torch.cat([q[:1], q[:-1]])
    gonfle = q * 1.2
    # Amplitude reduite de trente pour cent AUTOUR de la moyenne : gamma seul doit bouger,
    # le volume et le calendrier restant intacts par construction.
    ampute = q.mean() + (q - q.mean()) * 0.7

    r0 = float(differentiable_r_loss(q, q))
    b0 = float(differentiable_beta_loss(q, q))
    g0 = float(differentiable_gamma_loss(q, q))
    assert r0 == pytest.approx(0.0, abs=1e-4)
    assert b0 == pytest.approx(0.0, abs=1e-6)
    assert g0 == pytest.approx(0.0, abs=1e-6)

    # Le decalage degrade le calendrier sans toucher au volume.
    assert float(differentiable_r_loss(q, decale)) > 1e-3
    assert float(differentiable_beta_loss(q, decale)) < 1e-3
    # Le facteur d'echelle degrade le volume.
    assert float(differentiable_beta_loss(q, gonfle)) > 1e-2
    # L'amputation d'amplitude touche gamma et laisse le volume et le calendrier.
    assert float(differentiable_gamma_loss(q, ampute)) > 1e-3
    assert float(differentiable_beta_loss(q, ampute)) < 1e-3
    assert float(differentiable_r_loss(q, ampute)) < 1e-4


def test_les_poids_sont_nuls_par_defaut():
    """Sans declaration explicite, la perte est celle d'avant."""
    perte = HydroLoss(per_station=True)
    assert perte.w_r == 0.0
    assert perte.w_beta == 0.0
    assert perte.w_gamma == 0.0


def _perte(**poids):
    base = dict(w_kge=0.0, w_pbias=0.0, w_mse=0.0, w_nse=0.0, w_nrmse=0.0, w_log_nse=0.0,
                w_log_mse=0.0, w_dq=0.0, w_fdc_bas=0.0, w_dq_log=0.0, w_peak=0.0,
                per_station=True)
    return HydroLoss(**{**base, **poids})


def test_la_somme_ponderee_est_derivable_et_non_nulle():
    q = _serie().unsqueeze(1)
    sim = (q * 1.15).clone().requires_grad_(True)
    mk = torch.ones(1, dtype=torch.bool)
    f = _perte(w_r=1.0, w_beta=1.0, w_gamma=1.0)
    out = f(q_obs=q, q_sim=sim, station_mask=mk)
    val = out[0] if isinstance(out, tuple) else out
    assert float(val) > 0.0
    val.backward()
    assert sim.grad is not None and torch.isfinite(sim.grad).all()


def test_les_trois_facteurs_valent_le_kge_entier_sur_un_cas_simple():
    """Somme des trois carres et distance du KGE mesurent la meme chose, a la racine pres."""
    q = _serie()
    sim = q * 1.1
    r = float(differentiable_r_loss(q, sim))
    b = float(differentiable_beta_loss(q, sim))
    g = float(differentiable_gamma_loss(q, sim))
    attendu = (r ** 2 + b ** 2 + g ** 2) ** 0.5
    assert float(differentiable_kge_loss(q, sim)) == pytest.approx(attendu, rel=1e-3)


def test_les_composantes_sont_exposees_au_journal():
    q = _serie().unsqueeze(1)
    sim = q * 1.15
    mk = torch.ones(1, dtype=torch.bool)
    out = _perte(w_r=1.0, w_gamma=1.0)(q_obs=q, q_sim=sim, station_mask=mk)
    comps = out[1] if isinstance(out, tuple) and len(out) > 1 else {}
    for cle in ("r_loss", "beta_loss", "gamma_loss"):
        assert cle in comps, f"{cle} doit apparaitre dans la decomposition imprimee"
