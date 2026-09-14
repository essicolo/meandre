"""Tests de la tete de regression quantile.

Ecrits le 2026-09-11, APRES que deux defauts soient partis en production faute de test.

DEFAUT 1. Les ecarts cumules etaient appliques ADDITIVEMENT au debit, si bien que le
cinquieme centile predit devenait negatif : mesure sur cinq regions, il l'etait dans 21 a
44 % des pas de temps. Un debit negatif est impossible, donc aucune observation ne pouvait
tomber sous cette borne et la classe inferieure du diagramme de Talagrand etait vide PAR
CONSTRUCTION, ce qui donnait l'illusion d'un modele bien centre.

DEFAUT 2. Le correctif qui a suivi a applique les memes largeurs dans le LOGARITHME du
debit sans les remettre a l'echelle, alors qu'elles etaient calibrees en metres cubes par
seconde. Le cumul atteignait sa borne : quatre-vingt-quinzieme centile a 10^13 fois la
mediane, cinquieme centile numeriquement nul, couverture de 96 % pour un intervalle
annonce a 90.

Les deux defauts se voient en dix secondes sur des tenseurs quelconques. Les tests
ci-dessous les auraient arretes avant tout entrainement.
"""
import os

import pytest
import torch

from meandre.utils.quantile_head import QuantileHead

TAUS = (0.05, 0.10, 0.25, 0.75, 0.90, 0.95)


def _tete(graine=0, n_spatial=8):
    torch.manual_seed(graine)
    return QuantileHead(n_spatial_params=n_spatial, taus=TAUS)


def _quantiles(h, n_noeuds=6, n_pas=200, echelle=20.0, graine=1):
    torch.manual_seed(graine)
    sp = torch.randn(n_noeuds, 8)
    Q = torch.rand(n_pas, n_noeuds) * echelle + 0.05
    with torch.no_grad():
        return Q, Q.unsqueeze(-1) + h(sp, Q)


def test_quantiles_strictement_positifs():
    """DEFAUT 1 : un debit predit ne peut pas etre negatif."""
    for g in range(5):
        Q, q = _quantiles(_tete(g), graine=g)
        assert (q > 0).all(), f"quantile negatif ou nul (graine {g}), min {q.min():.3e}"


def test_quantiles_monotones():
    """Les niveaux ne doivent jamais se croiser."""
    for g in range(5):
        _, q = _quantiles(_tete(g), graine=g)
        assert (q.diff(dim=-1) > 0).all(), f"quantiles croises (graine {g})"


def test_enveloppe_plausible_a_l_initialisation():
    """DEFAUT 2 : l'enveloppe de depart doit etre du bon ordre de grandeur.

    Bornes volontairement larges : on n'attend pas une calibration a l'initialisation,
    seulement qu'un hydrologue puisse regarder la figure sans sursauter.
    """
    for g in range(5):
        Q, q = _quantiles(_tete(g), graine=g)
        r = q / Q.unsqueeze(-1)
        r05, r95 = r[..., 0].median().item(), r[..., -1].median().item()
        assert 0.05 < r05 < 0.98, f"cinquieme centile a {r05:.3e} fois la mediane (graine {g})"
        assert 1.02 < r95 < 20.0, f"quatre-vingt-quinzieme centile a {r95:.3e} fois la mediane (graine {g})"


def test_enveloppe_bornee_meme_sur_des_sorties_extremes():
    """Aucune entree ne doit pouvoir faire exploser l'enveloppe."""
    h = _tete()
    with torch.no_grad():
        for p in h.net.parameters():
            p.mul_(50.0)
    Q = torch.rand(50, 4) * 10 + 0.1
    with torch.no_grad():
        q = Q.unsqueeze(-1) + h(torch.randn(4, 8) * 10, Q)
    r = q / Q.unsqueeze(-1)
    assert (q > 0).all(), "quantile negatif avec des poids extremes"
    assert r.max().item() < 100.0, f"enveloppe explosive : rapport maximal {r.max().item():.3e}"


def test_enveloppe_proportionnelle_au_debit():
    """La largeur suit le debit : c'est ce que veut dire multiplicatif."""
    h = _tete()
    torch.manual_seed(3)
    sp = torch.randn(4, 8)
    for facteur in (1.0, 10.0, 100.0):
        Q = torch.full((30, 4), facteur)
        with torch.no_grad():
            r = (Q.unsqueeze(-1) + h(sp, Q)) / Q.unsqueeze(-1)
        if facteur == 1.0:
            ref = r.median(dim=0).values
        else:
            # la pente en log(Q) fait deriver le rapport, mais il reste du meme ordre
            assert torch.allclose(r.median(dim=0).values, ref, rtol=3.0), \
                "le rapport quantile/mediane change d'ordre de grandeur avec le debit"


def test_forme_additive_restituee_par_la_variable():
    """Le temoin de l'ancienne forme reste disponible pour comparaison."""
    h = _tete()
    Q = torch.rand(20, 3) * 5 + 0.5
    sp = torch.randn(3, 8)
    os.environ["MEANDRE_QUANTILE_ADDITIF"] = "1"
    try:
        with torch.no_grad():
            off_add = h(sp, Q)
    finally:
        del os.environ["MEANDRE_QUANTILE_ADDITIF"]
    with torch.no_grad():
        off_mul = h(sp, Q)
    assert not torch.allclose(off_add, off_mul), "la variable temoin n'a aucun effet"
    assert (off_add.diff(dim=-1) > 0).all(), "forme additive non monotone"


def test_gradient_fini():
    """La perte doit pouvoir remonter dans la tete."""
    h = _tete()
    Q = torch.rand(40, 3) * 8 + 0.2
    sp = torch.randn(3, 8, requires_grad=True)
    q = Q.unsqueeze(-1) + h(sp, Q)
    q.sum().backward()
    g = torch.cat([p.grad.flatten() for p in h.parameters() if p.grad is not None])
    assert torch.isfinite(g).all(), "gradient non fini dans la tete"


def test_la_tete_apprend_la_bonne_largeur():
    """La tete doit RETROUVER les quantiles d'une log-normale dont on connait la verite.

    Ce test manquait, et son absence a coute une file complete : la parametrisation etait
    juste mais le taux d'apprentissage du socle, 5e-4 decroissant a 5e-6, est un reglage
    d'affinage du champ physique, pas d'entrainement d'une tete neuve. La tete restait
    figee sur son enveloppe de depart et rendait des couvertures de 0,13 pour un intervalle
    annonce a 0,50.
    """
    torch.manual_seed(0)
    h = _tete()
    sp = torch.randn(4, 8)
    Q = torch.rand(4000, 4) * 20 + 1.0
    sigma = 0.6
    obs = Q * torch.exp(sigma * torch.randn_like(Q))
    t7 = torch.tensor([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    opt = torch.optim.Adam(h.parameters(), lr=1e-2)
    for _ in range(500):
        off = h(sp, Q)
        q = torch.cat([Q.unsqueeze(-1) + off[..., :3], Q.unsqueeze(-1),
                       Q.unsqueeze(-1) + off[..., 3:]], dim=-1)
        d = obs.unsqueeze(-1) - q
        perte = torch.maximum(t7 * d, (t7 - 1) * d).mean()
        opt.zero_grad(); perte.backward(); opt.step()
    with torch.no_grad():
        off = h(sp, Q)
        q05 = Q + off[..., 0]
        q95 = Q + off[..., -1]
        q25 = Q + off[..., 2]
        q75 = Q + off[..., 3]
        cov90 = float(((obs > q05) & (obs < q95)).float().mean())
        cov50 = float(((obs > q25) & (obs < q75)).float().mean())
    assert abs(cov90 - 0.90) < 0.05, f"couverture a 90 % apprise : {cov90:.3f}"
    assert abs(cov50 - 0.50) < 0.05, f"couverture a 50 % apprise : {cov50:.3f}"
