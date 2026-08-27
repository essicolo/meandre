"""Le gel depend des proprietes THERMIQUES du sol, apprises par noeud.

Motif (Essi, 2026-08-27 : « le gel doit etre physique, en fonction de la temperature et
de la diffusion de la chaleur dans le sol, ce qui est evidemment un champ NeRF rendu
identifiable avec des variables auxiliaires »). Rankinen portait trois SCALAIRES
GLOBAUX identiques sur les 25 656 troncons de la province, et sa propre docstring le
disait (« KT/CS/CIce uniformes donc le type n'entre pas »).

UNE diffusivite, pas un couple conductivite/capacite. Le premier essai exposait les deux,
et le champ a exploite la redondance : apres deux epochs elles etaient anti-correlees a
-0.920 d'un troncon a l'autre, ce qui est thermodynamiquement impossible puisque les deux
croissent avec la teneur en eau. La relaxation ne depend du sol que par leur RAPPORT ;
donner deux sorties pour un seul degre de liberte effectif invitait a les compenser.

Ce que ces tests verrouillent :
  1. l'ABSENCE des champs rend EXACTEMENT le clone C++ (fidelite preservee) ;
  2. deux sols de proprietes differentes gelent DIFFEREMMENT ;
  3. le SENS est physique -- plus diffusif, front plus profond ; plus d'amortissement
     nival, front moins profond ;
  4. le gradient traverse les deux, sans quoi ils ne seraient pas identifiables.
"""
import torch

from hydrotel_clone.frost import Rankinen, n_intervalles


def _profil(mod, n, tair=-15.0, neige=0.2, jours=60):
    """Fait descendre un front de gel sur `jours` a temperature constante."""
    tmin = torch.full((n,), tair - 3.0)
    tmax = torch.full((n,), tair + 3.0)
    hs = torch.full((n,), float(neige))
    nd = n_intervalles(3.2, mod.dz)
    p = mod.init_profil(tmin, tmax, hs, nd)
    z = torch.full((n,), 1.0)
    return tmin, tmax, hs, p, z, nd


def _descendre(mod, n, jours=60, **kw):
    tmin, tmax, hs, p, z, _ = _profil(mod, n)
    gel = None
    for _ in range(jours):
        p, gel = mod(tmin, tmax, hs, p, z, z, z, **kw)
    return gel


def test_sans_champs_le_clone_est_identique():
    """Regle de tout processus opt-in du projet : le defaut reste le clone fidele."""
    mod = Rankinen()
    a = _descendre(mod, 3)
    b = _descendre(mod, 3, alpha=None, fs=None)
    assert torch.equal(a, b)


def test_deux_sols_differents_gelent_differemment():
    """Le point de la correction : avec des scalaires uniformes, tous les troncons
    gelaient pareil, ce qui rendait le gel aveugle a la texture et a l'humidite."""
    mod = Rankinen()
    al = torch.tensor([0.6e-7, 1.6e-7, 4.0e-7])   # inerte -> reactif
    gel = _descendre(mod, 3, alpha=al)
    assert float(gel.max() - gel.min()) > 1.0, gel


def test_sens_physique_des_proprietes():
    """Un parametre qui bouge dans le mauvais sens serait pire qu'un scalaire fige :
    il donnerait au champ un levier qui compense au lieu de representer."""
    mod = Rankinen()
    n = 2
    # diffusivite : plus diffusif, le froid penetre plus vite
    g = _descendre(mod, n, alpha=torch.tensor([0.6e-7, 4.0e-7]))
    assert float(g[1]) > float(g[0]), g
    # amortissement nival : plus d'amortissement, le sol est mieux isole de l'air froid
    g = _descendre(mod, n, fs=torch.tensor([0.8, 5.0]))
    assert float(g[1]) < float(g[0]), g


def test_gradient_traverse_les_deux():
    """Un parametre dont le gradient ne passe pas n'est pas identifiable, et c'est
    l'identifiabilite qui est la promesse du projet."""
    mod = Rankinen()
    for nom, val in (("alpha", 1.6e-7), ("fs", 2.35)):
        x = torch.full((2,), val, requires_grad=True)
        tmin, tmax, hs, p, z, _ = _profil(mod, 2)
        for _ in range(20):
            p, gel = mod(tmin, tmax, hs, p, z, z, z, **{nom: x})
        gel.sum().backward()
        assert x.grad is not None and float(x.grad.abs().sum()) > 0.0, nom


def test_les_bornes_respectent_la_stabilite_du_schema():
    """La borne haute de la diffusivite est fixee par la NUMERIQUE, pas par la physique.

    Rankinen est un schema EXPLICITE : son taux de relaxation vaut dt*alpha/(2z)^2, donc
    8.64e6*alpha au noeud le plus superficiel (5 cm, pas journalier). Au-dela d'un taux
    de 2, le profil de temperature oscille en divergeant et le gel sort en NaN -- constate
    le 2026-08-27 avec une borne a 8e-7 empruntee a la litterature des sols, qui donne un
    taux de 6.9.

    Ce test existe parce que c'est exactement le genre de contrainte qu'on reintroduit
    sans y penser en elargissant une borne pour « laisser plus de liberte au champ ».
    """
    from meandre.spatial.field_network import SpatialFieldNetwork

    mod = Rankinen()
    n = SpatialFieldNetwork(n_territorial=8, n_nodes=3)
    n.init_from_literature()
    sp = n(torch.tensor([[-71.0, 46.0]] * 3), torch.zeros(3, 8))
    alpha_max = 2.0 * (2.0 * mod.dz) ** 2 / mod.dt
    assert float(sp.diff_gel.max()) < alpha_max, (float(sp.diff_gel.max()), alpha_max)

    # et le gel reste fini sur une descente complete
    gel = _descendre(mod, 3, alpha=sp.diff_gel.detach())
    assert torch.isfinite(gel).all(), gel
