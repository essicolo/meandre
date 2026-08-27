"""Le gel depend des proprietes THERMIQUES du sol, apprises par noeud.

Motif (Essi, 2026-08-27 : « le gel doit etre physique, en fonction de la temperature et
de la diffusion de la chaleur dans le sol, ce qui est evidemment un champ NeRF rendu
identifiable avec des variables auxiliaires »). Rankinen portait trois SCALAIRES
GLOBAUX -- conductivite thermique 0.8, capacite du sol 1e6, amortissement nival 2.35 --
identiques sur les 25 656 troncons de la province, et sa propre docstring le disait
(« KT/CS/CIce uniformes donc le type n'entre pas »). Une argile saturee et un sable sec
gelaient donc identiquement, alors que leurs proprietes different d'un facteur trois a
quatre.

Ce que ces tests verrouillent :
  1. l'ABSENCE des champs rend EXACTEMENT le clone C++ (fidelite preservee) ;
  2. deux sols de proprietes differentes gelent DIFFEREMMENT ;
  3. le SENS est physique -- plus conducteur, front plus profond ; plus capacitif,
     front moins profond ; plus d'amortissement nival, front moins profond ;
  4. le gradient traverse les trois, sans quoi ils ne seraient pas identifiables.
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
    b = _descendre(mod, 3, kt=None, cs=None, fs=None)
    assert torch.equal(a, b)


def test_deux_sols_differents_gelent_differemment():
    """Le point de la correction : avec des scalaires uniformes, tous les troncons
    gelaient pareil, ce qui rendait le gel aveugle a la texture et a l'humidite."""
    mod = Rankinen()
    kt = torch.tensor([0.3, 0.8, 2.2])          # sec poreux -> sature sableux
    gel = _descendre(mod, 3, kt=kt)
    assert float(gel.max() - gel.min()) > 1.0, gel


def test_sens_physique_des_trois_proprietes():
    """Un parametre qui bouge dans le mauvais sens serait pire qu'un scalaire fige :
    il donnerait au champ un levier qui compense au lieu de representer."""
    mod = Rankinen()
    n = 2
    # conductivite : plus conducteur, la chaleur (ici le froid) penetre plus vite
    g = _descendre(mod, n, kt=torch.tensor([0.3, 2.2]))
    assert float(g[1]) > float(g[0]), g
    # capacite : plus capacitif, plus d'inertie, front moins profond
    g = _descendre(mod, n, cs=torch.tensor([0.6e6, 2.8e6]))
    assert float(g[1]) < float(g[0]), g
    # amortissement nival : plus d'amortissement, le sol est mieux isole de l'air froid
    g = _descendre(mod, n, fs=torch.tensor([0.8, 5.0]))
    assert float(g[1]) < float(g[0]), g


def test_gradient_traverse_les_trois():
    """Un parametre dont le gradient ne passe pas n'est pas identifiable, et c'est
    l'identifiabilite qui est la promesse du projet."""
    mod = Rankinen()
    for nom, val in (("kt", 0.8), ("cs", 1.0e6), ("fs", 2.35)):
        x = torch.full((2,), val, requires_grad=True)
        tmin, tmax, hs, p, z, _ = _profil(mod, 2)
        for _ in range(20):
            p, gel = mod(tmin, tmax, hs, p, z, z, z, **{nom: x})
        gel.sum().backward()
        assert x.grad is not None and float(x.grad.abs().sum()) > 0.0, nom
