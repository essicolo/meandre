"""Verrouille les proprietes de l'ilr AVANT tout branchement (chantier 1.1).

Ecrit le jour ou la faute a ete trouvee (rang 2 sur les trois colonnes granulometriques
normalisees d'OUTV : la contrainte de somme survit a la normalisation) pour que le
module soit teste le jour ou on le branche, pas la veille d'un entrainement.
"""
import math

import torch

from meandre.spatial.compositions import ilr, remplacement_zeros


def test_zeros_remplaces_somme_preservee():
    p = torch.tensor([[0.7, 0.3, 0.0], [0.0, 0.0, 1.0]])
    r = remplacement_zeros(p, delta=0.005)
    assert torch.all(r > 0)
    assert torch.allclose(r.sum(dim=1), torch.ones(2), atol=1e-6)


def test_ilr_dimensions_et_rang():
    torch.manual_seed(0)
    p = torch.rand(500, 3)
    z = ilr(p)
    assert z.shape == (500, 2)
    # les coordonnees ilr sont de plein rang : la degenerescence du simplexe a disparu
    r = torch.linalg.matrix_rank(z - z.mean(0))
    assert int(r) == 2


def test_ilr_invariance_d_echelle():
    """Multiplier toutes les parts par une constante ne change pas les coordonnees :
    c'est la propriete d'Aitchison qui fait que des pourcentages mal fermes (sommes a
    98 ou 102) donnent le meme resultat que des fractions exactes."""
    p = torch.tensor([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]])
    assert torch.allclose(ilr(p), ilr(p * 37.0), atol=1e-5)


def test_ilr_balance_interpretable():
    """Premiere coordonnee = balance sable contre limon : elle doit croitre avec le
    rapport sable/limon a argile fixee."""
    p = torch.tensor([[0.5, 0.3, 0.2], [0.6, 0.2, 0.2]])
    z = ilr(p)
    assert z[1, 0] > z[0, 0]
