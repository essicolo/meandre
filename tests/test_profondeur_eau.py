"""La profondeur d'eau, seul endroit du modèle où le coefficient de Manning agit.

Le modèle n'avait aucune géométrie de lit et ne produisait que des débits. L'altimétrie
satellitaire, elle, mesure une élévation de surface libre : sans profondeur, elle est
inutilisable, et le coefficient de Manning sortait du champ spatial sans que rien ne le lise.

Manning en section large, où le rayon hydraulique se confond avec la profondeur :
h = (Q n / (w racine(pente)))^(3/5).
"""
import pytest
import torch

from meandre.model import HydroModel

PROFONDEUR = HydroModel._profondeur_deau


class _Graphe:
    def __init__(self, w=None, s=None):
        self.reach_width_m = w
        self.node_slope = s


class _Champ:
    def __init__(self, n=0.035, taille=2):
        self.manning_n = torch.full((taille,), n)


def test_sans_largeur_rien_nest_produit():
    """Une profondeur inventee serait pire qu'une absence."""
    assert PROFONDEUR(torch.ones(3, 2), _Graphe(None, torch.full((2,), 0.01)), _Champ()) is None


def test_sans_pente_rien_nest_produit():
    assert PROFONDEUR(torch.ones(3, 2), _Graphe(torch.full((2,), 50.0), None), _Champ()) is None


def test_la_profondeur_croit_avec_le_debit():
    g = _Graphe(torch.full((2,), 50.0), torch.full((2,), 0.001))
    h = PROFONDEUR(torch.tensor([[10.0, 100.0]]), g, _Champ())
    assert float(h[0, 1]) > float(h[0, 0])


def test_lexposant_vaut_trois_cinquiemes():
    """Un debit multiplie par dix doit multiplier la profondeur par dix puissance 0,6."""
    g = _Graphe(torch.full((2,), 50.0), torch.full((2,), 0.001))
    h = PROFONDEUR(torch.tensor([[10.0, 100.0]]), g, _Champ())
    assert float(h[0, 1] / h[0, 0]) == pytest.approx(10.0 ** 0.6, rel=1e-4)


def test_une_riviere_plus_large_est_moins_profonde():
    g = _Graphe(torch.tensor([20.0, 200.0]), torch.full((2,), 0.001))
    h = PROFONDEUR(torch.tensor([[100.0, 100.0]]), g, _Champ())
    assert float(h[0, 0]) > float(h[0, 1])


def test_une_pente_plus_forte_donne_moins_de_profondeur():
    g = _Graphe(torch.full((2,), 50.0), torch.tensor([0.0001, 0.01]))
    h = PROFONDEUR(torch.tensor([[100.0, 100.0]]), g, _Champ())
    assert float(h[0, 0]) > float(h[0, 1])


def test_lordre_de_grandeur_est_hydraulique():
    """Cent metres cubes par seconde sur cinquante metres de large, pente d'un pour mille."""
    g = _Graphe(torch.full((2,), 50.0), torch.full((2,), 0.001))
    h = float(PROFONDEUR(torch.tensor([[100.0, 100.0]]), g, _Champ())[0, 0])
    assert 0.5 < h < 5.0, f"profondeur hors plage hydraulique : {h:.2f} m"


def test_le_gradient_atteint_le_coefficient_de_manning():
    """C'est tout l'interet : rendre Manning identifiable par l'altimetrie."""
    n = torch.full((2,), 0.035, requires_grad=True)

    class _C:
        manning_n = n

    g = _Graphe(torch.full((2,), 50.0), torch.full((2,), 0.001))
    PROFONDEUR(torch.tensor([[100.0, 100.0]]), g, _C()).sum().backward()
    assert float(n.grad.abs().sum()) > 0
