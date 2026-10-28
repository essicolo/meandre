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

DEPTH = HydroModel._water_depth


class _Graph:
    def __init__(self, w=None, s=None):
        self.reach_width_m = w
        self.node_slope = s


class _Field:
    def __init__(self, n=0.035, size=2):
        self.manning_n = torch.full((size,), n)


def test_without_width_nothing_is_produced():
    """Une profondeur inventee serait pire qu'une absence."""
    assert DEPTH(torch.ones(3, 2), _Graph(None, torch.full((2,), 0.01)), _Field()) is None


def test_without_slope_nothing_is_produced():
    assert DEPTH(torch.ones(3, 2), _Graph(torch.full((2,), 50.0), None), _Field()) is None


def test_depth_grows_with_discharge():
    g = _Graph(torch.full((2,), 50.0), torch.full((2,), 0.001))
    h = DEPTH(torch.tensor([[10.0, 100.0]]), g, _Field())
    assert float(h[0, 1]) > float(h[0, 0])


def test_exponent_is_three_fifths():
    """Un debit multiplie par dix doit multiplier la profondeur par dix puissance 0,6."""
    g = _Graph(torch.full((2,), 50.0), torch.full((2,), 0.001))
    h = DEPTH(torch.tensor([[10.0, 100.0]]), g, _Field())
    assert float(h[0, 1] / h[0, 0]) == pytest.approx(10.0 ** 0.6, rel=1e-4)


def test_wider_river_is_shallower():
    g = _Graph(torch.tensor([20.0, 200.0]), torch.full((2,), 0.001))
    h = DEPTH(torch.tensor([[100.0, 100.0]]), g, _Field())
    assert float(h[0, 0]) > float(h[0, 1])


def test_steeper_slope_gives_less_depth():
    g = _Graph(torch.full((2,), 50.0), torch.tensor([0.0001, 0.01]))
    h = DEPTH(torch.tensor([[100.0, 100.0]]), g, _Field())
    assert float(h[0, 0]) > float(h[0, 1])


def test_order_of_magnitude_is_hydraulic():
    """Cent metres cubes par seconde sur cinquante metres de large, pente d'un pour mille."""
    g = _Graph(torch.full((2,), 50.0), torch.full((2,), 0.001))
    h = float(DEPTH(torch.tensor([[100.0, 100.0]]), g, _Field())[0, 0])
    assert 0.5 < h < 5.0, f"profondeur hors plage hydraulique : {h:.2f} m"


def test_gradient_reaches_manning_coefficient():
    """C'est tout l'interet : rendre Manning identifiable par l'altimetrie."""
    n = torch.full((2,), 0.035, requires_grad=True)

    class _C:
        manning_n = n

    g = _Graph(torch.full((2,), 50.0), torch.full((2,), 0.001))
    DEPTH(torch.tensor([[100.0, 100.0]]), g, _C()).sum().backward()
    assert float(n.grad.abs().sum()) > 0
