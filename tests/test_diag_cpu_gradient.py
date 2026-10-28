"""Les diagnostics accumulés sur le processeur doivent rester dérivables.

`MEANDRE_DIAG_CPU=1` déplace les diagnostics sur le processeur pour tenir dans les 8 Go d'une
carte portable. L'implémentation d'origine les DÉTACHAIT au passage, au motif écrit dans le
code que les diagnostics ne sont jamais dérivés. C'était exact quand la variable a été
introduite ; ce ne l'est plus. Quatre termes de perte dérivent aujourd'hui de diagnostics :
l'évapotranspiration MODIS de `etr`, la neige de `swe`, les niveaux de puits de `s_gw` ou de
`profondeur_nappe`, et la gravimétrie de `theta1`, `theta2`, `theta3`, `swe` et `s_gw`.

Détachés, ces quatre termes gardent une VALEUR, qui s'affiche dans le journal, et ne produisent
aucun gradient. Deux entraînements identiques, l'un avec la contrainte et l'autre sans, rendent
alors le même résultat à quatre décimales, ce qui est le symptôme qui a conduit ici le
2026-09-20 sur le Saint-Laurent nord-ouest.
"""
import torch

from meandre.model import HydroModel


def _diagnostic_list_factory(on_cpu: bool, monkeypatch):
    """Construit la liste d'accumulation telle que le modèle la construit."""
    monkeypatch.setenv("MEANDRE_DIAG_CPU", "1" if on_cpu else "0")
    return HydroModel._diagnostic_list_factory()


DERIVEES = ("etr", "swe", "theta1", "theta2", "theta3", "s_gw", "profondeur_nappe_m")


def test_derived_diagnostics_keep_their_gradient(monkeypatch):
    fabrique = _diagnostic_list_factory(True, monkeypatch)
    for field_name in DERIVEES:
        source = torch.ones(3, requires_grad=True)
        acc = fabrique(field_name)
        acc.append(source * 2.0)
        empile = torch.stack(acc, dim=0)
        assert empile.requires_grad, f"{field_name} : le diagnostic est détaché, sa perte ne peut rien"
        empile.sum().backward()
        assert source.grad is not None and torch.all(source.grad == 2.0)


def test_underived_diagnostic_is_detached(monkeypatch):
    """Le reste doit rester détaché, sans quoi la mémoire de la carte n'est pas économisée."""
    fabrique = _diagnostic_list_factory(True, monkeypatch)
    source = torch.ones(3, requires_grad=True)
    acc = fabrique("canopy")
    acc.append(source * 2.0)
    assert not torch.stack(acc, dim=0).requires_grad


def test_without_variable_nothing_is_detached(monkeypatch):
    fabrique = _diagnostic_list_factory(False, monkeypatch)
    source = torch.ones(3, requires_grad=True)
    for field_name in ("canopy", "etr"):
        acc = fabrique(field_name)
        acc.append(source * 2.0)
        assert torch.stack(acc, dim=0).requires_grad
