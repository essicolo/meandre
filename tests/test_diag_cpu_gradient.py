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


def _liste_diagnostics(sur_processeur: bool, monkeypatch):
    """Construit la liste d'accumulation telle que le modèle la construit."""
    monkeypatch.setenv("MEANDRE_DIAG_CPU", "1" if sur_processeur else "0")
    return HydroModel._liste_diagnostics()


DERIVEES = ("etr", "swe", "theta1", "theta2", "theta3", "s_gw", "profondeur_nappe_m")


def test_les_diagnostics_derives_gardent_leur_gradient(monkeypatch):
    fabrique = _liste_diagnostics(True, monkeypatch)
    for nom in DERIVEES:
        source = torch.ones(3, requires_grad=True)
        acc = fabrique(nom)
        acc.append(source * 2.0)
        empile = torch.stack(acc, dim=0)
        assert empile.requires_grad, f"{nom} : le diagnostic est détaché, sa perte ne peut rien"
        empile.sum().backward()
        assert source.grad is not None and torch.all(source.grad == 2.0)


def test_un_diagnostic_non_derive_est_bien_detache(monkeypatch):
    """Le reste doit rester détaché, sans quoi la mémoire de la carte n'est pas économisée."""
    fabrique = _liste_diagnostics(True, monkeypatch)
    source = torch.ones(3, requires_grad=True)
    acc = fabrique("canopy")
    acc.append(source * 2.0)
    assert not torch.stack(acc, dim=0).requires_grad


def test_sans_la_variable_rien_nest_detache(monkeypatch):
    fabrique = _liste_diagnostics(False, monkeypatch)
    source = torch.ones(3, requires_grad=True)
    for nom in ("canopy", "etr"):
        acc = fabrique(nom)
        acc.append(source * 2.0)
        assert torch.stack(acc, dim=0).requires_grad
