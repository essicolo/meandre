"""Les multiplicateurs de champ, qui rendent la sensibilité mesurable.

Un facteur par champ du réseau spatial, valant un par défaut, donne une dérivée SANS
DIMENSION : la réponse d'un observable à une variation relative uniforme du champ. C'est ce
qui permet de comparer la sensibilité d'une conductivité en mètres par heure et celle d'un
seuil de fonte en degrés. Le mécanisme doit être rigoureusement neutre quand il n'est pas
posé, faute de quoi tout entraînement en hériterait.
"""
import dataclasses

import pytest
import torch

from meandre.spatial.field_network import SpatialFieldNetwork


class _Faux(SpatialFieldNetwork):
    """Réseau minimal : on n'exerce que l'application des multiplicateurs."""

    def __init__(self):
        pass


def _params(n=3):
    from meandre.spatial.field_network import SpatialParams

    field_names = {f.name: torch.full((n,), 2.0) for f in dataclasses.fields(SpatialParams)}
    return SpatialParams(**field_names)


def test_without_multiplier_field_is_intact():
    r, sp = _Faux(), _params()
    step_output = r._apply_multipliers(sp)
    assert step_output is sp, "aucune copie ne doit etre faite quand rien n'est pose"


def test_multiplier_acts_only_on_target_field():
    r, sp = _Faux(), _params()
    r.multipliers = {"K_sat_1": torch.tensor(3.0)}
    step_output = r._apply_multipliers(sp)
    assert float(step_output.K_sat_1[0]) == pytest.approx(6.0)
    assert float(step_output.K_sat_2[0]) == pytest.approx(2.0)


def test_factor_one_changes_nothing():
    r, sp = _Faux(), _params()
    r.multipliers = {"K_sat_1": torch.tensor(1.0)}
    assert float(r._apply_multipliers(sp).K_sat_1[0]) == pytest.approx(2.0)


def test_gradient_reaches_the_factor():
    """Sans cela le banc de sensibilite mesurerait zero partout."""
    r, sp = _Faux(), _params()
    m = torch.ones((), requires_grad=True)
    r.multipliers = {"porosity_1": m}
    step_output = r._apply_multipliers(sp)
    step_output.porosity_1.sum().backward()
    assert float(m.grad) == pytest.approx(6.0), "d(somme)/d(facteur) = somme du champ"


def test_nonexistent_field_fails_loudly():
    r, sp = _Faux(), _params()
    r.multipliers = {"champ_invente": torch.tensor(1.0)}
    with pytest.raises(AttributeError, match="absent du champ spatial"):
        r._apply_multipliers(sp)


def test_empty_dict_is_neutral():
    r, sp = _Faux(), _params()
    r.multipliers = {}
    assert r._apply_multipliers(sp) is sp
