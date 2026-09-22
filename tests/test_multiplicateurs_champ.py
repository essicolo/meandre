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

    champs = {f.name: torch.full((n,), 2.0) for f in dataclasses.fields(SpatialParams)}
    return SpatialParams(**champs)


def test_sans_multiplicateur_le_champ_est_intact():
    r, sp = _Faux(), _params()
    sortie = r._applique_multiplicateurs(sp)
    assert sortie is sp, "aucune copie ne doit etre faite quand rien n'est pose"


def test_un_multiplicateur_agit_sur_le_seul_champ_vise():
    r, sp = _Faux(), _params()
    r.multiplicateurs = {"K_sat_1": torch.tensor(3.0)}
    sortie = r._applique_multiplicateurs(sp)
    assert float(sortie.K_sat_1[0]) == pytest.approx(6.0)
    assert float(sortie.K_sat_2[0]) == pytest.approx(2.0)


def test_le_facteur_un_ne_change_rien():
    r, sp = _Faux(), _params()
    r.multiplicateurs = {"K_sat_1": torch.tensor(1.0)}
    assert float(r._applique_multiplicateurs(sp).K_sat_1[0]) == pytest.approx(2.0)


def test_le_gradient_remonte_jusqu_au_facteur():
    """Sans cela le banc de sensibilite mesurerait zero partout."""
    r, sp = _Faux(), _params()
    m = torch.ones((), requires_grad=True)
    r.multiplicateurs = {"porosity_1": m}
    sortie = r._applique_multiplicateurs(sp)
    sortie.porosity_1.sum().backward()
    assert float(m.grad) == pytest.approx(6.0), "d(somme)/d(facteur) = somme du champ"


def test_un_champ_inexistant_echoue_bruyamment():
    r, sp = _Faux(), _params()
    r.multiplicateurs = {"champ_invente": torch.tensor(1.0)}
    with pytest.raises(AttributeError, match="absent du champ spatial"):
        r._applique_multiplicateurs(sp)


def test_un_dictionnaire_vide_est_neutre():
    r, sp = _Faux(), _params()
    r.multiplicateurs = {}
    assert r._applique_multiplicateurs(sp) is sp
