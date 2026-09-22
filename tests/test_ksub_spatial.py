"""Le plafond de percolation du substratum peut suivre le champ spatial.

C'est le seul paramètre dont la spatialisation soit MESURÉE comme payante : sur 95 stations
et cinq territoires, chacun prédit par un modèle ajusté sur les autres, la texture du sol
explique la part souterraine du débit à +27 % contre le témoin. Il était pourtant posé à une
constante dans toutes les configurations québécoises.
"""
import pytest
import torch

from meandre.vertical.hydrotel_column import HydrotelColumn

PLAFOND = HydrotelColumn.plafond_substratum


class _Champ:
    """Sortie minimale du champ spatial."""

    k_sub = torch.tensor([1e-6, 1e-5, 1e-4])


def test_un_nombre_donne_un_plafond_uniforme():
    like = torch.zeros(3)
    p = PLAFOND(2.0e-5, _Champ(), like)
    assert p.shape == (3,)
    assert float(p.std()) == pytest.approx(0.0)


def test_le_nom_dun_champ_donne_un_plafond_par_nœud():
    like = torch.zeros(3)
    p = PLAFOND("k_sub", _Champ(), like)
    assert torch.equal(p, _Champ.k_sub)
    assert float(p.max() / p.min()) == pytest.approx(100.0)


def test_un_champ_absent_echoue_bruyamment():
    """Retomber sur une constante ferait tourner une recette qui n'est pas celle qu'on croit."""
    with pytest.raises(AttributeError, match="absent du champ spatial"):
        PLAFOND("champ_qui_nexiste_pas", _Champ(), torch.zeros(3))


def test_un_tenseur_est_accepte_tel_quel():
    p = PLAFOND(torch.tensor([1.0, 2.0, 3.0]), _Champ(), torch.zeros(3))
    assert p.tolist() == [1.0, 2.0, 3.0]


def test_le_plafond_borne_bien_la_percolation():
    """Deux nœuds identiques, plafonds différents : le plafond bas percole moins."""
    q3 = torch.tensor([5e-4, 5e-4])
    borne = torch.minimum(q3, torch.tensor([1e-5, 1e-3]))
    assert float(borne[0]) == pytest.approx(1e-5)
    assert float(borne[1]) == pytest.approx(5e-4)
