"""Le partage des puits entre la perte et la validation indépendante.

Les niveaux mesurés sont la seule corroboration indépendante que la physique de drainage
profond ait reçue, et elle vaut PARCE QUE les puits n'étaient dans aucune perte. Les y mettre
tous détruirait la preuve en même temps qu'elle servirait. Le partage doit donc être stable
d'une exécution à l'autre, sans quoi la part tenue de côté change à chaque passe et plus rien
ne se compare.
"""
import pytest


def partager(wells, part):
    """Reproduit la règle du pilote : ordre stable, pas de tirage."""
    kept = []
    if 0.0 < part < 1.0 and len(wells) >= 4:
        rank = sorted(range(len(wells)), key=lambda i: str(wells[i]))
        n_val = max(1, int(round(part * len(wells))))
        kept = sorted(rank[::max(len(wells) // n_val, 1)][:n_val])
    trained = [i for i in range(len(wells)) if i not in set(kept)]
    if not trained:
        trained, kept = list(range(len(wells))), []
    return trained, kept


WELL = ["04300001", "04020001", "04630001", "04300011", "04020002", "04020005",
         "04060003", "04010003"]


def test_split_is_stable():
    assert partager(WELL, 0.5) == partager(WELL, 0.5)
    assert partager(list(reversed(WELL)), 0.5)[1] == partager(list(reversed(WELL)), 0.5)[1]


def test_both_shares_are_disjoint_and_complete():
    e, g = partager(WELL, 0.5)
    assert set(e) & set(g) == set()
    assert sorted(e + g) == list(range(len(WELL)))


def test_requested_share_is_respected():
    _e, g = partager(WELL, 0.5)
    assert len(g) == 4
    _e, g = partager(WELL, 0.25)
    assert len(g) == 2


def test_no_well_held_out_when_share_is_zero():
    e, g = partager(WELL, 0.0)
    assert g == [] and len(e) == len(WELL)


def test_poor_territory_keeps_everything_for_the_loss():
    """Sous quatre puits, tenir la moitié de côté ne laisserait presque rien."""
    e, g = partager(["a", "b", "c"], 0.5)
    assert g == [] and len(e) == 3


def test_held_out_share_independent_of_input_order():
    """Deux exécutions qui lisent les puits dans un ordre différent tiennent les mêmes."""
    _e1, g1 = partager(WELL, 0.5)
    other = sorted(WELL)
    _e2, g2 = partager(other, 0.5)
    assert {WELL[i] for i in g1} == {other[i] for i in g2}
