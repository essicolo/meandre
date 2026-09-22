"""Le partage des puits entre la perte et la validation indépendante.

Les niveaux mesurés sont la seule corroboration indépendante que la physique de drainage
profond ait reçue, et elle vaut PARCE QUE les puits n'étaient dans aucune perte. Les y mettre
tous détruirait la preuve en même temps qu'elle servirait. Le partage doit donc être stable
d'une exécution à l'autre, sans quoi la part tenue de côté change à chaque passe et plus rien
ne se compare.
"""
import pytest


def partager(puits, part):
    """Reproduit la règle du pilote : ordre stable, pas de tirage."""
    garde = []
    if 0.0 < part < 1.0 and len(puits) >= 4:
        rang = sorted(range(len(puits)), key=lambda i: str(puits[i]))
        n_val = max(1, int(round(part * len(puits))))
        garde = sorted(rang[::max(len(puits) // n_val, 1)][:n_val])
    entraine = [i for i in range(len(puits)) if i not in set(garde)]
    if not entraine:
        entraine, garde = list(range(len(puits))), []
    return entraine, garde


PUITS = ["04300001", "04020001", "04630001", "04300011", "04020002", "04020005",
         "04060003", "04010003"]


def test_le_partage_est_stable():
    assert partager(PUITS, 0.5) == partager(PUITS, 0.5)
    assert partager(list(reversed(PUITS)), 0.5)[1] == partager(list(reversed(PUITS)), 0.5)[1]


def test_les_deux_parts_sont_disjointes_et_completes():
    e, g = partager(PUITS, 0.5)
    assert set(e) & set(g) == set()
    assert sorted(e + g) == list(range(len(PUITS)))


def test_la_part_demandee_est_respectee():
    _e, g = partager(PUITS, 0.5)
    assert len(g) == 4
    _e, g = partager(PUITS, 0.25)
    assert len(g) == 2


def test_aucun_puits_tenu_de_cote_si_la_part_est_nulle():
    e, g = partager(PUITS, 0.0)
    assert g == [] and len(e) == len(PUITS)


def test_un_territoire_pauvre_garde_tout_pour_la_perte():
    """Sous quatre puits, tenir la moitié de côté ne laisserait presque rien."""
    e, g = partager(["a", "b", "c"], 0.5)
    assert g == [] and len(e) == 3


def test_la_part_tenue_de_cote_ne_depend_pas_de_lordre_dentree():
    """Deux exécutions qui lisent les puits dans un ordre différent tiennent les mêmes."""
    _e1, g1 = partager(PUITS, 0.5)
    autre = sorted(PUITS)
    _e2, g2 = partager(autre, 0.5)
    assert {PUITS[i] for i in g1} == {autre[i] for i in g2}
