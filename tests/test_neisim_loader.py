"""Le chargeur NEISIM aligne bien la grille et l'axe de temps.

Les deux pièges mesurés à l'écriture : NEISIM horodate ses journées à 05 h UTC alors que les
jeux au sol sont à minuit, si bien que l'intersection des index est vide sans normalisation ;
et son champ est stocké en entiers avec facteur d'échelle, si bien que les valeurs manquantes
arrivent comme valeur de remplissage plutôt que comme absence.
"""
import numpy as np
import pandas as pd
import pytest

from meandre.data import neisim_loader as nl


def test_les_cellules_sont_les_plus_proches(monkeypatch):
    lon = np.array([-80.0, -79.0, -78.0])
    lat = np.array([45.0, 46.0])
    temps = pd.to_datetime(["2000-01-01 05:00", "2000-01-02 05:00"])
    monkeypatch.setattr(nl, "_grille", lambda f: (lon, lat, temps.normalize()))
    cellules, rang, _lo, _la, _t = nl.cellules_des_noeuds([-78.9, -80.1, -78.9], [45.9, 45.1, 45.9])
    assert rang[0] == rang[2], "deux nœuds au même endroit doivent partager la cellule"
    assert len(cellules) == 2
    proche = cellules[rang[1]]
    assert lon[proche[0]] == -80.0 and lat[proche[1]] == 45.0


def test_lhorodatage_a_cinq_heures_sintersecte_avec_minuit():
    """Sans normalisation à la date, aucun jour ne se recoupe."""
    a = pd.to_datetime(["2001-03-01 05:00", "2001-03-02 05:00"])
    b = pd.to_datetime(["2001-03-01 00:00", "2001-03-02 00:00"])
    assert len(a.intersection(b)) == 0
    assert len(a.normalize().intersection(b.normalize())) == 2


def test_les_jours_hors_periode_restent_absents(monkeypatch):
    lon, lat = np.array([-75.0]), np.array([46.0])
    temps = pd.to_datetime(["2001-01-02", "2001-01-03"])
    monkeypatch.setattr(nl, "_grille", lambda f: (lon, lat, temps))
    monkeypatch.setattr(nl, "lire_cellules", lambda c, f=None, **k: np.array([[10.0, 20.0]], dtype="float32"))
    axe = pd.date_range("2001-01-01", "2001-01-04", freq="D")
    valeurs, couverture = nl.charger_aux_noeuds([-75.0], [46.0], axe)
    assert valeurs.shape == (4, 1)
    assert np.isnan(valeurs[0, 0]) and np.isnan(valeurs[3, 0])
    assert valeurs[1, 0] == pytest.approx(10.0)
    assert valeurs[2, 0] == pytest.approx(20.0)
    assert couverture == pytest.approx(0.5)
