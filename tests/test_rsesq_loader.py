"""Chargeur des niveaux de nappe mesurés : filtres de recevabilité et grille mensuelle.

Les données vivent hors du dépôt ; le test se saute quand elles sont absentes plutôt que
d'échouer, mais il vérifie la logique dès qu'elles sont là.
"""
import os

import numpy as np
import pandas as pd
import pytest

from meandre.adapters.physitel.rsesq_loader import _path_default, index_monthly, read_rsesq

_present = os.path.exists(f"{_path_default()}/rsesq-puits.parquet")
pytestmark = pytest.mark.skipif(not _present, reason="niveaux de nappe absents de la machine")

TIMES = pd.date_range("2000-01-01", "2024-12-31", freq="D")


def test_grille_mensuelle_couvre_exactement_la_periode():
    c = read_rsesq("outv", TIMES)
    assert len(c.months) == 300
    assert c.level.shape == (300, c.n_wells)
    assert c.mask.shape == c.level.shape


def test_le_filtre_de_recevabilite_retire_des_puits():
    """Les puits captifs et influencés doivent disparaître, donc le compte doit baisser."""
    tout = read_rsesq("outv", TIMES, unconfined_only=False, uninfluenced=False)
    libres = read_rsesq("outv", TIMES)
    assert libres.n_wells < tout.n_wells


def test_aucun_puits_sous_le_minimum_de_mois():
    c = read_rsesq("slso", TIMES, min_months=60)
    assert (c.mask.sum(axis=0) >= 60).all()


def test_les_noeuds_apparies_sont_des_indices_valides():
    c = read_rsesq("gasp", TIMES)
    assert c.node_idx.dtype == np.int64
    assert (c.node_idx >= 0).all()
    assert len(c.node_idx) == c.n_wells


def test_index_mensuel_est_croissant_et_complet():
    idx = index_monthly(TIMES)
    assert idx[0] == 0
    assert idx[-1] == 299
    assert (np.diff(idx) >= 0).all()
