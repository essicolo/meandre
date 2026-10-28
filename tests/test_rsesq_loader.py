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


def test_monthly_grid_covers_exactly_the_period():
    c = read_rsesq("outv", TIMES)
    assert len(c.months) == 300
    assert c.level.shape == (300, c.n_wells)
    assert c.mask.shape == c.level.shape


def test_admissibility_filter_removes_wells():
    """Les puits captifs et influencés doivent disparaître, donc le compte doit baisser."""
    tout = read_rsesq("outv", TIMES, unconfined_only=False, uninfluenced=False)
    free = read_rsesq("outv", TIMES)
    assert free.n_wells < tout.n_wells


def test_no_well_below_minimum_months():
    c = read_rsesq("slso", TIMES, min_months=60)
    assert (c.mask.sum(axis=0) >= 60).all()


def test_matched_nodes_are_valid_indices():
    c = read_rsesq("gasp", TIMES)
    assert c.node_idx.dtype == np.int64
    assert (c.node_idx >= 0).all()
    assert len(c.node_idx) == c.n_wells


def test_monthly_index_is_increasing_and_complete():
    idx = index_monthly(TIMES)
    assert idx[0] == 0
    assert idx[-1] == 299
    assert (np.diff(idx) >= 0).all()
