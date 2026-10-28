"""L'exposant de vidange mesuré sur les hydrogrammes d'un territoire.

Il a été mesuré le 2026-09-20 qu'aucune covariable de terrain ne prédit cet exposant : texture
à −8 pour cent contre le témoin, socle à −22, relief à −26. Il ne sort donc pas du champ
spatial. Mais il se mesure sur les débits observés, sans simulation et sans circularité, et
entre par la loi des ancrages comme l'évapotranspiration de Linacre et les taux de fonte.
"""
import numpy as np
import pytest

from meandre.data.recession_anchor import (B_MAX, N_DEFAULT, N_MAX, anchor_result, recession_exponent,
                                           recession_segments)


def _drainage(n, day_count=3000, fraction=0.002, s0=100.0, sub_step=20, seed=0):
    """Hydrogramme synthétique d'un réservoir Q = c·S^n, rechargé par impulsions."""
    rng = np.random.default_rng(seed)
    c = fraction * s0 ** (1.0 - n)
    s, dt, q = s0, 1.0 / sub_step, []
    for j in range(day_count):
        if rng.random() < 0.05:
            s += rng.uniform(5.0, 40.0)
        discharge = c * s ** n
        for _ in range(sub_step):
            s = max(s - c * s ** n * dt, 1e-9)
        q.append(discharge)
    return np.array(q)


@pytest.mark.parametrize("n", [1.0, 2.0, 3.0])
def test_exponent_recovers_the_reservoir_one(n):
    """Sur un réservoir dont on connaît la loi, la mesure doit retrouver b = 2 − 1/n."""
    b = recession_exponent(_drainage(n))
    assert b is not None
    assert b == pytest.approx(2.0 - 1.0 / n, abs=0.15)


def test_territory_returns_median_of_its_stations():
    q = np.column_stack([_drainage(2.0, seed=g) for g in range(8)])
    a = anchor_result(q)
    assert a.n_stations == 8 and a.reliable
    assert a.storage_exponent == pytest.approx(2.0, abs=0.4)
    assert a.etendue[0] <= a.storage_exponent <= a.etendue[1]


def test_territory_without_usable_station_returns_boussinesq():
    """Le défaut est la loi en carré de la charge, pas une valeur inventée."""
    a = anchor_result(np.full((3000, 2), np.nan))
    assert a.storage_exponent == N_DEFAULT and not a.reliable


def test_exponent_is_bounded_to_avoid_blowup():
    """n = 1/(2−b) diverge quand b approche 2 : la borne doit tenir."""
    q = np.column_stack([_drainage(n, seed=7) for n in (1.0, 2.0, 5.0)])
    a = anchor_result(q)
    assert 1.0 <= a.storage_exponent <= N_MAX
    assert a.recession_exponent < B_MAX


def test_segments_drop_post_peak_drainage():
    q = np.array([1.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.5, 1.2])
    good = recession_segments(q, delay=2, min_duration=5)
    assert not good[1] and not good[2], "les deux jours suivant la pointe sont ecartes"
    assert good[4:9].all(), "le coeur de la decrue est retenu"


def test_too_short_or_flat_series_returns_None():
    assert recession_exponent(np.ones(300)) is None
    assert recession_exponent(np.ones(3000)) is None
