"""Tests des corrections generiques de forcage (meandre/data/forcing_correction.py)."""
import numpy as np
import pandas as pd
import pytest

from meandre.data.forcing_correction import (local_day_index, dedrizzle, scale_volume,
                                             monthly_climatology, monthly_ratio_merge,
                                             lapse_rate_adjust, attenuate_ratio,
                                             confidence_from_uncertainty)


def test_local_day_index_shift():
    t = pd.DatetimeIndex(["2020-06-01 02:00", "2020-12-01 02:00"])
    out = local_day_index(t, shift_h=-5)
    assert out[0] == pd.Timestamp("2020-05-31 21:00")
    assert out[1] == pd.Timestamp("2020-11-30 21:00")


def test_local_day_index_dst():
    t = pd.DatetimeIndex(["2020-06-01 04:30", "2020-12-01 04:30"])
    out = local_day_index(t, shift_h=-5, dst=True)
    assert out[0] == pd.Timestamp("2020-06-01 00:30")   # UTC-4 en ete
    assert out[1] == pd.Timestamp("2020-11-30 23:30")   # UTC-5 en hiver


def test_dedrizzle_threshold_and_no_mutation():
    h = np.array([[0.1, 0.3, 2.0]])
    out = dedrizzle(h, 0.3)
    assert np.allclose(out, [[0.0, 0.3, 2.0]])
    assert np.allclose(h, [[0.1, 0.3, 2.0]])  # entree intacte


def test_scale_volume():
    p = np.full((730, 3), 2.0)  # 730.5 mm/an
    out, f = scale_volume(p, 1147.0)
    assert np.isclose(out.mean() * 365.25, 1147.0)
    assert np.isclose(f, 1147.0 / 730.5)


def test_attenuate_ratio():
    ratio = np.array([[2.0, 0.5], [1.0, 1.5]])
    assert np.allclose(attenuate_ratio(ratio, 1.0), ratio)          # confiance pleine
    assert np.allclose(attenuate_ratio(ratio, 0.0), 1.0)            # neutralise
    assert np.allclose(attenuate_ratio(ratio, 0.5), [[1.5, 0.75], [1.0, 1.25]])
    with pytest.raises(ValueError):
        attenuate_ratio(ratio, 1.5)


def test_confidence_from_uncertainty():
    clim = np.array([2.0, 2.0, 2.0])
    w = confidence_from_uncertainty(np.array([0.0, 2.0, 20.0]), clim)
    assert np.isclose(w[0], 1.0)                                    # reference sure
    assert np.isclose(w[1], 0.5)                                    # erreur relative 1
    assert w[2] < 0.01                                              # extrapolation
    assert (w >= 0).all() and (w <= 1).all()


def test_monthly_ratio_merge_confidence_neutralizes():
    t = _daily_index(4)
    p_src = np.full((len(t), 2), 3.0)
    p_ref = np.full((len(t), 2), 2.0)
    conf = np.array([1.0, 0.0])                                     # noeud 1 : ref non fiable
    p_m, ratio, _ = monthly_ratio_merge(p_src, t, p_ref, t, confidence=conf)
    assert np.allclose(ratio[:, 0], 2.0 / 3.0)                      # ratio plein
    assert np.allclose(ratio[:, 1], 1.0)                            # neutralise
    assert np.allclose(p_m[:, 1], 3.0)                              # source intacte


def test_lapse_rate_adjust():
    t = np.array([[10.0, 10.0], [0.0, 0.0]])
    z_src = np.array([300.0, 300.0])    # altitude grille CaSR
    z_dst = np.array([500.0, 100.0])    # noeud plus haut / plus bas
    out = lapse_rate_adjust(t, z_src, z_dst)
    assert np.allclose(out, [[9.0, 11.0], [-1.0, 1.0]])   # -0.5 degC/100 m
    assert np.allclose(t, [[10.0, 10.0], [0.0, 0.0]])     # entree intacte
    with pytest.raises(ValueError):
        lapse_rate_adjust(t, z_src, z_dst[:1])


def _daily_index(n_years=3):
    return pd.date_range("2001-01-01", periods=int(365 * n_years), freq="D")


def test_monthly_climatology_shape():
    t = _daily_index()
    p = np.ones((len(t), 4))
    clim = monthly_climatology(p, t.month.values)
    assert clim.shape == (12, 4)
    assert np.allclose(clim, 1.0)


def test_monthly_ratio_merge_adopts_structure_and_keeps_sequence():
    t = _daily_index(4)
    rng = np.random.default_rng(0)
    p_src = rng.gamma(0.5, 4.0, size=(len(t), 5))
    p_src[rng.random(p_src.shape) < 0.5] = 0.0          # jours secs
    # reference = source x structure mensuelle connue (ex. hiver plus sec)
    monthly_factor = np.linspace(0.7, 1.3, 12)
    p_ref = p_src * monthly_factor[t.month.values - 1][:, None]
    p_m, ratio, f = monthly_ratio_merge(p_src, t, p_ref, t)
    # le ratio retrouve la structure imposee
    assert np.allclose(np.median(ratio, axis=1), monthly_factor, atol=0.02)
    # sequence intacte : memes jours secs, correlation 1 dans chaque mois
    assert ((p_m == 0) == (p_src == 0)).all()
    # sans volume cible, le volume converge vers la reference
    assert np.isclose(p_m.mean(), p_ref.mean(), rtol=0.01)
    assert f == 1.0


def test_monthly_ratio_merge_target_volume():
    t = _daily_index(4)
    p_src = np.full((len(t), 2), 3.0)
    p_ref = np.full((len(t), 2), 2.0)
    p_m, _, f = monthly_ratio_merge(p_src, t, p_ref, t, target_vol_mm_yr=1147.0)
    assert np.isclose(p_m.mean() * 365.25, 1147.0)
    assert f > 1.0  # la reference (730 mm/an) est sous la cible, rescale vers le haut


def test_monthly_ratio_merge_bounds_and_errors():
    t = _daily_index(4)
    p_src = np.full((len(t), 2), 0.01)                  # climatologie sous le plancher
    p_ref = np.full((len(t), 2), 5.0)
    _, ratio, _ = monthly_ratio_merge(p_src, t, p_ref, t, bounds=(0.5, 2.0))
    assert np.allclose(ratio, 2.0)                       # borne haute atteinte
    with pytest.raises(ValueError):
        monthly_ratio_merge(p_src[:100], t[:100], p_ref[:100], t[:100])  # periode trop courte
    with pytest.raises(ValueError):
        monthly_ratio_merge(p_src, t, p_ref[:, :1], t)   # noeuds incompatibles
