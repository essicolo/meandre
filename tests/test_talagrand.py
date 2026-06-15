"""Tests de la normalisation Candille-Talagrand (meandre.diagnostics.talagrand).

Trois comportements attendus :
  1. PIT uniforme i.i.d.            -> delta_iid ~ 1, delta_eff ~ 1, p grand
  2. PIT uniforme mais autocorrélé  -> tau > 1 (la correction travaille),
                                       delta_eff ~ 1, p grand
  3. PIT mal calibré (U-shape)      -> delta_eff >> 1, p petit
"""
import numpy as np
import pytest
from scipy.stats import norm

from meandre.diagnostics.talagrand import candille_talagrand

T, S = 3000, 8


def _ar1_uniform(rho: float, rng) -> np.ndarray:
    """PIT (T, S) à marginales uniformes et autocorrélation AR(1) en gaussien."""
    z = np.empty((T, S))
    z[0] = rng.standard_normal(S)
    innov = rng.standard_normal((T, S)) * np.sqrt(1 - rho**2)
    for t in range(1, T):
        z[t] = rho * z[t - 1] + innov[t]
    return norm.cdf(z)


def test_iid_uniform_is_reliable():
    rng = np.random.default_rng(42)
    pit = rng.uniform(size=(T, S))
    ct = candille_talagrand(pit, n_boot=300, seed=1)
    assert 0.4 < ct["delta_iid"] < 2.5
    assert 0.3 < ct["delta_eff"] < 3.0
    assert ct["p_value"] > 0.01


def test_autocorrelated_uniform_corrected():
    rng = np.random.default_rng(7)
    pit = _ar1_uniform(rho=0.95, rng=rng)
    ct = candille_talagrand(pit, n_boot=300, seed=2)
    # la dépendance gonfle le bruit : tau doit le capter
    assert ct["tau"] > 2.0
    # une fois corrigé, compatible avec la fiabilité parfaite
    assert ct["delta_eff"] < 3.0
    assert ct["p_value"] > 0.01


def test_miscalibrated_is_detected():
    rng = np.random.default_rng(3)
    # U-shape (sous-dispersion) : Beta(0.4, 0.4), même avec autocorrélation
    z = _ar1_uniform(rho=0.9, rng=rng)
    pit = np.clip(np.abs(2 * z - 1) ** 0.5 * np.sign(z - 0.5) * 0.5 + 0.5, 0, 1)
    ct = candille_talagrand(pit, n_boot=300, seed=4)
    assert ct["delta_eff"] > 3.0
    assert ct["p_value"] < 0.05


def test_nan_handling_and_1d():
    rng = np.random.default_rng(11)
    pit = rng.uniform(size=(T, S))
    pit[rng.uniform(size=pit.shape) < 0.3] = np.nan
    ct = candille_talagrand(pit, n_boot=100, seed=5)
    assert ct["n"] == int(np.isfinite(pit).sum())
    # vecteur 1-D accepté
    ct1 = candille_talagrand(rng.uniform(size=2000), n_boot=100, seed=6)
    assert 0.3 < ct1["delta_iid"] < 3.0


def test_too_few_values_raises():
    with pytest.raises(ValueError):
        candille_talagrand(np.random.default_rng(0).uniform(size=50), n_bins=20)
