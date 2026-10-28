"""La hierarchie de drainage : echelle ecoforestiere et regle de preseance."""
import importlib.util
import os

import numpy as np

_p = os.path.join(os.path.dirname(__file__), "..", "meandre", "adapters", "physitel", "builders", "drainage_maps.py")
_spec = importlib.util.spec_from_file_location("croiser_drainage", _p)
cd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cd)


def test_ecoforest_rank_tens_plus_one():
    r = cd.ecoforest_rank(np.array([0, 10, 16, 34, 60, 71, np.nan], dtype=float))
    assert np.allclose(r[[0, 1, 3, 4]], [1, 2, 4, 7])
    # Complexe, hors echelle et absent restent vides.
    assert np.isnan(r[[2, 5, 6]]).all()


def test_rank_first_source_that_answers():
    n = np.nan
    a = np.array([[1, n, n, n]], dtype=np.float32)
    b = np.array([[5, 2, n, n]], dtype=np.float32)
    c = np.array([[6, 6, 3, n]], dtype=np.float32)
    v, s = cd.rank_layers([a, b, c])
    assert np.allclose(v[0, :3], [1, 2, 3]) and np.isnan(v[0, 3])
    assert s.tolist() == [[1, 2, 3, 0]]


def test_sigeom_key_covers_main_deposits():
    for code in ("Tm", "T", "O", "L", "Gx", "MGa", "At", "Cg"):
        assert code in cd.CODE_TO_GROUP
    assert "R" not in cd.CODE_TO_GROUP


def test_ecoforest_deposit_rank_family_and_prefixes():
    r = cd.ecoforest_deposit_rank(["1A", "1AY", "M1A", "R1A", "2AE", "3D", "5A", "5S", "7E", "8E", "R", "RC", "AN", None])
    assert np.allclose(r[:4], 4.0)
    assert np.allclose(r[4:10], [1.0, 1.0, 7.0, 2.5, 7.0, 1.0])
    assert np.isnan(r[10:]).all()
