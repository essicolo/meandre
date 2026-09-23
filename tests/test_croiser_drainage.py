"""La hierarchie de drainage : echelle ecoforestiere et regle de preseance."""
import importlib.util
import os

import numpy as np

_p = os.path.join(os.path.dirname(__file__), "..", ".runs", "quebec", "croiser_drainage.py")
_spec = importlib.util.spec_from_file_location("croiser_drainage", _p)
cd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cd)


def test_rang_ecoforestier_dizaine_plus_un():
    r = cd.rang_ecoforestier(np.array([0, 10, 16, 34, 60, 71, np.nan], dtype=float))
    assert np.allclose(r[[0, 1, 3, 4]], [1, 2, 4, 7])
    # Complexe, hors echelle et absent restent vides.
    assert np.isnan(r[[2, 5, 6]]).all()


def test_hierarchiser_premiere_source_qui_repond():
    n = np.nan
    a = np.array([[1, n, n, n]], dtype=np.float32)
    b = np.array([[5, 2, n, n]], dtype=np.float32)
    c = np.array([[6, 6, 3, n]], dtype=np.float32)
    v, s = cd.hierarchiser([a, b, c])
    assert np.allclose(v[0, :3], [1, 2, 3]) and np.isnan(v[0, 3])
    assert s.tolist() == [[1, 2, 3, 0]]


def test_cle_sigeom_couvre_les_depots_principaux():
    for code in ("Tm", "T", "O", "L", "Gx", "MGa", "At", "Cg"):
        assert code in cd.CODE_VERS_GROUPE
    assert "R" not in cd.CODE_VERS_GROUPE
