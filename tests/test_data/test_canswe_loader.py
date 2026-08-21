"""CanSWE : appariement aux noeuds et normalisation des drapeaux."""
import numpy as np
import pytest

from meandre.data.canswe_loader import _haversine_km, _as_text


def test_as_text_decode_les_octets():
    """Les drapeaux CanSWE sont stockes en OCTETS (|S1). Un str() naif rend "b''" au
    lieu de "", et tout filtre de qualite rejette alors 100 % des mesures : c'est
    arrive le 2026-08-20, 45 037 mesures ecartees en silence sur OUTV."""
    a = np.array([b"", b"R", b"M"], dtype="S1")
    assert list(_as_text(a)) == ["", "R", "M"]


def test_as_text_tolere_as_text_et_none():
    a = np.array(["A", None, " B "], dtype=object)
    assert list(_as_text(a)) == ["A", "", "B"]


def test_as_text_preserve_la_forme():
    a = np.array([[b"", b"R"], [b"M", b""]], dtype="S1")
    assert _as_text(a).shape == (2, 2)


def test_haversine_distance_connue():
    # Montreal -> Quebec, ~233 km a vol d'oiseau
    d = _haversine_km(45.51, -73.57, np.array([46.81]), np.array([-71.21]))
    assert 225.0 < float(d[0]) < 240.0


def test_haversine_nulle_sur_le_meme_point():
    d = _haversine_km(46.0, -74.0, np.array([46.0]), np.array([-74.0]))
    assert float(d[0]) == pytest.approx(0.0, abs=1e-9)


def test_haversine_vectorise():
    d = _haversine_km(46.0, -74.0, np.array([46.0, 47.0, 45.0]), np.array([-74.0, -74.0, -74.0]))
    assert d.shape == (3,)
    assert float(d[0]) < float(d[1])
