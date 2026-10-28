"""Audit du rattachement station-tronçon (2026-10-28) : lecture des fiches CEHQ et de troncon.trl."""
import os
from importlib.machinery import SourceFileLoader

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
audit = SourceFileLoader("audit_stations", os.path.join(HERE, "..", "meandre", "adapters", "physitel", "builders", "audit_stations.py")).load_module()


def test_cehq_record_coordinates():
    header = ("Station:        062002         Rouge - en amont du ponceau sur la route 169\n"
            "Bassin versant: 26.8 km²       Régime: Non influencé\n"
            "Coordonnées:    (NAD83) 48° 48' 14\" // -72° 29' 31\"\n")
    lon, lat, area, regime = audit.parse_header(header)
    assert abs(lat - (48 + 48 / 60 + 14 / 3600)) < 1e-9
    assert abs(lon + (72 + 29 / 60 + 31 / 3600)) < 1e-9
    assert area == 26.8 and regime == "Non influencé"


def test_record_without_coordinates():
    lon, lat, area, _ = audit.parse_header("Bassin versant: 1051 km²\nCoordonnées:    (NAD83)           //          \n")
    assert np.isnan(lon) and np.isnan(lat) and area == 1051.0


def test_read_troncon_trl(tmp_path):
    """Une rivière (unités 48 49 50) se jette dans un lac (noeud amont 3) ; le lac porte ses unités."""
    (tmp_path / "physitel").mkdir()
    (tmp_path / "physitel" / "troncon.trl").write_text(
        "2\n2\nTRONCONS\n"
        "2 2 2 2 3 4 644075.25 365.66 74.6 1.5 2 3 4 60\n"
        "3 1 3 43 9389.95 3.43 0.04 3 48 49 50 1\n", encoding="latin-1")
    reach_of_unit, downstream, upstream = audit.read_reaches(str(tmp_path))
    assert reach_of_unit[48] == 3 and reach_of_unit[50] == 3 and reach_of_unit[3] == 2
    assert downstream[3] == 2 and downstream[2] is None
    assert upstream[2] == [3]
