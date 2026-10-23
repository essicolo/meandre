"""Restriction d'un territoire au sous-bassin en amont d'une station (.runs/quebec/subbasin.py)."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".runs", "quebec"))

from meandre.routing.graph import RiverGraph
from meandre.spatial.territorial import TerritorialFeatures
import subbasin


def _graphe():
    # 0 -> 2, 1 -> 2, 2 -> 4, 3 -> 4, 4 -> 5 ; 6 isole, deverse dans 5
    src = [0, 1, 2, 3, 4, 6]
    dst = [2, 2, 4, 4, 5, 5]
    return RiverGraph(
        edge_index=torch.tensor([src, dst]),
        edge_attr=torch.arange(6, dtype=torch.float32)[:, None].repeat(1, 3),
        topo_order=torch.tensor([0, 1, 3, 6, 2, 4, 5]),
        is_lake=torch.tensor([False, True, False, False, False, False, False]),
        travel_time_days=torch.zeros(6, dtype=torch.long),
    )


def test_amont_comprend_le_troncon_et_seulement_l_amont():
    assert subbasin.upstream(_graphe(), 4).tolist() == [0, 1, 2, 3, 4]
    assert subbasin.upstream(_graphe(), 2).tolist() == [0, 1, 2]
    assert subbasin.upstream(_graphe(), 6).tolist() == [6]


def test_graphe_reindexe_garde_aretes_ordre_et_lacs():
    keep = subbasin.upstream(_graphe(), 4)
    g = subbasin.restrict_graph(_graphe(), keep)
    assert g.edge_index.tolist() == [[0, 1, 2, 3], [2, 2, 4, 4]]
    assert g.edge_attr[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert g.topo_order.tolist() == [0, 1, 3, 2, 4]
    assert g.is_lake.tolist() == [False, True, False, False, False]


def test_attributs_et_series_suivent_les_troncons_gardes():
    keep = np.array([1, 3])
    terr = TerritorialFeatures(data=torch.arange(14, dtype=torch.float32).reshape(7, 2), columns=["a", "b"], physical={"area_km2_local": torch.arange(7, dtype=torch.float32)})
    t = subbasin.restrict_territorial(terr, keep)
    assert t.data.tolist() == [[2.0, 3.0], [6.0, 7.0]]
    assert t.physical["area_km2_local"].tolist() == [1.0, 3.0]
    serie = torch.arange(21, dtype=torch.float32).reshape(3, 7)
    assert subbasin.restrict_nodes(serie, keep).tolist() == [[1.0, 3.0], [8.0, 10.0], [15.0, 17.0]]
    assert subbasin.restrict_nodes(None, keep) is None


def test_station_d_un_autre_territoire_ignoree(monkeypatch):
    monkeypatch.setenv("ETL_SUBBASIN", "labi:089907")
    assert subbasin.requested("labi") == "089907"
    assert subbasin.requested("cndc") is None
