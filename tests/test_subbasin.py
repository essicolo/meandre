"""Restriction d'un territoire au sous-bassin en amont d'une station (meandre/run/subbasin.py)."""
import os
import sys

import numpy as np
import torch



from meandre.routing.graph import RiverGraph
from meandre.spatial.territorial import TerritorialFeatures
from meandre.run import subbasin


def _graph():
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


def test_upstream_includes_the_reach_and_only_upstream():
    assert subbasin.upstream(_graph(), 4).tolist() == [0, 1, 2, 3, 4]
    assert subbasin.upstream(_graph(), 2).tolist() == [0, 1, 2]
    assert subbasin.upstream(_graph(), 6).tolist() == [6]


def test_reindexed_graph_keeps_edges_order_and_lakes():
    keep = subbasin.upstream(_graph(), 4)
    g = subbasin.restrict_graph(_graph(), keep)
    assert g.edge_index.tolist() == [[0, 1, 2, 3], [2, 2, 4, 4]]
    assert g.edge_attr[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert g.topo_order.tolist() == [0, 1, 3, 2, 4]
    assert g.is_lake.tolist() == [False, True, False, False, False]


def test_attributes_and_series_follow_kept_reaches():
    keep = np.array([1, 3])
    terr = TerritorialFeatures(data=torch.arange(14, dtype=torch.float32).reshape(7, 2), columns=["a", "b"], physical={"area_km2_local": torch.arange(7, dtype=torch.float32)})
    t = subbasin.restrict_territorial(terr, keep)
    assert t.data.tolist() == [[2.0, 3.0], [6.0, 7.0]]
    assert t.physical["area_km2_local"].tolist() == [1.0, 3.0]
    series = torch.arange(21, dtype=torch.float32).reshape(3, 7)
    assert subbasin.restrict_nodes(series, keep).tolist() == [[1.0, 3.0], [8.0, 10.0], [15.0, 17.0]]
    assert subbasin.restrict_nodes(None, keep) is None


def test_station_from_other_territory_ignored(monkeypatch):
    monkeypatch.setenv("ETL_SUBBASIN", "labi:089907")
    assert subbasin.requested("labi") == "089907"
    assert subbasin.requested("cndc") is None
