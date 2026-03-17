"""River network graph construction and utilities.

Builds a PyTorch Geometric Data object from a river network description.
Nodes are subbasins/reaches; edges are upstream -> downstream connections.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class RiverGraph:
    """Directed acyclic graph of the river network.

    Attributes
    ----------
    edge_index: (2, n_edges) int64 — [source, target] pairs (upstream->downstream)
    edge_attr:  (n_edges, n_edge_features) float — reach length, slope, travel time
    topo_order: (n_nodes,) int — topological sort order (process upstream first)
    is_lake:    (n_nodes,) bool — True for lake/reservoir nodes
    travel_time_days: (n_edges,) int — estimated travel time in whole days

    Node features (geometric) are stored in TerritorialFeatures,
    not here, to keep the graph structure separate from node attributes.
    """

    edge_index: Tensor          # (2, n_edges) int64
    edge_attr: Tensor           # (n_edges, n_edge_features)
    topo_order: Tensor          # (n_nodes,) int
    is_lake: Tensor             # (n_nodes,) bool
    travel_time_days: Tensor    # (n_edges,) int

    @property
    def n_nodes(self) -> int:
        return int(self.topo_order.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def upstream_nodes(self, node: int) -> list[int]:
        """Return list of direct upstream neighbours of *node*."""
        mask = self.edge_index[1] == node
        return self.edge_index[0][mask].tolist()

    def upstream_travel_times(self, node: int) -> list[int]:
        """Return travel times (days) for edges incoming to *node*."""
        mask = self.edge_index[1] == node
        return self.travel_time_days[mask].tolist()

    def to(self, device: torch.device) -> "RiverGraph":
        return RiverGraph(
            edge_index=self.edge_index.to(device),
            edge_attr=self.edge_attr.to(device),
            topo_order=self.topo_order.to(device),
            is_lake=self.is_lake.to(device),
            travel_time_days=self.travel_time_days.to(device),
        )


def synthetic_linear_graph(n_nodes: int, tau_days: int = 1) -> RiverGraph:
    """Build a simple linear chain graph for testing: 0->1->2->...->N-1."""
    src = torch.arange(n_nodes - 1, dtype=torch.long)
    dst = torch.arange(1, n_nodes, dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)
    edge_attr = torch.ones(n_nodes - 1, 3)  # [length, slope, travel_time]
    topo_order = torch.arange(n_nodes, dtype=torch.long)
    is_lake = torch.zeros(n_nodes, dtype=torch.bool)
    travel_time_days = torch.full((n_nodes - 1,), tau_days, dtype=torch.long)
    return RiverGraph(edge_index, edge_attr, topo_order, is_lake, travel_time_days)
