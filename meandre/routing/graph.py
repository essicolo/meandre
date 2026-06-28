"""River network graph construction and utilities.

Builds a PyTorch Geometric Data object from a river network description.
Nodes are subbasins/reaches; edges are upstream -> downstream connections.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass

import torch
from torch import Tensor


def _topological_sort(edge_index: Tensor, n_nodes: int) -> Tensor:
    """Kahn's algorithm — returns node indices in topological order.

    Raises
    ------
    ValueError
        If the graph contains a cycle (not a valid DAG).
    """
    in_degree = [0] * n_nodes
    adj: dict[int, list[int]] = collections.defaultdict(list)
    for e in range(edge_index.shape[1]):
        src = int(edge_index[0, e].item())
        dst = int(edge_index[1, e].item())
        adj[src].append(dst)
        in_degree[dst] += 1

    queue = collections.deque(i for i in range(n_nodes) if in_degree[i] == 0)
    order: list[int] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for nb in adj[node]:
            in_degree[nb] -= 1
            if in_degree[nb] == 0:
                queue.append(nb)

    if len(order) != n_nodes:
        raise ValueError(
            f"Graph has cycles — not a valid DAG. "
            f"Only {len(order)}/{n_nodes} nodes resolved in topological sort."
        )
    return torch.tensor(order, dtype=torch.long)


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

    Raises
    ------
    ValueError
        If the graph contains a cycle (not a valid DAG) or if the
        node count inferred from edge_index does not match other tensors.
    """

    edge_index: Tensor          # (2, n_edges) int64
    edge_attr: Tensor           # (n_edges, n_edge_features)
    topo_order: Tensor          # (n_nodes,) int
    is_lake: Tensor             # (n_nodes,) bool
    travel_time_days: Tensor    # (n_edges,) int

    def __post_init__(self) -> None:
        n_nodes = int(self.topo_order.shape[0])
        if self.edge_index.numel() == 0:
            return   # graphe sans arête (nœud isolé / sous-bassin minimal) : rien à valider
        if self.edge_index.max().item() >= n_nodes:
            raise ValueError(
                f"edge_index contains node index >= n_nodes ({n_nodes}). "
                f"Found max index {int(self.edge_index.max().item())}."
            )
        if self.edge_index.min().item() < 0:
            raise ValueError("edge_index contains negative node indices.")
        # Validate DAG via Kahn's algorithm — cheap O(V+E) check that also
        # confirms topo_order is consistent with the edge structure.
        try:
            topo_check = _topological_sort(self.edge_index, n_nodes)
        except ValueError as exc:
            raise ValueError(f"RiverGraph construction failed: {exc}") from exc
        if not torch.equal(topo_check.to(self.topo_order.device), self.topo_order):
            raise ValueError(
                "Provided topo_order is inconsistent with edge_index. "
                "Omit topo_order when constructing RiverGraph; it will be "
                "computed automatically via Kahn's algorithm."
            )

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