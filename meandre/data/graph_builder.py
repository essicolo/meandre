"""Build a RiverGraph from shapefiles or tabular network description.

Inputs:
    - River network shapefile (reaches with from/to node IDs)
    - DEM-derived travel time estimates per reach

Output:
    meandre.routing.graph.RiverGraph
"""

from __future__ import annotations

from pathlib import Path

import torch

from meandre.routing.graph import RiverGraph


def from_shapefile(
    network_shp: str | Path,
    from_col: str = "FROM_NODE",
    to_col: str = "TO_NODE",
    length_col: str = "LENGTH_M",
    slope_col: str = "SLOPE",
    tau_col: str = "TAU_DAYS",
    lake_col: str | None = "IS_LAKE",
) -> RiverGraph:
    """Build RiverGraph from a river network shapefile.

    Args:
        network_shp: Path to shapefile with reach attributes.
        from_col, to_col: Column names for upstream/downstream node IDs.
        length_col: Reach length column (m).
        slope_col: Reach slope column (m/m).
        tau_col: Travel time column (integer days).
        lake_col: Optional boolean column for lake/reservoir nodes.
    Returns:
        RiverGraph ready for routing.
    """
    try:
        import geopandas as gpd
    except ImportError as e:
        raise ImportError("geopandas is required for graph_builder") from e

    gdf = gpd.read_file(network_shp)

    # Build node index (0-based)
    all_nodes = sorted(
        set(gdf[from_col].tolist() + gdf[to_col].tolist())
    )
    node_to_idx = {n: i for i, n in enumerate(all_nodes)}
    n_nodes = len(all_nodes)

    src = torch.tensor([node_to_idx[n] for n in gdf[from_col]], dtype=torch.long)
    dst = torch.tensor([node_to_idx[n] for n in gdf[to_col]], dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)

    lengths = torch.tensor(gdf[length_col].values, dtype=torch.float)
    slopes = torch.tensor(gdf[slope_col].values, dtype=torch.float)
    taus = torch.tensor(gdf[tau_col].values, dtype=torch.long)
    edge_attr = torch.stack([lengths, slopes, taus.float()], dim=-1)

    # Topological sort via Kahn's algorithm
    topo_order = _topological_sort(edge_index, n_nodes)

    is_lake = torch.zeros(n_nodes, dtype=torch.bool)
    if lake_col and lake_col in gdf.columns:
        for _, row in gdf.iterrows():
            if row[lake_col]:
                is_lake[node_to_idx[row[to_col]]] = True

    return RiverGraph(edge_index, edge_attr, topo_order, is_lake, taus)


def _topological_sort(edge_index: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """Kahn's algorithm — returns node indices in topological order."""
    import collections

    in_degree = [0] * n_nodes
    adj = collections.defaultdict(list)
    for e in range(edge_index.shape[1]):
        src, dst = edge_index[0, e].item(), edge_index[1, e].item()
        adj[src].append(dst)
        in_degree[dst] += 1

    queue = collections.deque(i for i in range(n_nodes) if in_degree[i] == 0)
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for nb in adj[node]:
            in_degree[nb] -= 1
            if in_degree[nb] == 0:
                queue.append(nb)

    if len(order) != n_nodes:
        raise ValueError("Graph has cycles — not a valid DAG")
    return torch.tensor(order, dtype=torch.long)
