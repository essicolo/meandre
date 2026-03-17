"""Rebuild DuckDB graph edges from troncon.trl using junction-based topology.

The original graph was built using the ds_reach field (last token per line),
which represents HYDROTEL's simplified routing — many tributaries mapped
directly to a few collector troncons.  This produced a flat star topology
where 50% of nodes flowed directly into troncon 1.

The correct physical topology uses junction node IDs:
  from_junct = downstream junction, to_junct = upstream junction.
  troncon A feeds into troncon B if A.from_junct == B.to_junct.

Usage:
    uv run python scripts/rebuild_graph.py /path/to/SLSO/physitel/troncon.trl
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meandre.data.physitel_loader import _parse_troncon, _build_graph

import collections
import numpy as np


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/rebuild_graph.py <path/to/troncon.trl>")
        print("       python scripts/rebuild_graph.py <path/to/PHYSITEL_PROJECT_DIR>")
        sys.exit(1)

    trl_path = Path(sys.argv[1])
    if trl_path.is_dir():
        trl_path = trl_path / "physitel" / "troncon.trl"
    if not trl_path.exists():
        print(f"ERROR: {trl_path} not found")
        sys.exit(1)

    db_path = Path("notebooks/data/slso.duckdb")
    if not db_path.exists():
        print(f"ERROR: {db_path} not found")
        sys.exit(1)

    # ── 1. Parse troncon.trl with new parser (extracts junction IDs) ──
    print(f"Parsing {trl_path} ...")
    troncons = _parse_troncon(trl_path)
    print(f"  {len(troncons)} troncons parsed")

    # ── 2. Build graph using junction-based connectivity ──
    print("Building junction-based graph ...")
    graph, node_ids, troncon_idx = _build_graph(troncons, velocity_m_s=1.0, device=None)
    n_nodes = len(node_ids)
    print(f"  {n_nodes} nodes, {graph.n_edges} edges")

    # ── 3. Verify topology ──
    # Count headwaters
    dst_set = set(graph.edge_index[1].tolist())
    headwaters = sum(1 for i in range(n_nodes) if i not in dst_set)
    print(f"  Headwater nodes: {headwaters} ({100*headwaters/n_nodes:.0f}%)")

    # Top junction nodes by in-degree
    in_count = collections.Counter()
    for e in range(graph.n_edges):
        in_count[graph.edge_index[1, e].item()] += 1
    print("  Top 5 junction nodes by upstream edge count:")
    idx_to_tid = {v: k for k, v in troncon_idx.items()}
    for ni, count in in_count.most_common(5):
        print(f"    troncon {idx_to_tid[ni]}: {count} upstream edges")

    # ── 4. Compute cumulative areas ──
    import duckdb
    con = duckdb.connect(str(db_path))
    area_df = con.execute(
        "SELECT node_idx, area_km2_local FROM territorial ORDER BY node_idx"
    ).df()
    local_area = dict(zip(area_df["node_idx"], area_df["area_km2_local"]))

    # Build downstream map
    downstream = {}
    for e in range(graph.n_edges):
        s = graph.edge_index[0, e].item()
        d = graph.edge_index[1, e].item()
        downstream[s] = d

    cum_area = {i: local_area.get(i, 0.0) for i in range(n_nodes)}
    for ni in graph.topo_order.tolist():
        ds = downstream.get(ni)
        if ds is not None:
            cum_area[ds] += cum_area[ni]

    # ── 5. Verify station drainage areas ──
    station_troncons = {
        "023402": (25, 5820), "023429": (246, 3085),
        "024007": (786, 2330), "024014": (796, 2163),
        "030103": (958, 1550), "023303": (2388, 1152),
    }
    print("\nStation drainage area verification:")
    for sid, (tid, expected) in station_troncons.items():
        idx = troncon_idx.get(tid)
        actual = cum_area.get(idx, 0) if idx is not None else 0
        ratio = actual / expected if expected > 0 else 0
        print(f"  {sid}: troncon {tid}: cum_area={actual:.0f} km2, "
              f"expected={expected} km2, ratio={ratio:.2f}")

    max_area_node = max(cum_area, key=cum_area.get)
    print(f"\nMax cumulative area: {cum_area[max_area_node]:.0f} km2 "
          f"at troncon {idx_to_tid[max_area_node]}")

    # ── 6. Update DuckDB ──
    print("\nUpdating DuckDB ...")

    # Update edges
    con.execute("DELETE FROM edges")
    ei = graph.edge_index.numpy()
    ea = graph.edge_attr.numpy()
    tt = graph.travel_time_days.numpy()
    import pandas as pd
    edge_df = pd.DataFrame({
        "src": ei[0],
        "dst": ei[1],
        "edge_attr_0": ea[:, 0],
        "edge_attr_1": ea[:, 1],
        "edge_attr_2": ea[:, 2],
        "travel_time_days": tt,
    })
    con.execute("INSERT INTO edges SELECT * FROM edge_df")

    # Update topo_order in nodes
    topo = graph.topo_order.numpy()
    rank = np.empty(n_nodes, dtype=np.int64)
    rank[topo] = np.arange(n_nodes)
    for ni in range(n_nodes):
        con.execute("UPDATE nodes SET topo_order = ? WHERE node_idx = ?",
                     [int(rank[ni]), ni])

    # Update is_lake
    for ni in range(n_nodes):
        con.execute("UPDATE nodes SET is_lake = ? WHERE node_idx = ?",
                     [bool(graph.is_lake[ni]), ni])

    # Update cumulative area
    for ni in range(n_nodes):
        con.execute(
            "UPDATE territorial SET area_km2_physical = ? WHERE node_idx = ?",
            [float(max(cum_area[ni], 1e-3)), ni],
        )

    con.close()
    print("DuckDB updated successfully!")
    print("\nNext: delete any stale checkpoint and retrain.")


if __name__ == "__main__":
    main()
