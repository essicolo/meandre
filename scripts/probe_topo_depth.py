"""Mesure la profondeur topologique du DAG slso et la distribution
des nodes par niveau Kahn.

Question clé : combien d'iterations Python le routing loop fait
par timestep, et combien de nodes par level (parallélisable) ?
"""
from __future__ import annotations
import duckdb
import numpy as np

BASIN_DB = "notebooks/slso/data/slso.duckdb"


def main():
    con = duckdb.connect(BASIN_DB, read_only=True)
    edges = con.execute("SELECT src, dst FROM edges").df()
    n = con.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    src = edges.src.values
    dst = edges.dst.values

    # Kahn's algorithm to compute levels
    in_deg = np.bincount(dst, minlength=n).astype(int)
    children: list[list[int]] = [[] for _ in range(n)]
    for s, d in zip(src.tolist(), dst.tolist()):
        children[s].append(d)

    level = np.full(n, -1, dtype=int)
    queue = [i for i in range(n) if in_deg[i] == 0]
    for i in queue:
        level[i] = 0
    cur_level = 0
    in_deg_work = in_deg.copy()
    while queue:
        next_q = []
        for node in queue:
            for c in children[node]:
                in_deg_work[c] -= 1
                if in_deg_work[c] == 0:
                    level[c] = cur_level + 1
                    next_q.append(c)
        queue = next_q
        cur_level += 1
    max_level = level.max()

    print(f"Total nodes        : {n}")
    print(f"Total edges        : {len(edges):,}")
    print(f"Profondeur DAG (max level) : {max_level}")
    print(f"Nb levels totaux   : {max_level + 1}")
    print()
    print("Distribution nodes par level :")
    counts = np.bincount(level)
    cum = np.cumsum(counts)
    # Show every 10th level or significant
    for L in range(max_level + 1):
        bar = "#" * min(int(counts[L] / counts.max() * 60), 60)
        flag = ""
        if counts[L] >= counts.max() * 0.5:
            flag = "  <-- PEAK"
        if counts[L] == 1 and L > 0:
            continue  # skip 1-node levels
        print(f"  L{L:3d} : {counts[L]:5d} nodes  {bar}{flag}")
    print()
    print(f"Levels avec >= 5 nodes : {(counts >= 5).sum()}")
    print(f"Levels avec 1 node     : {(counts == 1).sum()}")
    print(f"Nodes dans top-10% des levels les plus peuplés : "
          f"{int(counts[counts >= np.percentile(counts, 90)].sum())} / {n}")


if __name__ == "__main__":
    main()
