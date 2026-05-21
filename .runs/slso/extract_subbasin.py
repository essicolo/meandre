"""Extract a sub-basin from the SLSO DuckDB for fast iteration.

Traces upstream from a chosen outlet node, remaps indices to 0..N-1,
copies all tables, and subsets the forcing NetCDF.

Usage:
    python notebooks/slso/extract_subbasin.py
"""

import os
import subprocess
from collections import deque
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import xarray as xr

os.chdir(subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], text=True
).strip())

# ── Configuration ──────────────────────────────────────────────────────
SOURCE_DB   = Path("notebooks/slso/data/slso.duckdb")
TARGET_DB   = Path("notebooks/slso/data/slso-sub.duckdb")
SOURCE_NC   = Path("notebooks/slso/data/forcing.nc")
TARGET_NC   = Path("notebooks/slso/data/forcing-sub.nc")
OUTLET_NODE = 252          # station 023448, ~204 nodes upstream

# ── 1. Find upstream nodes ─────────────────────────────────────────────
con = duckdb.connect(str(SOURCE_DB), read_only=True)

upstream_df = con.execute(f"""
    WITH RECURSIVE upstream AS (
        SELECT {OUTLET_NODE} AS node_idx
        UNION ALL
        SELECT DISTINCT e.src
        FROM edges e JOIN upstream u ON e.dst = u.node_idx
    )
    SELECT DISTINCT node_idx FROM upstream ORDER BY node_idx
""").df()

old_indices = sorted(upstream_df["node_idx"].tolist())
old_to_new = {old: new for new, old in enumerate(old_indices)}
n_nodes = len(old_indices)
print(f"Sub-basin: {n_nodes} nodes (outlet={OUTLET_NODE})")

# ── 2. Extract and remap tables ───────────────────────────────────────
old_set = set(old_indices)

# Nodes
nodes_df = con.execute("SELECT * FROM nodes ORDER BY node_idx").df()
nodes_df = nodes_df[nodes_df["node_idx"].isin(old_set)].copy()
nodes_df["node_idx"] = nodes_df["node_idx"].map(old_to_new)
nodes_df["node_id"] = nodes_df["node_idx"] + 1

# Recompute topo_order via BFS from sources
edges_df = con.execute("SELECT * FROM edges").df()
edges_df = edges_df[
    edges_df["src"].isin(old_set) & edges_df["dst"].isin(old_set)
].copy()
edges_df["src"] = edges_df["src"].map(old_to_new)
edges_df["dst"] = edges_df["dst"].map(old_to_new)

# Topological sort (Kahn's algorithm)
in_degree = np.zeros(n_nodes, dtype=int)
children: dict[int, list[int]] = {i: [] for i in range(n_nodes)}
for _, row in edges_df.iterrows():
    s, d = int(row["src"]), int(row["dst"])
    in_degree[d] += 1
    children[s].append(d)

queue = deque(int(i) for i in range(n_nodes) if in_degree[i] == 0)
topo_order = np.zeros(n_nodes, dtype=int)
rank = 0
while queue:
    node = queue.popleft()
    topo_order[node] = rank
    rank += 1
    for child in children[node]:
        in_degree[child] -= 1
        if in_degree[child] == 0:
            queue.append(child)

nodes_df["topo_order"] = nodes_df["node_idx"].map(lambda i: int(topo_order[i]))
print(f"Edges: {len(edges_df)}")

# Territorial
terr_df = con.execute("SELECT * FROM territorial").df()
terr_df = terr_df[terr_df["node_idx"].isin(old_set)].copy()
terr_df["node_idx"] = terr_df["node_idx"].map(old_to_new)

# Initial state
state_df = con.execute("SELECT * FROM initial_state").df()
state_df = state_df[state_df["node_idx"].isin(old_set)].copy()
state_df["node_idx"] = state_df["node_idx"].map(old_to_new)

# Stations
stations_df = con.execute("SELECT * FROM stations").df()
stations_df = stations_df[stations_df["node_idx"].isin(old_set)].copy()
stations_df["node_idx"] = stations_df["node_idx"].map(old_to_new)
print(f"Stations: {len(stations_df)} ({', '.join(stations_df['station_id'].tolist())})")

# Observations (filter by station_id)
station_ids = stations_df["station_id"].tolist()
placeholders = ", ".join(f"'{s}'" for s in station_ids)
obs_df = con.execute(
    f"SELECT * FROM observations WHERE station_id IN ({placeholders})"
).df()
print(f"Observations: {len(obs_df)} rows")

# Withdrawals
withdrawal_ids = ", ".join(str(i) for i in old_indices)
wd_df = con.execute(
    f"SELECT * FROM withdrawals WHERE node_idx IN ({withdrawal_ids})"
).df()
wd_df["node_idx"] = wd_df["node_idx"].map(old_to_new)
print(f"Withdrawals: {len(wd_df)} rows")

con.close()

# ── 3. Write target DuckDB ────────────────────────────────────────────
if TARGET_DB.exists():
    TARGET_DB.unlink()

out = duckdb.connect(str(TARGET_DB))

# Create schema (same as BasinCache._create_schema)
out.execute("""
    CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT);
    CREATE TABLE nodes (
        node_idx INTEGER PRIMARY KEY, node_id INTEGER,
        lon FLOAT, lat FLOAT, is_lake BOOLEAN, topo_order INTEGER
    );
    CREATE TABLE edges (
        src INTEGER, dst INTEGER,
        edge_attr_0 FLOAT, edge_attr_1 FLOAT, edge_attr_2 FLOAT,
        travel_time_days INTEGER
    );
    CREATE TABLE initial_state (
        node_idx INTEGER PRIMARY KEY,
        theta1 FLOAT, theta2 FLOAT, theta3 FLOAT,
        swe FLOAT, t_soil FLOAT, canopy_storage FLOAT, wetland_storage FLOAT
    );
    CREATE TABLE stations (
        station_id TEXT PRIMARY KEY, node_idx INTEGER,
        lon DOUBLE, lat DOUBLE, drainage_area_km2 DOUBLE
    );
    CREATE TABLE observations (
        station_id TEXT, date DATE, discharge FLOAT,
        PRIMARY KEY (station_id, date)
    );
    CREATE TABLE warm_states (
        state_date TEXT, node_idx INTEGER,
        theta1 FLOAT, theta2 FLOAT, theta3 FLOAT,
        swe FLOAT, t_soil FLOAT, canopy_storage FLOAT, wetland_storage FLOAT,
        lake_storage FLOAT, q_out_prev FLOAT,
        PRIMARY KEY (state_date, node_idx)
    );
    CREATE TABLE encoder_states (
        state_date TEXT, run_id TEXT, data BLOB,
        PRIMARY KEY (state_date, run_id)
    );
""")

# Also create territorial with same columns
terr_cols = ", ".join(
    f"{c} {'BIGINT' if c == 'node_idx' else 'FLOAT'}"
    for c in terr_df.columns
)
out.execute(f"CREATE TABLE territorial ({terr_cols})")

# Also create withdrawals
out.execute("CREATE TABLE withdrawals (date DATE, node_idx INTEGER, net_withdrawal FLOAT)")

# Insert data
out.register("nodes_v", nodes_df)
out.execute("INSERT INTO nodes SELECT * FROM nodes_v")

out.register("edges_v", edges_df)
out.execute("INSERT INTO edges SELECT * FROM edges_v")

out.register("terr_v", terr_df)
out.execute("INSERT INTO territorial SELECT * FROM terr_v")

out.register("state_v", state_df)
out.execute("INSERT INTO initial_state SELECT * FROM state_v")

out.register("stations_v", stations_df)
out.execute("INSERT INTO stations SELECT * FROM stations_v")

out.register("obs_v", obs_df)
out.execute("INSERT INTO observations SELECT * FROM obs_v")

out.register("wd_v", wd_df)
out.execute("INSERT INTO withdrawals SELECT * FROM wd_v")

# Metadata
out.execute("INSERT INTO metadata VALUES ('n_nodes', ?), ('source', ?), ('created_at', CURRENT_TIMESTAMP::TEXT)",
            [str(n_nodes), f"SLSO sub-basin (outlet={OUTLET_NODE})"])

out.close()
print(f"\nDuckDB written: {TARGET_DB}")

# ── 4. Subset forcing NetCDF ──────────────────────────────────────────
if SOURCE_NC.exists():
    ds = xr.open_dataset(SOURCE_NC)
    # Index along the node dimension with old indices
    node_dim = [d for d in ds.dims if "node" in d.lower() or ds.sizes[d] == 2889]
    if node_dim:
        dim_name = node_dim[0]
    else:
        # Fallback: assume second dimension is nodes
        var0 = list(ds.data_vars)[0]
        dim_name = ds[var0].dims[1]

    ds_sub = ds.isel(**{dim_name: old_indices})
    ds_sub.to_netcdf(TARGET_NC)
    ds.close()
    print(f"Forcing written: {TARGET_NC}  (shape: {dict(ds_sub.sizes)})")
else:
    print(f"[!] Source forcing not found: {SOURCE_NC}")

print("\nDone! Use config/slso-sub.toml to train on this sub-basin.")
