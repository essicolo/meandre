"""Ingest the corrected withdrawal parquet into the SLSO DuckDB.

Replaces the old `withdrawals` table (backup kept as `withdrawals_old`)
with the SLSO subset of `io-eau-meandre.parquet`.

Convention (BasinCache.import_withdrawals + routing/withdrawals.py):
    positive = water added (effluent / return flow)
    negative = water removed (pumping / consumptive use)
"""
from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import duckdb
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from meandre.data.basin_cache import BasinCache

with open("notebooks/slso/config/slso.toml", "rb") as f:
    cfg = tomllib.load(f)
db_path = cfg["paths"]["basin_db"]
parquet = "notebooks/slso/data/io-eau-meandre.parquet"

# ── 1. Backup + drop existing withdrawals ──────────────────────────
con = duckdb.connect(db_path)
tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
if "withdrawals" in tables:
    n_old = con.execute("SELECT COUNT(*) FROM withdrawals").fetchone()[0]
    nodes_old = con.execute("SELECT COUNT(DISTINCT node_idx) FROM withdrawals").fetchone()[0]
    print(f"[backup] existing withdrawals: {n_old:,} rows, {nodes_old} active nodes")

    if "withdrawals_old" in tables:
        con.execute("DROP TABLE withdrawals_old")
    con.execute("CREATE TABLE withdrawals_old AS SELECT * FROM withdrawals")
    con.execute("DROP TABLE withdrawals")
    print("[backup] saved → withdrawals_old, dropped withdrawals")
con.close()

# ── 2. Filter parquet to SLSO troncons ─────────────────────────────
df = pd.read_parquet(parquet)
df["date"] = pd.to_datetime(df["date"])
slso = df[df["IDTRONCON"].str.startswith("SLSO")].copy()
print(f"[filter] kept {len(slso):,} rows (SLSO subset) of {len(df):,}")
print(f"  unique sites:    {slso['site_id'].nunique()}")
print(f"  unique troncons: {slso['IDTRONCON'].nunique()}")
print(f"  date range:      {slso['date'].min()} to {slso['date'].max()}")
print(f"  sources:         {slso['source'].value_counts().to_dict()}")

# Drop NULL source rows (can't classify surface/gw)
n_null = (slso["source"] == "NULL").sum()
if n_null:
    slso = slso[slso["source"] != "NULL"]
    print(f"  dropped {n_null} rows with NULL source")

# ── 3. Sanity: yearly net before import ────────────────────────────
slso["year"] = slso["date"].dt.year
yearly = slso.groupby("year").agg(
    n_records=("net_withdrawal", "count"),
    sum_net=("net_withdrawal", "sum"),
    surface_only=("source", lambda s: (s == "Surface").sum()),
).round(2)
print("\n[before import] yearly stats (SLSO subset):")
print(yearly.tail(10))

# ── 4. Import via cache helper (snaps via lon/lat) ─────────────────
cache = BasinCache(db_path)
print("\n[import] snapping sites to nearest model nodes...")
n_inserted = cache.import_withdrawals(
    source=slso,
    date_col="date",
    net_col="net_withdrawal",
    lon_col="lon",
    lat_col="lat",
    site_col="site_id",
    source_col="source",
    max_snap_km=10.0,
)
print(f"[import] inserted {n_inserted:,} rows into withdrawals")

# ── 5. Verify post-import ──────────────────────────────────────────
con = duckdb.connect(db_path, read_only=True)
print("\n[verify] new withdrawals table:")
print(con.execute("""
    SELECT EXTRACT(YEAR FROM date) AS year,
           COUNT(*) AS rows,
           SUM(net_surface) AS sum_surf,
           SUM(net_gw) AS sum_gw,
           AVG(net_surface) AS avg_surf
    FROM withdrawals
    WHERE EXTRACT(YEAR FROM date) BETWEEN 2017 AND 2024
    GROUP BY year ORDER BY year
""").fetchdf())

print("\n[verify] active nodes:")
print(con.execute("""
    SELECT COUNT(DISTINCT node_idx) AS n_active_nodes,
           COUNT(*) AS total_rows,
           MIN(date) AS first_date, MAX(date) AS last_date
    FROM withdrawals
""").fetchdf())

print("\n[verify] comparison with backup:")
print(con.execute("""
    SELECT 'NEW' AS src, COUNT(*) AS rows,
           COUNT(DISTINCT node_idx) AS nodes,
           SUM(net_surface) AS sum_surf
    FROM withdrawals
    WHERE EXTRACT(YEAR FROM date) BETWEEN 2017 AND 2024
    UNION ALL
    SELECT 'OLD', COUNT(*),
           COUNT(DISTINCT node_idx),
           SUM(net_surface)
    FROM withdrawals_old
    WHERE EXTRACT(YEAR FROM date) BETWEEN 2017 AND 2024
""").fetchdf())
con.close()

print("\n[done] withdrawals replaced.  Old data preserved in `withdrawals_old`.")
