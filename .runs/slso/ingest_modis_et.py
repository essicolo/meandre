"""Ingest MODIS MOD16A2 ETR into the SLSO DuckDB.

Fetches 8-day ETR composites from the Microsoft Planetary Computer STAC
catalogue for the SLSO basin extent and date range, then upserts them into
the ``modis_et`` table of the basin DuckDB.

Usage
-----
    python .runs/slso/ingest_modis_et.py
    python .runs/slso/ingest_modis_et.py .runs/slso/config/slso-kendall-gal-v2.toml

The script is idempotent: re-running it only adds new rows (INSERT OR REPLACE).
"""
import os
import sys
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[2])

import tomllib
import torch
import numpy as np

from meandre.utils.paths import run_dir_from_config, resolve_run_path

CFG_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".runs/slso/config/slso.toml")
with open(CFG_PATH, "rb") as f:
    cfg = tomllib.load(f)

RUN_DIR = run_dir_from_config(CFG_PATH)

def _p(key: str) -> Path:
    return resolve_run_path(cfg["paths"][key], RUN_DIR)

BASIN_DB = _p("basin_db")
DATE_START = cfg["temporal"]["date_start"]
DATE_END = cfg["temporal"]["date_end"]

print(f"Config   : {CFG_PATH}")
print(f"Basin DB : {BASIN_DB}")
print(f"Period   : {DATE_START} → {DATE_END}")

# ── Load basin metadata ────────────────────────────────────────────────────
from meandre.data.basin_cache import BasinCache

cache = BasinCache(BASIN_DB)
hydro = cache.load(device=torch.device("cpu"))
node_coords = hydro["node_coords"].cpu().numpy()  # (N, 2) lon/lat
n_nodes = node_coords.shape[0]

lons, lats = node_coords[:, 0], node_coords[:, 1]
bbox = (float(lons.min()) - 0.1,
        float(lats.min()) - 0.1,
        float(lons.max()) + 0.1,
        float(lats.max()) + 0.1)

print(f"Nodes    : {n_nodes}")
print(f"Bbox     : {bbox}")

# ── Check if already ingested ──────────────────────────────────────────────
if cache.has_modis_et():
    ans = input(
        "\nmodis_et table already exists. Re-fetch and overwrite? [y/N] "
    ).strip().lower()
    if ans != "y":
        print("Aborted.")
        sys.exit(0)

# ── Fetch ──────────────────────────────────────────────────────────────────
print(f"\nFetching MODIS MOD16A2 from Planetary Computer…")
from meandre.data.modis_loader import fetch_modis_et

df = fetch_modis_et(
    bbox=bbox,
    date_start=DATE_START,
    date_end=DATE_END,
    node_coords=node_coords,
    node_indices=np.arange(n_nodes),
)

if df.empty:
    print("No data fetched — check bbox and date range.")
    sys.exit(1)

n_composites = df["date"].nunique()
n_good = df["quality_ok"].sum()
print(f"\nFetched  : {len(df):,} rows, {n_composites} composites, "
      f"{n_good:,} good-quality ({n_good/len(df):.1%})")
print(f"ETR range: {df['etr_mm_day'].min():.3f} – {df['etr_mm_day'].max():.3f} mm/day")

# ── Ingest ─────────────────────────────────────────────────────────────────
n_inserted = cache.import_modis_et(df)
print(f"\nIngested : {n_inserted:,} rows into modis_et")
print("Done. Run slso.py to use MODIS ETR in training (set w_nll_et > 0).")
