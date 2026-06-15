"""Ingestion ETR MOD16A2GF 8-jours dans modis_et (remplace l'annuel A3GF).

Année par année, resumable : les granules sont cachés sur D: (gros volume),
l'import DuckDB est idempotent (INSERT OR REPLACE). Sur restart, les granules
déjà téléchargés sont sautés et les années ré-importées sans dommage.

Lancer (WSL) :
  ./.venv-wsl/bin/python .runs/slso/ingest_modis_et8.py
"""
import os, sys
from pathlib import Path
os.chdir(Path(__file__).resolve().parents[2])

import torch, duckdb, numpy as np
from meandre.data.basin_cache import BasinCache
from meandre.data.modis_loader import fetch_modis_et_8day

BASIN_DB = ".runs/slso/data/slso.duckdb"
CACHE_DIR = "/mnt/d/meandre-data/modis8"   # gros volume sur D:
YEAR_START, YEAR_END = 2000, 2024

cache = BasinCache(BASIN_DB)
hydro = cache.load(device=torch.device("cpu"))
nc = hydro["node_coords"].cpu().numpy()
n_nodes = len(nc)
bbox = (float(nc[:, 0].min()) - 0.1, float(nc[:, 1].min()) - 0.1,
        float(nc[:, 0].max()) + 0.1, float(nc[:, 1].max()) + 0.1)
node_indices = np.arange(n_nodes)
print(f"nodes {n_nodes} | bbox {tuple(round(b,2) for b in bbox)} | cache {CACHE_DIR}", flush=True)

# Nettoyer l'annuel MOD16A3GF UNE fois (skip si du 8-jours est déjà présent)
con = duckdb.connect(BASIN_DB)
tabs = [t[0] for t in con.execute("show tables").fetchall()]
if "modis_et" in tabs:
    has_8day = con.execute(
        "select count(*) from modis_et where not (month(date)=1 and day(date)=1)"
    ).fetchone()[0]
    if has_8day == 0:
        n = con.execute("select count(*) from modis_et").fetchone()[0]
        con.execute("DELETE FROM modis_et")
        print(f"Table annuelle nettoyée ({n} lignes supprimées)", flush=True)
    else:
        print(f"8-jours déjà présent ({has_8day} lignes non-Jan1) — pas de nettoyage", flush=True)
con.close()

for year in range(YEAR_START, YEAR_END + 1):
    df = fetch_modis_et_8day(
        bbox, f"{year}-01-01", f"{year}-12-31", nc, node_indices, cache_dir=CACHE_DIR,
    )
    if df.empty:
        print(f"{year}: aucun granule", flush=True)
        continue
    n = cache.import_modis_et(df)
    good = df["quality_ok"].mean()
    print(f"{year}: {len(df):,} lignes ({df['date'].nunique()} composites, "
          f"{good:.0%} quality) -> import {n:,}", flush=True)

print("INGESTION TERMINÉE", flush=True)
