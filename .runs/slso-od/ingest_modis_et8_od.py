"""Ingestion ETR MOD16A2GF 8-jours dans modis_et du bassin OPEN-DATA.

Variante de .runs/slso/ingest_modis_et8.py pointée sur slso-od (chemins Windows
natifs). Le 8-jours est le cadençage attendu par le trainer (et_obs « 8-day
agrégé en daily », trainer.py:283) — l'annuel MOD16A3GF placé sur une seule date
par an comparerait l'ET d'un jour d'hiver à la moyenne annuelle (non-sens).

Année par année, resumable (granules cachés sur D:, import idempotent).

  uv run python .runs/slso-od/ingest_modis_et8_od.py
"""
import os, sys
from pathlib import Path
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass
os.chdir(Path(__file__).resolve().parents[2])

import torch, duckdb, numpy as np
from meandre.data.basin_cache import BasinCache
from meandre.data.modis_loader import fetch_modis_et_8day

BASIN_DB = ".runs/slso-od/data/basin.duckdb"
# Lancé via WSL (.venv-wsl) car pyhdf (HDF4) ne charge pas sa DLL sous Windows
# natif. Cache WSL partagé avec le run PHYSITEL 8-jours → granules MODIS (mêmes
# tuiles région SLSO) déjà téléchargés, ré-extraction rapide pour les 6166 nœuds.
CACHE_DIR = "/mnt/d/meandre-data/modis8"
YEAR_START, YEAR_END = 2000, 2024

cache = BasinCache(BASIN_DB)
hydro = cache.load(device=torch.device("cpu"))
nc = hydro["node_coords"].cpu().numpy()
n_nodes = len(nc)
bbox = (float(nc[:, 0].min()) - 0.1, float(nc[:, 1].min()) - 0.1,
        float(nc[:, 0].max()) + 0.1, float(nc[:, 1].max()) + 0.1)
node_indices = np.arange(n_nodes)
print(f"nodes {n_nodes} | bbox {tuple(round(b,2) for b in bbox)} | cache {CACHE_DIR}", flush=True)

# Nettoyer un éventuel annuel MOD16A3GF (skip si 8-jours déjà présent)
con = duckdb.connect(BASIN_DB)
tabs = [t[0] for t in con.execute("show tables").fetchall()]
if "modis_et" in tabs:
    has_8day = con.execute(
        "select count(*) from modis_et where not (month(date)=1 and day(date)=1)"
    ).fetchone()[0]
    if has_8day == 0:
        n = con.execute("select count(*) from modis_et").fetchone()[0]
        con.execute("DELETE FROM modis_et")
        print(f"Table annuelle nettoyee ({n} lignes supprimees)", flush=True)
    else:
        print(f"8-jours deja present ({has_8day} lignes non-Jan1) — pas de nettoyage", flush=True)
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

print("INGESTION_ET8_TERMINEE", flush=True)
