"""Construit le bassin SLSO open data, piloté par .runs/slso-od/config/slso-od.toml.

Tout est paramétré dans le TOML ([basin.build] + [paths] + [temporal]).
Sources géo et forçage : cache global partagé D:/meandre-data (re-créable).
Sorties : .runs/slso-od/data/basin.duckdb (graphe + territorial + longueurs de
tronçon D8 + observations HYDAT).

  python .runs/slso-od/build_basin_od.py
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
import tomllib
from pathlib import Path
import geopandas as gpd

from meandre.data.open_data import (
    download_all, download_observations_hydat, populate_basin_observations,
)
from meandre.data.basin_builder import build_basin

CFG = Path(".runs/slso-od/config/slso-od.toml")
cfg = tomllib.load(open(CFG, "rb"))
B = cfg["basin"]["build"]
P = cfg["paths"]
T = cfg["temporal"]

# Mêmes règles de résolution que slso.py : relatif au dossier du cas
# (parent de config/), absolu (D:/...) gardé tel quel.
RUN_DIR = CFG.resolve().parent.parent
def _rp(key: str) -> Path:
    p = Path(P[key])
    return p if p.is_absolute() else RUN_DIR / p

GEO_CACHE = _rp("geo_cache")                       # D:/meandre-data/geo_cache/slso-od
HYDAT_CACHE = Path("D:/meandre-data/hydat")        # cache HYDAT global partagé
BASIN_DB = _rp("basin_db")
WITHDRAWALS = Path(".runs/slso/data/io-eau-meandre.parquet")
BASIN_DB.parent.mkdir(parents=True, exist_ok=True)
if BASIN_DB.exists():
    BASIN_DB.unlink()

print(f"Config       : {CFG}")
print(f"Geo cache    : {GEO_CACHE}")
print(f"Sortie       : {BASIN_DB}")
print(f"bbox={B['bbox']} outlet={B['outlet']} min_area={B['min_area_km2']} "
      f"seg_area={B['max_segment_area_km2']} seg_len={B['max_segment_length_km']}")

# 1. Données géo (réutilise le cache global, aucun téléchargement si présent)
rasters = download_all(bbox=tuple(B["bbox"]), cache_dir=GEO_CACHE)

# 2. Jauges HYDAT (téléchargées AVANT le build pour ancrer le réseau). Les
# emplacements de stations deviennent des nœuds préservés de la fusion, afin
# que chaque jauge ait un nœud à la bonne aire drainée (corrige le biais
# d'aire systématique +38 % diagnostiqué 2026-06-10).
import numpy as np, pandas as pd
hydat = download_observations_hydat(
    bbox=tuple(B["bbox"]), cache_dir=HYDAT_CACHE,
    start_date=T["date_start"], end_date=T["date_end"],
)
anchor_coords = anchor_areas = None
sta_path = obs_path = None
if hydat is not None:
    sta_path, obs_path = hydat
    sta = pd.read_parquet(sta_path)
    w, s, e, n = B["bbox"]
    inb = (sta["lon"].between(w, e)) & (sta["lat"].between(s, n))
    sta_in = sta[inb]
    anchor_coords = sta_in[["lon", "lat"]].to_numpy(dtype=float)
    anchor_areas = sta_in["drainage_area_km2"].to_numpy(dtype=float)
    print(f"Ancres jauges : {len(anchor_coords)} stations HYDAT dans le bbox")

# 3. Construction du bassin (graphe + territorial + longueurs D8 + ancrage jauges)
slso_mask = gpd.read_parquet("data/regions.parquet")
slso_mask = slso_mask[slso_mask["layer"] == "slso"]

cache = build_basin(
    dem_path=rasters["dem"],
    landcover_path=rasters["landcover"],
    soil_dir=rasters["soil_dir"],
    outlet=tuple(B["outlet"]),
    basin_db=BASIN_DB,
    min_area_km2=B["min_area_km2"],
    max_subcatchments=B["max_subcatchments"],
    max_segment_area_km2=B["max_segment_area_km2"],
    max_segment_length_km=B["max_segment_length_km"],
    max_dem_pixels=B["max_dem_pixels"],
    min_lake_area_km2=B["min_lake_area_km2"],
    water_occurrence_path=rasters["water_occurrence"],
    lai_path=rasters["lai"],
    nrcan_lc_path=rasters["nrcan_lc"],
    basin_mask_gdf=slso_mask,
    anchor_coords=anchor_coords,
    anchor_areas=anchor_areas,
)

# 4. Insertion des observations (snap final vers le réseau désormais ancré)
if sta_path is not None:
    n_obs = populate_basin_observations(BASIN_DB, sta_path, obs_path)
    print(f"HYDAT : {n_obs:,} observations insérées")
else:
    print("[!] HYDAT indisponible")

obs = cache.load_observations(
    date_start=T["date_start"], date_end=T["date_end"], min_valid_days=365,
)
print(f"Stations retenues : {obs['n_stations']}")

# 4. Prélèvements et rejets (parquet curé manuellement, optionnel)
if WITHDRAWALS.exists():
    n = cache.import_withdrawals(WITHDRAWALS, site_col="site_id")
    print(f"Prélèvements importés : {n} lignes")
else:
    print(f"[!] Prélèvements non trouvés : {WITHDRAWALS}")

print("BUILD_OD_DONE", flush=True)
