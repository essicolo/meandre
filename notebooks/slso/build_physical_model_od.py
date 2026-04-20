# ---
# jupyter:
#   jupytext:
#     text_representation:
#       format_name: percent
#   kernelspec:
#     display_name: meandre
#     language: python
#     name: meandre
# ---

# %% [markdown]
# # Construction du modèle physique — SLSO (Planetary Computer)
#
# Version alternative du cache DuckDB construite entièrement à partir de
# données ouvertes téléchargées via Planetary Computer (Copernicus DEM 30m,
# ESA WorldCover 10m) et ISRIC SoilGrids 250m.
#
# **Objectif** : comparer la qualité du modèle Physitel (`slso.duckdb`) avec
# une version dérivée automatiquement des données ouvertes (`slso-od.duckdb`).
#
# **Prérequis** : `pip install meandre[geo]`
# (pystac-client, planetary-computer, rioxarray, pysheds, rasterstats, rasterio)
#
# **À exécuter une seule fois**, ou lorsque les paramètres du bassin changent.

# %%
import os
import subprocess
from pathlib import Path

os.chdir(subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], text=True
).strip())

import tomllib

with open("notebooks/slso/config/slso.toml", "rb") as f:
    cfg = tomllib.load(f)

# Paths
PC_DB        = Path("notebooks/slso/data/slso-od.duckdb")
GEO_CACHE    = Path("notebooks/slso/data/geo_cache")   # DEM / landcover / sol mis en cache
PHYSITEL_DB  = Path(cfg["paths"]["basin_db"])           # référence pour comparer

# Emprise du bassin SLSO — resserrée pour limiter la mémoire pysheds.
# À 30m : ~1° ≈ 3 700 pixels → 3° × 3° ≈ 123 M pixels (trop grand).
# On restreint à l'enveloppe réelle du bassin + 0.1° de marge.
# (west, south, east, north) en EPSG:4326
BBOX = (-73.0, 44.5, -69.6, 47.7)

# Point d'exutoire : embouchure de la Chaudiere dans le Saint-Laurent, pres de Levis.
# Source : noeud le plus aval du modele Physitel (topo_order max).
OUTLET = (-71.27, 46.77)   # (lon, lat)

MIN_AREA_KM2      = 1.5    # seuil drainage minimal — augmenter réduit le nb de nœuds
MAX_SUBCATCHMENTS = 3500   # nb max de sous-bassins (confluences retenues)

print(f"DuckDB PC   : {PC_DB}")
print(f"Cache géo   : {GEO_CACHE}")
print(f"Bbox        : {BBOX}")
print(f"Exutoire    : {OUTLET}")

# %% [markdown]
# ## 1. Téléchargement des données géospatiales

# %%
from meandre.data.open_data import download_all

print("Téléchargement des données géospatiales (mis en cache après la 1ère fois)...")
rasters = download_all(bbox=BBOX, cache_dir=GEO_CACHE)

dem_path       = rasters["dem"]
landcover_path = rasters["landcover"]
soil_dir       = rasters["soil_dir"]
water_occ_path = rasters["water_occurrence"]          # JRC surface water
lai_path       = rasters["lai"]                       # MODIS LAI
nrcan_lc_path  = rasters["nrcan_lc"]                  # NRCan land cover (None si indisponible)
grhq_path      = rasters["grhq"]                      # Géobase RHQ (None si indisponible)

print(f"\nDEM        : {dem_path}")
print(f"Land cover : {landcover_path}")
print(f"Sol        : {soil_dir}")
print(f"GRHQ       : {grhq_path}")

# %% [markdown]
# ## 2. Construction du bassin

# %%
import geopandas as gpd
from meandre.data.basin_builder import build_basin

# Load SLSO polygon mask (replaces single-outlet delineation)
_regions = gpd.read_parquet("notebooks/regions.parquet")
slso_mask = _regions[_regions["layer"] == "slso"]

if PC_DB.exists():
    PC_DB.unlink()
    print(f"Ancien cache supprime: {PC_DB}")

PC_DB.parent.mkdir(parents=True, exist_ok=True)

cache = build_basin(
    dem_path              = dem_path,
    landcover_path        = landcover_path,
    soil_dir              = soil_dir,
    outlet                = OUTLET,
    basin_db              = PC_DB,
    min_area_km2          = MIN_AREA_KM2,
    max_subcatchments     = MAX_SUBCATCHMENTS,
    water_occurrence_path = water_occ_path,
    lai_path              = lai_path,
    nrcan_lc_path         = nrcan_lc_path,
    basin_mask_gdf        = slso_mask,
)

print(f"\nCache créé: {PC_DB}")

# %% [markdown]
# ## 3. Géométrie des tronçons (GRHQ)

# %%
import pandas as pd

REACH_PARQUET = PC_DB.parent / "reaches_od.parquet"

if grhq_path is not None:
    from meandre.data.open_data import build_reach_parquet as _build_reach

    # Charger les nœuds depuis le DuckDB
    import duckdb as _ddb
    _con = _ddb.connect(str(PC_DB), read_only=True)
    _nodes_df = _con.execute(
        "SELECT node_idx, node_id, lon, lat FROM nodes ORDER BY node_idx"
    ).df()
    _con.close()

    _build_reach(
        nodes_df     = _nodes_df,
        grhq_path    = grhq_path,
        out_path     = REACH_PARQUET,
        max_dist_deg = 0.05,          # ~5.5 km — augmenter si tronçons trop larges
    )
    print(f"Géométrie des tronçons : {REACH_PARQUET}")
else:
    print("[!] GRHQ non disponible — reaches_od.parquet non créé.")

# %% [markdown]
# ## 4. Vérification du réseau hydrographique

# %%
import numpy as np
import torch

hydro  = cache.load(device=torch.device("cpu"))
graph  = hydro["graph"]
coords = hydro["node_coords"].numpy()
lon, lat = coords[:, 0], coords[:, 1]
n_nodes  = hydro["n_nodes"]

print(f"Nœuds : {n_nodes}")
print(f"Arêtes: {graph.n_edges}")
print(f"Lacs  : {graph.is_lake.sum().item()}")

# Arêtes longues
edge_src = graph.edge_index[0].numpy()
edge_dst = graph.edge_index[1].numpy()
dx = lon[edge_src] - lon[edge_dst]
dy = lat[edge_src] - lat[edge_dst]
dist_km = np.sqrt(dx**2 + dy**2) * 111.0

if len(dist_km) == 0:
    print("\n[!] Aucune arête — vérifier le bassin versant.")
else:
    print(f"\nDistance des arêtes (km) :")
    print(f"  Médiane : {np.median(dist_km):.2f}")
    print(f"  P95     : {np.percentile(dist_km, 95):.2f}")
    print(f"  Max     : {np.max(dist_km):.2f}")

    LONG_EDGE_KM = 15.0
    long_edges = np.where(dist_km > LONG_EDGE_KM)[0]
    if len(long_edges):
        print(f"\n[!] {len(long_edges)} arêtes > {LONG_EDGE_KM} km")
        for i in long_edges[:10]:
            s, d = edge_src[i], edge_dst[i]
            print(f"  {s} -> {d}  ({dist_km[i]:.1f} km)")
    else:
        print(f"\nOK Aucune arete > {LONG_EDGE_KM} km")

# %% [markdown]
# ## 5. Observations hydrométriques

# %%
import duckdb

STATIONS_FILE = Path(r"C:\Users\parse01\documents-locaux\rqh-local\rqh_2026-04\data\07_stations\stations_concatenees.nc")

cache.import_observations(STATIONS_FILE, basin_prefix="SLSO")

obs = cache.load_observations(
    date_start     = cfg["temporal"]["date_start"],
    date_end       = cfg["temporal"]["date_end"],
    min_valid_days = 365,
)
print(f"Stations retenues : {obs['n_stations']}")

# %% [markdown]
# ## 6. Prélèvements et rejets

# %%
WITHDRAWALS_FILE = Path("notebooks/io-eau-meandre.parquet")

if WITHDRAWALS_FILE.exists():
    n_imported = cache.import_withdrawals(WITHDRAWALS_FILE, site_col="site_id")
    print(f"Prélèvements importés : {n_imported} lignes")
else:
    print(f"[!] Fichier non trouvé : {WITHDRAWALS_FILE}")

# %% [markdown]
# ## 7. Comparaison rapide Physitel vs Planetary Computer

# %%
if PHYSITEL_DB.exists():
    con_od  = duckdb.connect(str(PC_DB), read_only=True)
    con_phy = duckdb.connect(str(PHYSITEL_DB), read_only=True)

    for label, con in [("Physitel", con_phy), ("PC", con_od)]:
        n_nodes = con.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        n_edges = con.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        n_lakes = con.execute("SELECT COUNT(*) FROM nodes WHERE is_lake").fetchone()[0]
        n_sta   = con.execute("SELECT COUNT(DISTINCT station_id) FROM stations").fetchone()[0]
        print(f"{label:10s}  nœuds={n_nodes:5d}  arêtes={n_edges:5d}  lacs={n_lakes:4d}  stations={n_sta}")

    con_od.close()
    con_phy.close()
