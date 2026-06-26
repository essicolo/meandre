"""Télécharge l'historique MODIS snow (MOD10A1) sur la période TRAIN (2014-2018)
via Planetary Computer, importé dans la base (upsert, ne touche pas le 2023-24
déjà présent). Année par année pour que le progrès partiel survive.
  python .runs/slso/fetch_snow_history.py [DB] [year_start] [year_end]
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np
from meandre.data.basin_cache import BasinCache
from meandre.data.modis_loader import fetch_modis_snow

DB = sys.argv[1] if len(sys.argv) > 1 else "/home/essi/slso-data/slso.duckdb"
Y0 = int(sys.argv[2]) if len(sys.argv) > 2 else 2014
Y1 = int(sys.argv[3]) if len(sys.argv) > 3 else 2018

cache = BasinCache(DB)
h = cache.load(device="cpu")
nc = h["node_coords"].cpu().numpy().astype(np.float64)   # (n,2) lon,lat
lon, lat = nc[:, 0], nc[:, 1]
m = 0.2
bbox = (float(lon.min() - m), float(lat.min() - m), float(lon.max() + m), float(lat.max() + m))
node_idx = np.arange(nc.shape[0])
print(f"DB={DB}  n_nodes={nc.shape[0]}  bbox={bbox}  years {Y0}-{Y1}")

total = 0
for yr in range(Y0, Y1 + 1):
    print(f"\n=== {yr} ===", flush=True)
    try:
        df = fetch_modis_snow(bbox, f"{yr}-01-01", f"{yr}-12-31", nc, node_idx)
        if df is not None and len(df):
            cache.import_modis_snow(df)
            valid = int((df["quality_ok"]).sum())
            total += valid
            print(f"  [{yr}] {len(df):,} lignes, {valid:,} valides — importé (cumul {total:,})", flush=True)
        else:
            print(f"  [{yr}] aucune donnée", flush=True)
    except Exception as e:
        print(f"  [{yr}] ERREUR : {e}", flush=True)

print(f"\nFETCH SNOW DONE : {total:,} obs valides 2014-2018")
