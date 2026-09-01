"""DISTANCE DE BASCULE entre grille krigée (stations) et CaSR, mesurée SANS DÉBIT.
Validation croisée météo pure : on retire une station, on compare à son emplacement la
précipitation des deux produits à ses observations, et on regarde comment la qualité de
chacun varie avec la distance aux stations restantes. Le point de croisement donne le
poids de mélange continu par nœud — critère météorologique, non circulaire (le débit
n'intervient jamais). Motivation : la carte hybride du 3 août perd 40-50 % du volume
d'eau là où le réseau est clairsemé (labi beta 0.57, abit 0.60, cndb 0.50).

  PYTHONIOENCODING=utf-8 python .runs/quebec/meteo_bascule.py 2015 2020
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, xarray as xr
from scipy.spatial import cKDTree
from meandre.data.eccc_loader import fetch_stations, fetch_daily

# Racines portables (portage grappe, 2026-09-01) : les chemins absolus rendaient toute
# execution hors du poste d'origine impossible. Defauts inchanges.
import os as _osp
_RQH_ROOT = _osp.environ.get("MEANDRE_RQH", "C:/Users/parse01/documents-locaux/rqh-local")
_DATA_ROOT = _osp.environ.get("MEANDRE_DATA", "D:/meandre-data")

Y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
Y1 = int(sys.argv[2]) if len(sys.argv) > 2 else 2020
BBOX = (-80.0, 44.5, -60.0, 53.5)
ZARR = f"{_RQH_ROOT}/io_2026-04/data/03_imputation/quebec.zarr"

st = fetch_stations(BBOX)
for c in ("lon", "lat"):
    v = st[c].astype(float)
    st[c] = np.where(np.abs(v) > 1000, v / 1e7, v)
st = st[st.lon.between(-80, -60) & st.lat.between(44, 54)].drop_duplicates("climate_id")
print(f"stations inventaire : {len(st)}", flush=True)
# L API expire sur une requête province entière : on découpe en tuiles de 4°
# de longitude, année par année (fetch_daily met en cache par tuile+année, donc une
# reprise après coupure ne retélécharge rien).
parts = []
for lon0 in range(-80, -60, 4):
    for y in range(Y0, Y1 + 1):
        bb = (float(lon0), BBOX[1], float(min(lon0 + 4, -60)), BBOX[3])
        try:
            d = fetch_daily(bb, y, y)
        except Exception as ex:
            print(f"  tuile {bb[0]:.0f}..{bb[2]:.0f} {y} : ECHEC {type(ex).__name__}", flush=True)
            continue
        if d is not None and len(d):
            parts.append(d)
            print(f"  tuile {bb[0]:.0f}..{bb[2]:.0f} {y} : {len(d):,} lignes", flush=True)
if not parts:
    raise SystemExit("aucune donnée ECCC récupérée")
df = pd.concat(parts, ignore_index=True).drop_duplicates()
print(f"observations quotidiennes : {len(df):,} lignes | colonnes {list(df.columns)[:8]}", flush=True)
df.to_parquet(f"{_DATA_ROOT}/quebec/eccc_daily_{Y0}_{Y1}.parquet")
print("-> eccc_daily parquet écrit")
