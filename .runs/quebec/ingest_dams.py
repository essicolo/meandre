"""Phase 0 (design_modules_appris.md) : ingère le Répertoire des barrages MELCCFP
(repertoire_des_barrages.xls, ~8600 ouvrages) dans les 15 bases régionales.
Table `dams` : ouvrage -> nœud le plus proche (haversine) + attributs utiles
(catégorie, utilisation, hauteur de retenue, capacité m3, sup. réservoir).
Les ouvrages hors domaine restent filtrables par dist_km.

  python .runs/quebec/ingest_dams.py
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
import duckdb
from meandre.data.basin_cache import BasinCache

# Racines portables (portage grappe, 2026-09-01) : les chemins absolus rendaient toute
# execution hors du poste d'origine impossible. Defauts inchanges.
import os as _osp
_DATA_ROOT = _osp.environ.get("MEANDRE_DATA", "D:/meandre-data")

XLS = f"{_DATA_ROOT}/quebec/barrages/repertoire_des_barrages.xls"
REGIONS = ["abit", "cnda", "cndb", "cndc", "cndd", "cnde", "gasp", "labi", "mont",
           "outm", "outv", "sagu", "slno", "slso", "vaud"]
DBS = {"slso": ".runs/slso/data/slso.duckdb"}

df = pd.read_excel(XLS, header=1)
df.columns = [str(c).strip() for c in df.columns]
df = df.rename(columns={
    "Numéro du barrage": "dam_id", "Nom du barrage": "name",
    "Latitude (NAD 83)": "lat", "Longitude (NAD 83)": "lon",
    "Bassin": "basin", "Catégorie administrative": "category",
    "Utilisation": "usage", "Hauteur de retenue (m)": "height_m",
    "Sup. bassin (km2)": "drainage_km2", "Année de construction": "year_built",
    "Capacité de retenue (m3)": "capacity_m3", "Sup. réservoir (ha)": "reservoir_ha",
    "Longueur de refoulement (m)": "backwater_m", "Propriétaire": "owner",
})
cols = ["dam_id", "name", "lat", "lon", "basin", "category", "usage", "height_m",
        "drainage_km2", "year_built", "capacity_m3", "reservoir_ha", "backwater_m", "owner"]
df = df[cols].copy()
for c in ["lat", "lon", "height_m", "drainage_km2", "capacity_m3", "reservoir_ha", "backwater_m"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["lat", "lon"])
print(f"[dams] répertoire : {len(df)} ouvrages géolocalisés | capacité totale {df.capacity_m3.sum()/1e9:.1f} km3")

R = 6371.0
for reg in REGIONS:
    db = DBS.get(reg, f"{_DATA_ROOT}/quebec/{reg}.duckdb")
    cache = BasinCache(db)
    h = cache.load(device="cpu")
    coords = h["node_coords"].numpy()
    lat_col = 0 if 40 < float(np.nanmean(coords[:, 0])) < 62 else 1
    nlat, nlon = coords[:, lat_col], coords[:, 1 - lat_col]

    m = (df.lat >= nlat.min() - 0.1) & (df.lat <= nlat.max() + 0.1) & \
        (df.lon >= nlon.min() - 0.1) & (df.lon <= nlon.max() + 0.1)
    sub = df[m].copy()
    if len(sub) == 0:
        print(f"[{reg}] 0 barrage dans le bbox"); continue
    la1, lo1 = np.radians(sub.lat.values)[:, None], np.radians(sub.lon.values)[:, None]
    la2, lo2 = np.radians(nlat)[None, :], np.radians(nlon)[None, :]
    d = 2 * R * np.arcsin(np.sqrt(np.sin((la2 - la1) / 2) ** 2 +
                                  np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2))
    sub["node_idx"] = d.argmin(axis=1)
    sub["dist_km"] = d.min(axis=1)

    con = duckdb.connect(db)
    con.execute("drop table if exists dams")
    con.execute("create table dams as select * from sub")
    con.close()
    big = sub[(sub.dist_km < 5) & (sub.capacity_m3 > 1e7)]
    print(f"[{reg}] {len(sub)} ouvrages (bbox) | <5 km du réseau et >10 hm3 : {len(big)} | "
          f"capacité mappée {sub[sub.dist_km < 5].capacity_m3.sum()/1e9:.2f} km3", flush=True)
print("[dams] DONE")
