"""Apparie les nœuds-lacs de méandre aux plans d'eau de HydroLAKES v1.0 (Messager et al.
2016, Nature Communications) par proximité de l'EXUTOIRE (Pour_long/Pour_lat).

Motif : le déficit contre l'ensemble Hydrotel est concentré sur les stations lacustres
(-0.22 contre -0.03 ailleurs) et la tête de lac, une fois libérée, apprend de la dispersion
sans structure physique (corrélation nulle avec la surface). Il manquait le descripteur :
on n'avait que la surface du tronçon, alors que la relation stockage-débit dépend du
VOLUME, de la PROFONDEUR et surtout du TEMPS DE SÉJOUR. Avec beta = 1, la loi implémentée
Q = k*(S/A)^beta*A se réduit à Q = k*S, donc k = 1/temps_de_sejour : le coefficient de
vidange devient l'inverse d'une quantité MESURÉE par lac, sans seuil arbitraire.

Sortie : D:/meandre-data/quebec/lacs_hydrolakes.parquet (region, node_idx, hylak_id,
lake_area_km2, vol_total_mcm, depth_avg_m, res_time_j, dis_avg_m3s, dist_km).
"""
import os, sys, glob
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, geopandas as gpd
from scipy.spatial import cKDTree
from meandre.data.basin_cache import BasinCache

# Racines portables (portage grappe, 2026-09-01) : les chemins absolus rendaient toute
# execution hors du poste d'origine impossible. Defauts inchanges.
import os as _osp
_DATA_ROOT = _osp.environ.get("MEANDRE_DATA", "D:/meandre-data")

SHP = glob.glob(f"{_DATA_ROOT}/hydrolakes/**/HydroLAKES_points_v10.shp", recursive=True)[0]
REGS = ["gasp", "sagu", "mont", "labi", "abit", "cnda", "cndb", "cndc", "cndd", "cnde",
        "outm", "outv", "slno", "slso", "vaud"]
DMAX = float(os.environ.get("HL_DMAX_KM", "10"))

g = gpd.read_file(SHP, bbox=(-80.0, 44.0, -60.0, 54.0))
print(f"[hydrolakes] {len(g)} plans d'eau dans le Québec méridional", flush=True)
lon = g["Pour_long"].values.astype(float); lat = g["Pour_lat"].values.astype(float)
lat0 = float(np.nanmean(lat))
def proj(lo, la): return np.c_[np.asarray(lo)*111.32*np.cos(np.radians(lat0)), np.asarray(la)*110.57]
tree = cKDTree(proj(lon, lat))

out = []
for reg in REGS:
    db = ".runs/slso/data/slso.duckdb" if reg == "slso" else f"{_DATA_ROOT}/quebec/{reg}.duckdb"
    try:
        h = BasinCache(db).load(device="cpu")
    except Exception as e:
        print(f"[{reg}] {type(e).__name__}"); continue
    nc = h["node_coords"].numpy()
    lat_col = 0 if 40 < float(np.nanmean(nc[:, 0])) < 62 else 1
    lac = h["graph"].is_lake.bool().numpy()
    idx = np.flatnonzero(lac)
    if not len(idx):
        continue
    P = proj(nc[idx, 1-lat_col], nc[idx, lat_col])
    d, j = tree.query(P, k=1)
    ok = d <= DMAX
    sub = g.iloc[j[ok]]
    out.append(pd.DataFrame(dict(
        region=reg, node_idx=idx[ok], hylak_id=sub["Hylak_id"].values,
        lake_area_km2=sub["Lake_area"].values, vol_total_mcm=sub["Vol_total"].values,
        depth_avg_m=sub["Depth_avg"].values, res_time_j=sub["Res_time"].values,
        dis_avg_m3s=sub["Dis_avg"].values, dist_km=d[ok])))
    print(f"[{reg}] {ok.sum()}/{len(idx)} nœuds-lacs appariés (<{DMAX:.0f} km) | "
          f"temps de séjour méd {np.median(sub['Res_time'].values):.0f} j | "
          f"profondeur méd {np.median(sub['Depth_avg'].values):.1f} m", flush=True)

df = pd.concat(out, ignore_index=True)
df.to_parquet(f"{_DATA_ROOT}/quebec/lacs_hydrolakes.parquet")
print(f"\n{len(df)} nœuds-lacs appariés au total -> lacs_hydrolakes.parquet")
print(df[["lake_area_km2", "depth_avg_m", "res_time_j", "dist_km"]].describe().round(2).to_string())
