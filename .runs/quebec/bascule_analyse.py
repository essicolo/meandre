"""COURBE DE BASCULE station vs CaSR, SANS DÉBIT.
Pour chaque station ECCC tenue à l'écart, on estime sa précipitation quotidienne de deux
façons et on les compare à ses observations :
  (a) interpolation depuis les AUTRES stations (inverse de la distance, 4 voisines) —
      c'est ce que fait un produit krigé type SIMAT, la station retirée ne s'auto-informe
      donc pas (lire quebec.zarr à sa position serait optimiste : elle y est incluse) ;
  (b) CaSR, lu au nœud méandre le plus proche (les 28035 nœuds couvrent la province).
On range ensuite les stations par distance à leur plus proche voisine CONSERVÉE et on
trace les deux qualités. Le croisement donne la distance de bascule, et donc le poids de
mélange continu par nœud. Aucune observation de débit n'entre dans ce calcul.

  PYTHONIOENCODING=utf-8 python .runs/quebec/bascule_analyse.py 2016 2019
"""
import os, sys, glob
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, xarray as xr
from scipy.spatial import cKDTree
from meandre.data.basin_cache import BasinCache

Y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2016
Y1 = int(sys.argv[2]) if len(sys.argv) > 2 else 2019
REGS = ["gasp","sagu","mont","labi","abit","cnda","cndb","cndc","cndd","cnde","outm","outv","slno","slso","vaud"]

d = pd.concat([pd.read_parquet(f) for f in glob.glob("D:/meandre-data/eccc/daily_*.parquet")],
              ignore_index=True)
d = d.drop_duplicates(["climate_id", "date"])
d["date"] = pd.to_datetime(d["date"])
d = d[(d.date.dt.year >= Y0) & (d.date.dt.year <= Y1) & d.P.notna()]
d = d[d.lon.between(-80, -60) & d.lat.between(44, 54)]
print(f"[eccc] {len(d):,} obs | {d.climate_id.nunique()} stations | {Y0}-{Y1}", flush=True)

cnt = d.groupby("climate_id").size()
keep = cnt[cnt >= 0.6 * 365 * (Y1 - Y0 + 1)].index
d = d[d.climate_id.isin(keep)]
pos = d.groupby("climate_id")[["lon", "lat"]].first()
print(f"[eccc] {len(pos)} stations suffisamment complètes (>=60 % des jours)", flush=True)

lat0 = float(pos.lat.mean())
def proj(lon, lat): return np.c_[np.asarray(lon)*111.32*np.cos(np.radians(lat0)), np.asarray(lat)*110.57]
Pst = proj(pos.lon.values, pos.lat.values)
tree = cKDTree(Pst)

# CaSR au nœud le plus proche de chaque station (tous les nœuds de la province)
noeuds, casr = [], []
for reg in REGS:
    db = ".runs/slso/data/slso.duckdb" if reg == "slso" else f"D:/meandre-data/quebec/{reg}.duckdb"
    fx = f"D:/meandre-data/quebec/forcing-{reg}.nc"
    if not os.path.exists(fx):
        continue
    h = BasinCache(db).load(device="cpu"); nc = h["node_coords"].numpy()
    lat_col = 0 if 40 < float(np.nanmean(nc[:, 0])) < 62 else 1
    ds = xr.open_dataset(fx)
    t = pd.to_datetime(ds["time"].values)
    m = (t.year >= Y0) & (t.year <= Y1)
    P = ds["forcing"].values[m][:, :, 0]; ds.close()
    noeuds.append(proj(nc[:, 1-lat_col], nc[:, lat_col])); casr.append(P)
    print(f"  [{reg}] {P.shape[1]} nœuds, {P.shape[0]} jours", flush=True)
Pn = np.vstack(noeuds); Pcasr = np.hstack(casr)
tn = pd.to_datetime(xr.open_dataset(f"D:/meandre-data/quebec/forcing-gasp.nc")["time"].values)
tn = tn[(tn.year >= Y0) & (tn.year <= Y1)]
tree_n = cKDTree(Pn)
print(f"[casr] {Pn.shape[0]} nœuds | {Pcasr.shape[0]} jours", flush=True)

piv = d.pivot_table(index="date", columns="climate_id", values="P", aggfunc="first")
piv = piv.reindex(tn)
ids = list(pos.index)
dist_all, _ = tree.query(Pst, k=2)
rows = []
for i, sid in enumerate(ids):
    obs = piv[sid].values if sid in piv.columns else None
    if obs is None: continue
    # (a) IDW depuis les 4 stations voisines (la station elle-même exclue)
    dd, jj = tree.query(Pst[i], k=5)
    dd, jj = dd[1:], jj[1:]
    w = 1.0 / np.clip(dd, 1.0, None) ** 2; w /= w.sum()
    vois = np.array([piv[ids[j]].values if ids[j] in piv.columns else np.full(len(tn), np.nan) for j in jj])
    idw = np.nansum(vois * w[:, None], axis=0)
    idw[np.all(~np.isfinite(vois), axis=0)] = np.nan
    # (b) CaSR au nœud le plus proche
    _, jn = tree_n.query(Pst[i], k=1)
    cas = Pcasr[:, int(jn)]
    v = np.isfinite(obs) & np.isfinite(idw) & np.isfinite(cas)
    if v.sum() < 500: continue
    def sk(x):
        r = np.corrcoef(obs[v], x[v])[0, 1]
        return r, float(x[v].sum() / max(obs[v].sum(), 1e-6))
    r_i, b_i = sk(idw); r_c, b_c = sk(cas)
    rows.append(dict(climate_id=sid, d_voisine_km=float(dd[0]), n=int(v.sum()),
                     r_stations=round(r_i, 3), biais_stations=round(b_i, 3),
                     r_casr=round(r_c, 3), biais_casr=round(b_c, 3)))
df = pd.DataFrame(rows)
df.to_csv("reports/bascule_meteo.csv", index=False)
b = [0, 10, 15, 20, 30, 40, 60, 100, 1e9]
df["classe"] = pd.cut(df.d_voisine_km, b)
g = df.groupby("classe", observed=True).agg(n=("climate_id", "size"),
    r_st=("r_stations", "median"), r_casr=("r_casr", "median"),
    b_st=("biais_stations", "median"), b_casr=("biais_casr", "median")).round(3)
print(f"\n{len(df)} stations évaluées\n")
print(g.to_string())
