"""PRODUIT MÉTÉO MIXTE, continu sur le territoire (aucune frontière régionale).
P/Tmin/Tmax = combinaison, par nœud et par jour, de deux estimateurs :
  (a) STATIONS : pondération inverse du carré de la distance sur les k stations ECCC les
      plus proches AYANT une observation ce jour-là (les poids sont recalculés chaque
      jour, une station en panne ne laisse pas de trou) ;
  (b) CaSR : la valeur du nœud dans forcing-<reg>.nc.
Le poids vient de la COURBE DE BASCULE mesurée sans débit (2026-08-03, 236 stations tenues
à l'écart) : les stations dominent en corrélation jusqu'à ~60 km (0.82-0.85 contre
0.68-0.80), CaSR au-delà (0.82-0.91 contre 0.76-0.59). Les biais de volume sont OPPOSÉS
(stations 0.82-0.93, CaSR 1.00-1.31), donc le mélange en annule une bonne part.
w_stations = 1 / (1 + (d/d0)^p), d0 = 55 km (croisement mesuré), p = 3 (transition nette
mais dérivable). Les canaux d'énergie (R_n, u2, e_a) restent CaSR : aucune station ne les
mesure. Sortie : forcing-<reg>-mix.nc, 6 canaux, même grille temporelle.

  PYTHONIOENCODING=utf-8 python .runs/quebec/build_forcing_mix.py gasp
"""
import os, sys, glob
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, xarray as xr
from scipy.spatial import cKDTree
from meandre.data.basin_cache import BasinCache

REG = sys.argv[1].lower()
K = int(os.environ.get("MIX_K", "6"))          # voisines retenues
D0 = float(os.environ.get("MIX_D0", "55"))     # km, croisement mesuré
PEXP = float(os.environ.get("MIX_P", "3"))
DB = ".runs/slso/data/slso.duckdb" if REG == "slso" else f"D:/meandre-data/quebec/{REG}.duckdb"
FX = f"D:/meandre-data/quebec/forcing-{REG}.nc"
OUT = f"D:/meandre-data/quebec/forcing-{REG}-mix.nc"

ds = xr.open_dataset(FX)
F = ds["forcing"].values.copy()                # (T, N, 6) : P, Tmin, Tmax, R_n, u2, e_a
times = pd.to_datetime(ds["time"].values)
h = BasinCache(DB).load(device="cpu"); nc = h["node_coords"].numpy()
lat_col = 0 if 40 < float(np.nanmean(nc[:, 0])) < 62 else 1
lon_n, lat_n = nc[:, 1 - lat_col], nc[:, lat_col]

d = pd.concat([pd.read_parquet(f) for f in glob.glob("D:/meandre-data/eccc/daily_*.parquet")],
              ignore_index=True).drop_duplicates(["climate_id", "date"])
d["date"] = pd.to_datetime(d["date"])
d = d[d.date.isin(times)]
lat0 = float(lat_n.mean())
def proj(lon, lat): return np.c_[np.asarray(lon)*111.32*np.cos(np.radians(lat0)), np.asarray(lat)*110.57]
pos = d.groupby("climate_id")[["lon", "lat"]].first()
tree = cKDTree(proj(pos.lon.values, pos.lat.values))
dd, jj = tree.query(proj(lon_n, lat_n), k=K)   # (N, K)
w_st = 1.0 / (1.0 + (dd[:, 0] / D0) ** PEXP)   # poids du produit stations, par nœud
print(f"[{REG}] {len(pos)} stations | d voisine méd {np.median(dd[:,0]):.1f} km | "
      f"poids stations méd {np.median(w_st):.2f} (min {w_st.min():.2f}, max {w_st.max():.2f})", flush=True)

ids = list(pos.index)
canaux = {0: "P", 1: "Tmin", 2: "Tmax"}
for ic, var in canaux.items():
    piv = d.pivot_table(index="date", columns="climate_id", values=var, aggfunc="first")
    piv = piv.reindex(times).reindex(columns=ids)
    V = piv.values                              # (T, S)
    obs = np.isfinite(V)
    Vz = np.where(obs, V, 0.0)
    wk = 1.0 / np.clip(dd, 1.0, None) ** 2      # (N, K)
    # somme pondérée sur les K voisines seulement : une matrice (T, N) par voisine,
    # au lieu d'une (T, N, S) qui ne tiendrait pas en mémoire
    num = np.zeros((len(times), len(lon_n)), dtype=np.float32)
    den = np.zeros_like(num)
    for k in range(K):
        col = jj[:, k]                          # (N,) index de station
        num += Vz[:, col] * (wk[:, k] * obs[:, col].astype(np.float32))
        den += wk[:, k] * obs[:, col].astype(np.float32)
    est = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
    casr = F[:, :, ic]
    mel = np.where(np.isfinite(est), w_st[None, :] * est + (1 - w_st[None, :]) * casr, casr)
    print(f"  {var}: stations dispo {np.isfinite(est).mean()*100:.1f} % des couples | "
          f"moy CaSR {np.nanmean(casr):.3f} -> mixte {np.nanmean(mel):.3f}", flush=True)
    F[:, :, ic] = mel

out = xr.Dataset({"forcing": (("time", "node", "channel"), F.astype(np.float32))},
                 coords={"time": times, "node": np.arange(F.shape[1]), "channel": np.arange(6)})
out.attrs["produit"] = f"mixte stations ECCC (IDW k={K}) + CaSR, w=1/(1+(d/{D0})^{PEXP}), courbe de bascule 2026-08-03"
out.to_netcdf(OUT)
print(f"[ok] {OUT}")
