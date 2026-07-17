"""Forçage MONT hybride pour le test d'attribution forçage (mont-v5) :
P/Tmin/Tmax = météo krigée MELCCFP de la plateforme (MONT.nc, 727 stations, 2020-2026),
R_n/u2/e_a = repris du forçage CaSR (fenêtre 2020-01-01..2024-12-31).
Interpolation stations -> nœuds : IDW 4 plus proches voisins (lat/lon).
Sortie : D:/meandre-data/quebec/forcing-mont-krig.nc (2020-2024, 6 canaux).
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, xarray as xr
from scipy.spatial import cKDTree
from meandre.data.basin_cache import BasinCache

T0, T1 = "2020-01-01", "2024-12-31"
MET = "C:/Users/parse01/documents-locaux/GitHub/plateformes-hydrotel/LN24HA/MONT_LN24HA_2020/meteo/MONT.nc"
CASR_F = "D:/meandre-data/quebec/forcing-mont.nc"
OUT = "D:/meandre-data/quebec/forcing-mont-krig.nc"

h = BasinCache("D:/meandre-data/quebec/mont.duckdb").load(device="cpu")
nc_ = h["node_coords"].numpy()  # (n, 2) lon/lat
d = xr.open_dataset(MET)
t = pd.to_datetime(d["time"].values)
sl = (t >= T0) & (t <= T1)
slat = d["lat"].values; slon = d["lon"].values
pr = d["pr"].values[sl]; tmx = d["tasmax"].values[sl]; tmn = d["tasmin"].values[sl]
d.close(); tk = t[sl]
# IDW 4-NN en degrés (domaine petit, anisotropie négligeable pour du krigé déjà lisse)
tree = cKDTree(np.c_[slon, slat])
dist, idx = tree.query(nc_, k=4)
w = 1.0 / np.maximum(dist, 1e-6); w /= w.sum(axis=1, keepdims=True)  # (n, 4)
def interp(F):  # (T, S) -> (T, N)
    return np.einsum("tsk,nk->tn", F[:, idx].reshape(F.shape[0], len(nc_), 4), w) if False else \
           (F[:, idx] * w[None, :, :]).sum(axis=2)
P_k = interp(np.nan_to_num(pr, nan=0.0))
Tmx_k = interp(tmx); Tmn_k = interp(tmn)
if np.nanmax(P_k) < 1.0: P_k = P_k * 86400.0  # kg/m2/s -> mm/j si nécessaire
print(f"krigé : P {np.nanmean(P_k)*365.25:.0f} mm/an | Tmax moy {np.nanmean(Tmx_k):.1f} | {P_k.shape}")

b = xr.open_dataset(CASR_F); Fc = b["forcing"].values; V = list(b["var"].values.astype(str))
tc = pd.to_datetime(b["time"].values); b.close()
slc = (tc >= T0) & (tc <= T1)
Fc = Fc[slc]; tcc = tc[slc]
assert len(tcc) == len(tk), f"{len(tcc)} vs {len(tk)}"
F = Fc.copy()
F[:, :, 0] = P_k.astype(np.float32)
F[:, :, 1] = Tmn_k.astype(np.float32)
F[:, :, 2] = Tmx_k.astype(np.float32)
assert not np.isnan(F).any(), "NaN dans le forçage"
if os.path.exists(OUT): os.remove(OUT)
xr.Dataset({"forcing": (("time", "node", "var"), F.astype(np.float32))},
           coords={"time": tcc, "node": np.arange(F.shape[1]), "var": V}).to_netcdf(OUT)
print(f"[ok] {OUT} ({F.shape[0]} jours, P krigé + T krigé + R_n/u2/e_a CaSR)")
