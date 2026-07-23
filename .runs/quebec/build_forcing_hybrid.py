"""Forçage HYBRIDE pleine fenêtre (décision krigeage 2026-07-23) :
P/Tmin/Tmax = grille krigée MELCCFP (quebec.zarr, 2000-2026, stations sans ébauche modèle),
R_n/u2/e_a = CaSR (énergie physiquement cohérente, forcing-{reg}.nc).
Interpolation grille -> nœuds : cellule valide la plus proche (la grille est déjà lisse).
Sortie : D:/meandre-data/quebec/forcing-{reg}-hyb.nc (2000-2024, 6 canaux).

  python .runs/quebec/build_forcing_hybrid.py gasp
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, xarray as xr
from scipy.spatial import cKDTree
from meandre.data.basin_cache import BasinCache

REG = sys.argv[1].lower()
T0, T1 = "2000-01-01", "2024-12-31"
ZARR = "C:/Users/parse01/documents-locaux/rqh-local/io_2026-04/data/03_imputation/quebec.zarr"
CASR_F = f"D:/meandre-data/quebec/forcing-{REG}.nc"
OUT = f"D:/meandre-data/quebec/forcing-{REG}-hyb.nc"
DB = ".runs/slso/data/slso.duckdb" if REG == "slso" else f"D:/meandre-data/quebec/{REG}.duckdb"

h = BasinCache(DB).load(device="cpu")
nc_ = h["node_coords"].numpy()
lat_col = 0 if 40 < float(np.nanmean(nc_[:, 0])) < 62 else 1
nlat, nlon = nc_[:, lat_col], nc_[:, 1 - lat_col]

d = xr.open_zarr(ZARR)
t = pd.to_datetime(d["time"].values).normalize()   # timestamps zarr à 05:00 UTC -> jour civil
sl = (t >= T0) & (t <= T1)
glat, glon = d["latitude"].values, d["longitude"].values
pr = d["pr"].values[sl]; tmx = d["tasmax"].values[sl]; tmn = d["tasmin"].values[sl]
d.close(); tk = t[sl]
print(f"[{REG}] zarr {pr.shape} | fenêtre {tk[0].date()} -> {tk[-1].date()}", flush=True)

# cellules valides (certaines cellules de la grille sont NaN hors couverture)
valid = ~np.isnan(pr[0]) & ~np.isnan(tmx[0])
gy, gx = np.nonzero(valid)
tree = cKDTree(np.c_[glat[gy], glon[gx]])
dist, k = tree.query(np.c_[nlat, nlon], k=1)
print(f"[{REG}] distance nœud->cellule valide : méd {np.median(dist):.3f}° | max {dist.max():.3f}°", flush=True)
assert dist.max() < 0.75, f"nœuds hors grille krigée (max {dist.max():.2f}°)"
iy, ix = gy[k], gx[k]

P_k = pr[:, iy, ix]; Tmx_k = tmx[:, iy, ix]; Tmn_k = tmn[:, iy, ix]
if np.nanmax(P_k) < 1.0: P_k = P_k * 86400.0
for name, arr in [("P", P_k), ("Tmax", Tmx_k), ("Tmin", Tmn_k)]:
    assert np.isnan(arr).mean() < 0.001, f"{name}: {np.isnan(arr).mean():.1%} NaN"
    np.nan_to_num(arr, copy=False, nan=0.0 if name == "P" else np.nanmean(arr))
print(f"[{REG}] krigé : P {np.nanmean(P_k)*365.25:.0f} mm/an | Tmax moy {np.nanmean(Tmx_k):.1f} °C", flush=True)

b = xr.open_dataset(CASR_F)
Fc = b["forcing"].values[:, :, :6]
V = [str(v) for v in b["var"].values[:6]]
tc = pd.to_datetime(b["time"].values); b.close()
slc = (tc >= T0) & (tc <= T1)
Fc = Fc[slc]; tcc = tc[slc]
if len(tcc) != len(tk):
    # jours manquants dans le zarr -> réindexation sur le calendrier CaSR (plus proche voisin)
    missing = tcc.difference(tk)
    print(f"[{REG}] {len(missing)} jour(s) manquant(s) dans le krigé ({[str(m)[:10] for m in missing[:5]]}) — réindexé", flush=True)
    pos = np.searchsorted(tk.values, tcc.values)
    pos = np.clip(pos, 0, len(tk) - 1)
    P_k, Tmx_k, Tmn_k = P_k[pos], Tmx_k[pos], Tmn_k[pos]
    tk = tcc
F = Fc.copy()
F[:, :, 0] = P_k.astype(np.float32)
F[:, :, 1] = Tmn_k.astype(np.float32)
F[:, :, 2] = Tmx_k.astype(np.float32)
assert not np.isnan(F).any(), "NaN dans le forçage final"
if os.path.exists(OUT): os.remove(OUT)
xr.Dataset({"forcing": (("time", "node", "var"), F.astype(np.float32))},
           coords={"time": tcc, "node": np.arange(F.shape[1]), "var": V}).to_netcdf(OUT)
print(f"[ok] {OUT} ({F.shape[0]} jours, P/T krigés MELCCFP + R_n/u2/e_a CaSR)", flush=True)
