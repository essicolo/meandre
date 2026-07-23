"""Forçage PyGMET+CaSR pour UNE région (VRAI krigeage de stations, remplace SIMAT).
Extrait la grille déterministe PyGMET (Grid_Regression : prcp_boxcox, tmean, trange)
aux nœuds, reconstruit P/Tmin/Tmax, et blende l'énergie (R_n/u2/e_a) depuis CaSR.
Sortie : D:/meandre-data/quebec/forcing-{reg}-pgm.nc (6 canaux, pleine fenêtre).

  python .runs/quebec/build_forcing_pygmet.py gasp
"""
import os, sys, glob
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, xarray as xr
from scipy.interpolate import RegularGridInterpolator
from meandre.data.basin_cache import BasinCache

REG = sys.argv[1].lower()
T0, T1 = "2000-01-01", "2024-12-31"
PGM_DIR = f"D:/meandre-data/pygmet/{REG}/PyGMET_output/regression_outputs"
CASR_F = f"D:/meandre-data/quebec/forcing-{REG}.nc"
OUT = f"D:/meandre-data/quebec/forcing-{REG}-pgm.nc"
DB = ".runs/slso/data/slso.duckdb" if REG == "slso" else f"D:/meandre-data/quebec/{REG}.duckdb"


def boxcox_retransform(d, texp=4):
    d = np.where(d < -texp, -texp, d)
    return (d / texp + 1) ** texp


nc = BasinCache(DB).load(device="cpu")["node_coords"].numpy()
lat_col = 0 if 40 < float(np.nanmean(nc[:, 0])) < 62 else 1
nlat, nlon = nc[:, lat_col], nc[:, 1 - lat_col]

# concatène tous les fichiers Grid_Regression (PyGMET peut sortir par batch)
files = sorted(glob.glob(f"{PGM_DIR}/{REG}_full_Grid_Regression_*.nc"))
assert files, f"aucune sortie grille PyGMET dans {PGM_DIR}"
g = xr.open_mfdataset(files, combine="by_coords") if len(files) > 1 else xr.open_dataset(files[0])
xg, yg = g["x"].values, g["y"].values
times = pd.to_datetime(g["time"].values).normalize()
P = boxcox_retransform(g["prcp_boxcox"].values)          # (y,x,t) mm
Tmin = (g["tmean"].values - g["trange"].values / 2.0)
Tmax = (g["tmean"].values + g["trange"].values / 2.0)
g.close()
print(f"[{REG}] grille PyGMET {P.shape} | {times[0].date()}..{times[-1].date()}", flush=True)


def to_nodes(F):  # (y,x,t) -> (t,N) bilinéaire, ordre y croissant
    oy = np.argsort(yg)
    itp = RegularGridInterpolator((yg[oy], xg), np.moveaxis(F, 2, 0)[:, oy, :],
                                  bounds_error=False, fill_value=None)
    return itp(np.c_[nlat, nlon]) if False else \
        np.stack([RegularGridInterpolator((yg[oy], xg), F[oy, :, t], bounds_error=False, fill_value=None)(np.c_[nlat, nlon])
                  for t in range(F.shape[2])])


Pn = np.clip(to_nodes(P), 0, None).astype(np.float32)
Tmn = to_nodes(Tmin).astype(np.float32)
Tmx = to_nodes(Tmax).astype(np.float32)
# combler d'éventuels NaN (jours sans station) par la clim mensuelle du nœud
for arr, fill in [(Pn, 0.0), (Tmn, None), (Tmx, None)]:
    if np.isnan(arr).any():
        mo = times.month.values
        for m in range(1, 13):
            sel = mo == m
            col = np.nanmean(arr[sel], axis=0)
            idx = np.isnan(arr[sel])
            a = arr[sel]; a[idx] = np.take(col, np.where(idx)[1]); arr[sel] = a
print(f"[{REG}] aux nœuds : P {np.nanmean(Pn)*365.25:.0f} mm/an | Tmax moy {np.nanmean(Tmx):.1f} °C", flush=True)

# ── blend énergie CaSR ───────────────────────────────────────────────────────
b = xr.open_dataset(CASR_F)
Fc = b["forcing"].values[:, :, :6]; V = [str(v) for v in b["var"].values[:6]]
tc = pd.to_datetime(b["time"].values).normalize(); b.close()
# aligne PyGMET sur le calendrier CaSR
pos = pd.Series(np.arange(len(times)), index=times)
common = tc.intersection(times)
assert len(common) > 9000, f"recouvrement calendrier faible ({len(common)})"
ci = pd.Series(np.arange(len(tc)), index=tc).reindex(times).values
F = Fc.copy()
map_pgm = pd.Series(np.arange(len(times)), index=times).reindex(tc)
valid = map_pgm.notna().values
pj = map_pgm.values
for t_casr in np.where(valid)[0]:
    j = int(pj[t_casr])
    F[t_casr, :, 0] = Pn[j]; F[t_casr, :, 1] = Tmn[j]; F[t_casr, :, 2] = Tmx[j]
assert not np.isnan(F).any(), "NaN dans le forçage final"
if os.path.exists(OUT): os.remove(OUT)
xr.Dataset({"forcing": (("time", "node", "var"), F.astype(np.float32))},
           coords={"time": tc, "node": np.arange(F.shape[1]), "var": V}).to_netcdf(OUT)
print(f"[ok] {OUT} ({F.shape[0]} jours, P/T PyGMET krigé stations + énergie CaSR)", flush=True)
