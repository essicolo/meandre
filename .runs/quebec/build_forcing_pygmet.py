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
from scipy.spatial import cKDTree
from meandre.data.basin_cache import BasinCache

# Racines portables (portage grappe, 2026-09-01) : les chemins absolus rendaient toute
# execution hors du poste d'origine impossible. Defauts inchanges.
import os as _osp
_DATA_ROOT = _osp.environ.get("MEANDRE_DATA", "D:/meandre-data")

REG = sys.argv[1].lower()
T0, T1 = "2000-01-01", "2024-12-31"
PGM_DIR = f"{_DATA_ROOT}/pygmet/{REG}/PyGMET_output/regression_outputs"
CASR_F = f"{_DATA_ROOT}/quebec/forcing-{REG}.nc"
OUT = f"{_DATA_ROOT}/quebec/forcing-{REG}-pgm.nc"
DB = ".runs/slso/data/slso.duckdb" if REG == "slso" else f"{_DATA_ROOT}/quebec/{REG}.duckdb"


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
# sortie déterministe = prcp PHYSIQUE (mm) ; ensemble/regression brute = prcp_boxcox.
# force l'ordre d'axes (y, x, time) quel que soit le stockage.
pvar = "prcp" if "prcp" in g else "prcp_boxcox"
gp = g[pvar].transpose("y", "x", "time").values
P = gp if pvar == "prcp" else boxcox_retransform(gp)
tmean = g["tmean"].transpose("y", "x", "time").values
trange = g["trange"].transpose("y", "x", "time").values
Tmin = tmean - trange / 2.0
Tmax = tmean + trange / 2.0
g.close()
print(f"[{REG}] grille PyGMET {P.shape} | {times[0].date()}..{times[-1].date()}", flush=True)


# chaque nœud -> cellule VALIDE la plus proche (masque statique = cellules krigées).
# Évite les NaN des cellules masquées ; plus proche voisin (grille 0.05° ~ 5 km, plus
# fin que les reaches). Un seul mapping, appliqué sur toute la série temporelle.
LONg, LATg = np.meshgrid(xg, yg)
valid = np.isfinite(P).all(axis=2)          # cellules définies à tous les pas
vy, vx = np.nonzero(valid)
tree = cKDTree(np.c_[LATg[vy, vx], LONg[vy, vx]])
_, k = tree.query(np.c_[nlat, nlon])
iy, ix = vy[k], vx[k]
Pn = np.clip(P[iy, ix, :].T, 0, None).astype(np.float32)   # (T, N)
Tmn = Tmin[iy, ix, :].T.astype(np.float32)
Tmx = Tmax[iy, ix, :].T.astype(np.float32)
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
