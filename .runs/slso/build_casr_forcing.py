"""Construit le cache de forçage du modèle (time, node, 6) = [P, Tmin, Tmax, R_n,
u2, e_a] à partir des tuiles CaSR v3.2 téléchargées. Fusionne les 2 tuiles SLSO
(concat rlat) + les 7 tranches (concat temps), regrid rotated->nœuds (KDTree plus
proche voisin), agrège horaire->journalier, dérive R_n/e_a/u2. Sortie compatible
gridded_forcing (var "forcing").
  python .runs/slso/build_casr_forcing.py
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, xarray as xr
from scipy.spatial import cKDTree
from meandre.data.basin_cache import BasinCache

CASR = ".runs/slso/data/casr"
OUT = ".runs/slso/data/forcing-casr.nc"
TILES = ["rlon526-560_rlat351-385", "rlon526-560_rlat386-420"]
CHUNKS = ["2000-2003", "2004-2007", "2008-2011", "2012-2015", "2016-2019", "2020-2023", "2024-2024"]
SIGMA = 5.670374e-8   # Stefan-Boltzmann
ALBEDO = 0.23; EMIS = 0.95

h = BasinCache(".runs/slso/data/slso.duckdb").load(device="cpu")
nc = h["node_coords"].numpy(); nlon, nlat = nc[:, 0], nc[:, 1]
n_nodes = len(nc)
print(f"nœuds SLSO : {n_nodes}")

# KDTree construit une fois sur la grille fusionnée des 2 tuiles (concat rlat)
def _merged_grid_coords():
    parts = []
    f0 = f"{CASR}/CaSR_v3.2_A_TT_1.5m_{{t}}_2000-2003.nc"
    for t in TILES:
        ds = xr.open_dataset(f0.format(t=t))
        parts.append((ds.lon.values - 360.0, ds.lat.values)); ds.close()
    lon = np.concatenate([p[0] for p in parts], axis=0)   # (70,35)
    lat = np.concatenate([p[1] for p in parts], axis=0)
    return lon, lat
glon, glat = _merged_grid_coords()
tree = cKDTree(np.c_[glon.ravel(), glat.ravel()])
gdist, gidx = tree.query(np.c_[nlon, nlat])
print(f"regrid : dist médiane {np.median(gdist):.3f}° max {gdist.max():.3f}° (grille {glon.shape})")

def load_var(var, agg):
    """(n_days_total, n_nodes) agrégé journalier sur 2000-2024."""
    daily_all = []
    for ch in CHUNKS:
        arrs, times = [], None
        for t in TILES:
            f = f"{CASR}/CaSR_v3.2_{var}_{t}_{ch}.nc"
            ds = xr.open_dataset(f)
            vname = [x for x in ds.data_vars if "CaSR" in x][0]
            arrs.append(ds[vname].values)   # (time, rlat, rlon)
            times = pd.to_datetime(ds.time.values); ds.close()
        merged = np.concatenate(arrs, axis=1)   # (time, 70, 35) concat rlat
        T = merged.shape[0]
        node_series = merged.reshape(T, -1)[:, gidx]   # (time, n_nodes)
        df = pd.DataFrame(node_series, index=times)
        d = getattr(df.resample("1D"), agg)()
        daily_all.append(d)
    out = pd.concat(daily_all)
    # Dédup des jours-frontière entre tranches (un jour peut être partiel dans 2
    # tranches de 4 ans) : ré-agrège par date avec la MÊME opération (exact pour
    # sum/min/max ; ~ pour mean, négligeable sur 1 jour/4ans).
    return getattr(out.groupby(out.index), agg)()

print("P (PR0 somme, m->mm)..."); P = load_var("A_PR0_SFC", "sum") * 1000.0
print("Tmin/Tmax (TT)..."); TT_min = load_var("A_TT_1.5m", "min"); TT_max = load_var("A_TT_1.5m", "max")
TT_mean = (TT_min + TT_max) / 2.0
print("TD (point rosée -> e_a)..."); TD = load_var("A_TD_1.5m", "mean")
print("FB/FI (W/m2 -> R_n)..."); FB = load_var("P_FB_SFC", "mean"); FI = load_var("P_FI_SFC", "mean")
print("UVC (vent -> u2)..."); UVC = load_var("P_UVC_10m", "mean")

# Aligner sur l'index commun (toutes mêmes dates normalement)
idx = P.index
def al(x): return x.reindex(idx)
P, TT_min, TT_max, TT_mean, TD, FB, FI, UVC = [al(x) for x in (P, TT_min, TT_max, TT_mean, TD, FB, FI, UVC)]

# Dérivations FAO-56
e_a = 0.6108 * np.exp(17.27 * TD / (TD + 237.3))                       # kPa (es au point de rosée)
T_K = TT_mean + 273.15
R_nl = EMIS * SIGMA * T_K**4 - FI                                       # net longwave sortant (W/m2)
R_n_W = (1 - ALBEDO) * FB - R_nl                                        # net (W/m2)
R_n = (R_n_W * 0.0864).clip(lower=0.0)                                  # MJ/m2/jour
u2 = UVC * 0.748                                                        # 10m -> 2m

# Empile (time, node, 6) = [P, Tmin, Tmax, R_n, u2, e_a]
chans = [P, TT_min, TT_max, R_n, u2, e_a]
idx_slice = (idx >= pd.Timestamp("2000-01-01")) & (idx <= pd.Timestamp("2024-12-31"))
idx = idx[idx_slice]
P, TT_min, TT_max, R_n, u2, e_a = [x.reindex(idx) for x in (P, TT_min, TT_max, R_n, u2, e_a)]
chans = [P, TT_min, TT_max, R_n, u2, e_a]
forcing = np.stack([c.values.astype(np.float32) for c in chans], axis=-1)  # (T, n, 6)
print(f"forcing {forcing.shape}  NaN={np.isnan(forcing).any()}  "
      f"P_moy_an={float(np.nanmean(P.values))*365:.0f}mm  Rn_moy={float(np.nanmean(R_n)):.1f}MJ/j  u2_moy={float(np.nanmean(u2)):.1f}m/s")

ds = xr.Dataset({"forcing": (("time", "node", "var"), forcing)},
                coords={"time": idx.values, "node": np.arange(n_nodes), "var": ["P", "Tmin", "Tmax", "R_n", "u2", "e_a"]})
ds.to_netcdf(OUT)
print(f"[ok] cache forçage CaSR écrit : {OUT}")
