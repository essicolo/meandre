"""Modèle additif (GAM) des variables que les scénarios ESPO-G6 ne fournissent pas, à partir de P, Tmin,
Tmax et du calendrier : vent, rayonnement net, pression de vapeur (2026-11-11).

Chaque variable est prédite comme un écart à une base physique ou climatologique :
- vent à 2 m : écart à la normale du tronçon pour le jour de l'année ;
- rayonnement net : écart à la reconstruction FAO-56 (Hargreaves, point de rosée = Tmin) ;
- pression de vapeur : écart à la pression de saturation de Tmin.
Variables explicatives, toutes dans leur plage même en climat plus chaud : spline cyclique du jour de
l'année, splines de l'amplitude thermique, de l'anomalie de température (écart à la normale du jour),
de la pluie du jour et de la veille (racine carrée), et la latitude ; régression ridge sur les bases de
splines (GAM pénalisé), extrapolation linéaire hors de la plage d'entraînement.

Entraînement 2001-2015, vérification 2016-2020, sur un tronçon sur dix ; on compare à la base seule et à
la base recalée par tronçon et par mois (la reconstruction qui égalait CaSR, R545).

Usage : python gam_missing_vars.py <forçage CaSR (vent en m/s) .nc> <base .duckdb>
"""
import sys

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

from reconstruct_forcing_fao import SIGMA_MJ, es

ds = xr.open_dataset(sys.argv[1])
f = ds.forcing.values[:, ::10, :].astype(np.float64)
names = list(ds["var"].values)
from meandre.data.basin_cache import BasinCache
lat = BasinCache(sys.argv[2]).load(device="cpu")["node_coords"].numpy()[::10, 1]
days = pd.DatetimeIndex(ds.time.values)
T, N = f.shape[:2]
P, tmin, tmax, Rn, u2, ea = (f[..., names.index(k)] for k in ("P", "Tmin", "Tmax", "R_n", "u2", "e_a"))
if "-vent" not in sys.argv[1]:
    u2 = u2 * 0.514444  # forçage d'avant la correction du vent (R544) : noeuds convertis en m/s
J = days.dayofyear.to_numpy()[:, None] * np.ones((1, N))
phi = np.deg2rad(lat)[None, :]
dr = 1 + 0.033 * np.cos(2 * np.pi * J / 365)
delta = 0.409 * np.sin(2 * np.pi * J / 365 - 1.39)
ws = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1, 1))
Ra = 24 * 60 / np.pi * 0.0820 * dr * (ws * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(ws))
Rs = 0.16 * np.sqrt(np.clip(tmax - tmin, 0, None)) * Ra
ea_fao = es(tmin)
Rnl = SIGMA_MJ * ((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2 * (0.34 - 0.14 * np.sqrt(ea_fao)) * (1.35 * np.clip(Rs / np.maximum(0.75 * Ra, 1e-6), 0, 1) - 0.35)
Rn_fao = np.clip(0.77 * Rs - Rnl, 0, None)

train = (days.year >= 2001) & (days.year <= 2015)
test = (days.year >= 2016) & (days.year <= 2020)
doy = days.dayofyear.to_numpy()
tmean = 0.5 * (tmin + tmax)
norm = lambda x: pd.DataFrame(x[train]).groupby(doy[train]).mean().reindex(doy).to_numpy()
t_anom = tmean - norm(tmean)
u2_clim = norm(u2)
P_prev = np.vstack([P[:1], P[:-1]])
X = np.stack([J, tmax - tmin, t_anom, np.sqrt(P), np.sqrt(P_prev), np.broadcast_to(lat[None, :], (T, N))], axis=-1)
cols = ["doy", "dtr", "t_anom", "p", "p_prev", "lat"]
splines = ColumnTransformer([("doy", SplineTransformer(n_knots=12, degree=3, extrapolation="periodic"), [0])] + [(c, SplineTransformer(n_knots=8, degree=3, extrapolation="linear"), [i]) for i, c in enumerate(cols) if i > 0])
months = days.month.to_numpy()


def recalibrated(base, target):
    """Base recalée par tronçon et par mois sur la période d'entraînement (méthode de R545)."""
    out = base.copy()
    for m in range(1, 13):
        s, st = months == m, (months == m) & train
        if np.isfinite(target).all() and base is not u2_clim:
            out[s] = base[s] * target[st].mean(axis=0) / np.maximum(base[st].mean(axis=0), 1e-6)
    return out


def score(pred, obs):
    e = pred[test] - obs[test]
    return np.sqrt((e ** 2).mean()), 1 - (e ** 2).mean() / obs[test].var()


rows = []
for name, obs, base in (("vent à 2 m (m/s)", u2, u2_clim), ("rayonnement net (MJ/m²/j)", Rn, Rn_fao), ("pression de vapeur (kPa)", ea, ea_fao)):
    model = make_pipeline(splines, RidgeCV(alphas=np.logspace(-3, 3, 7)))
    model.fit(X[train].reshape(-1, len(cols))[::7], (obs - base)[train].ravel()[::7])
    gam = base + model.predict(X.reshape(-1, len(cols))).reshape(T, N)
    if name.startswith("rayonnement"):
        gam = np.clip(gam, 0, None)
    for label, pred in (("base", base), ("base recalée", recalibrated(base, obs) if base is not u2_clim else base), ("GAM", gam)):
        rmse, r2 = score(pred, obs)
        rows.append(dict(variable=name, methode=label, rmse=rmse, r2=r2, biais=(pred[test] - obs[test]).mean()))
pd.set_option("display.width", 200)
print(pd.DataFrame(rows).round(3).to_string(index=False))
