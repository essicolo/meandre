"""Forçage de meandre tiré d'un membre d'ESPO-G6-R2 (P, Tmin, Tmax seulement, téléchargés par
download_espo_g6.py), pour un territoire, au format des forçages CaSR : (time, node, var) avec
var = P, Tmin, Tmax, R_n, u2, e_a.

Les trois variables absentes d'ESPO-G6 sont prédites par le modèle additif de gam_missing_vars.py
(R546), ajusté à chaque appel sur le forçage CaSR du territoire (vent en m/s) :
- vent à 2 m : écart à la normale CaSR du tronçon pour le jour de l'année (normale supposée inchangée) ;
- rayonnement net : écart à la reconstruction FAO-56 (Hargreaves, point de rosée = Tmin) ;
- pression de vapeur : écart à la pression de saturation de Tmin.
L'anomalie de température est prise par rapport à la normale de la série elle-même (CaSR à
l'ajustement, membre ESPO sur sa période de trente ans) : un climat plus chaud ne sort donc pas
de la plage d'entraînement, seul l'écart au temps typique de la saison est lu.

Calendrier sans 29 février : le 29 février est copié du 28. Interpolation linéaire aux tronçons
sur la triangulation des centres des mailles de terre RDRS.

Usage :
    python build_forcing_espo.py <dossier ESPO> <modèle global> <étiquette> <territoire> <suffixe> [dossier de sortie]
    python build_forcing_espo.py ~/meandre-data/climat/espo-g6-r2 BCC_BCC-CSM2-MR reference_1991-2020 mont -espo-bcc-ref
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

from build_forcing_rcm import weights
from reconstruct_forcing_fao import SIGMA_MJ, es
from meandre.data.basin_cache import BasinCache

COLS = ["doy", "dtr", "t_anom", "p", "p_prev", "lat"]


def smooth_doy_normal(x, doy):
    """Normale par jour de l'année et par tronçon, lissée sur 31 jours (fenêtre circulaire)."""
    n = pd.DataFrame(x).groupby(doy).mean().reindex(range(1, 367)).interpolate(limit_direction="both")
    wrapped = pd.concat([n.iloc[-15:], n, n.iloc[:15]])
    return wrapped.rolling(31, center=True).mean().iloc[15:-15].to_numpy()[doy - 1]


def fao_base(tmin, tmax, doy, lat):
    """Rayonnement net FAO-56 de Hargreaves et pression de vapeur à Tmin (R545)."""
    J = doy[:, None]
    phi = np.deg2rad(lat)[None, :]
    dr = 1 + 0.033 * np.cos(2 * np.pi * J / 365)
    delta = 0.409 * np.sin(2 * np.pi * J / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1, 1))
    Ra = 24 * 60 / np.pi * 0.0820 * dr * (ws * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(ws))
    Rs = 0.16 * np.sqrt(np.clip(tmax - tmin, 0, None)) * Ra
    ea = es(tmin)
    Rnl = SIGMA_MJ * ((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2 * (0.34 - 0.14 * np.sqrt(ea)) * (1.35 * np.clip(Rs / np.maximum(0.75 * Ra, 1e-6), 0, 1) - 0.35)
    return np.clip(0.77 * Rs - Rnl, 0, None), ea


def features(P, tmin, tmax, doy, lat):
    T, N = P.shape
    tmean = 0.5 * (tmin + tmax)
    t_anom = tmean - smooth_doy_normal(tmean, doy)
    P_prev = np.vstack([P[:1], P[:-1]])
    J = np.broadcast_to(doy[:, None], (T, N))
    return np.stack([J, tmax - tmin, t_anom, np.sqrt(np.clip(P, 0, None)), np.sqrt(np.clip(P_prev, 0, None)), np.broadcast_to(lat[None, :], (T, N))], axis=-1)


def fit_gam(casr_path, lat):
    """Ajuste les trois modèles additifs sur CaSR 2001-2020 (un tronçon sur dix, un jour sur sept)."""
    ds = xr.open_dataset(casr_path)
    names = list(ds["var"].values)
    days = pd.DatetimeIndex(ds.time.values)
    keep = (days.year >= 2001) & (days.year <= 2020)
    f = ds.forcing.values[keep][:, ::10, :].astype(np.float64)
    ds.close()
    doy = np.minimum(days[keep].dayofyear.to_numpy(), 365)
    sub = lat[::10]
    P, tmin, tmax, Rn, u2, ea = (f[..., names.index(k)] for k in ("P", "Tmin", "Tmax", "R_n", "u2", "e_a"))
    Rn_fao, ea_fao = fao_base(tmin, tmax, doy, sub)
    X = features(P, tmin, tmax, doy, sub).reshape(-1, len(COLS))[::7]
    models = {}
    for name, obs, base in (("u2", u2, smooth_doy_normal(u2, doy)), ("R_n", Rn, Rn_fao), ("e_a", ea, ea_fao)):
        splines = ColumnTransformer([("doy", SplineTransformer(n_knots=12, degree=3, extrapolation="periodic"), [0])] + [(c, SplineTransformer(n_knots=8, degree=3, extrapolation="linear"), [i]) for i, c in enumerate(COLS) if i > 0])
        models[name] = make_pipeline(splines, RidgeCV(alphas=np.logspace(-3, 3, 7))).fit(X, (obs - base).ravel()[::7])
    # Normale de vent par tronçon (tous les tronçons), base du modèle de vent en scénario.
    ds = xr.open_dataset(casr_path)
    u2_all = ds.forcing.values[keep][:, :, names.index("u2")].astype(np.float64)
    ds.close()
    u2_normal = smooth_doy_normal(u2_all, doy)
    return models, u2_normal, doy


def espo_series(folder, model, label, var, idx, w):
    """Série (jour sans 29 février, tronçon) interpolée aux tronçons ; découpe d'abord la boîte utile."""
    path = glob.glob(f"{folder}/espo-g6-r2_{model}_{label}_{var}.nc")[0]
    ds = xr.open_dataset(path)
    rr, cc = np.unravel_index(np.unique(idx), ds.lat.shape)
    box = ds[var].isel(rlat=slice(rr.min(), rr.max() + 1), rlon=slice(cc.min(), cc.max() + 1)).values
    ny, nx = rr.max() - rr.min() + 1, cc.max() - cc.min() + 1
    r, c = np.unravel_index(idx, ds.lat.shape)
    local = (r - rr.min()) * nx + (c - cc.min())
    out = (box.reshape(box.shape[0], ny * nx)[:, local] * w[None]).sum(axis=-1)
    first = ds.attrs["first_day"]
    ds.close()
    return out, int(first[:4])


def main():
    folder, model, label, territory, suffix = sys.argv[1:6]
    data = os.environ.get("MEANDRE_DATA", "D:/meandre-data")
    out_dir = sys.argv[6] if len(sys.argv) > 6 else f"{data}/quebec"
    coords = BasinCache(f"{data}/quebec/{territory}.duckdb").load(device="cpu")["node_coords"].numpy()
    lat = coords[:, 1]
    models, u2_normal, casr_doy = fit_gam(f"{data}/quebec/forcing-{territory}-casr-brut-vent.nc", lat)
    ref = xr.open_dataset(glob.glob(f"{folder}/espo-g6-r2_{model}_{label}_pr.nc")[0])
    glon = np.where(ref.lon.values > 180, ref.lon.values - 360, ref.lon.values).ravel()
    # Triangulation sur les seules mailles de terre : une maille d'eau (Saint-Laurent, grands lacs)
    # n'a pas de valeur et rendrait le forçage manquant aux tronçons riverains.
    land = np.flatnonzero(np.isfinite(ref.pr.isel(time=0).values).ravel())
    idx, w = weights(glon[land], ref.lat.values.ravel()[land], coords[:, 0], coords[:, 1])
    idx = land[idx]
    ref.close()
    P, y0 = espo_series(folder, model, label, "pr", idx, w)
    tmin, _ = espo_series(folder, model, label, "tasmin", idx, w)
    tmax, _ = espo_series(folder, model, label, "tasmax", idx, w)
    n_years = P.shape[0] // 365
    noleap = pd.DatetimeIndex([d for d in pd.date_range(f"{y0}-01-01", f"{y0 + n_years - 1}-12-31", freq="D") if not (d.month == 2 and d.day == 29)])
    doy = np.tile(np.arange(1, 366), n_years)
    swap = tmin > tmax
    tmin, tmax = np.where(swap, tmax, tmin), np.where(swap, tmin, tmax)
    Rn_fao, ea_fao = fao_base(tmin, tmax, doy, lat)
    X = features(P, tmin, tmax, doy, lat)
    # Prédiction par année : les bases de splines de toute la période d'un coup dépassent 10 Go.
    pred = {k: np.concatenate([m.predict(X[a:a + 365].reshape(-1, len(COLS))).reshape(-1, P.shape[1]) for a in range(0, P.shape[0], 365)]) for k, m in models.items()}
    # Normale de vent CaSR par jour de l'année, indexée sur les jours du membre.
    u2_base = pd.DataFrame(u2_normal).groupby(casr_doy).mean().reindex(range(1, 366)).to_numpy()[doy - 1]
    u2 = np.clip(u2_base + pred["u2"], 0.1, None)
    Rn = np.clip(Rn_fao + pred["R_n"], 0, None)
    ea = np.clip(ea_fao + pred["e_a"], 0.01, es(tmax))
    F = np.stack([np.clip(P, 0, None), tmin, tmax, Rn, u2, ea], axis=-1).astype(np.float32)
    days = pd.date_range(noleap[0], noleap[-1], freq="D")
    F = pd.DataFrame(F.reshape(len(noleap), -1), index=noleap).reindex(days).ffill().to_numpy().reshape(len(days), *F.shape[1:])
    out = f"{out_dir}/forcing-{territory}{suffix}.nc"
    xr.Dataset({"forcing": (("time", "node", "var"), F)}, coords={"time": days, "node": np.arange(len(coords)), "var": ["P", "Tmin", "Tmax", "R_n", "u2", "e_a"]}, attrs={"source": f"ESPO-G6-R2 {model} {label}, R_n u2 e_a par GAM ajusté sur CaSR"}).to_netcdf(out)
    print(f"{out} : {F.shape} | swaps Tmin>Tmax {swap.mean() * 100:.2f} % | P {F[..., 0].mean() * 365.25:.0f} mm/an | Tmoy {F[..., 1:3].mean():.1f} °C | Rn {F[..., 3].mean():.2f} | u2 {F[..., 4].mean():.2f} | ea {F[..., 5].mean():.3f}")


if __name__ == "__main__":
    main()
