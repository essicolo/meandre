"""Forçage de meandre tiré d'une simulation climatique régionale (MRCC5 piloté par CMIP6, téléchargé par
download_crcm5_cmip6.py), pour un territoire, au format des forçages CaSR : (time, node, var) avec
var = P, Tmin, Tmax, R_n, u2, e_a.

Les six variables se calculent comme celles de CaSR (build_forcing_region.py) :
- P : précipitation journalière, en mm/j, sans calage de volume ni dé-crachinage (comme -casr-brut) ;
- Tmin, Tmax : températures minimale et maximale journalières à 2 m, en °C ;
- R_n : rayonnement net FAO-56 à partir des rayonnements DESCENDANTS (albédo 0,23, émissivité 0,95,
  émission à la température moyenne de l'air), en MJ/m²/j, et non le bilan radiatif du modèle
  climatique, pour garder la définition du forçage d'entraînement ;
- u2 : vent à 2 m, 0,748 fois le vent à 10 m ;
- e_a : pression de vapeur, en kPa, de l'humidité spécifique et de la pression de surface.

Interpolation linéaire aux tronçons sur la triangulation des centres de maille (coordonnées
géographiques), poids calculés une fois. Le jour est le jour UTC du modèle climatique (CaSR est
décalé au jour local, cinq heures plus tôt).

Usage :
    python build_forcing_rcm.py <dossier pilote/expérience> <territoire> <suffixe> <première année> <dernière année>
    python build_forcing_rcm.py ~/meandre-data/climat/crcm5-cmip6/ERA5/evaluation mont -mrcc5-era5 2000 2020
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import Delaunay

from meandre.data.basin_cache import BasinCache

SIGMA = 5.670374e-8
ALBEDO = 0.23
EMIS = 0.95


def weights(grid_lon, grid_lat, lon, lat):
    """Indices et poids barycentriques de chaque point (lon, lat) dans la triangulation des mailles."""
    pts = np.c_[grid_lon.ravel(), grid_lat.ravel()]
    tri = Delaunay(pts)
    simplex = tri.find_simplex(np.c_[lon, lat])
    if (simplex < 0).any():
        raise ValueError(f"{(simplex < 0).sum()} tronçons hors de la grille du modèle climatique")
    T = tri.transform[simplex]
    b = np.einsum("nij,nj->ni", T[:, :2], np.c_[lon, lat] - T[:, 2])
    w = np.c_[b, 1.0 - b.sum(axis=1)]
    return tri.simplices[simplex], w


def series(folder, variable, idx, w, years):
    """Série journalière (temps, tronçon) d'une variable, interpolée aux tronçons."""
    parts = []
    for f in sorted(glob.glob(f"{folder}/{variable}_*.nc")):
        a, b = (int(x[:4]) for x in os.path.basename(f)[:-3].split("_")[-1].split("-"))
        if b < years[0] or a > years[1]:
            continue
        ds = xr.open_dataset(f)
        v = ds[variable].values.reshape(ds.sizes["time"], -1)
        days = pd.DatetimeIndex(ds.indexes["time"].to_datetimeindex() if hasattr(ds.indexes["time"], "to_datetimeindex") else ds.indexes["time"]).normalize()
        parts.append(pd.DataFrame((v[:, idx] * w[None]).sum(axis=-1), index=days))
        ds.close()
    out = pd.concat(parts)
    return out[~out.index.duplicated()].sort_index()


def main():
    folder, territory, suffix, y0, y1 = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
    data = os.environ.get("MEANDRE_DATA", "D:/meandre-data")
    nodes = BasinCache(f"{data}/quebec/{territory}.duckdb").load(device="cpu")["node_coords"].numpy()
    ref = xr.open_dataset(sorted(glob.glob(f"{folder}/pr_*.nc"))[0])
    lon = np.where(ref.lon.values > 180, ref.lon.values - 360, ref.lon.values)
    idx, w = weights(lon, ref.lat.values, nodes[:, 0], nodes[:, 1])
    ref.close()
    get = lambda v: series(folder, v, idx, w, (y0, y1))
    P = get("pr") * 86400.0
    Tmin = get("tasmin") - 273.15
    Tmax = get("tasmax") - 273.15
    Tmean = 0.5 * (Tmin + Tmax)
    q, ps = get("huss"), get("ps")
    e_a = q * ps / (0.622 + 0.378 * q) / 1000.0
    R_n = (((1 - ALBEDO) * get("rsds") - (EMIS * SIGMA * (Tmean + 273.15) ** 4 - get("rlds"))) * 0.0864).clip(lower=0.0)
    u2 = get("sfcWind") * 0.748
    days = pd.date_range(f"{y0}-01-01", f"{y1}-12-31", freq="D")
    F = np.stack([x.reindex(days).to_numpy(dtype=np.float32) for x in (P, Tmin, Tmax, R_n, u2, e_a)], axis=-1)
    missing = np.isnan(F).any(axis=(1, 2))
    if missing.any():
        # Calendrier sans 29 février des simulations pilotées par un modèle global : jour copié de la veille.
        F = pd.DataFrame(F.reshape(len(days), -1), index=days).ffill().to_numpy().reshape(F.shape)
        print(f"{missing.sum()} jours absents du modèle climatique, copiés de la veille")
    out = f"{data}/quebec/forcing-{territory}{suffix}.nc"
    xr.Dataset({"forcing": (("time", "node", "var"), F)}, coords={"time": days, "node": np.arange(len(nodes)), "var": ["P", "Tmin", "Tmax", "R_n", "u2", "e_a"]}).to_netcdf(out)
    print(f"{out} : {F.shape} | P {np.nanmean(F[..., 0]) * 365.25:.0f} mm/an | Tmoy {np.nanmean(F[..., 1:3]):.1f} °C | Rn {np.nanmean(F[..., 3]):.1f} | u2 {np.nanmean(F[..., 4]):.2f} | ea {np.nanmean(F[..., 5]):.3f}")


if __name__ == "__main__":
    main()
