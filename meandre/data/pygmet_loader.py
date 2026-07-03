"""Loader d'ensemble PyGMET -> forçage méandre aux nœuds PHYSITEL.

PyGMET (NCAR, Guo et al.) génère un ENSEMBLE de champs météo grillés par
interpolation probabiliste des stations : chaque membre est une réalisation
plausible de la vraie météo. Sortie typique : NetCDF sur grille lat/lon régulière,
journalier, variables de précip (pcp/prcp) et température (tmin/tmax). Un ensemble
= plusieurs fichiers (ens_001.nc, ...) ou une dimension `ens`.

Ce module extrait chaque membre aux coordonnées des nœuds (interp bilinéaire sur
grille régulière, bien plus simple que la grille tournée de CaSR) et produit un
cache de forçage par membre, compatible avec le modèle (canaux P, Tmin, Tmax ;
McGuinness n'a besoin que de T). L'inférence d'ensemble (run_ensemble.py) propage
ensuite ces membres pour obtenir l'incertitude de FORÇAGE sur le débit.

Le nom des variables est CONFIGURABLE (var_map) pour s'adapter à la sortie exacte
des collègues sans toucher au code.
"""
from __future__ import annotations
import glob
import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

# Noms de variables par défaut (convention GMET/PyGMET) ; surchargeables.
DEFAULT_VAR_MAP = {"pcp": ["pcp", "prcp", "precip", "PRCP"],
                   "tmin": ["tmin", "t_min", "TMIN"],
                   "tmax": ["tmax", "t_max", "TMAX"]}


def _find_var(ds: xr.Dataset, candidates: list[str]) -> str:
    for c in candidates:
        if c in ds.variables:
            return c
    raise KeyError(f"aucune variable parmi {candidates} dans {list(ds.data_vars)}")


def _coord_names(ds: xr.Dataset) -> tuple[str, str, str]:
    lat = next((c for c in ("lat", "latitude", "y") if c in ds.coords or c in ds.dims), None)
    lon = next((c for c in ("lon", "longitude", "x") if c in ds.coords or c in ds.dims), None)
    tim = next((c for c in ("time", "date", "t") if c in ds.coords or c in ds.dims), None)
    if not (lat and lon and tim):
        raise KeyError(f"coords lat/lon/time introuvables dans {list(ds.coords)}")
    return lat, lon, tim


def load_pygmet_member(nc_path: str, node_lon: np.ndarray, node_lat: np.ndarray,
                       var_map: dict | None = None) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Extrait un membre PyGMET aux nœuds. Retourne (T, N, 3) [P, Tmin, Tmax] + dates.

    Interp bilinéaire de la grille lat/lon régulière vers les coordonnées des nœuds.
    Points hors grille -> plus proche bord (bounds_error=False, fill par nearest).
    """
    vm = var_map or DEFAULT_VAR_MAP
    ds = xr.open_dataset(nc_path)
    latn, lonn, timn = _coord_names(ds)
    lat = ds[latn].values; lon = ds[lonn].values
    times = pd.to_datetime(ds[timn].values).normalize()
    # grille croissante requise par RegularGridInterpolator
    lat_inc = lat[0] < lat[-1]
    lat_ax = lat if lat_inc else lat[::-1]
    pts = np.column_stack([np.clip(node_lat, lat_ax.min(), lat_ax.max()),
                           np.clip(node_lon, lon.min(), lon.max())])
    out = np.empty((len(times), len(node_lon), 3), dtype=np.float32)
    for c, key in enumerate(("pcp", "tmin", "tmax")):
        v = ds[_find_var(ds, vm[key])].values  # (T, lat, lon)
        if not lat_inc:
            v = v[:, ::-1, :]
        # interp par pas de temps (vectorisé sur les nœuds)
        for t in range(len(times)):
            f = RegularGridInterpolator((lat_ax, lon), v[t], bounds_error=False,
                                        fill_value=None)  # None = extrapole au bord
            out[t, :, c] = f(pts)
    ds.close()
    return out, times


def build_pygmet_ensemble(pygmet_glob: str, node_coords: np.ndarray, out_dir: str,
                          date0: str, date1: str, var_map: dict | None = None,
                          prefix: str = "forcing-pygmet-ens") -> list[str]:
    """Construit un cache de forçage par membre PyGMET.

    pygmet_glob : motif des fichiers membres (ex '.../pygmet/ens_*.nc').
    node_coords : (N, 2) [lon, lat] des nœuds (depuis BasinCache).
    Écrit {out_dir}/{prefix}{i:03d}.nc (canaux P, Tmin, Tmax). Retourne les chemins.
    Le modèle attend n_forcing canaux ; McGuinness (T-only) tourne avec ces 3.
    """
    files = sorted(glob.glob(pygmet_glob))
    if not files:
        raise FileNotFoundError(f"aucun membre PyGMET pour {pygmet_glob}")
    os.makedirs(out_dir, exist_ok=True)
    nlon, nlat = node_coords[:, 0], node_coords[:, 1]
    written = []
    for i, f in enumerate(files):
        arr, times = load_pygmet_member(f, nlon, nlat, var_map)
        sl = (times >= pd.Timestamp(date0)) & (times <= pd.Timestamp(date1))
        arr = arr[sl]; t = times[sl]
        out = xr.Dataset(
            {"forcing": (("time", "node", "var"), arr)},
            coords={"time": t.values, "node": np.arange(len(nlon)),
                    "var": ["P", "Tmin", "Tmax"]})
        path = os.path.join(out_dir, f"{prefix}{i:03d}.nc")
        out.to_netcdf(path, engine="h5netcdf"); out.close()
        written.append(path)
    return written
