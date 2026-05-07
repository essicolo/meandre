"""Interpolate gridded daily climate to model node centroids and derive forcing.

Supported input formats: Zarr (.zarr), NetCDF (.nc), GRIB (.grib/.grib2/.grb).

The gridded dataset must provide three daily variables on a regular grid:
    pr      Precipitation (mm/day)
    tasmin  Minimum temperature (°C)
    tasmax  Maximum temperature (°C)

Coordinate names ``latitude``/``longitude`` are expected.  Common aliases
(``lat``/``lon``) are normalized automatically.

Three additional variables required by HydroModel (n_forcing=6) are derived:
    R_n   Net radiation      (MJ/m²/day) — Hargreaves–Samani + FAO-56
    u2    Wind speed at 2 m  (m/s)       — constant (no data available)
    e_a   Actual vapour pres (kPa)       — e_s(T_min), FAO-56 eq. 48

Output column order matches meandre.data.forcing.FORCING_VARS:
    [P, T_min, T_max, R_n, u2, e_a]

Spatial method: nearest-grid-cell interpolation.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


def extract_forcing(
    zarr_path: str | Path,
    node_coords: Tensor,
    node_elev: Tensor | None,
    date_start: str,
    date_end: str,
    cache_nc: str | Path | None = None,
    kRS: float = 0.17,
    u2_default: float = 2.0,
    device: torch.device | None = None,
) -> Tensor:
    """Interpolate gridded climate to node centroids and derive full forcing.

    Parameters
    ----------
    zarr_path:
        Path to a gridded climate dataset (Zarr, NetCDF, or GRIB) with
        ``pr``, ``tasmin``, ``tasmax`` variables on a regular
        ``latitude`` × ``longitude`` grid.
    node_coords:
        (N, 2) tensor [lon, lat] — troncon centroid coordinates.
    node_elev:
        (N,) tensor of mean elevation (m).  Normalised values trigger a 300 m
        fallback (used only for the clear-sky radiation term R_so).
    date_start, date_end:
        ISO 8601 date strings selecting the time window (inclusive).
    cache_nc:
        Optional path.  If the file exists, load from cache and skip the Zarr.
        If the file does not exist, write the result after computing.
    kRS:
        Hargreaves–Samani radiation coefficient.
        0.17 = interior/continental (default);  0.20 = coastal.
    u2_default:
        Assumed wind speed (m/s) when no wind data are available.
    device:
        Target PyTorch device.

    Returns
    -------
    Tensor of shape (T, N, 6), float32.
    Columns: [P, T_min, T_max, R_n, u2, e_a]
    """
    import xarray as xr
    import pandas as pd

    cache_nc = Path(cache_nc) if cache_nc else None

    if cache_nc is not None:
        # Support both legacy .nc and preferred .zarr
        cache_path = cache_nc
        _is_zarr = cache_path.suffix == ".zarr" or cache_path.name.endswith(".zarr")
        if cache_path.exists():
            ds_cache = xr.open_zarr(cache_path) if _is_zarr else xr.open_dataset(cache_path)
            cached_times = ds_cache["forcing"].coords["time"].values
            ds_cache.close()
            cache_start = str(cached_times[0])[:10]
            cache_end   = str(cached_times[-1])[:10]
            if cache_start == date_start[:10] and cache_end == date_end[:10]:
                print(f"[gridded_forcing] Loading cached forcing from {cache_path}")
                ds2 = xr.open_zarr(cache_path) if _is_zarr else xr.open_dataset(cache_path)
                arr = ds2["forcing"].values.astype(np.float32)
                ds2.close()
                arr = _fill_nan(arr)
                t = torch.from_numpy(arr)
                return t.to(device) if device else t
            print(
                f"[gridded_forcing] Cache date mismatch "
                f"(cached {cache_start}:{cache_end}, requested {date_start}:{date_end}) "
                "— recomputing."
            )

    print(f"[gridded_forcing] Extracting {date_start} to {date_end}")

    ds = _open_gridded(zarr_path)
    ds = _normalize_coords(ds)
    ds_slice = ds.sel(time=slice(date_start, date_end))
    n_time = len(ds_slice.time)
    dates = ds_slice.time.values

    doy = np.array([pd.Timestamp(t).day_of_year for t in dates], dtype=np.float64)

    coords_np = node_coords.detach().cpu().numpy()
    n_nodes = coords_np.shape[0]

    lats_grid = ds.latitude.values
    lons_grid = ds.longitude.values

    lat_min = coords_np[:, 1].min() - 0.5
    lat_max = coords_np[:, 1].max() + 0.5
    lon_min = coords_np[:, 0].min() - 0.5
    lon_max = coords_np[:, 0].max() + 0.5

    lat_mask = (lats_grid >= lat_min) & (lats_grid <= lat_max)
    lon_mask = (lons_grid >= lon_min) & (lons_grid <= lon_max)
    sub_lats = lats_grid[lat_mask]
    sub_lons = lons_grid[lon_mask]

    print(f"[gridded_forcing] Grid subset: {n_time} x {lat_mask.sum()} x {lon_mask.sum()} cells")
    pr_raw   = ds_slice.pr.sel(latitude=sub_lats, longitude=sub_lons).values.astype(np.float32)
    tmax_raw = ds_slice.tasmax.sel(latitude=sub_lats, longitude=sub_lons).values.astype(np.float32)
    tmin_raw = ds_slice.tasmin.sel(latitude=sub_lats, longitude=sub_lons).values.astype(np.float32)
    # Optional 10 m wind speed — converted to u2 below if present
    has_wind = "sfcWind" in ds_slice.data_vars
    wind_raw = (
        ds_slice.sfcWind.sel(latitude=sub_lats, longitude=sub_lons).values.astype(np.float32)
        if has_wind else None
    )
    ds.close()

    lat_idx = np.argmin(np.abs(sub_lats[:, None] - coords_np[:, 1][None, :]), axis=0)
    lon_idx = np.argmin(np.abs(sub_lons[:, None] - coords_np[:, 0][None, :]), axis=0)

    pr   = pr_raw[:, lat_idx, lon_idx]
    tmax = tmax_raw[:, lat_idx, lon_idx]
    tmin = tmin_raw[:, lat_idx, lon_idx]
    wind10 = wind_raw[:, lat_idx, lon_idx] if has_wind else None

    pr = np.maximum(pr, 0.0)
    swap = tmin > tmax
    if swap.any():
        warnings.warn(f"{swap.sum()} (t,n) cells have Tmin > Tmax — swapping", stacklevel=2)
        tmin[swap], tmax[swap] = tmax[swap], tmin[swap]

    if node_elev is not None:
        elev_np = node_elev.detach().cpu().numpy().astype(np.float64)
        if elev_np.min() < 0:
            elev_np = np.full(n_nodes, 300.0)
    else:
        elev_np = np.zeros(n_nodes, dtype=np.float64)

    lat_rad = np.deg2rad(coords_np[:, 1])
    R_a  = _extraterrestrial_radiation(lat_rad, doy)
    dT   = np.maximum(tmax.astype(np.float64) - tmin.astype(np.float64), 0.0)
    R_s  = kRS * np.sqrt(dT) * R_a
    R_so = (0.75 + 2e-5 * elev_np[None, :]) * R_a
    e_a  = _saturation_vapour_pressure(tmin.astype(np.float64))
    R_ns = (1.0 - 0.23) * R_s
    Tmean_K = (tmax.astype(np.float64) + tmin.astype(np.float64)) / 2.0 + 273.16
    ratio = np.clip(np.where(R_so > 0.01, R_s / R_so, 0.5), 0.0, 1.0)
    R_nl = (4.903e-9 * Tmean_K**4 *
            (0.34 - 0.14 * np.sqrt(np.maximum(e_a, 1e-6))) *
            (1.35 * ratio - 0.35))
    R_n  = (R_ns - R_nl).astype(np.float32)
    e_a  = e_a.astype(np.float32)
    if has_wind:
        # FAO-56 eq. 47: u2 = u_z * 4.87 / ln(67.8 * z - 5.42), z=10 m -> ratio 0.748
        u2 = (np.nan_to_num(wind10, nan=u2_default) * 0.748).astype(np.float32)
        print(f"[gridded_forcing] Using sfcWind from forcing (mean u2 = {u2.mean():.2f} m/s)")
    else:
        u2 = np.full((n_time, n_nodes), u2_default, dtype=np.float32)

    forcing = np.stack(
        [pr, tmin.astype(np.float32), tmax.astype(np.float32), R_n, u2, e_a],
        axis=-1,
    )

    forcing = _fill_nan(forcing)

    if cache_nc is not None:
        _is_zarr = cache_nc.suffix == ".zarr" or cache_nc.name.endswith(".zarr")
        if _is_zarr:
            _save_forcing_zarr(forcing, dates, cache_nc)
        else:
            _save_forcing_nc(forcing, dates, cache_nc)
        print(f"[gridded_forcing] Cached to {cache_nc}")

    t = torch.from_numpy(forcing)
    return t.to(device) if device else t


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _open_gridded(path: str | Path):
    """Open a gridded climate dataset (Zarr, NetCDF, or GRIB) as xarray."""
    import xarray as xr
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".zarr" or path.name.endswith(".zarr"):
        return xr.open_zarr(path)
    elif suffix in (".grib", ".grib2", ".grb"):
        return xr.open_dataset(path, engine="cfgrib")
    else:
        # Default: NetCDF (.nc) or any other format xarray can handle
        return xr.open_dataset(path)


def _normalize_coords(ds):
    """Rename common coordinate aliases to ``latitude``/``longitude``."""
    rename = {}
    for alias, canonical in [("lat", "latitude"), ("lon", "longitude")]:
        if alias in ds.coords and canonical not in ds.coords:
            rename[alias] = canonical
    if rename:
        ds = ds.rename(rename)
    return ds


def _extraterrestrial_radiation(lat_rad: np.ndarray, doy: np.ndarray) -> np.ndarray:
    """R_a in MJ/m²/day.  lat_rad: (N,), doy: (T,) → (T, N)."""
    dr    = 1 + 0.033 * np.cos(2 * np.pi / 365 * doy)
    delta = 0.409 * np.sin(2 * np.pi / 365 * doy - 1.39)
    arg   = -np.tan(lat_rad)[np.newaxis, :] * np.tan(delta)[:, np.newaxis]
    ws    = np.arccos(np.clip(arg, -1.0, 1.0))
    Gsc   = 0.0820
    R_a   = (24 * 60 / np.pi * Gsc * dr[:, np.newaxis] *
             (ws * np.sin(lat_rad)[np.newaxis, :] * np.sin(delta)[:, np.newaxis]
              + np.cos(lat_rad)[np.newaxis, :] * np.cos(delta)[:, np.newaxis] * np.sin(ws)))
    return np.maximum(R_a, 0.0)


def _saturation_vapour_pressure(T_celsius: np.ndarray) -> np.ndarray:
    """e_s(T) in kPa — FAO-56 eq. 11."""
    return 0.6108 * np.exp(17.27 * T_celsius / (T_celsius + 237.3))


def _fill_nan(forcing: np.ndarray) -> np.ndarray:
    """Forward-fill NaN values along the time axis (axis 0) per node and variable.

    Remaining NaN at the start of the record (no prior valid value) are
    backward-filled. If all values for a (node, variable) are NaN, replaced with 0.
    """
    if not np.isnan(forcing).any():
        return forcing
    T, N, F = forcing.shape
    out = forcing.copy()
    n_filled = 0
    for n in range(N):
        for f in range(F):
            col = out[:, n, f]
            nan_mask = np.isnan(col)
            if not nan_mask.any():
                continue
            n_filled += int(nan_mask.sum())
            # Forward-fill
            last = np.nan
            for t in range(T):
                if not np.isnan(col[t]):
                    last = col[t]
                elif not np.isnan(last):
                    col[t] = last
            # Backward-fill any leading NaN
            last = np.nan
            for t in range(T - 1, -1, -1):
                if not np.isnan(col[t]):
                    last = col[t]
                elif not np.isnan(last):
                    col[t] = last
            # All-NaN fallback
            if np.isnan(col).any():
                col[:] = 0.0
    if n_filled > 0:
        warnings.warn(
            f"[gridded_forcing] Forward-filled {n_filled} NaN values in forcing.",
            stacklevel=3,
        )
    return out


def _save_forcing_zarr(forcing: np.ndarray, dates: np.ndarray, path: Path) -> None:
    import xarray as xr
    import shutil
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.rmtree(path)  # Zarr stores are directories; overwrite cleanly
    FORCING_VARS = ["P", "T_min", "T_max", "R_n", "u2", "e_a"]
    ds = xr.Dataset(
        {"forcing": xr.DataArray(
            forcing, dims=["time", "node", "variable"],
            attrs={"variables": ", ".join(FORCING_VARS),
                   "units": "mm/day, degC, degC, MJ/m2/day, m/s, kPa"},
        )},
        coords={"time": dates, "variable": FORCING_VARS},
    )
    ds.to_zarr(path)


def _save_forcing_nc(forcing: np.ndarray, dates: np.ndarray, path: Path) -> None:
    import xarray as xr
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    FORCING_VARS = ["P", "T_min", "T_max", "R_n", "u2", "e_a"]
    ds = xr.Dataset(
        {"forcing": xr.DataArray(
            forcing, dims=["time", "node", "variable"],
            attrs={"variables": ", ".join(FORCING_VARS),
                   "units": "mm/day, degC, degC, MJ/m2/day, m/s, kPa"},
        )},
        coords={"time": dates, "variable": FORCING_VARS},
    )
    ds.to_netcdf(path)
