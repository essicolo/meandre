"""Load observed daily discharge and map to model nodes.

Two backends:
1. DuckDB (preferred) — via ``BasinCache.load_observations()``.
2. NetCDF (legacy) — ``load_hydrometric_stations()`` reads the old
   ``stations_concatenees.nc`` format directly.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np


def load_hydrometric_stations(
    stations_nc: str | Path,
    troncon_ids: list[int],
    date_start: str,
    date_end: str,
    basin_prefix: str,
    min_valid_days: int = 365,
) -> dict:
    """Load observed discharge from NetCDF and map stations to troncon nodes.

    .. deprecated::
        Prefer ``BasinCache.load_observations()`` which reads from DuckDB.

    Parameters
    ----------
    stations_nc:
        Path to a NetCDF with dimensions ``station_id`` and ``time``,
        variable ``discharge``, and a ``troncon_id`` variable whose values
        are strings like ``"SLSO02878"``.
    troncon_ids:
        Ordered list of troncon IDs from ``load_hydrotel`` (``node_ids``).
        The list position is the node index used by the model.
    date_start, date_end:
        Time window to extract (inclusive).
    basin_prefix:
        Prefix used in ``troncon_id`` to identify the target basin
        (e.g. ``"SLSO"``).
    min_valid_days:
        Minimum number of non-NaN days within the window for a station to
        be retained.

    Returns
    -------
    dict with keys:
        ``discharge``        : (T, N) float32 array — NaN at ungauged nodes
        ``station_node_map`` : {station_id (str): node_index (int)}
        ``dates``            : (T,) datetime64 array
        ``n_stations``       : number of stations retained
    """
    import xarray as xr

    ds = xr.open_dataset(stations_nc)
    ds_slice = ds.sel(time=slice(date_start, date_end))

    prefix = basin_prefix.upper()
    basin_mask = np.array(
        [str(t).upper().startswith(prefix) for t in ds.troncon_id.values]
    )
    ds_basin       = ds_slice.isel(station_id=basin_mask)
    tids_in_basin  = ds.troncon_id.values[basin_mask]
    station_ids    = ds.station_id.values[basin_mask]

    troncon_idx = {tid: i for i, tid in enumerate(troncon_ids)}

    dates         = ds_basin.time.values
    n_time        = len(dates)
    n_nodes_total = len(troncon_ids)

    discharge_full: np.ndarray = np.full(
        (n_time, n_nodes_total), np.nan, dtype=np.float32
    )
    station_node_map: dict[str, int] = {}
    n_kept = 0

    for i, (sid, tid_str) in enumerate(zip(station_ids, tids_in_basin)):
        numeric_part = str(tid_str)[len(prefix):]
        try:
            tid_int = int(numeric_part)
        except ValueError:
            warnings.warn(
                f"Cannot parse troncon_id '{tid_str}' — skipping station {sid}",
                stacklevel=2,
            )
            continue

        node_idx = troncon_idx.get(tid_int)
        if node_idx is None:
            continue

        q = ds_basin.discharge.isel(station_id=i).values.astype(np.float32)
        if (~np.isnan(q)).sum() < min_valid_days:
            continue

        # Multiple stations on the same troncon: keep first by file order
        if node_idx in station_node_map.values():
            continue

        discharge_full[:, node_idx] = q
        station_node_map[str(sid)]  = node_idx
        n_kept += 1

    ds.close()
    print(f"[station_obs] {n_kept} stations retained ({prefix}, {date_start} to {date_end})")

    return {
        "discharge":        discharge_full,
        "station_node_map": station_node_map,
        "dates":            dates,
        "n_stations":       n_kept,
    }
