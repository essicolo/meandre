"""GRACE / GRACE-FO terrestrial water storage anomaly loader.

Downloads JPL GRACE Mascon RL06v3 monthly TWS anomaly (cm water equivalent)
for the basin bounding box and returns basin-average ΔStorage in mm/month.

Data source
-----------
JPL GRACE Mascon RL06v3 — NASA Physical Oceanography DAAC (PO.DAAC):
    https://podaac.jpl.nasa.gov/dataset/TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06V3_V4

The product provides monthly 0.5° × 0.5° grids of TWS anomaly relative to
the 2004-2009 mean baseline (cm water equivalent). One grid cell often covers
the entire SLSO basin (~3°×3°), so the basin-average is essentially a single
cell value. This constrains the total storage dynamics:

    ΔS_model = ΔSWE + Δsoil_1 + Δsoil_2 + Δsoil_3 + Δaquifer
    ΔS_grace = observed

Download requirement
--------------------
NASA Earthdata login is required. Credentials can be provided via:
    1. Environment variables: EARTHDATA_LOGIN, EARTHDATA_PASSWORD
    2. ~/.netrc entry for urs.earthdata.nasa.gov
    3. earthaccess library (https://github.com/nsidc/earthaccess)

The ``fetch_grace_tws`` function handles authentication automatically when
earthaccess is installed and credentials are available.

References
----------
Watkins M. et al. (2015) J. Geophys. Res. 120, 2648-2671 — RL06 Mascons
Wiese D. et al. (2019) J. Geophys. Res. 124, 5468-5489  — CRI filtering
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# JPL Mascon short name on PO.DAAC / Earthdata
GRACE_SHORTNAME = "TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V4"
GRACE_VAR = "lwe_thickness"   # cm water equivalent anomaly
CM_TO_MM = 10.0


def fetch_grace_tws(
    bbox: tuple[float, float, float, float],
    date_start: str,
    date_end: str,
) -> pd.DataFrame:
    """Download and spatially aggregate GRACE-FO TWS for the basin bbox.

    Parameters
    ----------
    bbox : (lon_min, lat_min, lon_max, lat_max)
    date_start, date_end : ISO date strings

    Returns
    -------
    pd.DataFrame with columns:
        date         : datetime64[ns] — first day of the monthly composite
        tws_mm       : float — basin-average TWS anomaly in mm w.e.
        uncertainty  : float — propagated measurement uncertainty in mm (if available)
        quality_ok   : bool
    """
    try:
        import earthaccess
    except ImportError:
        raise ImportError(
            "GRACE loader requires the earthaccess library. "
            "Install with: uv add earthaccess"
        )

    import os
    # Try token first (EARTHDATA_TOKEN env var), then netrc, then guest.
    # GRACE mascons are public (no auth required to download), so guest works.
    # Interactive mode is explicitly disabled — fails in non-interactive runs.
    token = os.environ.get("EARTHDATA_TOKEN")
    if token:
        _auth = earthaccess.login(strategy="environment")
    else:
        try:
            _auth = earthaccess.login(strategy="netrc")
        except Exception:
            _auth = earthaccess.login(strategy="guest")

    # GRACE data is global — bounding_box filter in search is not reliable.
    # We download all granules for the period and filter spatially after.
    # count=-1 lifts the default limit (which may be 1 or 10).
    results = earthaccess.search_data(
        short_name=GRACE_SHORTNAME,
        temporal=(date_start, date_end),
        count=-1,
    )
    if not results:
        logger.warning(
            f"[GRACE] No data found for period={date_start}/{date_end}"
        )
        return pd.DataFrame(columns=["date", "tws_mm", "uncertainty", "quality_ok"])

    print(f"  [GRACE] {len(results)} granules trouvés — téléchargement en cours…")

    import tempfile, xarray as xr, os

    # Convert bbox to 0-360 longitude convention used by JPL GRACE mascons
    lon_min_360 = bbox[0] % 360  # e.g. -73 → 287
    lon_max_360 = bbox[2] % 360  # e.g. -69.6 → 290.4
    lat_min, lat_max = bbox[1], bbox[3]

    rows = []
    with tempfile.TemporaryDirectory() as tmpdir:
        files = earthaccess.download(results, local_path=tmpdir)
        for fpath in sorted(files):
            try:
                ds = xr.open_dataset(fpath)

                # Detect variable name — varies by version
                var = None
                for candidate in ("lwe_thickness", "LWE_thickness", "lwe",
                                  "land_water_equivalent_thickness"):
                    if candidate in ds:
                        var = candidate
                        break
                if var is None:
                    logger.warning(f"  [GRACE] {Path(fpath).name}: variable inconnue "
                                   f"({list(ds.data_vars)[:5]})")
                    ds.close()
                    continue

                # Spatial subset — handle both lon conventions
                lat_dim = [d for d in ds[var].dims if "lat" in d.lower()][0]
                lon_dim = [d for d in ds[var].dims if "lon" in d.lower()][0]
                lon_vals = ds[lon_dim].values

                # Detect 0-360 vs -180-180
                if lon_vals.max() > 180:
                    lwe = ds[var].sel(
                        {lat_dim: slice(lat_min, lat_max),
                         lon_dim: slice(lon_min_360, lon_max_360)}
                    )
                else:
                    lwe = ds[var].sel(
                        {lat_dim: slice(lat_min, lat_max),
                         lon_dim: slice(bbox[0], bbox[2])}
                    )

                if lwe.size == 0:
                    logger.warning(f"  [GRACE] {Path(fpath).name}: aucun pixel dans bbox")
                    ds.close()
                    continue

                basin_mean_cm = float(np.nanmean(lwe.values))
                tws_mm = basin_mean_cm * CM_TO_MM

                # Uncertainty (optional)
                uncert = np.nan
                for uvar in ("uncertainty", "lwe_uncertainty", "LWE_uncertainty"):
                    if uvar in ds:
                        u = ds[uvar].sel(
                            {lat_dim: slice(lat_min, lat_max),
                             lon_dim: slice(lon_min_360 if lon_vals.max() > 180
                                            else bbox[0],
                                            lon_max_360 if lon_vals.max() > 180
                                            else bbox[2])}
                        )
                        uncert = float(np.nanmean(u.values)) * CM_TO_MM
                        break

                # Date
                t = ds.coords.get("time", ds.coords.get("TIME"))
                date = pd.Timestamp(
                    t.values[0] if t is not None else "1970-01-01"
                ).to_period("M").to_timestamp()

                # GRACE mascon RL06.3 is distributed as one file with all
                # time steps along the 'time' dimension. Loop over them.
                t_coord = None
                for tc in ("time", "TIME", "month"):
                    if tc in ds.coords:
                        t_coord = tc
                        break

                if t_coord is not None and ds[t_coord].size > 1:
                    # Multi-time file: extract one row per time step
                    for ti in range(ds[t_coord].size):
                        lwe_t = lwe.isel({t_coord: ti}) if t_coord in lwe.dims else lwe
                        val_cm = float(np.nanmean(lwe_t.values))
                        val_mm = val_cm * CM_TO_MM
                        dt = pd.Timestamp(ds[t_coord].values[ti]).to_period("M").to_timestamp()

                        uncert_t = np.nan
                        for uvar in ("uncertainty", "lwe_uncertainty", "LWE_uncertainty"):
                            if uvar in ds:
                                u_t = ds[uvar].isel({t_coord: ti}) if t_coord in ds[uvar].dims else ds[uvar]
                                u_sub = u_t.sel(
                                    {lat_dim: slice(lat_min, lat_max),
                                     lon_dim: slice(lon_min_360 if lon_vals.max() > 180
                                                    else bbox[0],
                                                    lon_max_360 if lon_vals.max() > 180
                                                    else bbox[2])}
                                )
                                uncert_t = float(np.nanmean(u_sub.values)) * CM_TO_MM
                                break

                        rows.append({"date": dt, "tws_mm": val_mm,
                                     "uncertainty": uncert_t,
                                     "quality_ok": np.isfinite(val_mm)})
                else:
                    # Single time step
                    rows.append({"date": date, "tws_mm": tws_mm,
                                 "uncertainty": uncert,
                                 "quality_ok": np.isfinite(tws_mm)})

                ds.close()
                ds.close()
            except Exception as exc:
                logger.warning(f"[GRACE] Skipped {fpath}: {exc}")

    if not rows:
        return pd.DataFrame(columns=["date", "tws_mm", "uncertainty", "quality_ok"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)
