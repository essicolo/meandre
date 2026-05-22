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
GRACE_SHORTNAME = "TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06V3_V4"
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
        _auth = earthaccess.login(strategy="environment")  # reads EARTHDATA_LOGIN env var
    except ImportError:
        raise ImportError(
            "GRACE loader requires the earthaccess library. "
            "Install with: pip install earthaccess"
        )

    results = earthaccess.search_data(
        short_name=GRACE_SHORTNAME,
        temporal=(date_start, date_end),
        bounding_box=bbox,
    )
    if not results:
        logger.warning(
            f"[GRACE] No data found for bbox={bbox}, period={date_start}/{date_end}"
        )
        return pd.DataFrame(columns=["date", "tws_mm", "uncertainty", "quality_ok"])

    logger.info(f"[GRACE] Downloading {len(results)} monthly granules…")

    import tempfile, xarray as xr, os

    rows = []
    with tempfile.TemporaryDirectory() as tmpdir:
        files = earthaccess.download(results, local_path=tmpdir)
        for fpath in sorted(files):
            try:
                ds = xr.open_dataset(fpath)
                lwe = ds[GRACE_VAR].sel(
                    lat=slice(bbox[1], bbox[3]),
                    lon=slice(bbox[0], bbox[2]),
                )
                basin_mean_cm = float(lwe.mean().values)
                tws_mm = basin_mean_cm * CM_TO_MM

                # Uncertainty (not in all versions — optional)
                uncert = None
                for uvar in ("uncertainty", "lwe_uncertainty"):
                    if uvar in ds:
                        u = ds[uvar].sel(
                            lat=slice(bbox[1], bbox[3]),
                            lon=slice(bbox[0], bbox[2]),
                        )
                        uncert = float(u.mean().values) * CM_TO_MM
                        break

                # Date: first day of the monthly composite
                t = ds.coords.get("time", ds.coords.get("TIME"))
                date = pd.Timestamp(
                    t.values[0] if t is not None else "1970-01-01"
                ).to_period("M").to_timestamp()

                rows.append({
                    "date": date,
                    "tws_mm": tws_mm,
                    "uncertainty": uncert if uncert is not None else np.nan,
                    "quality_ok": np.isfinite(tws_mm),
                })
                ds.close()
            except Exception as exc:
                logger.warning(f"[GRACE] Skipped {fpath}: {exc}")

    if not rows:
        return pd.DataFrame(columns=["date", "tws_mm", "uncertainty", "quality_ok"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)
