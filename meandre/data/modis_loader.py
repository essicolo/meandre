"""MODIS multi-product loader.

Products and sources
--------------------
MOD16A2 ETR  500 m  8-day   → NASA Earthdata (earthaccess, token required)
MOD10A1 snow 500 m  daily   → Planetary Computer STAC (no auth)
MOD13A1 NDVI 500 m  16-day  → Planetary Computer STAC (no auth)

MOD16A2 is not available on Planetary Computer (only MOD16A3GF annual is).
It is fetched via earthaccess using the same token as GRACE.

What each product constrains in meandre
-----------------------------------------
ETR    K_c, K_sat, f_vert
snow   C_f, T_melt, T_snow
NDVI   K_c seasonal variation, LAI proxy

References
----------
Mu Q. et al. (2011) Remote Sens. Environ. 115, 1781-1800  — MOD16A2
Hall D. & Riggs G. (2007) Hydrol. Process. 21, 1534-1547  — MOD10A1
Huete A. et al. (2002) Remote Sens. Environ. 83, 195-213   — MOD13A1/A2
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── MOD16A2 ETR (NASA Earthdata) ───────────────────────────────────────────

ET_SHORTNAME = "MOD16A2"           # NASA Earthdata short name
ET_VAR = "ET_500m"
ET_SCALE = 0.1                     # raw integer → mm/8day
ET_FILL = 32761
ET_MAX_MM_8DAY = 3000
ET_DAYS = 8.0

# ─── MOD10A1 snow (Planetary Computer) ──────────────────────────────────────

SNOW_COLLECTION = "modis-10A1-061"
SNOW_ASSET = "NDSI_Snow_Cover"
SNOW_CLOUD = 200
SNOW_FILL = 250
SNOW_MAX = 100

# ─── MOD13A1 NDVI (Planetary Computer, 500m 16-day) ─────────────────────────

NDVI_COLLECTION = "modis-13A1-061"   # 500m 16-day — confirmed on PC
NDVI_ASSET = "500m_16_days_NDVI"
NDVI_SCALE = 0.0001
NDVI_FILL = -3000


# ─── Shared helpers ──────────────────────────────────────────────────────────

def _pc_sign_modifier():
    try:
        import planetary_computer as pc
        return pc.sign_inplace
    except ImportError:
        return None


def _pc_sign(item):
    try:
        import planetary_computer as pc
        return pc.sign(item)
    except Exception:
        return item


def _open_catalog():
    import pystac_client
    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=_pc_sign_modifier(),
    )


def _nearest_node_lookup(da, lons, lats, node_indices):
    """For each (lon, lat) pair, extract the nearest pixel value from da.
    Returns list of (node_idx, float_value) — value may be NaN.
    """
    x_coords = da.coords.get("x", da.coords.get("lon", da.coords.get("longitude")))
    y_coords = da.coords.get("y", da.coords.get("lat", da.coords.get("latitude")))
    if x_coords is None or y_coords is None:
        raise ValueError(f"Cannot find spatial coords in DataArray: {list(da.coords)}")
    x_arr = x_coords.values
    y_arr = y_coords.values
    arr = da.values if da.values.ndim == 2 else da.values.squeeze()

    rows = []
    for ni, lon, lat in zip(node_indices, lons, lats):
        ix = int(np.argmin(np.abs(x_arr - lon)))
        iy = int(np.argmin(np.abs(y_arr - lat)))
        val = float(arr[iy, ix]) if arr.ndim == 2 else float(arr[ix])
        rows.append((int(ni), val))
    return rows


# ─── MOD16A2 ETR via earthaccess ─────────────────────────────────────────────

def fetch_modis_et(
    bbox: tuple[float, float, float, float],
    date_start: str,
    date_end: str,
    node_coords: "np.ndarray",
    node_indices: "np.ndarray | None" = None,
) -> pd.DataFrame:
    """Fetch MOD16A2 8-day ETR via NASA Earthdata (earthaccess).

    Requires EARTHDATA_TOKEN env var (same token used for GRACE).
    Returns DataFrame(date, node_idx, etr_mm_day, quality_ok).
    """
    try:
        import earthaccess, xarray as xr, rioxarray  # noqa
    except ImportError as e:
        raise ImportError("fetch_modis_et requires earthaccess + rioxarray") from e

    node_coords = np.asarray(node_coords, dtype=np.float64)
    n = node_coords.shape[0]
    if node_indices is None:
        node_indices = np.arange(n)
    lons, lats = node_coords[:, 0], node_coords[:, 1]

    import os
    if os.environ.get("EARTHDATA_TOKEN"):
        earthaccess.login(strategy="environment")
    else:
        earthaccess.login(strategy="all")

    results = earthaccess.search_data(
        short_name=ET_SHORTNAME,
        temporal=(date_start, date_end),
        bounding_box=bbox,
        count=-1,
    )
    if not results:
        logger.warning(f"[MOD16A2] No granules found for {date_start}/{date_end}")
        return pd.DataFrame(columns=["date", "node_idx", "etr_mm_day", "quality_ok"])

    print(f"  [MOD16A2] {len(results)} granules → téléchargement…")
    rows = []
    with tempfile.TemporaryDirectory() as tmpdir:
        files = earthaccess.download(results, local_path=tmpdir)
        for fpath in sorted(files):
            try:
                da = xr.open_dataarray(fpath, engine="rasterio").squeeze("band", drop=True)
                if da.rio.crs is not None and da.rio.crs.to_epsg() != 4326:
                    da = da.rio.reproject("EPSG:4326")
                # Date from filename e.g. MOD16A2.A2020153...
                fname = Path(fpath).stem
                try:
                    year = int(fname.split(".A")[1][:4])
                    doy = int(fname.split(".A")[1][4:7])
                    composite_date = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=doy - 1)
                except Exception:
                    composite_date = pd.Timestamp("1970-01-01")

                for ni, raw in _nearest_node_lookup(da, lons, lats, node_indices):
                    if np.isnan(raw) or raw >= ET_FILL or raw < 0:
                        rows.append({"date": composite_date, "node_idx": ni,
                                     "etr_mm_day": np.nan, "quality_ok": False})
                    else:
                        etr = min(raw * ET_SCALE, ET_MAX_MM_8DAY) / ET_DAYS
                        rows.append({"date": composite_date, "node_idx": ni,
                                     "etr_mm_day": etr, "quality_ok": True})
                da.close()
            except Exception as exc:
                logger.warning(f"[MOD16A2] Skipped {Path(fpath).name}: {exc}")

    if not rows:
        return pd.DataFrame(columns=["date", "node_idx", "etr_mm_day", "quality_ok"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "node_idx"]).reset_index(drop=True)


# ─── MOD10A1 snow (Planetary Computer) ───────────────────────────────────────

def fetch_modis_snow(
    bbox: tuple[float, float, float, float],
    date_start: str,
    date_end: str,
    node_coords: "np.ndarray",
    node_indices: "np.ndarray | None" = None,
) -> pd.DataFrame:
    """Fetch MOD10A1 daily snow cover fraction via Planetary Computer.

    Returns DataFrame(date, node_idx, snow_frac, quality_ok).
    snow_frac ∈ [0, 1]. NaN on cloudy / fill days.
    """
    import xarray as xr, rioxarray  # noqa

    node_coords = np.asarray(node_coords, dtype=np.float64)
    n = node_coords.shape[0]
    if node_indices is None:
        node_indices = np.arange(n)
    lons, lats = node_coords[:, 0], node_coords[:, 1]

    catalog = _open_catalog()
    items = list(catalog.search(
        collections=[SNOW_COLLECTION], bbox=list(bbox),
        datetime=f"{date_start}/{date_end}",
    ).items())
    print(f"  [MOD10A1] {len(items)} items → extraction…")

    rows = []
    for item in items:
        try:
            item = _pc_sign(item)
            href = item.assets[SNOW_ASSET].href
            da = xr.open_dataarray(href, engine="rasterio").squeeze("band", drop=True)
            if da.rio.crs is not None and da.rio.crs.to_epsg() != 4326:
                da = da.rio.reproject("EPSG:4326")
            # Date from RANGEBEGINNINGDATE attr (item.datetime is None on PC)
            _rbd = da.attrs.get("RANGEBEGINNINGDATE", "")
            try:
                obs_date = pd.Timestamp(_rbd) if _rbd else pd.NaT
            except Exception:
                obs_date = pd.NaT
            for ni, raw in _nearest_node_lookup(da, lons, lats, node_indices):
                # raw may be NaN (no-data pixel after reproject)
                if np.isnan(raw):
                    rows.append({"date": obs_date, "node_idx": ni,
                                 "snow_frac": np.nan, "quality_ok": False})
                    continue
                raw_i = int(raw)
                if raw_i >= SNOW_CLOUD or raw_i == SNOW_FILL or raw_i > SNOW_MAX:
                    rows.append({"date": obs_date, "node_idx": ni,
                                 "snow_frac": np.nan, "quality_ok": False})
                else:
                    rows.append({"date": obs_date, "node_idx": ni,
                                 "snow_frac": raw_i / 100.0, "quality_ok": True})
            da.close()
        except Exception as exc:
            logger.warning(f"[MOD10A1] Skipped {item.id}: {exc}")

    if not rows:
        return pd.DataFrame(columns=["date", "node_idx", "snow_frac", "quality_ok"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "node_idx"]).reset_index(drop=True)


# ─── MOD13A1 NDVI (Planetary Computer) ───────────────────────────────────────

def fetch_modis_ndvi(
    bbox: tuple[float, float, float, float],
    date_start: str,
    date_end: str,
    node_coords: "np.ndarray",
    node_indices: "np.ndarray | None" = None,
) -> pd.DataFrame:
    """Fetch MOD13A1 16-day NDVI (500m) via Planetary Computer.

    Returns DataFrame(date, node_idx, ndvi, quality_ok).
    ndvi ∈ [-1, 1].
    """
    import xarray as xr, rioxarray  # noqa

    node_coords = np.asarray(node_coords, dtype=np.float64)
    n = node_coords.shape[0]
    if node_indices is None:
        node_indices = np.arange(n)
    lons, lats = node_coords[:, 0], node_coords[:, 1]

    catalog = _open_catalog()
    items = list(catalog.search(
        collections=[NDVI_COLLECTION], bbox=list(bbox),
        datetime=f"{date_start}/{date_end}",
    ).items())
    print(f"  [MOD13A1] {len(items)} items → extraction…")

    rows = []
    for item in items:
        try:
            item = _pc_sign(item)
            href = item.assets[NDVI_ASSET].href
            da = xr.open_dataarray(href, engine="rasterio").squeeze("band", drop=True)
            if da.rio.crs is not None and da.rio.crs.to_epsg() != 4326:
                da = da.rio.reproject("EPSG:4326")
            # Date from RANGEBEGINNINGDATE attr (item.datetime is None on PC)
            _rbd = da.attrs.get("RANGEBEGINNINGDATE", "")
            try:
                composite_date = pd.Timestamp(_rbd) if _rbd else pd.NaT
            except Exception:
                composite_date = pd.NaT
            # PC COG stores raw × 10000 relative to the HDF int16 values.
            # Effective scale: raw × 1e-8 → NDVI in [-1, 1].
            # Valid COG range: [-20_000_000, 100_000_000]; fill ≈ -286_720_000.
            for ni, raw in _nearest_node_lookup(da, lons, lats, node_indices):
                if np.isnan(raw) or raw < -20_000_000 or raw > 100_000_000:
                    rows.append({"date": composite_date, "node_idx": ni,
                                 "ndvi": np.nan, "quality_ok": False})
                else:
                    rows.append({"date": composite_date, "node_idx": ni,
                                 "ndvi": raw * 1e-8, "quality_ok": True})
            da.close()
        except Exception as exc:
            logger.warning(f"[MOD13A1] Skipped {item.id}: {exc}")

    if not rows:
        return pd.DataFrame(columns=["date", "node_idx", "ndvi", "quality_ok"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "node_idx"]).reset_index(drop=True)
