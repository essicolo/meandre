"""MODIS multi-product loader — Planetary Computer STAC.

Loads three MODIS products relevant to hydrological modelling:

Product        Collection             Variable          Resolution  Cadence
---------------------------------------------------------------------------
MOD16A2 ETR    modis-16A2-061         ET_500m           500 m       8-day
MOD10A1 snow   modis-10A1-061         NDSI_Snow_Cover   500 m       daily
MOD13A2 NDVI   modis-13A2-061         500m_16_days_NDVI 1 km        16-day

All products are fetched via the Microsoft Planetary Computer STAC catalogue
(no API key required) and aggregated to basin nodes via nearest-neighbour.

The long-format DataFrames returned are ready for DuckDB ingestion via
``BasinCache.import_modis_et/snow/ndvi``. The Gaussian NLL in HydroLoss
skips NaN automatically, so no interpolation is performed between composites.

What each product constrains in meandre
-----------------------------------------
ETR    K_c (crop coeff.), K_sat (indirectly), f_vert — most direct constraint
snow   C_f (degree-day melt factor), T_melt, T_snow
NDVI   K_c seasonal variation, LAI proxy, vegetation phenology

References
----------
Mu Q. et al. (2011) Remote Sens. Environ. 115, 1781-1800  — MOD16A2
Hall D. & Riggs G. (2007) Hydrol. Process. 21, 1534-1547  — MOD10A1
Huete A. et al. (2002) Remote Sens. Environ. 83, 195-213   — MOD13A2
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── MOD16A2 ETR constants ───────────────────────────────────────────────────

ET_COLLECTION = "modis-16A2-061"
ET_ASSET = "ET_500m"
ET_SCALE = 0.1             # kg/m²/8day raw → mm/8day
ET_FILL = 32761
ET_MAX_MM_8DAY = 3000
ET_DAYS = 8.0

# ─── MOD10A1 snow constants ───────────────────────────────────────────────────

SNOW_COLLECTION = "modis-10A1-061"
SNOW_ASSET = "NDSI_Snow_Cover"
SNOW_FILL = 250            # land, no snow
SNOW_CLOUD = 200           # cloud mask
SNOW_MAX = 100             # fraction in %

# ─── MOD13A2 NDVI constants ──────────────────────────────────────────────────

NDVI_COLLECTION = "modis-13A2-061"
NDVI_ASSET = "500m_16_days_NDVI"
NDVI_SCALE = 0.0001        # raw integer → dimensionless [-1, 1]
NDVI_FILL = -3000
NDVI_DAYS = 16.0


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
    """For each (lon, lat) pair, extract the nearest pixel value from da."""
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


# ─── MOD16A2 ETR ─────────────────────────────────────────────────────────────

def fetch_modis_et(
    bbox: tuple[float, float, float, float],
    date_start: str,
    date_end: str,
    node_coords: "np.ndarray",
    node_indices: "np.ndarray | None" = None,
) -> pd.DataFrame:
    """Fetch MOD16A2 8-day ETR and aggregate to basin nodes.

    Returns DataFrame(date, node_idx, etr_mm_day, quality_ok).
    """
    import xarray as xr
    import rioxarray  # noqa

    node_coords = np.asarray(node_coords, dtype=np.float64)
    n = node_coords.shape[0]
    if node_indices is None:
        node_indices = np.arange(n)
    lons, lats = node_coords[:, 0], node_coords[:, 1]

    catalog = _open_catalog()
    items = list(catalog.search(
        collections=[ET_COLLECTION], bbox=list(bbox),
        datetime=f"{date_start}/{date_end}",
    ).items())

    logger.info(f"[MOD16A2] {len(items)} items found")
    rows = []
    for item in items:
        try:
            item = _pc_sign(item)
            href = item.assets[ET_ASSET].href
            da = xr.open_dataarray(href, engine="rasterio").squeeze("band", drop=True)
            if da.rio.crs is not None and da.rio.crs.to_epsg() != 4326:
                da = da.rio.reproject("EPSG:4326")
            composite_date = pd.Timestamp(
                item.datetime or item.properties.get("start_datetime")
            )
            for ni, raw in _nearest_node_lookup(da, lons, lats, node_indices):
                if raw >= ET_FILL or raw < 0:
                    rows.append({"date": composite_date, "node_idx": ni,
                                 "etr_mm_day": np.nan, "quality_ok": False})
                else:
                    etr = min(raw * ET_SCALE, ET_MAX_MM_8DAY) / ET_DAYS
                    rows.append({"date": composite_date, "node_idx": ni,
                                 "etr_mm_day": etr, "quality_ok": True})
            da.close()
        except Exception as exc:
            logger.warning(f"[MOD16A2] Skipped {item.id}: {exc}")

    if not rows:
        return pd.DataFrame(columns=["date", "node_idx", "etr_mm_day", "quality_ok"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "node_idx"]).reset_index(drop=True)


# ─── MOD10A1 snow ────────────────────────────────────────────────────────────

def fetch_modis_snow(
    bbox: tuple[float, float, float, float],
    date_start: str,
    date_end: str,
    node_coords: "np.ndarray",
    node_indices: "np.ndarray | None" = None,
) -> pd.DataFrame:
    """Fetch MOD10A1 daily snow cover fraction and aggregate to basin nodes.

    Returns DataFrame(date, node_idx, snow_frac, quality_ok).
    snow_frac ∈ [0, 1] — fraction of pixel covered by snow.
    NaN on cloudy days.

    In meandre, this constrains C_f (degree-day melt factor) and T_melt by
    comparing modelled SWE > 0 area with observed snow cover extent.
    """
    import xarray as xr
    import rioxarray  # noqa

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

    logger.info(f"[MOD10A1] {len(items)} items found")
    rows = []
    for item in items:
        try:
            item = _pc_sign(item)
            href = item.assets[SNOW_ASSET].href
            da = xr.open_dataarray(href, engine="rasterio").squeeze("band", drop=True)
            if da.rio.crs is not None and da.rio.crs.to_epsg() != 4326:
                da = da.rio.reproject("EPSG:4326")
            obs_date = pd.Timestamp(item.datetime)
            for ni, raw in _nearest_node_lookup(da, lons, lats, node_indices):
                raw = int(raw)
                if raw >= SNOW_CLOUD:
                    rows.append({"date": obs_date, "node_idx": ni,
                                 "snow_frac": np.nan, "quality_ok": False})
                elif raw == SNOW_FILL or raw > SNOW_MAX:
                    rows.append({"date": obs_date, "node_idx": ni,
                                 "snow_frac": np.nan, "quality_ok": False})
                else:
                    rows.append({"date": obs_date, "node_idx": ni,
                                 "snow_frac": raw / 100.0, "quality_ok": True})
            da.close()
        except Exception as exc:
            logger.warning(f"[MOD10A1] Skipped {item.id}: {exc}")

    if not rows:
        return pd.DataFrame(columns=["date", "node_idx", "snow_frac", "quality_ok"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "node_idx"]).reset_index(drop=True)


# ─── MOD13A2 NDVI ────────────────────────────────────────────────────────────

def fetch_modis_ndvi(
    bbox: tuple[float, float, float, float],
    date_start: str,
    date_end: str,
    node_coords: "np.ndarray",
    node_indices: "np.ndarray | None" = None,
) -> pd.DataFrame:
    """Fetch MOD13A2 16-day NDVI and aggregate to basin nodes.

    Returns DataFrame(date, node_idx, ndvi, quality_ok).
    ndvi ∈ [-1, 1], dimensionless.

    In meandre, NDVI is a proxy for K_c and seasonal LAI variation.
    It can constrain the temporal pattern of actual ETR independently
    from the spatial calibration.
    """
    import xarray as xr
    import rioxarray  # noqa

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

    logger.info(f"[MOD13A2] {len(items)} items found")
    rows = []
    for item in items:
        try:
            item = _pc_sign(item)
            href = item.assets[NDVI_ASSET].href
            da = xr.open_dataarray(href, engine="rasterio").squeeze("band", drop=True)
            if da.rio.crs is not None and da.rio.crs.to_epsg() != 4326:
                da = da.rio.reproject("EPSG:4326")
            composite_date = pd.Timestamp(
                item.datetime or item.properties.get("start_datetime")
            )
            for ni, raw in _nearest_node_lookup(da, lons, lats, node_indices):
                if raw <= NDVI_FILL or raw > 10000:
                    rows.append({"date": composite_date, "node_idx": ni,
                                 "ndvi": np.nan, "quality_ok": False})
                else:
                    rows.append({"date": composite_date, "node_idx": ni,
                                 "ndvi": raw * NDVI_SCALE, "quality_ok": True})
            da.close()
        except Exception as exc:
            logger.warning(f"[MOD13A2] Skipped {item.id}: {exc}")

    if not rows:
        return pd.DataFrame(columns=["date", "node_idx", "ndvi", "quality_ok"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "node_idx"]).reset_index(drop=True)
