"""MODIS multi-product loader.

Products and sources
--------------------
MOD16A3GF ETR  500 m  annual  → Planetary Computer STAC (no auth)
MOD10A1   snow 500 m  daily   → Planetary Computer STAC (no auth)
MOD13A1   NDVI 500 m  16-day  → Planetary Computer STAC (no auth)

NOTE on ETR product choice: MOD16A2 (8-day) is HDF4 on NASA Earthdata, and
GDAL builds shipped with rasterio on Windows lack the HDF4 driver. We use
MOD16A3GF (annual, gap-filled) which is available as cloud-optimised GeoTIFF
on Planetary Computer. Trade-off: loses 8-day seasonality, keeps 25 annual
ETR observations per node — sufficient for spatial identifiability of K_c
and the gradient diagnostic. For 8-day resolution, install HDF4-enabled
GDAL or use AppEEARS pre-processing.

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
from meandre.utils import paths as _paths

logger = logging.getLogger(__name__)


# ─── MOD16A3GF ETR (Planetary Computer, annual) ────────────────────────────

ET_COLLECTION = "modis-16A3GF-061"
ET_ASSET = "ET_500m"
ET_FILL = 32761                    # MODIS native fill
ET_MAX_MM_YEAR = 30000             # cap raw before scaling
# PC COG ET_500m units are mm/year directly (no scale_factor application
# needed — verified empirically on Quebec forest: 565 mm/year typical).
ET_DAYS_PER_YEAR = 365.0

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


# ─── MOD16A2GF ETR 8-jours via earthaccess + pyhdf (HDF-EOS sinusoïdal) ──────

ET8_SHORT = "MOD16A2GF"
ET8_VERSION = "061"
ET8_SCALE = 0.1            # raw → kg/m²/8day (= mm/8day)
ET8_VALID_MAX = 32700      # > valid_range max → fill/eau/etc.
_SINU = ("+proj=sinu +lon_0=0 +x_0=0 +y_0=0 "
         "+R=6371007.181 +units=m +no_defs")


def _parse_structmeta(sd):
    """UL/LR (m) et dims depuis StructMetadata.0 d'un granule HDF-EOS."""
    import re
    sm = sd.attributes().get("StructMetadata.0", "")
    def grab(key):
        m = re.search(rf"{key}=\(([-0-9.,]+)\)", sm)
        return tuple(float(v) for v in m.group(1).split(",")) if m else None
    def grab_int(key):
        m = re.search(rf"{key}=(\d+)", sm)
        return int(m.group(1)) if m else None
    ulx, uly = grab("UpperLeftPointMtrs")
    lrx, lry = grab("LowerRightMtrs")
    nx, ny = grab_int("XDim"), grab_int("YDim")
    return ulx, uly, lrx, lry, nx, ny


def fetch_modis_et_8day(
    bbox: tuple[float, float, float, float],
    date_start: str,
    date_end: str,
    node_coords: np.ndarray,
    node_indices: np.ndarray,
    cache_dir: str | Path = ".runs/slso/data/modis8",
    token_path: str = "~/.edl_token",
) -> pd.DataFrame:
    """Fetch MOD16A2GF.061 ETR 8-jours via NASA Earthdata (token bearer) et
    extrait aux nœuds par reprojection sinusoïdale (pyhdf + pyproj).

    Retourne DataFrame(date, node_idx, etr_mm_day, quality_ok). date = début du
    composite ; etr_mm_day = raw·0.1 / (jours du composite, 8 sauf dernier).
    """
    import earthaccess
    import requests
    from pyhdf.SD import SD, SDC
    from pyproj import Transformer

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    token = open(Path(token_path).expanduser()).read().strip()
    headers = {"Authorization": f"Bearer {token}"}

    lons = node_coords[:, 0].astype(float)
    lats = node_coords[:, 1].astype(float)
    node_indices = np.asarray(node_indices)
    tr = Transformer.from_crs("EPSG:4326", _SINU, always_xy=True)
    nx_sinu, ny_sinu = tr.transform(lons, lats)  # coords sinusoïdales des nœuds (m)

    granules = earthaccess.search_data(
        short_name=ET8_SHORT, version=ET8_VERSION,
        bounding_box=tuple(bbox), temporal=(date_start, date_end),
    )
    logger.info(f"[MOD16A2GF] {len(granules)} granules 8-jours trouvés")

    rows = []
    for g in granules:
        url = next((l for l in g.data_links() if l.endswith(".hdf")), None)
        if url is None:
            continue
        fn = url.split("/")[-1]
        # MOD16A2GF.A{YYYYDDD}.h..v..  → date composite + DOY
        ayyyyddd = fn.split(".")[1]  # A2018177
        year, doy = int(ayyyyddd[1:5]), int(ayyyyddd[5:8])
        cdate = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=doy - 1)
        days_in_year = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
        period = min(8, days_in_year - doy + 1)  # dernier composite = 5/6 jours

        local = cache_dir / fn
        if not local.exists():
            r = requests.get(url, headers=headers, allow_redirects=True, timeout=300)
            r.raise_for_status()
            local.write_bytes(r.content)

        sd = SD(str(local), SDC.READ)
        ulx, uly, lrx, lry, gnx, gny = _parse_structmeta(sd)
        psx = (lrx - ulx) / gnx
        psy = (uly - lry) / gny  # positif
        et = sd.select("ET_500m").get()
        sd.end()

        # nœuds tombant dans cette tuile
        col = np.floor((nx_sinu - ulx) / psx).astype(int)
        row = np.floor((uly - ny_sinu) / psy).astype(int)
        inside = (col >= 0) & (col < gnx) & (row >= 0) & (row < gny)
        for k in np.nonzero(inside)[0]:
            raw = float(et[row[k], col[k]])
            ok = raw <= ET8_VALID_MAX and raw >= -32767
            etr = (raw * ET8_SCALE / period) if ok else np.nan
            rows.append((cdate, int(node_indices[k]), etr, bool(ok)))

    if not rows:
        return pd.DataFrame(columns=["date", "node_idx", "etr_mm_day", "quality_ok"])
    return pd.DataFrame(rows, columns=["date", "node_idx", "etr_mm_day", "quality_ok"])


def fetch_modis_snow_daily(
    bbox: tuple[float, float, float, float],
    date_start: str,
    date_end: str,
    node_coords: np.ndarray,
    node_indices: np.ndarray,
    cache_dir: str | Path = f"{_paths.DATA_ROOT}/modis10",
    token_path: str = "~/.edl_token",
) -> pd.DataFrame:
    """Fetch MOD10A1.061 NDSI Snow Cover QUOTIDIEN via NASA Earthdata (même chemin
    que fetch_modis_et_8day : token bearer + pyhdf sinusoïdal). Retourne
    DataFrame(date, node_idx, snow_frac, quality_ok). snow_frac = NDSI_Snow_Cover/100
    (0-1) ; raw > 100 (nuage 250, nuit 211, eau 237/239, fill 255...) -> NaN/not ok."""
    import earthaccess
    import requests
    from pyhdf.SD import SD, SDC
    from pyproj import Transformer

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    token = open(Path(token_path).expanduser()).read().strip()
    headers = {"Authorization": f"Bearer {token}"}

    lons = node_coords[:, 0].astype(float)
    lats = node_coords[:, 1].astype(float)
    node_indices = np.asarray(node_indices)
    tr = Transformer.from_crs("EPSG:4326", _SINU, always_xy=True)
    nx_sinu, ny_sinu = tr.transform(lons, lats)

    granules = earthaccess.search_data(
        short_name="MOD10A1", version="61",
        bounding_box=tuple(bbox), temporal=(date_start, date_end),
    )
    logger.info(f"[MOD10A1] {len(granules)} granules quotidiens trouvés")

    rows = []
    for g in granules:
        url = next((l for l in g.data_links() if l.endswith(".hdf")), None)
        if url is None:
            continue
        fn = url.split("/")[-1]
        ayyyyddd = fn.split(".")[1]
        year, doy = int(ayyyyddd[1:5]), int(ayyyyddd[5:8])
        cdate = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=doy - 1)

        local = cache_dir / fn
        if not local.exists():
            r = requests.get(url, headers=headers, allow_redirects=True, timeout=300)
            r.raise_for_status()
            local.write_bytes(r.content)

        sd = SD(str(local), SDC.READ)
        ulx, uly, lrx, lry, gnx, gny = _parse_structmeta(sd)
        psx = (lrx - ulx) / gnx
        psy = (uly - lry) / gny
        snow = sd.select("NDSI_Snow_Cover").get()
        sd.end()

        col = np.floor((nx_sinu - ulx) / psx).astype(int)
        row = np.floor((uly - ny_sinu) / psy).astype(int)
        inside = (col >= 0) & (col < gnx) & (row >= 0) & (row < gny)
        for k in np.nonzero(inside)[0]:
            raw = float(snow[row[k], col[k]])
            ok = raw <= 100.0
            rows.append((cdate, int(node_indices[k]), raw / 100.0 if ok else np.nan, bool(ok)))

    if not rows:
        return pd.DataFrame(columns=["date", "node_idx", "snow_frac", "quality_ok"])
    return pd.DataFrame(rows, columns=["date", "node_idx", "snow_frac", "quality_ok"])


# ─── MOD16A2 ETR via earthaccess ─────────────────────────────────────────────

# ─── MOD10A1 snow (Planetary Computer) ───────────────────────────────────────

# ─── MOD13A1 NDVI (Planetary Computer) ───────────────────────────────────────
