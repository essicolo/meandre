"""Download geospatial rasters from public sources.

Planetary Computer (STAC):
    - Copernicus DEM 30m (``cop-dem-glo-30``)
    - ESA WorldCover 10m (``esa-worldcover``, 2021)

SoilGrids (ISRIC REST API):
    - Sand / silt / clay content (%)
    - Depth to bedrock (cm)

All downloads are cached in ``cache_dir`` so repeated calls are free.

Requires optional dependencies::

    pip install meandre[geo]
    # or: pip install pystac-client planetary-computer rioxarray
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _check_geo_deps() -> None:
    """Raise ImportError with helpful message if geo deps are missing."""
    missing = []
    for pkg in ("pystac_client", "planetary_computer", "rioxarray"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg.replace("_", "-"))
    if missing:
        raise ImportError(
            f"Missing packages: {', '.join(missing)}. "
            "Install with: pip install meandre[geo]"
        )


# ── Copernicus DEM 30m ──────────────────────────────────────────────────────


def _windowed_read(href: str, bbox: tuple[float, float, float, float]) -> tuple:
    """Read only the *bbox* window from a Cloud-Optimized GeoTIFF.

    Returns (data_array, transform, crs, nodata) — memory-efficient for large
    remote tiles.  Uses conservative GDAL cache settings to avoid OOM on
    memory-constrained environments (e.g. Google Colab).
    """
    import rasterio
    from rasterio.windows import from_bounds

    env_opts = {
        "GDAL_CACHEMAX": 64,           # MB — limit GDAL block cache
        "CPL_VSIL_CURL_CACHE_SIZE": 16_000_000,  # 16 MB curl cache
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "VSI_CACHE": "TRUE",
        "VSI_CACHE_SIZE": 16_000_000,
    }
    with rasterio.Env(**env_opts):
        with rasterio.open(href) as src:
            window = from_bounds(*bbox, transform=src.transform)
            # Clamp window to valid raster extent
            window = window.intersection(rasterio.windows.Window(
                0, 0, src.width, src.height,
            ))
            data = src.read(1, window=window)
            transform = src.window_transform(window)
            return data, transform, src.crs, src.nodata


def _save_windowed(
    out_path: Path,
    arrays: list[tuple],
    dtype: str = "float32",
) -> None:
    """Merge windowed reads and save to GeoTIFF."""
    import rasterio
    from rasterio.merge import merge
    from rasterio.transform import array_bounds
    import tempfile, os

    if len(arrays) == 1:
        data, transform, crs, nodata = arrays[0]
        with rasterio.open(
            str(out_path), "w", driver="GTiff",
            height=data.shape[0], width=data.shape[1],
            count=1, dtype=dtype, crs=crs, transform=transform,
            nodata=nodata,
        ) as dst:
            dst.write(data.astype(dtype), 1)
        return

    # Multiple tiles: write temp files, merge
    tmp_files = []
    try:
        for data, transform, crs, nodata in arrays:
            tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
            tmp_files.append(tmp.name)
            with rasterio.open(
                tmp.name, "w", driver="GTiff",
                height=data.shape[0], width=data.shape[1],
                count=1, dtype=dtype, crs=crs, transform=transform,
                nodata=nodata,
            ) as dst:
                dst.write(data.astype(dtype), 1)

        datasets = [rasterio.open(f) for f in tmp_files]
        mosaic, out_transform = merge(datasets)
        for ds in datasets:
            ds.close()

        with rasterio.open(
            str(out_path), "w", driver="GTiff",
            height=mosaic.shape[1], width=mosaic.shape[2],
            count=1, dtype=dtype, crs=crs, transform=out_transform,
            nodata=nodata,
        ) as dst:
            dst.write(mosaic[0].astype(dtype), 1)
    finally:
        for f in tmp_files:
            os.unlink(f)


def download_dem(
    bbox: tuple[float, float, float, float],
    cache_dir: str | Path,
    collection: str = "cop-dem-glo-30",
) -> Path:
    """Download Copernicus 30m DEM tiles, mosaic, and clip to *bbox*.

    Parameters
    ----------
    bbox : (west, south, east, north) in EPSG:4326 degrees.
    cache_dir : Directory for cached GeoTIFF.
    collection : STAC collection name (default ``cop-dem-glo-30``).

    Returns
    -------
    Path to the clipped DEM GeoTIFF (``{cache_dir}/dem.tif``).
    """
    _check_geo_deps()
    import planetary_computer
    from pystac_client import Client

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / "dem.tif"
    if out_path.exists():
        print(f"[open_data] DEM cached: {out_path}")
        return out_path

    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(collections=[collection], bbox=bbox)
    items = list(search.items())
    if not items:
        raise RuntimeError(f"No DEM tiles found for bbox={bbox}")

    import gc

    print(f"[open_data] Downloading {len(items)} DEM tile(s)...")
    arrays = []
    for i, item in enumerate(items):
        href = item.assets["data"].href
        print(f"[open_data]   tile {i + 1}/{len(items)}: {item.id}")
        arrays.append(_windowed_read(href, bbox))
        gc.collect()

    _save_windowed(out_path, arrays, dtype="float32")
    print(f"[open_data] DEM saved: {out_path} ({arrays[0][0].shape})")
    return out_path


# ── ESA WorldCover 10m ──────────────────────────────────────────────────────


def download_landcover(
    bbox: tuple[float, float, float, float],
    cache_dir: str | Path,
    year: int = 2021,
) -> Path:
    """Download ESA WorldCover 10m land cover, clip to *bbox*.

    Parameters
    ----------
    bbox : (west, south, east, north) in EPSG:4326.
    cache_dir : Directory for cached GeoTIFF.
    year : WorldCover version year (2020 or 2021).

    Returns
    -------
    Path to the clipped land cover GeoTIFF.

    ESA WorldCover classes::

        10  Tree cover
        20  Shrubland
        30  Grassland
        40  Cropland
        50  Built-up
        60  Bare / sparse vegetation
        70  Snow and ice
        80  Permanent water bodies
        90  Herbaceous wetland
        95  Mangroves
        100 Moss and lichen
    """
    _check_geo_deps()
    import planetary_computer
    from pystac_client import Client

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / "landcover.tif"
    if out_path.exists():
        print(f"[open_data] Land cover cached: {out_path}")
        return out_path

    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    # Search without version filter — Planetary Computer may not support
    # the esa_worldcover:product_version query field consistently.
    search = catalog.search(
        collections=["esa-worldcover"],
        bbox=bbox,
    )
    items = list(search.items())

    # Filter by year if multiple versions found
    if year and len(items) > 1:
        version_str = str(year)
        filtered = [it for it in items if version_str in (it.id or "")]
        if filtered:
            items = filtered

    if not items:
        raise RuntimeError(f"No WorldCover tiles found for bbox={bbox}")

    import gc

    print(f"[open_data] Downloading {len(items)} WorldCover tile(s)...")
    arrays = []
    for i, item in enumerate(items):
        href = item.assets["map"].href
        print(f"[open_data]   tile {i + 1}/{len(items)}: {item.id}")
        arrays.append(_windowed_read(href, bbox))
        gc.collect()

    _save_windowed(out_path, arrays, dtype="uint8")
    print(f"[open_data] Land cover saved: {out_path} ({arrays[0][0].shape})")
    return out_path


# ── SoilGrids 250m (ISRIC) ─────────────────────────────────────────────────


def download_soil(
    bbox: tuple[float, float, float, float],
    cache_dir: str | Path,
    depth: str = "0-5cm",
) -> Path:
    """Download SoilGrids sand/silt/clay content and depth to bedrock.

    Uses the ISRIC WCS (Web Coverage Service) endpoint.

    Parameters
    ----------
    bbox : (west, south, east, north) in EPSG:4326.
    cache_dir : Directory for cached GeoTIFFs.
    depth : SoilGrids depth layer (default ``"0-5cm"``).

    Returns
    -------
    Path to cache_dir (contains ``sand.tif``, ``silt.tif``, ``clay.tif``,
    ``bdod.tif``).
    """
    import urllib.request

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # SoilGrids WCS base URL
    wcs_base = "https://maps.isric.org/mapserv?map=/map/{layer}.map"

    layers = {
        "sand": f"sand_{depth}_mean",
        "silt": f"silt_{depth}_mean",
        "clay": f"clay_{depth}_mean",
        "bdod": f"bdod_{depth}_mean",  # bulk density (proxy for depth info)
    }

    west, south, east, north = bbox
    # Homolosine projection bbox (approximate, WCS handles reprojection)
    for name, layer_id in layers.items():
        out_path = cache_dir / f"{name}.tif"
        if out_path.exists():
            print(f"[open_data] SoilGrids {name} cached: {out_path}")
            continue

        # Use WCS GetCoverage request
        url = (
            f"https://maps.isric.org/mapserv?map=/map/{name}.map"
            f"&SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage"
            f"&COVERAGEID={layer_id}"
            f"&FORMAT=image/tiff"
            f"&SUBSET=long({west},{east})"
            f"&SUBSET=lat({south},{north})"
            f"&SUBSETTINGCRS=http://www.opengis.net/def/crs/EPSG/0/4326"
        )
        print(f"[open_data] Downloading SoilGrids {name}...")
        urllib.request.urlretrieve(url, str(out_path))
        print(f"[open_data] Saved: {out_path}")

    return cache_dir


# ── Convenience: download all ────────────────────────────────────────────────


def download_all(
    bbox: tuple[float, float, float, float],
    cache_dir: str | Path,
) -> dict[str, Path]:
    """Download DEM, land cover, and soil data for a bounding box.

    Returns dict with keys ``"dem"``, ``"landcover"``, ``"soil_dir"``.
    """
    cache_dir = Path(cache_dir)
    return {
        "dem": download_dem(bbox, cache_dir),
        "landcover": download_landcover(bbox, cache_dir),
        "soil_dir": download_soil(bbox, cache_dir),
    }
