"""Download geospatial rasters from public sources.

Planetary Computer (STAC):
    - Copernicus DEM 30m (``cop-dem-glo-30``)
    - ESA WorldCover 10m (``esa-worldcover``, 2021)
    - JRC Global Surface Water
    - MODIS LAI (MOD15A2H)
    - NRCan annual land cover (Canada)

SoilGrids (ISRIC REST API):
    - Sand / silt / clay content (%)
    - Bulk density (depth-to-bedrock proxy)

OpenStreetMap Overpass API:
    - River LineStrings (``waterway=river``) — cosmetic, used for the
      reach parquet viz layer; topology comes from DEM/D8 in basin_builder.

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
        try:
            _src_ctx = rasterio.open(href)
        except Exception:
            return None
        with _src_ctx as src:
            # Reproject bbox to the raster's native CRS
            if src.crs:
                try:
                    from rasterio.warp import transform_bounds
                    w, s, e, n = transform_bounds("EPSG:4326", src.crs, *bbox)
                except Exception:
                    w, s, e, n = bbox
            else:
                w, s, e, n = bbox

            # Clip to raster extent — avoids out-of-bounds windows entirely
            rb = src.bounds
            w = max(w, rb.left)
            s = max(s, rb.bottom)
            e = min(e, rb.right)
            n = min(n, rb.top)
            if w >= e or s >= n:
                return None   # bbox does not overlap this tile

            window = from_bounds(w, s, e, n, transform=src.transform)
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

    if not arrays:
        raise RuntimeError(f"_save_windowed: no valid arrays to write to {out_path}")

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
        try:
            mosaic, out_transform = merge(datasets)
        finally:
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
            try:
                os.unlink(f)
            except OSError:
                pass


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
        result = _windowed_read(href, bbox)
        if result is not None:
            arrays.append(result)
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
        result = _windowed_read(href, bbox)
        if result is not None:
            arrays.append(result)
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


# ── JRC Global Surface Water ─────────────────────────────────────────────────


def download_jrc_surface_water(
    bbox: tuple[float, float, float, float],
    cache_dir: str | Path,
) -> Path | None:
    """Download JRC Global Surface Water occurrence (v1.4, 2021) for *bbox*.

    Tiles are freely distributed via Google Storage (no authentication).
    Returns path to a merged ``water_occurrence.tif`` (0–100 % occurrence),
    or ``None`` if no tiles are available for the bbox.

    JRC tiles are 10°×10°.  Two naming conventions are tried in order
    (datasets have used both in different releases):
    - latitude = north (top) edge, e.g. ``80W_50N`` for the tile 40–50°N
    - latitude = south (bottom) edge, e.g. ``80W_40N`` for the same tile

    Parameters
    ----------
    bbox : (west, south, east, north) in EPSG:4326.
    cache_dir : Directory for cached GeoTIFF.
    """
    import math
    import gc
    import urllib.request

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / "water_occurrence.tif"
    if out_path.exists():
        print(f"[open_data] JRC surface water cached: {out_path}")
        return out_path

    west, south, east, north = bbox

    base_url = (
        "https://storage.googleapis.com/global-surface-water"
        "/downloads2021/occurrence/occurrence_{lon}{ew}_{lat}{ns}v1_4_2021.tif"
    )

    lon_starts = range(int(math.floor(west / 10)) * 10,
                       int(math.floor(east  / 10)) * 10 + 10, 10)
    lat_starts = range(int(math.floor(south / 10)) * 10,
                       int(math.floor(north / 10)) * 10 + 10, 10)

    def _try_download(lon0: int, lat_val: int) -> Path | None:
        """Try to download one tile; return local path or None on failure."""
        ew  = "W" if lon0 < 0 else "E"
        ns  = "S" if lat_val < 0 else "N"
        name = f"{abs(lon0)}{ew}_{abs(lat_val)}{ns}"
        dst  = cache_dir / f"jrc_{name}.tif"
        if dst.exists():
            return dst
        url = base_url.format(lon=abs(lon0), ew=ew, lat=abs(lat_val), ns=ns)
        print(f"[open_data] Downloading JRC tile {name}...")
        try:
            urllib.request.urlretrieve(url, str(dst))
            return dst
        except Exception as e:
            print(f"[open_data]   skipped ({e})")
            dst.unlink(missing_ok=True)
            return None

    tmp_paths = []
    for lon0 in lon_starts:
        for lat0 in lat_starts:
            # JRC tiles are named by their NORTH (top) edge = lat0 + 10
            p = _try_download(lon0, lat0 + 10)
            if p is not None:
                tmp_paths.append(p)

    if not tmp_paths:
        print("[open_data] [!] No JRC tiles found — water_occurrence will be unavailable.")
        return None

    arrays = []
    import rasterio
    for p in tmp_paths:
        result = _windowed_read(str(p), bbox)
        if result is not None:
            arrays.append(result)
        gc.collect()

    _save_windowed(out_path, arrays, dtype="uint8")
    print(f"[open_data] JRC surface water saved: {out_path}")
    return out_path


# ── MODIS LAI (MOD15A2H, 500 m, 8-day) ───────────────────────────────────────


def download_modis_lai(
    bbox: tuple[float, float, float, float],
    cache_dir: str | Path,
    year: int = 2020,
) -> Path:
    """Download MODIS LAI (MOD15A2H) annual mean for *bbox* via Planetary Computer.

    Returns path to ``lai_mean.tif`` with annual mean LAI (scale factor 0.1 applied).
    """
    _check_geo_deps()
    import gc
    import planetary_computer
    from pystac_client import Client
    import numpy as np

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / "lai_mean.tif"
    if out_path.exists():
        print(f"[open_data] MODIS LAI cached: {out_path}")
        return out_path

    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(
        collections=["modis-15A2H-061"],
        bbox=bbox,
        datetime=f"{year}-01-01/{year}-12-31",
    )
    items = list(search.items())
    if not items:
        raise RuntimeError(f"No MODIS LAI items found for bbox={bbox}, year={year}")

    print(f"[open_data] Downloading MODIS LAI ({len(items)} scenes, computing mean)...")
    stacks = []
    crs = transform = None
    for i, item in enumerate(items):
        if "Lai_500m" not in item.assets:
            continue
        href = item.assets["Lai_500m"].href
        result = _windowed_read(href, bbox)
        if result is None:
            gc.collect()
            continue
        data, transform, crs, nodata = result
        # Scale: 0.1, fill value = 255
        arr = data.astype(np.float32)
        arr[arr == 255] = np.nan
        arr *= 0.1
        stacks.append(arr)
        gc.collect()

    if not stacks:
        raise RuntimeError("No valid MODIS LAI assets found.")

    # Multiple MODIS tiles may produce arrays of different shapes (partial
    # coverage of the bbox).  Keep only the dominant shape so np.stack works.
    from collections import Counter
    shape_counts = Counter(arr.shape for arr in stacks)
    ref_shape = shape_counts.most_common(1)[0][0]
    stacks = [arr for arr in stacks if arr.shape == ref_shape]

    mean_lai = np.nanmean(np.stack(stacks, axis=0), axis=0)
    mean_lai = np.nan_to_num(mean_lai, nan=0.0).astype(np.float32)

    import rasterio
    with rasterio.open(
        str(out_path), "w", driver="GTiff",
        height=mean_lai.shape[0], width=mean_lai.shape[1],
        count=1, dtype="float32", crs=crs, transform=transform,
    ) as dst:
        dst.write(mean_lai, 1)

    print(f"[open_data] MODIS LAI saved: {out_path}")
    return out_path


# ── NRCan Annual Land Cover (Canada) ─────────────────────────────────────────


def download_nrcan_landcover(
    bbox: tuple[float, float, float, float],
    cache_dir: str | Path,
    year: int = 2020,
) -> Path:
    """Download NRCan Annual Land Cover for Canada via Planetary Computer.

    Classes relevant for hydrology:
        1  Temperate/sub-polar needleleaf forest (conifères)
        2  Sub-polar taiga needleleaf forest (conifères)
        5  Temperate/sub-polar broadleaf deciduous forest (feuillus)
        6  Mixed forest
        8  Temperate/sub-polar shrubland
       14  Wetland (tourbières + milieux humides)
       15  Cropland
       17  Urban
       18  Water

    Returns path to ``nrcan_lc.tif``.
    """
    _check_geo_deps()
    import gc
    import planetary_computer
    from pystac_client import Client

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / "nrcan_lc.tif"
    if out_path.exists():
        print(f"[open_data] NRCan land cover cached: {out_path}")
        return out_path

    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    try:
        search = catalog.search(
            collections=["nrcan-landcover"],
            bbox=bbox,
            datetime=f"{year}-01-01/{year}-12-31",
        )
        items = list(search.items())
    except Exception:
        items = []

    if not items:
        # Fallback: try without year filter (collection may have fixed items)
        try:
            search = catalog.search(collections=["nrcan-landcover"], bbox=bbox)
            items = list(search.items())
        except Exception:
            items = []

    if not items:
        print("[open_data] NRCan land cover not found on Planetary Computer — skipping.")
        return None

    print(f"[open_data] Downloading NRCan land cover ({len(items)} tile(s))...")
    arrays = []
    for i, item in enumerate(items):
        # Prefer "landcover" asset; fall back to first non-metadata asset
        asset_key = next(
            (k for k in item.assets if k not in ("metadata", "tilejson", "rendered_preview")),
            next(iter(item.assets)),
        )
        href = item.assets[asset_key].href
        print(f"[open_data]   tile {i+1}/{len(items)}: {item.id}")
        result = _windowed_read(href, bbox)
        if result is not None:
            arrays.append(result)
        gc.collect()

    _save_windowed(out_path, arrays, dtype="uint8")
    print(f"[open_data] NRCan land cover saved: {out_path}")
    return out_path


# ── River-network lines via OpenStreetMap Overpass API ──────────────────────


def download_river_lines(
    bbox: tuple[float, float, float, float],
    cache_dir: str | Path,
) -> Path | None:
    """Download river LineStrings clipped to *bbox* from OpenStreetMap Overpass.

    Returns path to ``rivers.parquet`` (LineString geometries, EPSG:4326),
    or ``None`` if the Overpass mirrors are unreachable.

    Only ``waterway=river`` is queried (streams/creeks would cause 504
    timeouts on larger bboxes). The geometries are cosmetic for the
    `reach_parquet` viz layer — the model topology comes from the DEM/D8
    flow routing in `basin_builder.py`, not from these lines.

    Requires **geopandas** (``pip install meandre[geo]``).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / "rivers.parquet"
    if out_path.exists():
        print(f"[open_data] Rivers cached: {out_path}")
        return out_path

    try:
        import geopandas as gpd
        from shapely.geometry import box as shapely_box
    except ImportError:
        print("[open_data] geopandas not installed — skipping rivers download. "
              "Install with: pip install geopandas")
        return None

    import urllib.request
    import urllib.parse
    import json

    west, south, east, north = bbox
    aoi = shapely_box(west, south, east, north)

    print("[open_data] Querying OSM Overpass for river lines...")
    query = (
        f"[out:json][timeout:120];"
        f'(way["waterway"="river"]({south},{west},{north},{east}););'
        f"out geom;"
    )
    data = urllib.parse.urlencode({"data": query}).encode()

    response = None
    for mirror in [
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass-api.de/api/interpreter",
    ]:
        try:
            with urllib.request.urlopen(mirror, data=data, timeout=180) as r:
                response = r.read()
            break
        except Exception as e:
            print(f"[open_data]   {mirror} failed ({e})")
            continue

    if response is None:
        print("[open_data] All Overpass mirrors failed — skipping rivers.")
        return None

    feats = []
    for el in json.loads(response).get("elements", []):
        if el.get("type") != "way" or "geometry" not in el:
            continue
        coords = [(n["lon"], n["lat"]) for n in el["geometry"]]
        if len(coords) < 2:
            continue
        tags = el.get("tags", {})
        feats.append({
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {
                "name":     tags.get("name", ""),
                "waterway": tags.get("waterway", ""),
            },
        })

    if not feats:
        print("[open_data] OSM: no river lines in bbox.")
        return None

    gdf = gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326")
    gdf = gdf[gdf.geometry.notna()]
    gdf = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])]
    gdf = gdf.clip(aoi)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]

    if len(gdf) == 0:
        print("[open_data] OSM: no river lines in bbox after clip.")
        return None

    gdf = gdf.reset_index(drop=True)
    gdf.to_parquet(str(out_path))
    print(f"[open_data] Rivers saved: {out_path} ({len(gdf)} segments)")
    return out_path


# ── Reach geometry builder ────────────────────────────────────────────────────


def build_reach_parquet(
    nodes_df: "pd.DataFrame",
    rivers_path: str | Path,
    out_path: str | Path,
    max_dist_deg: float = 0.05,
) -> Path:
    """Match model nodes to OSM river segments and save as parquet.

    For each node ``(lon, lat)`` in *nodes_df*, finds the nearest river
    segment within *max_dist_deg* degrees (~5 km at Quebec latitudes).
    Saves a GeoDataFrame with columns::

        node_idx  node_id  dist_deg  geometry (the matched river LineString)

    Unmatched nodes are omitted.

    Parameters
    ----------
    nodes_df :
        DataFrame with columns ``node_idx``, ``node_id``, ``lon``, ``lat``.
    rivers_path :
        Path to ``rivers.parquet`` produced by :func:`download_river_lines`.
    out_path :
        Output parquet path, e.g. ``data/reaches.parquet``.
    max_dist_deg :
        Matching distance threshold in degrees.  0.05° ≈ 5.5 km.

    Returns
    -------
    Path to the saved parquet file.
    """
    import geopandas as gpd
    import pandas as pd

    rivers = gpd.read_parquet(str(rivers_path))
    if rivers.crs is None:
        rivers = rivers.set_crs("EPSG:4326")
    elif rivers.crs.to_epsg() != 4326:
        rivers = rivers.to_crs("EPSG:4326")

    # Keep only geometry (and a river index) for the join
    rivers_geom = rivers[["geometry"]].copy().reset_index(drop=True)
    rivers_geom["_riv_idx"] = rivers_geom.index

    # Build node GeoDataFrame
    nodes_gdf = gpd.GeoDataFrame(
        nodes_df[["node_idx", "node_id"]].copy(),
        geometry=gpd.points_from_xy(nodes_df["lon"], nodes_df["lat"]),
        crs="EPSG:4326",
    )

    # Nearest-neighbour spatial join
    joined = gpd.sjoin_nearest(
        nodes_gdf,
        rivers_geom,
        how="left",
        max_distance=max_dist_deg,
        distance_col="dist_deg",
    )

    matched = joined.dropna(subset=["index_right"]).copy()
    if len(matched) == 0:
        raise RuntimeError(
            f"No nodes matched within {max_dist_deg}° — "
            "check that the rivers parquet covers the model domain."
        )

    # Replace point geometries with the matched river LineStrings
    river_geoms = rivers_geom.geometry.iloc[
        matched["index_right"].astype(int).values
    ].values

    out_gdf = gpd.GeoDataFrame(
        matched[["node_idx", "node_id", "dist_deg"]].reset_index(drop=True),
        geometry=river_geoms,
        crs="EPSG:4326",
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_gdf.to_parquet(str(out_path))

    n_matched = len(out_gdf)
    n_total = len(nodes_df)
    print(
        f"[open_data] Reach parquet: {n_matched}/{n_total} nodes matched "
        f"(max_dist={max_dist_deg}deg) -> {out_path}"
    )
    return out_path


# ── Water polygons via OpenStreetMap Overpass API ───────────────────────────


def download_water_polygons(
    bbox: tuple[float, float, float, float],
    cache_dir: str | Path,
) -> Path | None:
    """Download water polygons (lakes, reservoirs, basins) from OSM Overpass.

    Captures small lakes and reservoirs that JRC Global Surface Water (30 m)
    may miss. Combined with JRC at the basin_builder level for fine lake
    detection.

    Tags queried:
        - ``natural=water`` (lakes, ponds)
        - ``landuse=reservoir``
        - ``landuse=basin`` (retention basins)

    Returns path to ``water_polygons.parquet`` (Polygon geometries, EPSG:4326),
    or ``None`` if Overpass mirrors fail.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / "water_polygons.parquet"
    if out_path.exists():
        print(f"[open_data] Water polygons cached: {out_path}")
        return out_path

    try:
        import geopandas as gpd
        from shapely.geometry import box as shapely_box
    except ImportError:
        print("[open_data] geopandas not installed — skipping water polygons.")
        return None

    import json
    import urllib.parse
    import urllib.request

    west, south, east, north = bbox
    aoi = shapely_box(west, south, east, north)

    print("[open_data] Querying OSM Overpass for water polygons...")
    query = (
        f"[out:json][timeout:120];"
        f"("
        f'way["natural"="water"]({south},{west},{north},{east});'
        f'way["landuse"="reservoir"]({south},{west},{north},{east});'
        f'way["landuse"="basin"]({south},{west},{north},{east});'
        f");"
        f"out geom;"
    )
    data = urllib.parse.urlencode({"data": query}).encode()

    response = None
    for mirror in [
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass-api.de/api/interpreter",
    ]:
        try:
            with urllib.request.urlopen(mirror, data=data, timeout=180) as r:
                response = r.read()
            break
        except Exception as e:
            print(f"[open_data]   {mirror} failed ({e})")
            continue

    if response is None:
        print("[open_data] All Overpass mirrors failed — skipping water polygons.")
        return None

    feats = []
    for el in json.loads(response).get("elements", []):
        if el.get("type") != "way" or "geometry" not in el:
            continue
        coords = [(n["lon"], n["lat"]) for n in el["geometry"]]
        # Polygon needs at least 4 points and must be closed
        if len(coords) < 4 or coords[0] != coords[-1]:
            continue
        tags = el.get("tags", {})
        feats.append({
            "geometry": {"type": "Polygon", "coordinates": [coords]},
            "properties": {
                "name": tags.get("name", ""),
                "type": tags.get("natural") or tags.get("landuse", ""),
            },
        })

    if not feats:
        print("[open_data] OSM: no water polygons in bbox.")
        return None

    gdf = gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326")
    gdf = gdf[gdf.geometry.notna() & gdf.geometry.is_valid]
    gdf = gdf.clip(aoi)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]

    if len(gdf) == 0:
        print("[open_data] OSM: no water polygons after clip.")
        return None

    gdf = gdf.reset_index(drop=True)
    gdf.to_parquet(str(out_path))
    print(f"[open_data] Water polygons saved: {out_path} ({len(gdf)} polygons)")
    return out_path


# ── Daily forcing via Open-Meteo ERA5 archive ───────────────────────────────


def download_forcing_open_meteo(
    bbox: tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    cache_dir: str | Path,
    resolution_deg: float = 0.1,
    batch_size: int = 200,
) -> Path | None:
    """Download daily ``pr``/``tasmin``/``tasmax`` from Open-Meteo ERA5 archive.

    Builds a regular lat/lon grid over *bbox* at *resolution_deg* spacing and
    queries the Open-Meteo bulk endpoint (multi-location). Output is a
    netCDF compatible with :func:`meandre.data.gridded_forcing.extract_forcing`
    (which derives ``R_n``, ``u2``, ``e_a`` internally via FAO-56).

    Parameters
    ----------
    bbox :
        ``(west, south, east, north)`` in EPSG:4326.
    start_date, end_date :
        ISO 8601 dates (``YYYY-MM-DD``), inclusive.
    cache_dir :
        Output directory (``forcing_open_meteo.nc`` is cached here).
    resolution_deg :
        Grid spacing in degrees. 0.1 ≈ 11 km. Lower = more queries but more
        spatial detail. ERA5 native is 0.25° (~28 km) so going below 0.1°
        re-samples without adding information.
    batch_size :
        Locations per HTTP request. Open-Meteo accepts ~200 safely; raise
        for fewer requests but longer per-request payloads.

    Returns
    -------
    Path to ``forcing_open_meteo.nc`` or ``None`` if all queries fail.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / "forcing_open_meteo.nc"
    if out_path.exists():
        print(f"[open_data] Forcing cached: {out_path}")
        return out_path

    import json
    import urllib.parse
    import urllib.request

    import pandas as pd
    import xarray as xr

    west, south, east, north = bbox

    # Snap grid to multiples of resolution_deg so repeated calls reuse cells
    lat0 = np.floor(south / resolution_deg) * resolution_deg
    lat1 = np.ceil(north / resolution_deg) * resolution_deg
    lon0 = np.floor(west / resolution_deg) * resolution_deg
    lon1 = np.ceil(east / resolution_deg) * resolution_deg
    lats_grid = np.round(np.arange(lat0, lat1 + 1e-6, resolution_deg), 6)
    lons_grid = np.round(np.arange(lon0, lon1 + 1e-6, resolution_deg), 6)
    n_lat, n_lon = len(lats_grid), len(lons_grid)

    dates = pd.date_range(start_date, end_date, freq="D")
    n_days = len(dates)

    # Pre-allocate (may be ~100s of MB for full SLSO over 25 years)
    pr       = np.full((n_days, n_lat, n_lon), np.nan, dtype=np.float32)
    tasmin   = np.full((n_days, n_lat, n_lon), np.nan, dtype=np.float32)
    tasmax   = np.full((n_days, n_lat, n_lon), np.nan, dtype=np.float32)
    sfcWind  = np.full((n_days, n_lat, n_lon), np.nan, dtype=np.float32)

    flat_pts = [(i, j, lats_grid[i], lons_grid[j])
                for i in range(n_lat) for j in range(n_lon)]
    n_pts = len(flat_pts)
    n_batches = (n_pts + batch_size - 1) // batch_size

    print(f"[open_data] Open-Meteo grid: {n_lat}×{n_lon} = {n_pts} points  "
          f"× {n_days} days  ({n_batches} batches)")

    URL = "https://archive-api.open-meteo.com/v1/era5"
    DAILY = (
        "precipitation_sum,temperature_2m_min,temperature_2m_max,"
        "wind_speed_10m_max"
    )

    failed = 0
    for b in range(n_batches):
        batch = flat_pts[b * batch_size:(b + 1) * batch_size]
        lats_q = ",".join(f"{p[2]:.4f}" for p in batch)
        lons_q = ",".join(f"{p[3]:.4f}" for p in batch)
        params = {
            "latitude":        lats_q,
            "longitude":       lons_q,
            "start_date":      start_date,
            "end_date":        end_date,
            "daily":           DAILY,
            "timezone":        "GMT",
            "wind_speed_unit": "ms",   # m/s instead of default km/h
        }
        full_url = URL + "?" + urllib.parse.urlencode(params)

        try:
            with urllib.request.urlopen(full_url, timeout=300) as r:
                data = json.loads(r.read())
        except Exception as e:
            print(f"[open_data]   batch {b+1}/{n_batches} failed ({e})")
            failed += len(batch)
            continue

        # Single location -> dict, multiple -> list of dicts
        if not isinstance(data, list):
            data = [data]

        for loc, (i, j, _, _) in zip(data, batch):
            d = loc.get("daily") or {}
            try:
                pr[:, i, j]      = d["precipitation_sum"]
                tasmin[:, i, j]  = d["temperature_2m_min"]
                tasmax[:, i, j]  = d["temperature_2m_max"]
                sfcWind[:, i, j] = d["wind_speed_10m_max"]
            except (KeyError, ValueError) as e:
                print(f"[open_data]   pt ({lats_grid[i]}, {lons_grid[j]}): {e}")
                failed += 1

        if (b + 1) % 5 == 0 or b + 1 == n_batches:
            print(f"  batch {b+1}/{n_batches} done", flush=True)

    if failed > 0:
        print(f"[open_data] {failed}/{n_pts} grid points failed (NaN-filled)")

    if np.all(np.isnan(pr)):
        print("[open_data] All Open-Meteo queries failed.")
        return None

    ds = xr.Dataset(
        {
            "pr":      (("time", "latitude", "longitude"), pr),
            "tasmin":  (("time", "latitude", "longitude"), tasmin),
            "tasmax":  (("time", "latitude", "longitude"), tasmax),
            "sfcWind": (("time", "latitude", "longitude"), sfcWind),
        },
        coords={
            "time":      dates,
            "latitude":  lats_grid,
            "longitude": lons_grid,
        },
        attrs={
            "source":         "Open-Meteo ERA5 archive",
            "url":            URL,
            "resolution_deg": resolution_deg,
            "bbox":           f"{west},{south},{east},{north}",
        },
    )
    ds["pr"].attrs.update(units="mm/day", standard_name="precipitation_amount")
    ds["tasmin"].attrs.update(units="degC", standard_name="air_temperature")
    ds["tasmax"].attrs.update(units="degC", standard_name="air_temperature")
    ds["sfcWind"].attrs.update(
        units="m/s",
        standard_name="wind_speed",
        long_name="Daily max wind speed at 10 m (FAO-56 conversion to u2 done at extract)",
    )
    ds.to_netcdf(out_path)
    print(f"[open_data] Forcing saved: {out_path}  ({n_days} days × {n_lat}×{n_lon})")
    return out_path


# ── Convenience: download all ────────────────────────────────────────────────


def download_all(
    bbox: tuple[float, float, float, float],
    cache_dir: str | Path,
) -> dict[str, Path]:
    """Download DEM, land cover, soil, JRC water, MODIS LAI, NRCan LC, OSM rivers.

    Returns dict with keys:
        ``"dem"``, ``"landcover"``, ``"soil_dir"``,
        ``"water_occurrence"``, ``"lai"``, ``"nrcan_lc"``, ``"rivers"``
        (values are ``None`` when a source is unavailable).
    """
    cache_dir = Path(cache_dir)
    return {
        "dem":              download_dem(bbox, cache_dir),
        "landcover":        download_landcover(bbox, cache_dir),
        "soil_dir":         download_soil(bbox, cache_dir),
        "water_occurrence": download_jrc_surface_water(bbox, cache_dir),
        "lai":              download_modis_lai(bbox, cache_dir),
        "nrcan_lc":         download_nrcan_landcover(bbox, cache_dir),
        "rivers":           download_river_lines(bbox, cache_dir),
        "water_polygons":   download_water_polygons(bbox, cache_dir),
    }
