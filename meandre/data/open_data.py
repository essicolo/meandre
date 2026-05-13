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
        # Validate: only return cached if reasonably complete (NaN < 1%).
        # An incomplete download (e.g. abandoned after rate-limit) leaves a
        # mostly-NaN file behind; we want the resumable cache to take over.
        try:
            import xarray as _xr
            _ds = _xr.open_dataset(out_path)
            nan_frac = float(_ds.pr.isnull().sum() / _ds.pr.size)
            _ds.close()
        except Exception:
            nan_frac = 1.0
        if nan_frac < 0.01:
            print(f"[open_data] Forcing cached: {out_path}")
            return out_path
        print(f"[open_data] Cached forcing has {nan_frac*100:.1f}% NaN — "
              f"resuming from chunk cache to fill gaps")
        out_path.unlink()  # remove so we rebuild after chunk merge

    import json
    import time
    import urllib.error
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

    # Open-Meteo limits the response payload — split temporally if the request
    # would exceed ~200k daily values (4 vars × N days × M points).
    # Strategy: chunk the period into 1-year windows, loop over location batches.
    MAX_VALUES_PER_REQUEST = 200_000
    N_VARS = 4
    days_per_chunk = max(1, MAX_VALUES_PER_REQUEST // (N_VARS * batch_size))
    if days_per_chunk >= n_days:
        # Single time chunk covering the whole period
        chunks = [(dates[0], dates[-1])]
    else:
        # Year-by-year chunking is the simplest stable strategy
        chunks = []
        chunk_start = dates[0]
        while chunk_start <= dates[-1]:
            chunk_end = min(
                chunk_start + pd.DateOffset(years=1) - pd.Timedelta(days=1),
                dates[-1],
            )
            chunks.append((chunk_start, chunk_end))
            chunk_start = chunk_end + pd.Timedelta(days=1)

    n_batches = (n_pts + batch_size - 1) // batch_size
    n_chunks = len(chunks)
    print(f"[open_data] Open-Meteo grid: {n_lat}x{n_lon} = {n_pts} points  "
          f"x {n_days} days  ({n_batches} loc batches x {n_chunks} time chunks)")

    URL = "https://archive-api.open-meteo.com/v1/era5"
    DAILY = (
        "precipitation_sum,temperature_2m_min,temperature_2m_max,"
        "wind_speed_10m_max"
    )

    # Resumable cache: 1 file per chunk (year). Lets a run survive Open-Meteo
    # rate-limit hits — the next run picks up where this one left off.
    chunks_dir = cache_dir / "forcing_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    def _chunk_path(start: pd.Timestamp, end: pd.Timestamp) -> Path:
        return (chunks_dir
                / f"chunk_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.nc")

    # Map each global date to its index in the (T, lat, lon) arrays
    date_idx = {d: i for i, d in enumerate(dates)}
    failed = 0
    total_calls = n_batches * n_chunks
    call_idx = 0
    chunks_loaded_from_cache = 0
    chunks_just_downloaded = 0

    for chunk_start, chunk_end in chunks:
        chunk_file = _chunk_path(chunk_start, chunk_end)
        chunk_dates = pd.date_range(chunk_start, chunk_end, freq="D")
        chunk_t_idx = np.array([date_idx[d] for d in chunk_dates])

        # ── Resume from cached chunk if available ──
        if chunk_file.exists():
            try:
                cds = xr.open_dataset(chunk_file)
                pr[chunk_t_idx, :, :]      = cds["pr"].values
                tasmin[chunk_t_idx, :, :]  = cds["tasmin"].values
                tasmax[chunk_t_idx, :, :]  = cds["tasmax"].values
                sfcWind[chunk_t_idx, :, :] = cds["sfcWind"].values
                cds.close()
                call_idx += n_batches  # advance counter
                chunks_loaded_from_cache += 1
                if chunks_loaded_from_cache % 5 == 0:
                    print(f"  resumed {chunks_loaded_from_cache} chunks from cache",
                          flush=True)
                continue
            except Exception as e:
                print(f"  cache file {chunk_file.name} unreadable ({e}); re-downloading")

        cstart_str = chunk_start.strftime("%Y-%m-%d")
        cend_str = chunk_end.strftime("%Y-%m-%d")
        chunk_failed_any = False

        # Allocate per-chunk arrays so we can persist on success
        chunk_pr      = np.full((len(chunk_dates), n_lat, n_lon), np.nan, dtype=np.float32)
        chunk_tasmin  = np.full_like(chunk_pr, np.nan)
        chunk_tasmax  = np.full_like(chunk_pr, np.nan)
        chunk_sfcWind = np.full_like(chunk_pr, np.nan)

        for b in range(n_batches):
            call_idx += 1
            batch = flat_pts[b * batch_size:(b + 1) * batch_size]
            lats_q = ",".join(f"{p[2]:.4f}" for p in batch)
            lons_q = ",".join(f"{p[3]:.4f}" for p in batch)
            params = {
                "latitude":        lats_q,
                "longitude":       lons_q,
                "start_date":      cstart_str,
                "end_date":        cend_str,
                "daily":           DAILY,
                "timezone":        "GMT",
                "wind_speed_unit": "ms",
            }
            full_url = URL + "?" + urllib.parse.urlencode(params)

            data = None
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(full_url, timeout=300) as r:
                        data = json.loads(r.read())
                    break
                except urllib.error.HTTPError as e:
                    if e.code == 429 and attempt < 3:
                        # Rate limit — exponential backoff
                        wait = 30 * (2 ** attempt)  # 30, 60, 120, 240 s
                        print(f"  call {call_idx}/{total_calls}: HTTP 429, "
                              f"waiting {wait}s (attempt {attempt+1}/4)...",
                              flush=True)
                        time.sleep(wait)
                        continue
                    print(f"[open_data]   call {call_idx}/{total_calls} "
                          f"({cstart_str}..{cend_str}, batch {b+1}/{n_batches}) "
                          f"failed (HTTP {e.code})")
                    break
                except Exception as e:
                    print(f"[open_data]   call {call_idx}/{total_calls} "
                          f"({cstart_str}..{cend_str}, batch {b+1}/{n_batches}) "
                          f"failed ({e})")
                    break

            if data is None:
                failed += len(batch)
                chunk_failed_any = True
                # Pace ourselves to avoid further 429s
                time.sleep(1.0)
                continue

            # Pacing: ~120 calls/min keeps us comfortably under Open-Meteo's
            # 600/min free-tier limit even with parallel users on the same IP.
            time.sleep(0.5)

            if not isinstance(data, list):
                data = [data]

            for loc, (i, j, _, _) in zip(data, batch):
                d = loc.get("daily") or {}
                try:
                    chunk_pr[:, i, j]      = d["precipitation_sum"]
                    chunk_tasmin[:, i, j]  = d["temperature_2m_min"]
                    chunk_tasmax[:, i, j]  = d["temperature_2m_max"]
                    chunk_sfcWind[:, i, j] = d["wind_speed_10m_max"]
                except (KeyError, ValueError) as e:
                    print(f"[open_data]   pt ({lats_grid[i]}, {lons_grid[j]}): {e}")
                    failed += 1
                    chunk_failed_any = True

            if call_idx % 10 == 0 or call_idx == total_calls:
                print(f"  call {call_idx}/{total_calls} done", flush=True)

        # ── Persist this chunk if it succeeded fully ──
        if not chunk_failed_any:
            chunk_ds = xr.Dataset(
                {
                    "pr":      (("time", "latitude", "longitude"), chunk_pr),
                    "tasmin":  (("time", "latitude", "longitude"), chunk_tasmin),
                    "tasmax":  (("time", "latitude", "longitude"), chunk_tasmax),
                    "sfcWind": (("time", "latitude", "longitude"), chunk_sfcWind),
                },
                coords={
                    "time": chunk_dates,
                    "latitude": lats_grid,
                    "longitude": lons_grid,
                },
            )
            chunk_ds.to_netcdf(chunk_file)
            chunks_just_downloaded += 1

        # Copy chunk arrays into the global ones (regardless of full success,
        # we keep what we got — but only chunks fully OK are cached)
        pr[chunk_t_idx, :, :]      = chunk_pr
        tasmin[chunk_t_idx, :, :]  = chunk_tasmin
        tasmax[chunk_t_idx, :, :]  = chunk_tasmax
        sfcWind[chunk_t_idx, :, :] = chunk_sfcWind

    if chunks_loaded_from_cache or chunks_just_downloaded:
        print(f"[open_data] Chunks: {chunks_loaded_from_cache} from cache, "
              f"{chunks_just_downloaded} downloaded this run "
              f"(of {n_chunks} total)", flush=True)

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


# ── Daily forcing via DestinE Earth Data Hub (ERA5-LAND) ────────────────────


def _read_destine_token() -> str | None:
    """Read DestinE personal access token from ~/.netrc (or _netrc on Windows).

    Expected entry::

        machine data.earthdatahub.destine.eu
        password edh_pat_...
    """
    import netrc as _netrc
    try:
        nrc = _netrc.netrc()
        auth = nrc.authenticators("data.earthdatahub.destine.eu")
    except Exception:
        return None
    if auth is None:
        return None
    _, _, password = auth
    return password


def download_forcing_era5_land_destine(
    bbox: tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    cache_dir: str | Path,
) -> Path | None:
    """Download daily forcing from ERA5-LAND via DestinE Earth Data Hub.

    Uses the public zarr at ``data.earthdatahub.destine.eu`` (auth via .netrc).
    The hourly raw data is aggregated to daily server-side-ish: lazy zarr
    chunks pull only the bbox subset, then a per-year chunk file is saved
    locally for resumability.

    Output netCDF schema matches :func:`download_forcing_open_meteo`:
    ``pr`` (mm/day), ``tasmin``/``tasmax`` (°C), ``sfcWind`` (m/s),
    dimensions ``(time, latitude, longitude)``.
    """
    import zarr
    import xarray as xr
    import pandas as pd

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / "forcing_era5_land.nc"
    chunks_dir = cache_dir / "forcing_chunks_destine"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        try:
            ds_chk = xr.open_dataset(out_path)
            nan_frac = float(ds_chk.pr.isnull().sum() / ds_chk.pr.size)
            ds_chk.close()
        except Exception:
            nan_frac = 1.0
        if nan_frac < 0.01:
            print(f"[destine] Forcing cached: {out_path}")
            return out_path
        out_path.unlink()

    token = _read_destine_token()
    if token is None:
        print("[destine] No DestinE token in .netrc — skipping ERA5-LAND")
        return None

    url = (
        f"https://edh:{token}@data.earthdatahub.destine.eu/"
        "era5/reanalysis-era5-land-no-antartica-v0.zarr"
    )

    print(f"[destine] Opening ERA5-LAND zarr...")
    store = zarr.storage.FsspecStore.from_url(url)
    root = zarr.open(store, mode="r")

    lats_full = np.asarray(root["latitude"][:])     # decreasing 90 to -57
    lons_full = np.asarray(root["longitude"][:])    # 0..360
    valid_time_hours = np.asarray(root["valid_time"][:])
    base = pd.Timestamp("1950-01-01")
    dates_all = base + pd.to_timedelta(valid_time_hours, unit="h")

    west, south, east, north = bbox
    lon_lo, lon_hi = west % 360, east % 360
    lat_north_idx = int(np.argmin(np.abs(lats_full - north)))
    lat_south_idx = int(np.argmin(np.abs(lats_full - south)))
    lat_lo = min(lat_north_idx, lat_south_idx)
    lat_hi = max(lat_north_idx, lat_south_idx)
    lon_west_idx = int(np.argmin(np.abs(lons_full - lon_lo)))
    lon_east_idx = int(np.argmin(np.abs(lons_full - lon_hi)))
    sub_lats = lats_full[lat_lo:lat_hi + 1]
    sub_lons_360 = lons_full[lon_west_idx:lon_east_idx + 1]
    # Express longitudes in -180..180 in the output (consistent with the rest)
    sub_lons = np.where(sub_lons_360 > 180, sub_lons_360 - 360, sub_lons_360)
    n_lat = len(sub_lats)
    n_lon = len(sub_lons)
    print(f"[destine] grid: {n_lat}x{n_lon}  bbox=({west},{south},{east},{north})")

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    years = list(range(start_ts.year, end_ts.year + 1))

    chunks_loaded = 0
    chunks_downloaded = 0
    for year in years:
        chunk_path = chunks_dir / f"chunk_{year}.nc"
        if chunk_path.exists():
            chunks_loaded += 1
            continue

        y_start = max(pd.Timestamp(f"{year}-01-01"), start_ts)
        y_end_inclusive = min(pd.Timestamp(f"{year}-12-31") + pd.Timedelta("23h"),
                              end_ts + pd.Timedelta("23h"))
        t_lo = int(np.searchsorted(dates_all, y_start))
        t_hi = int(np.searchsorted(dates_all, y_end_inclusive)) + 1
        if t_lo >= t_hi:
            continue

        print(f"[destine] year {year}: hours [{t_lo}:{t_hi}] "
              f"= {t_hi - t_lo}h  x {n_lat}x{n_lon}", flush=True)

        tp_h  = np.asarray(root["tp"][t_lo:t_hi, lat_lo:lat_hi+1, lon_west_idx:lon_east_idx+1])
        t2m_h = np.asarray(root["t2m"][t_lo:t_hi, lat_lo:lat_hi+1, lon_west_idx:lon_east_idx+1])
        u10_h = np.asarray(root["u10"][t_lo:t_hi, lat_lo:lat_hi+1, lon_west_idx:lon_east_idx+1])
        v10_h = np.asarray(root["v10"][t_lo:t_hi, lat_lo:lat_hi+1, lon_west_idx:lon_east_idx+1])
        hourly_times = dates_all[t_lo:t_hi]

        ds_hourly = xr.Dataset(
            {
                "tp":  (("time", "latitude", "longitude"), tp_h),
                "t2m": (("time", "latitude", "longitude"), t2m_h),
                "u10": (("time", "latitude", "longitude"), u10_h),
                "v10": (("time", "latitude", "longitude"), v10_h),
            },
            coords={"time": hourly_times, "latitude": sub_lats, "longitude": sub_lons},
        )

        # Daily aggregation. ERA5-LAND `tp` is cumulated through a forecast
        # cycle (resets every ~12h), so we recover the per-hour increment by
        # diff, clamp negatives (cycle boundaries) to 0, then sum daily.
        tp_inc = ds_hourly.tp.diff(dim="time", label="upper")
        tp_inc = tp_inc.where(tp_inc >= 0, 0)
        # Drop the first hour (no previous step) — its loss is bounded by
        # the typical 1h precip (<1 mm) and acceptable on 25-y averaging.
        pr      = (tp_inc.resample(time="1D").sum() * 1000.0).astype(np.float32)
        tasmin  = (ds_hourly.t2m.resample(time="1D").min() - 273.15).astype(np.float32)
        tasmax  = (ds_hourly.t2m.resample(time="1D").max() - 273.15).astype(np.float32)
        wind    = np.sqrt(ds_hourly.u10 ** 2 + ds_hourly.v10 ** 2)
        sfcWind = wind.resample(time="1D").mean().astype(np.float32)

        ds_daily = xr.Dataset(
            {"pr": pr, "tasmin": tasmin, "tasmax": tasmax, "sfcWind": sfcWind}
        )
        ds_daily["pr"].attrs.update(units="mm/day", standard_name="precipitation_amount")
        ds_daily["tasmin"].attrs.update(units="degC", standard_name="air_temperature")
        ds_daily["tasmax"].attrs.update(units="degC", standard_name="air_temperature")
        ds_daily["sfcWind"].attrs.update(units="m/s", standard_name="wind_speed",
                                         long_name="Daily mean wind speed at 10 m")
        ds_daily.to_netcdf(chunk_path)
        chunks_downloaded += 1

    print(f"[destine] Chunks: {chunks_loaded} from cache, "
          f"{chunks_downloaded} downloaded ({len(years)} years total)")

    chunk_files = sorted(chunks_dir.glob("chunk_*.nc"))
    if not chunk_files:
        return None
    full = xr.open_mfdataset(chunk_files, combine="by_coords")
    # Clip to exact requested period
    full = full.sel(time=slice(start_date, end_date))
    full.attrs.update(source="ERA5-LAND via DestinE Earth Data Hub",
                      url="https://data.earthdatahub.destine.eu/era5/reanalysis-era5-land-no-antartica-v0.zarr",
                      resolution_deg=0.1)
    full.to_netcdf(out_path)
    print(f"[destine] Forcing saved: {out_path}")
    return out_path


# ── Hydrometric observations via HYDAT (ECCC) ───────────────────────────────


def download_observations_hydat(
    bbox: tuple[float, float, float, float],
    cache_dir: str | Path,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[Path, Path] | None:
    """Download daily streamflow observations from HYDAT (ECCC) within bbox.

    HYDAT is the official Canadian hydrometric archive maintained by
    Environment and Climate Change Canada. The full SQLite snapshot is
    cached locally (~140 MB) on first call.

    Parameters
    ----------
    bbox :
        ``(west, south, east, north)`` in EPSG:4326.
    cache_dir :
        Directory where HYDAT SQLite and the parquet outputs are cached.
    start_date, end_date :
        ISO dates (``YYYY-MM-DD``) to filter observations. ``None`` keeps
        all available data.

    Returns
    -------
    Tuple ``(stations_parquet, observations_parquet)`` or ``None``.

    The parquet outputs are ready to ingest into ``basin.duckdb``:

    - **stations**: ``station_id, station_name, lon, lat,
      drainage_area_km2, regulated``
    - **observations**: ``station_id, date, Q_m3s``
    """
    import re
    import sqlite3
    import urllib.request
    import zipfile

    import pandas as pd

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    sqlite_path = cache_dir / "Hydat.sqlite3"

    if not sqlite_path.exists():
        # Discover latest snapshot URL
        base_url = "https://collaboration.cmc.ec.gc.ca/cmc/hydrometrics/www/"
        print("[hydat] Querying HYDAT snapshot index...")
        try:
            with urllib.request.urlopen(base_url, timeout=60) as r:
                html = r.read().decode("utf-8", errors="ignore")
            snapshots = sorted(
                set(re.findall(r"Hydat_sqlite3_\d{8}\.zip", html, re.I)),
                reverse=True,
            )
            if not snapshots:
                print("[hydat] No HYDAT snapshots found at " + base_url)
                return None
            zip_name = snapshots[0]
        except Exception as e:
            print(f"[hydat] Snapshot index query failed ({e})")
            return None

        snapshot_url = base_url + zip_name
        zip_path = cache_dir / zip_name
        if not zip_path.exists():
            print(f"[hydat] Downloading {snapshot_url} (~140 MB)...")
            try:
                urllib.request.urlretrieve(snapshot_url, str(zip_path))
            except Exception as e:
                print(f"[hydat] Download failed ({e})")
                return None

        print("[hydat] Extracting SQLite from zip...")
        with zipfile.ZipFile(zip_path) as zf:
            extracted = False
            for name in zf.namelist():
                if name.lower().endswith(".sqlite3"):
                    with zf.open(name) as src, open(sqlite_path, "wb") as dst:
                        dst.write(src.read())
                    extracted = True
                    break
            if not extracted:
                print("[hydat] No .sqlite3 found inside zip")
                return None
        # Free the zip (~283 MB) — sqlite3 is what we need from now on
        zip_path.unlink(missing_ok=True)

    # ── Query stations within bbox ────────────────────────────────────
    # REGULATED lives in STN_REGULATION (year_from/year_to ranges per station);
    # we take MAX so a station ever-regulated is flagged 1.
    west, south, east, north = bbox
    con = sqlite3.connect(str(sqlite_path))
    stations = pd.read_sql(
        "SELECT s.STATION_NUMBER as station_id, "
        "s.STATION_NAME as station_name, "
        "s.LATITUDE as lat, s.LONGITUDE as lon, "
        "s.DRAINAGE_AREA_GROSS as drainage_area_km2, "
        "s.HYD_STATUS as hyd_status, "
        "s.REAL_TIME as real_time, "
        "COALESCE(MAX(r.REGULATED), 0) as regulated "
        "FROM STATIONS s "
        "LEFT JOIN STN_REGULATION r ON s.STATION_NUMBER = r.STATION_NUMBER "
        "WHERE s.LONGITUDE BETWEEN ? AND ? "
        "AND s.LATITUDE BETWEEN ? AND ? "
        "GROUP BY s.STATION_NUMBER",
        con,
        params=(west, east, south, north),
    )

    if len(stations) == 0:
        print(f"[hydat] No HYDAT stations in bbox {bbox}")
        con.close()
        return None
    print(f"[hydat] {len(stations)} stations in bbox")

    # ── Query daily flows (DLY_FLOWS is wide: FLOW1..FLOW31 per month) ──
    placeholders = ",".join("?" for _ in stations["station_id"])
    flows_wide = pd.read_sql(
        f"SELECT * FROM DLY_FLOWS WHERE STATION_NUMBER IN ({placeholders})",
        con,
        params=tuple(stations["station_id"]),
    )
    con.close()

    if len(flows_wide) == 0:
        print("[hydat] No DLY_FLOWS rows for these stations")
        return None

    # Unpivot FLOW1..FLOW31 → long format
    flow_cols = [c for c in flows_wide.columns if re.fullmatch(r"FLOW\d+", c)]
    long = flows_wide.melt(
        id_vars=["STATION_NUMBER", "YEAR", "MONTH"],
        value_vars=flow_cols,
        var_name="_day_col",
        value_name="Q_m3s",
    )
    long["day"] = long["_day_col"].str.extract(r"FLOW(\d+)").astype(int)
    long = long.dropna(subset=["Q_m3s"])
    long["date"] = pd.to_datetime(
        long.rename(columns={"YEAR": "year", "MONTH": "month"})
            [["year", "month", "day"]],
        errors="coerce",
    )
    long = long.dropna(subset=["date"])
    long = (
        long[["STATION_NUMBER", "date", "Q_m3s"]]
        .rename(columns={"STATION_NUMBER": "station_id"})
        .sort_values(["station_id", "date"])
        .reset_index(drop=True)
    )

    if start_date:
        long = long[long["date"] >= pd.Timestamp(start_date)]
    if end_date:
        long = long[long["date"] <= pd.Timestamp(end_date)]

    if len(long) == 0:
        print(f"[hydat] No observations in date range")
        return None

    # ── Save parquets ─────────────────────────────────────────────────
    stations_path = cache_dir / "hydat_stations.parquet"
    obs_path = cache_dir / "hydat_observations.parquet"
    stations.to_parquet(stations_path)
    long.to_parquet(obs_path)
    print(f"[hydat] Saved {len(stations)} stations, {len(long):,} obs to "
          f"{stations_path.name}, {obs_path.name}")
    return stations_path, obs_path


def populate_basin_observations(
    basin_db: str | Path,
    stations_parquet: str | Path,
    observations_parquet: str | Path,
    max_snap_km: float = 5.0,
    max_drainage_ratio: float = 2.0,
    drainage_weight_km: float = 50.0,
) -> int:
    """Insert HYDAT stations and observations into a basin DuckDB.

    Each station is snapped to the node whose simulated cumulative drainage
    area best matches the HYDAT-published gauge drainage area, with proximity
    as a secondary criterion. This is topologically meaningful (the station
    measures Q at a known drainage point ; we map it to the segment whose
    outflow corresponds to that area) and avoids the failure mode of pure
    Euclidean snap where a small-tributary gauge gets attached to the much
    larger main-stem node next door.

    Cost function ::

        cost = distance_km + drainage_weight_km × |log(sim_area / hydat_area)|

    With default ``drainage_weight_km=50``, a 2.7× drainage mismatch costs
    the equivalent of 50 km. A station can accept a node 50 km further if it
    has a perfect drainage match. Hard filter : drainage ratio must lie in
    ``[1/max_drainage_ratio, max_drainage_ratio]`` (default ⇒ within 2×),
    AND distance ≤ ``max_snap_km``.

    Stations without a published HYDAT drainage_area_km2 fall back to pure
    Euclidean nearest-node (legacy behavior).

    The ``stations`` and ``observations`` tables are fully replaced
    (DELETE + INSERT) — call once per basin.

    Returns
    -------
    Number of observations inserted.
    """
    import duckdb
    import numpy as np
    import pandas as pd

    stations = pd.read_parquet(stations_parquet)
    obs = pd.read_parquet(observations_parquet)

    con = duckdb.connect(str(basin_db))
    nodes_with_area = con.execute(
        "SELECT n.node_idx, n.lon, n.lat, t.area_km2_physical AS sim_area "
        "FROM nodes n LEFT JOIN territorial t ON n.node_idx = t.node_idx"
    ).df()
    if len(nodes_with_area) == 0:
        print("[basin] basin DuckDB has no nodes — aborting observation insert")
        con.close()
        return 0

    node_lons = nodes_with_area["lon"].to_numpy()
    node_lats = nodes_with_area["lat"].to_numpy()
    node_areas = nodes_with_area["sim_area"].fillna(1e-3).to_numpy()
    node_idxs = nodes_with_area["node_idx"].to_numpy()
    cos_lat = float(np.cos(np.radians(stations["lat"].mean())))

    # Per-station results : aligned with original stations index ; NaN/None
    # for rejected stations (filtered out at the end).
    accepted_mask = np.zeros(len(stations), dtype=bool)
    snapped_idx_arr = np.zeros(len(stations), dtype=np.int64)
    snapped_dist_arr = np.full(len(stations), np.nan)
    snapped_ratio_arr = np.full(len(stations), np.nan)
    rejected: list[tuple[str, str]] = []  # (station_id, reason)

    for row_i, (_, s) in enumerate(stations.iterrows()):
        dx = (node_lons - s["lon"]) * 111.0 * cos_lat
        dy = (node_lats - s["lat"]) * 111.0
        dist = np.hypot(dx, dy)

        hydat_area = s.get("drainage_area_km2")
        if hydat_area is None or pd.isna(hydat_area) or hydat_area <= 0:
            # No HYDAT area : fall back to pure Euclidean (legacy behavior)
            i = int(np.argmin(dist))
            if dist[i] > max_snap_km:
                rejected.append((s["station_id"], f"{dist[i]:.1f} km, no area"))
                continue
            accepted_mask[row_i] = True
            snapped_idx_arr[row_i] = int(node_idxs[i])
            snapped_dist_arr[row_i] = float(dist[i])
            continue

        ratio = node_areas / max(float(hydat_area), 1e-3)
        log_ratio_abs = np.abs(np.log(np.clip(ratio, 1e-6, 1e6)))
        cost = dist + drainage_weight_km * log_ratio_abs

        # Hard filters : drainage within factor, distance within max
        valid = (
            (ratio < max_drainage_ratio)
            & (ratio > 1.0 / max_drainage_ratio)
            & (dist <= max_snap_km)
        )
        if not valid.any():
            # Diagnose : closest by distance, closest by drainage
            i_close = int(np.argmin(dist))
            i_match = int(np.argmin(log_ratio_abs))
            rejected.append((
                s["station_id"],
                f"hydat={hydat_area:.0f} km², closest sim={node_areas[i_close]:.0f}"
                f" ({dist[i_close]:.1f} km), best match sim={node_areas[i_match]:.0f}"
                f" ({dist[i_match]:.1f} km)",
            ))
            continue

        cost_valid = np.where(valid, cost, np.inf)
        i = int(np.argmin(cost_valid))
        accepted_mask[row_i] = True
        snapped_idx_arr[row_i] = int(node_idxs[i])
        snapped_dist_arr[row_i] = float(dist[i])
        snapped_ratio_arr[row_i] = float(ratio[i])

    if rejected:
        print(f"[basin] Rejected {len(rejected)} stations (no compatible node):")
        for sid, reason in rejected[:20]:
            print(f"  {sid}: {reason}")
        if len(rejected) > 20:
            print(f"  ... and {len(rejected) - 20} more")

    stations = stations.loc[accepted_mask].copy()
    stations["node_idx"] = snapped_idx_arr[accepted_mask]
    stations["snap_dist_km"] = snapped_dist_arr[accepted_mask]
    stations["snap_drainage_ratio"] = snapped_ratio_arr[accepted_mask]

    if len(stations) == 0:
        print(f"[basin] No HYDAT stations within {max_snap_km} km of any node")
        con.close()
        return 0

    obs = obs[obs["station_id"].isin(stations["station_id"])].copy()
    obs = obs.rename(columns={"Q_m3s": "discharge"})

    s_db = stations[["station_id", "node_idx", "lon", "lat",
                     "drainage_area_km2"]].copy()
    o_db = obs[["station_id", "date", "discharge"]].copy()

    con.execute("DELETE FROM stations")
    con.execute("DELETE FROM observations")
    con.register("s_df", s_db)
    con.register("o_df", o_db)
    con.execute("INSERT INTO stations SELECT * FROM s_df")
    con.execute("INSERT INTO observations SELECT * FROM o_df")
    con.close()

    print(f"[basin] Inserted {len(stations)} stations, {len(obs):,} obs into "
          f"{basin_db}")
    return len(obs)


# ── Convenience: download all ────────────────────────────────────────────────


def download_modis_et(
    bbox: tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    cache_dir: str | Path,
    node_coords: "Tensor | None" = None,
) -> "Path | None":
    """STUB — downloads MODIS MOD16A2 8-day ETR and aggregates to node × day.

    Pending implementation. Expected workflow:
      1. STAC query Planetary Computer for ``modis-16A2-061`` over bbox/period.
      2. Read the ``PET_500m`` / ``ET_500m`` bands (kg/m²/8day → mm/day).
      3. Zonal mean per node (using subcatchment masks from the basin DB).
      4. Linear interpolation to daily timestep (8-day composites carry mid-period
         values; observations should fall on the right calendar day).
      5. Save as NetCDF with dims (time, node) and ``etr`` variable.

    Returns the cached NetCDF path so ``meandre.data.basin_cache`` can load it
    into ``TrainingData.et_obs`` (NaN for unobserved days). Triggers the
    ``w_nll_et`` term in HydroLoss when populated.

    Reference: Mu, Q., Zhao, M., Running, S. W. (2011). Improvements to a MODIS
    global terrestrial evapotranspiration algorithm. Remote Sens. Environ.
    """
    raise NotImplementedError(
        "MODIS MOD16A2 ET loader is not yet implemented. "
        "See meandre/data/open_data.py for the expected interface."
    )


def download_modis_swe(
    bbox: tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    cache_dir: str | Path,
    node_coords: "Tensor | None" = None,
) -> "Path | None":
    """STUB — downloads MODIS NDSI snow cover (MOD10A1) → SWE proxy per node.

    Pending implementation. Expected workflow:
      1. STAC query for ``modis-10A1-061`` (daily snow cover fraction).
      2. Either:
         (a) Use NDSI fraction directly as a snow-presence indicator (less
             informative, no depth), or
         (b) Pair with SNODAS (CONUS only) for true SWE in mm.
      3. Zonal mean per node.
      4. Mask cloudy days (NaN — Gaussian NLL skips them automatically).
      5. Save as NetCDF with dims (time, node) and ``swe`` variable.

    Returns the cached NetCDF path. Triggers the ``w_nll_swe`` term in
    HydroLoss when populated — identifies C_f (degree-day melt factor) which
    is collapsed under KGE-on-Q alone.

    Reference: Hall, D. K., Riggs, G. A. (2007). Accuracy assessment of MODIS
    snow products. Hydrological Processes 21(12).
    """
    raise NotImplementedError(
        "MODIS NDSI snow loader is not yet implemented. "
        "See meandre/data/open_data.py for the expected interface."
    )


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
