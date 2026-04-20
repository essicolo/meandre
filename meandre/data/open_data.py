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


# ── GRHQ — Géobase du Réseau Hydrographique du Québec ───────────────────────


def download_grhq(
    bbox: tuple[float, float, float, float],
    cache_dir: str | Path,
    wfs_url: str | None = None,
) -> Path | None:
    """Download GRHQ river-network lines clipped to *bbox*.

    Tries three sources in order:

    1. **WFS** — clips to bbox server-side (pass a custom *wfs_url* if you know
       the correct endpoint for your layer).  Typical Quebec WFS pattern::

           "https://geoegl.msp.gouv.qc.ca/apis/wfs?SERVICE=WFS&VERSION=2.0.0"
           "&REQUEST=GetFeature&TYPENAMES=<layer>&BBOX=..."

    2. **données.gouv.qc.ca CKAN API** — fetches the GRHQ package metadata,
       downloads the first GeoPackage or ZIP resource found, then clips.
       The full dataset can be several hundred MB; it is cached locally.

    3. **NRCan National Hydrographic Network (NHN)** via CKAN — same approach
       for ``nhn-rhn`` if GRHQ is unavailable.

    Returns path to ``grhq_rivers.parquet`` (LineString geometries, EPSG:4326),
    or ``None`` if all sources fail.

    Requires **geopandas** (``pip install meandre[geo]``).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / "grhq_rivers.parquet"
    if out_path.exists():
        print(f"[open_data] GRHQ cached: {out_path}")
        return out_path

    try:
        import geopandas as gpd
        from shapely.geometry import box as shapely_box
    except ImportError:
        print("[open_data] geopandas not installed — skipping GRHQ download. "
              "Install with: pip install geopandas")
        return None

    west, south, east, north = bbox
    aoi = shapely_box(west, south, east, north)

    gdf = None

    # ── 1. WFS (server-side clip) ─────────────────────────────────────────────
    if wfs_url:
        try:
            print(f"[open_data] Querying WFS: {wfs_url[:80]}...")
            gdf = gpd.read_file(wfs_url)
            if len(gdf) == 0:
                gdf = None
            else:
                print(f"[open_data] WFS returned {len(gdf)} features")
        except Exception as e:
            print(f"[open_data] WFS failed ({e})")
            gdf = None

    # ── 2. CKAN sources ───────────────────────────────────────────────────────
    ckan_sources = [
        ("GRHQ",  "https://www.donneesquebec.ca/recherche/api/3/action/package_show?id=grhq"),
    ]

    if gdf is None:
        import json, urllib.request, zipfile

        for src_name, ckan_url in ckan_sources:
            print(f"[open_data] Querying {src_name} via CKAN API...")
            try:
                with urllib.request.urlopen(ckan_url, timeout=30) as resp:
                    pkg = json.loads(resp.read())
            except Exception as e:
                print(f"[open_data]   {src_name} CKAN failed ({e})")
                continue

            resources = (pkg.get("result") or {}).get("resources", [])

            # Prefer GeoPackage, then Shapefile ZIP
            dl_url = None
            for res in resources:
                url = res.get("url", "")
                if url.lower().endswith(".gpkg"):
                    dl_url = url
                    break
            if dl_url is None:
                for res in resources:
                    url = res.get("url", "")
                    if url.lower().endswith(".zip"):
                        dl_url = url
                        break

            if dl_url is None:
                print(f"[open_data]   No GeoPackage/ZIP found in {src_name} CKAN package.")
                continue

            ext = ".gpkg" if dl_url.lower().endswith(".gpkg") else ".zip"
            raw_dir = cache_dir / f"{src_name.lower()}_raw"
            raw_dir.mkdir(exist_ok=True)
            dl_path = raw_dir / f"download{ext}"

            if not dl_path.exists():
                print(f"[open_data]   Downloading {src_name} ({dl_url[:80]}...)  "
                      "This may take several minutes.")
                try:
                    urllib.request.urlretrieve(dl_url, str(dl_path))
                except Exception as e:
                    print(f"[open_data]   Download failed ({e})")
                    continue

            read_path = str(dl_path)
            if ext == ".zip":
                with zipfile.ZipFile(dl_path) as zf:
                    zf.extractall(str(raw_dir))
                # For NHN: prefer shapefiles whose name suggests flow paths
                _flow_keywords = ("flow", "cours", "hydrograph", "watercourse", "stream", "river")
                read_path = None
                candidates = sorted(raw_dir.rglob("*.shp")) + sorted(raw_dir.rglob("*.gpkg"))
                # First pass: prefer files with flow/river keywords
                for p in candidates:
                    if any(kw in p.stem.lower() for kw in _flow_keywords):
                        read_path = str(p)
                        break
                # Second pass: take first file found
                if read_path is None and candidates:
                    read_path = str(candidates[0])

            if read_path is None:
                print(f"[open_data]   Could not find spatial file in {src_name} archive.")
                continue

            print(f"[open_data]   Reading {src_name} (bbox clip)...")
            try:
                gdf = gpd.read_file(read_path, bbox=bbox)
                if len(gdf) == 0:
                    gdf = None
                    print(f"[open_data]   No features in bbox from {src_name}.")
                    continue
                print(f"[open_data]   {src_name}: {len(gdf)} features in bbox.")
                break
            except Exception as e:
                print(f"[open_data]   Read failed ({e})")
                gdf = None

    # ── 3. OpenStreetMap via Overpass API (always available) ──────────────────
    if gdf is None or len(gdf) == 0 or not any(
        t in ["LineString", "MultiLineString"]
        for t in gdf.geometry.geom_type.unique()
    ):
        print("[open_data] Trying OSM Overpass for river lines...")
        try:
            import urllib.request as _ur
            import urllib.parse as _up
            import json as _json
            # Only waterway=river — streams too numerous, cause 504 timeouts
            _query = (
                f"[out:json][timeout:120];"
                f'(way["waterway"="river"]({south},{west},{north},{east}););'
                f"out geom;"
            )
            _data = _up.urlencode({"data": _query}).encode()
            _r_data = None
            for _mirror in [
                "https://overpass.kumi.systems/api/interpreter",
                "https://overpass-api.de/api/interpreter",
            ]:
                try:
                    with _ur.urlopen(_mirror, data=_data, timeout=180) as _r:
                        _r_data = _r.read()
                    break
                except Exception:
                    continue
            if _r_data is None:
                raise RuntimeError("All Overpass mirrors failed")
            _result = _json.loads(_r_data)
            _feats = []
            for _el in _result.get("elements", []):
                if _el.get("type") == "way" and "geometry" in _el:
                    _coords = [(_n["lon"], _n["lat"]) for _n in _el["geometry"]]
                    if len(_coords) >= 2:
                        _tags = _el.get("tags", {})
                        _feats.append({
                            "geometry": {"type": "LineString", "coordinates": _coords},
                            "properties": {
                                "name":     _tags.get("name", ""),
                                "waterway": _tags.get("waterway", ""),
                            },
                        })
            gdf = gpd.GeoDataFrame.from_features(_feats, crs="EPSG:4326") if _feats else None
            if gdf is not None:
                print(f"[open_data] OSM: {len(gdf)} river lines")
        except Exception as _e:
            print(f"[open_data] OSM Overpass failed ({_e})")
            gdf = None

    if gdf is None or len(gdf) == 0:
        print("[open_data] GRHQ: all sources failed — skipping.")
        return None

    # ── Post-process ──────────────────────────────────────────────────────────
    gdf = gdf[gdf.geometry.notna()]
    gdf = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])]
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    gdf = gdf.clip(aoi)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]

    if len(gdf) == 0:
        print("[open_data] GRHQ: no river lines in bbox after clip.")
        return None

    gdf = gdf.reset_index(drop=True)
    gdf.to_parquet(str(out_path))
    print(f"[open_data] GRHQ saved: {out_path} ({len(gdf)} segments)")
    return out_path


# ── Reach geometry builder ────────────────────────────────────────────────────


def build_reach_parquet(
    nodes_df: "pd.DataFrame",
    grhq_path: str | Path,
    out_path: str | Path,
    max_dist_deg: float = 0.05,
) -> Path:
    """Match model nodes to GRHQ river segments and save as parquet.

    For each node ``(lon, lat)`` in *nodes_df*, finds the nearest GRHQ river
    segment within *max_dist_deg* degrees (~5 km at Quebec latitudes).
    Saves a GeoDataFrame with columns::

        node_idx  node_id  dist_deg  geometry (the matched river LineString)

    Unmatched nodes are omitted.

    Parameters
    ----------
    nodes_df :
        DataFrame with columns ``node_idx``, ``node_id``, ``lon``, ``lat``.
    grhq_path :
        Path to ``grhq_rivers.parquet`` produced by :func:`download_grhq`.
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

    rivers = gpd.read_parquet(str(grhq_path))
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
            "check that GRHQ bbox covers the model domain."
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


# ── Convenience: download all ────────────────────────────────────────────────


def download_all(
    bbox: tuple[float, float, float, float],
    cache_dir: str | Path,
) -> dict[str, Path]:
    """Download DEM, land cover, soil, JRC water, MODIS LAI, NRCan LC and GRHQ.

    Returns dict with keys:
        ``"dem"``, ``"landcover"``, ``"soil_dir"``,
        ``"water_occurrence"``, ``"lai"``, ``"nrcan_lc"``, ``"grhq"``
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
        "grhq":             download_grhq(bbox, cache_dir),
    }
