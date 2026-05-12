"""HydroBASINS upstream polygon resolution for bbox-aware basin delineation.

Downloads HydroBASINS Level 12 North America (HydroSHEDS) on first use, caches
locally. Provides ``upstream_polygon_from_point(lon, lat)`` which returns the
union of all sub-basins draining to a given point — used by ``bbox_to_basin.py``
to ensure the DEM download window covers the *full* upstream catchment, not just
an arbitrary bbox.

Without this, a point-of-interest near the centre of a large basin would have
its upstream tributaries truncated at the bbox edge, biasing drainage area and
introducing equifinality at training time.

HydroBASINS reference
---------------------
Lehner, B., Grill G. (2013). Global river hydrography and network routing:
baseline data and new approaches to study the world's large river systems.
Hydrological Processes 27(15): 2171-2186.
https://www.hydrosheds.org/products/hydrobasins
"""

from __future__ import annotations

import zipfile
from collections import defaultdict
from pathlib import Path
from urllib.request import Request, urlopen

import geopandas as gpd

# Level 12 = finest available (~10-100 km^2 per sub-basin in NA).
# "lake" variant includes lake polygons as separate basins.
_URL_HYBAS_NA_L12 = (
    "https://data.hydrosheds.org/file/HydroBASINS/standard/"
    "hybas_na_lev12_v1c.zip"
)


def _download_hydrobasins_na(cache_dir: Path) -> Path:
    """Download and extract HydroBASINS NA Level 12. Returns path to the .shp.

    Idempotent: skips download and extraction if already present.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    shp_path = cache_dir / "hybas_na_lev12_v1c.shp"
    if shp_path.exists():
        return shp_path

    zip_path = cache_dir / "hybas_na_lev12_v1c.zip"
    if not zip_path.exists():
        print(f"[hydrobasins] Downloading {_URL_HYBAS_NA_L12} (~80 MB)...")
        req = Request(_URL_HYBAS_NA_L12, headers={"User-Agent": "meandre/1.0"})
        with urlopen(req, timeout=120) as r, open(zip_path, "wb") as f:
            f.write(r.read())
        print(f"[hydrobasins] Saved {zip_path} ({zip_path.stat().st_size / 1e6:.0f} MB)")

    print(f"[hydrobasins] Extracting to {cache_dir}...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(cache_dir)
    if not shp_path.exists():
        raise FileNotFoundError(
            f"Expected {shp_path} after extracting {zip_path} but file is missing."
        )
    return shp_path


def load_hydrobasins_na(cache_dir: Path) -> gpd.GeoDataFrame:
    """Load HydroBASINS NA Lv12 as a GeoDataFrame (with parquet cache).

    The first call downloads + extracts + reads the shapefile (~1 min, ~600 MB
    on disk after extraction). Subsequent calls hit a parquet cache (~5 s).
    """
    parquet_cache = cache_dir / "hybas_na_lev12_v1c.parquet"
    if parquet_cache.exists():
        return gpd.read_parquet(parquet_cache)

    shp = _download_hydrobasins_na(cache_dir)
    print(f"[hydrobasins] Reading {shp.name} (first load, slow)...")
    gdf = gpd.read_file(shp)
    # Keep only columns we use to shrink the parquet.
    keep = [c for c in ["HYBAS_ID", "NEXT_DOWN", "SUB_AREA", "UP_AREA", "geometry"]
            if c in gdf.columns]
    gdf = gdf[keep].copy()
    print(f"[hydrobasins] Caching as parquet ({len(gdf)} basins)...")
    gdf.to_parquet(parquet_cache)
    return gdf


def upstream_polygon_from_point(
    lon: float,
    lat: float,
    cache_dir: Path,
    margin: float = 0.05,
) -> tuple[gpd.GeoDataFrame, tuple[float, float, float, float]]:
    """Compute the upstream drainage polygon from a point.

    Walks the HydroBASINS topology upstream from the basin containing
    ``(lon, lat)`` and returns the union of all ancestor sub-basins.

    Parameters
    ----------
    lon, lat :
        Point of interest (basin outlet or any in-network point) in EPSG:4326.
    cache_dir :
        Where to store the HydroBASINS download + parquet cache.
    margin :
        Fractional buffer applied to the bounding box of the polygon. 0.05 = 5%
        margin in each direction (so the DEM download window covers the polygon
        with some slack).

    Returns
    -------
    polygon_gdf :
        Single-row GeoDataFrame with the dissolved upstream polygon (EPSG:4326).
    bbox :
        (west, south, east, north) bounding box with ``margin`` applied,
        suitable for passing to ``--bbox`` of ``bbox_to_basin.py``.
    """
    from shapely.geometry import Point

    gdf = load_hydrobasins_na(cache_dir)
    point = Point(lon, lat)

    # Find the sub-basin containing the point (uses sindex automatically).
    candidates = gdf.iloc[list(gdf.sindex.query(point, predicate="intersects"))]
    containing = candidates[candidates.contains(point)]
    if len(containing) == 0:
        raise ValueError(
            f"No HydroBASINS sub-basin contains ({lon}, {lat}). "
            "Check the point is in North America and on land."
        )
    seed_id = int(containing.iloc[0]["HYBAS_ID"])
    seed_area = float(containing.iloc[0]["SUB_AREA"])
    print(f"[hydrobasins] Seed sub-basin HYBAS_ID={seed_id} (SUB_AREA={seed_area:.1f} km²)")

    # Build child→parent inverse: for each sub-basin, who flows into it.
    children = defaultdict(list)
    for hb, nd in zip(
        gdf["HYBAS_ID"].to_numpy(dtype="int64"),
        gdf["NEXT_DOWN"].to_numpy(dtype="int64"),
    ):
        if nd != 0:
            children[int(nd)].append(int(hb))

    # BFS upstream from the seed.
    visited: set[int] = set()
    stack = [seed_id]
    while stack:
        bid = stack.pop()
        if bid in visited:
            continue
        visited.add(bid)
        stack.extend(children.get(bid, []))

    upstream = gdf[gdf["HYBAS_ID"].isin(visited)]
    total_area = float(upstream["SUB_AREA"].sum())
    print(f"[hydrobasins] Upstream collected: {len(upstream)} sub-basins, "
          f"{total_area:.0f} km² total")

    polygon = upstream.geometry.union_all()
    polygon_gdf = gpd.GeoDataFrame(
        {"hybas_seed": [seed_id], "n_subbasins": [len(upstream)],
         "area_km2": [total_area]},
        geometry=[polygon], crs="EPSG:4326",
    )

    w, s, e, n = polygon.bounds
    dx, dy = (e - w) * margin, (n - s) * margin
    bbox = (w - dx, s - dy, e + dx, n + dy)
    return polygon_gdf, bbox
