"""Build a basin DuckDB from open-data rasters.

Pipeline: DEM + land cover + soil → subcatchments → zonal stats → DuckDB.

Requires optional dependencies::

    pip install meandre[geo]
    # or: pip install pysheds rasterstats rasterio

Usage::

    from meandre.data.basin_builder import build_basin

    build_basin(
        dem_path="data/dem.tif",
        landcover_path="data/landcover.tif",
        soil_dir="data/",
        outlet=(-69.5, 47.5),       # (lon, lat)
        basin_db="data/basin.duckdb",
    )
"""

from __future__ import annotations

import collections
import math
from pathlib import Path

import numpy as np

# pysheds uses np.in1d which was removed in NumPy 2.0
if not hasattr(np, "in1d"):
    np.in1d = np.isin

import rasterio
import torch
from torch import Tensor

from meandre.data.basin_cache import BasinCache
from meandre.routing.graph import RiverGraph
from meandre.spatial.territorial import TerritorialFeatures, DEFAULT_PHYSICAL_COLUMNS
from meandre.utils.state import HydroState


# ── Main entry point ────────────────────────────────────────────────────────


def build_basin(
    dem_path: str | Path,
    landcover_path: str | Path,
    soil_dir: str | Path,
    outlet: tuple[float, float] | None,
    basin_db: str | Path,
    min_area_km2: float = 2.0,
    max_subcatchments: int = 300,
    water_occurrence_path: str | Path | None = None,
    lai_path: str | Path | None = None,
    nrcan_lc_path: str | Path | None = None,
    water_polygons_path: str | Path | None = None,
    extra_stats: list[str] | None = None,
    normalise: bool = True,
    basin_mask_gdf=None,
    max_dem_pixels: int = 4_000_000,
) -> BasinCache:
    """Build a complete basin DuckDB from open-data rasters.

    Parameters
    ----------
    dem_path :
        Copernicus DEM 30m GeoTIFF.
    landcover_path :
        ESA WorldCover GeoTIFF.
    soil_dir :
        Directory with SoilGrids GeoTIFFs (sand.tif, silt.tif, clay.tif).
    outlet :
        (lon, lat) of the basin outlet in EPSG:4326. Pass ``None`` to
        auto-detect: the highest-accumulation cell on the raster edge is
        used (where water leaves the bbox).
    basin_db :
        Output DuckDB file path.
    min_area_km2 :
        Minimum subcatchment area for stream threshold.
    extra_stats :
        Additional zonal statistics to compute, e.g. ["elevation_std", "slope_p10"].
    normalise :
        If True, z-score normalise feature columns.

    Returns
    -------
    BasinCache ready for training.
    """
    dem_path = Path(dem_path)
    landcover_path = Path(landcover_path)
    soil_dir = Path(soil_dir)

    # Step 1: Hydrological conditioning and flow routing
    print("[basin_builder] Step 1: DEM conditioning and flow routing...")
    grid_data = _condition_dem(dem_path, max_dem_pixels=max_dem_pixels)

    # Step 2: Delineate subcatchments
    print("[basin_builder] Step 2: Delineating subcatchments...")
    subcatchments = _delineate_subcatchments(
        grid_data, outlet, min_area_km2=min_area_km2,
        max_subcatchments=max_subcatchments,
        basin_mask_gdf=basin_mask_gdf,
    )
    n_nodes = subcatchments["n_nodes"]
    print(f"  {n_nodes} subcatchments delineated")

    # Step 3: Build river network (graph)
    print("[basin_builder] Step 3: Building river network...")
    graph, node_ids, is_lake = _build_network(subcatchments)

    # Step 4: Zonal statistics
    print("[basin_builder] Step 4: Computing zonal statistics...")
    features, physical, columns = _compute_zonal_stats(
        subcatchments, dem_path, landcover_path, soil_dir,
        graph, extra_stats=extra_stats or [],
        water_occurrence_path=water_occurrence_path,
        lai_path=lai_path,
        nrcan_lc_path=nrcan_lc_path,
        water_polygons_path=water_polygons_path,
    )

    # Step 4b: Lake detection — flag nodes where >10 % of area is permanent water
    if "lake_fraction" in columns:
        lf_idx = columns.index("lake_fraction")
        lake_frac_raw = features[:, lf_idx]          # un-normalised here
        graph.is_lake = lake_frac_raw > 0.50
        n_lakes = int(graph.is_lake.sum())
        print(f"  {n_lakes} lacs détectés (lake_fraction > 50 %)", flush=True)

    # Step 5: Normalise features
    if normalise:
        mu = features.mean(dim=0, keepdim=True)
        sig = features.std(dim=0, keepdim=True)
        sig = torch.where(sig > 0, sig, torch.ones_like(sig))
        features = (features - mu) / sig

    territorial = TerritorialFeatures(
        data=features, columns=columns, physical=physical,
    )

    # Step 6: Node coordinates (subcatchment centroids)
    node_coords = torch.tensor(
        subcatchments["centroids"], dtype=torch.float32,
    )

    # Step 7: Initial state (zeros)
    initial_state = HydroState(
        theta1=torch.full((n_nodes,), 0.3),
        theta2=torch.full((n_nodes,), 0.3),
        theta3=torch.full((n_nodes,), 0.3),
        swe=torch.zeros(n_nodes),
        t_soil=torch.full((n_nodes,), 5.0),
        canopy_storage=torch.zeros(n_nodes),
        wetland_storage=torch.zeros(n_nodes),
        S_gw=torch.zeros(n_nodes),
        T_water=torch.full((n_nodes,), 10.0),
    )

    # Step 8: Write to DuckDB
    hydro = {
        "graph": graph,
        "territorial": territorial,
        "node_coords": node_coords,
        "initial_state": initial_state,
        "node_ids": node_ids,
        "n_nodes": n_nodes,
    }

    cache = BasinCache.from_dict(hydro, basin_db, source="open_data")
    print(f"[basin_builder] Basin DB written: {basin_db}")
    print(f"  {n_nodes} nodes, {graph.n_edges} edges, {len(columns)} features")
    return cache


# ── Step 1: DEM conditioning ────────────────────────────────────────────────


def _condition_dem(dem_path: Path, max_dem_pixels: int = 4_000_000) -> dict:
    """Fill depressions, compute flow direction and accumulation.

    Results are cached as compact .npy arrays next to the DEM so subsequent
    calls skip the expensive priority-flood + numba compilation (~3–5 min).
    fdir is stored as uint8 (~150 MB) and acc as float32 (~600 MB) instead
    of uncompressed GeoTIFFs (~1.2 GB each).

    Parameters
    ----------
    max_dem_pixels :
        If the DEM has more pixels than this, it is bilinear-downsampled
        before pysheds runs. Pysheds priority-flood depression filling
        recurses deeply on large rasters and segfaults on Windows past
        ~5 MP. 4 MP default is a safe cap (~150 m resolution on a
        3.4 deg x 3.2 deg bbox like SLSO).
    """
    import rasterio
    from rasterio.enums import Resampling
    from pysheds.grid import Grid
    from pysheds.sview import Raster, ViewFinder

    cache_dir  = dem_path.parent
    fdir_cache = cache_dir / "fdir.npy"
    acc_cache  = cache_dir  / "acc.npy"

    # Downsample if too large for pysheds in-memory routing.
    with rasterio.open(dem_path) as _src:
        n_pixels = _src.height * _src.width
        src_h, src_w = _src.height, _src.width

    routing_dem_path = dem_path
    if n_pixels > max_dem_pixels:
        routing_dem_path = cache_dir / "dem_routing.tif"
        if not routing_dem_path.exists():
            factor = float(np.sqrt(n_pixels / max_dem_pixels))
            new_h = int(src_h / factor)
            new_w = int(src_w / factor)
            print(f"[basin_builder] DEM = {n_pixels/1e6:.2f} MP > "
                  f"{max_dem_pixels/1e6:.2f} MP cap; "
                  f"downsampling by {factor:.2f} -> {new_w}x{new_h} = "
                  f"{new_h*new_w/1e6:.2f} MP", flush=True)
            with rasterio.open(dem_path) as src:
                data = src.read(
                    1, out_shape=(new_h, new_w),
                    resampling=Resampling.bilinear,
                )
                new_transform = src.transform * src.transform.scale(
                    src_w / new_w, src_h / new_h,
                )
                profile = src.profile.copy()
                profile.update(height=new_h, width=new_w,
                               transform=new_transform)
            with rasterio.open(routing_dem_path, "w", **profile) as dst:
                dst.write(data, 1)
            print(f"[basin_builder] Downsampled DEM -> {routing_dem_path}",
                  flush=True)
        else:
            print(f"[basin_builder] Using cached downsampled DEM: "
                  f"{routing_dem_path}", flush=True)

    grid = Grid.from_raster(str(routing_dem_path))
    dem  = grid.read_raster(str(routing_dem_path))

    if fdir_cache.exists() and acc_cache.exists():
        print("[basin_builder] DEM conditioning cached — loading fdir/acc...", flush=True)
        vf   = ViewFinder(affine=grid.affine, shape=grid.shape, crs=grid.crs)
        fdir = Raster(np.load(fdir_cache).astype(np.int32), viewfinder=vf)
        acc  = Raster(np.load(acc_cache).astype(np.float64), viewfinder=vf)
        return {"grid": grid, "dem": dem, "fdir": fdir, "acc": acc,
                "conditioned_dem": dem}

    print("[basin_builder] DEM conditioning (first run, may take several minutes)...",
          flush=True)
    pit_filled = grid.fill_pits(dem)
    flooded    = grid.fill_depressions(pit_filled)
    inflated   = grid.resolve_flats(flooded)
    fdir       = grid.flowdir(inflated)
    acc        = grid.accumulation(fdir)

    np.save(fdir_cache, np.asarray(fdir).astype(np.uint8))
    np.save(acc_cache,  np.asarray(acc).astype(np.float32))
    # Remove old .tif caches if they exist
    for old in (cache_dir / "fdir.tif", cache_dir / "acc.tif"):
        old.unlink(missing_ok=True)
    print(f"[basin_builder] Cached fdir/acc -> {cache_dir}", flush=True)

    return {
        "grid": grid,
        "dem": dem,
        "fdir": fdir,
        "acc": acc,
        "conditioned_dem": inflated,
    }


# ── Step 2: Subcatchment delineation ────────────────────────────────────────


def _delineate_subcatchments(
    grid_data: dict,
    outlet: tuple[float, float] | None,
    min_area_km2: float = 2.0,
    max_subcatchments: int = 300,
    basin_mask_gdf=None,
) -> dict:
    """Delineate subcatchments from flow accumulation threshold.

    If ``outlet`` is None, automatically pick the highest-accumulation cell
    on the raster edge — this is where the basin drains out of the bbox.
    """
    grid = grid_data["grid"]
    fdir = grid_data["fdir"]
    acc = grid_data["acc"]
    dem = grid_data["dem"]

    # Convert min_area_km2 to pixel count (approximate)
    # Copernicus DEM 30m: ~30m resolution → ~900 m² per pixel
    res_m = abs(grid.affine.a) * 111_000  # degrees to meters (approximate)
    pixel_area_km2 = (res_m ** 2) / 1e6
    min_pixels = max(int(min_area_km2 / pixel_area_km2), 100)
    print(f"  résolution={res_m:.1f} m  min_pixels={min_pixels}  "
          f"raster={grid.shape[0]}×{grid.shape[1]} px")

    # Auto-detect outlet if not given: max-accumulation cell on raster edge.
    # The edge is where water leaves the bbox, so the highest-accumulation
    # cell on the edge is the basin outlet.
    if outlet is None:
        acc_arr = np.asarray(acc)
        edge_mask = np.zeros_like(acc_arr, dtype=bool)
        edge_mask[0, :] = edge_mask[-1, :] = True
        edge_mask[:, 0] = edge_mask[:, -1] = True
        er, ec = np.unravel_index(
            np.argmax(np.where(edge_mask, acc_arr, 0.0)), acc_arr.shape
        )
        olon, olat = grid.affine * (ec + 0.5, er + 0.5)
        outlet = (float(olon), float(olat))
        print(f"  auto-detected outlet (max-acc on edge): "
              f"({outlet[0]:.4f}, {outlet[1]:.4f})  acc={acc_arr[er, ec]:.0f} px")

    # Snap outlet to nearest high-accumulation cell (used for reference even with mask).
    snap_pixels = max(min_pixels, int(25.0 / pixel_area_km2))
    lon, lat = outlet
    print(f"  snap outlet ({lon}, {lat})...")
    x_snap, y_snap = grid.snap_to_mask(acc > snap_pixels, (lon, lat))
    print(f"  snapped -> ({x_snap:.4f}, {y_snap:.4f})")

    if basin_mask_gdf is not None:
        # Use polygon boundary directly — supports multi-basin regions
        from rasterio.features import rasterize as _rasterize
        import geopandas as gpd
        _gdf = basin_mask_gdf.to_crs("EPSG:4326") if basin_mask_gdf.crs else basin_mask_gdf
        _shapes = [(geom, 1) for geom in _gdf.geometry if geom is not None]
        catch_arr = _rasterize(
            _shapes, out_shape=grid.shape, transform=grid.affine, fill=0,
            dtype="uint8", all_touched=False,
        ).astype(bool)
        print(f"  catchment cells (polygon mask): {catch_arr.sum()}", flush=True)
        # Wrap as pysheds-compatible object for downstream code
        catch = catch_arr
    else:
        # Delineate from outlet via D8 flow tracing
        print("  catchment delineation...")
        catch = grid.catchment(x=x_snap, y=y_snap, fdir=fdir, xytype="coordinate")
        catch_arr = np.asarray(catch).astype(bool)
        print(f"  catchment cells: {int(catch_arr.sum())}", flush=True)

    # Find pour points as D8 confluences within the catchment (guaranteed inside)
    pour_points = _find_pour_points(grid, fdir, acc, catch, min_pixels,
                                    max_points=max_subcatchments)
    print(f"  pour points retenus: {len(pour_points)}")

    if len(pour_points) == 0:
        # Fallback: single catchment
        pour_points = [(x_snap, y_snap)]

    # Assign each cell to nearest downstream pour point
    subcatch_labels, centroids, areas_km2 = _label_subcatchments(
        grid, fdir, acc, catch, pour_points, pixel_area_km2,
    )

    n_nodes = len(centroids)

    return {
        "grid": grid,
        "fdir": fdir,
        "acc": acc,
        "dem": dem,
        "labels": subcatch_labels,
        "centroids": centroids,  # (n_nodes, 2) [lon, lat]
        "areas_km2": areas_km2,  # (n_nodes,) local area
        "n_nodes": n_nodes,
        "pour_points": pour_points,
        "catch_mask": catch,
        "pixel_area_km2": pixel_area_km2,
    }


def _find_pour_points(
    grid,
    fdir,
    acc,
    catch_mask,
    min_pixels: int,
    max_points: int = 300,
) -> list[tuple[float, float]]:
    """Find D8 confluence pixels inside the catchment mask.

    A confluence is a catchment cell that receives flow from ≥2 other catchment
    cells.  All returned points are guaranteed to lie within catch_mask, so
    the BFS in _label_subcatchments can always find them.
    """
    fdir_arr = np.asarray(fdir)
    acc_arr  = np.asarray(acc)
    catch_arr = np.asarray(catch_mask).astype(bool)
    nrows, ncols = fdir_arr.shape
    affine = grid.affine

    # pysheds D8: N=64 NE=128 E=1 SE=2 S=4 SW=8 W=16 NW=32
    d8 = {64: (-1, 0), 128: (-1, 1), 1: (0, 1), 2: (1, 1),
           4: (1, 0),   8: (1, -1), 16: (0, -1), 32: (-1, -1)}

    # Count how many catchment cells flow into each catchment cell
    incoming = np.zeros((nrows, ncols), dtype=np.int16)
    rows_c, cols_c = np.where(catch_arr)
    for r, c in zip(rows_c.tolist(), cols_c.tolist()):
        d = int(fdir_arr[r, c])
        if d not in d8:
            continue
        dr, dc = d8[d]
        nr, nc = r + dr, c + dc
        if 0 <= nr < nrows and 0 <= nc < ncols and catch_arr[nr, nc]:
            incoming[nr, nc] += 1

    # Confluences: ≥2 incoming flows AND above the stream threshold
    conf_mask = catch_arr & (incoming >= 2) & (acc_arr >= min_pixels)
    conf_rows, conf_cols = np.where(conf_mask)

    if len(conf_rows) == 0:
        return []

    # Sort by accumulation descending, keep top max_points
    conf_acc = acc_arr[conf_rows, conf_cols]
    order = np.argsort(-conf_acc)[:max_points]

    pour_points = []
    for idx in order:
        r, c = int(conf_rows[idx]), int(conf_cols[idx])
        lon = affine.c + (c + 0.5) * affine.a
        lat = affine.f + (r + 0.5) * affine.e
        pour_points.append((lon, lat))

    return pour_points


def _label_subcatchments(
    grid, fdir, acc, catch_mask, pour_points, pixel_area_km2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Label each basin cell with the ID of its nearest downstream pour point.

    Uses BFS backwards along the reverse-flow graph.  Pour points are processed
    from most-downstream (highest accumulation) to most-upstream so each BFS
    flood-fill stops naturally when it hits cells already claimed by a more
    upstream pour point.  No calls to grid.catchment() are needed.
    """
    from collections import deque

    affine   = grid.affine
    fdir_arr = np.asarray(fdir)
    acc_arr  = np.asarray(acc)
    catch_arr = np.asarray(catch_mask).astype(bool)
    nrows, ncols = fdir_arr.shape

    # pysheds default dirmap: N=64 NE=128 E=1 SE=2 S=4 SW=8 W=16 NW=32
    d8 = {64: (-1,  0), 128: (-1,  1), 1: (0,  1), 2: (1,  1),
           4: ( 1,  0),   8: ( 1, -1), 16: (0, -1), 32: (-1, -1)}

    # Build reverse-flow map within the catchment: cell → upstream neighbours
    rev_map: dict[tuple[int,int], list[tuple[int,int]]] = {}
    rows_c, cols_c = np.where(catch_arr)
    for r, c in zip(rows_c.tolist(), cols_c.tolist()):
        d = int(fdir_arr[r, c])
        if d not in d8:
            continue
        dr, dc = d8[d]
        nr, nc = r + dr, c + dc
        if 0 <= nr < nrows and 0 <= nc < ncols and catch_arr[nr, nc]:
            rev_map.setdefault((nr, nc), []).append((r, c))

    # Convert pour points to pixel coords; keep only those inside the catchment
    def _to_pixel(px: float, py: float) -> tuple[int, int]:
        col = int((px - affine.c) / affine.a)
        row = int((py - affine.f) / affine.e)
        return max(0, min(row, nrows - 1)), max(0, min(col, ncols - 1))

    seen: set[tuple[int,int]] = set()
    pp_list: list[tuple[int, int, float]] = []   # (row, col, acc_value)
    for px, py in pour_points:
        r, c = _to_pixel(px, py)
        if catch_arr[r, c] and (r, c) not in seen:
            seen.add((r, c))
            pp_list.append((r, c, float(acc_arr[r, c])))

    # Always include the true outlet (max-acc cell) so downstream cells are covered
    max_idx = int(np.argmax(acc_arr * catch_arr.astype(np.float32)))
    outlet_r, outlet_c = np.unravel_index(max_idx, acc_arr.shape)
    if (outlet_r, outlet_c) not in seen:
        pp_list.append((outlet_r, outlet_c, float(acc_arr[outlet_r, outlet_c])))
        seen.add((outlet_r, outlet_c))

    if not pp_list:
        labels = catch_arr.astype(np.int32)
        centroid = np.array(pour_points[0:1]) if pour_points else np.zeros((1, 2))
        area = np.array([catch_arr.sum() * pixel_area_km2])
        return labels, centroid, area

    # Sort descending by accumulation (most downstream = highest acc first)
    pp_list.sort(key=lambda x: -x[2])

    labels = np.zeros((nrows, ncols), dtype=np.int32)
    # Pre-mark all pour-point pixels so BFS stops at them
    for label_id, (pr, pc, _) in enumerate(pp_list, start=1):
        labels[pr, pc] = label_id

    # BFS from each pour point going upstream; stop at already-labeled cells
    for label_id, (pr, pc, _) in enumerate(pp_list, start=1):
        queue = deque([(pr, pc)])
        while queue:
            r, c = queue.popleft()
            for nr, nc in rev_map.get((r, c), []):
                if catch_arr[nr, nc] and labels[nr, nc] == 0:
                    labels[nr, nc] = label_id
                    queue.append((nr, nc))

    # Any residual unlabeled cells → assign to most-downstream node (label 1)
    labels[catch_arr & (labels == 0)] = 1

    n_nodes = len(pp_list)
    centroids = np.zeros((n_nodes, 2))
    areas_km2 = np.zeros(n_nodes)

    # Vectorised centroid computation — one pass over catchment pixels only
    catch_rows, catch_cols = np.where(catch_arr)
    lbl_flat = labels[catch_rows, catch_cols]          # label for each catchment pixel
    valid    = lbl_flat > 0
    lbl0     = lbl_flat[valid] - 1                     # 0-indexed
    r_valid  = catch_rows[valid].astype(np.float64)
    c_valid  = catch_cols[valid].astype(np.float64)

    counts   = np.bincount(lbl0, minlength=n_nodes).astype(np.float64)
    sum_r    = np.bincount(lbl0, weights=r_valid, minlength=n_nodes)
    sum_c    = np.bincount(lbl0, weights=c_valid, minlength=n_nodes)

    for label_id, (pr, pc, _) in enumerate(pp_list, start=1):
        i = label_id - 1
        if counts[i] > 0:
            mean_c = sum_c[i] / counts[i]
            mean_r = sum_r[i] / counts[i]
            centroids[i, 0] = affine.c + (mean_c + 0.5) * affine.a   # lon
            centroids[i, 1] = affine.f + (mean_r + 0.5) * affine.e   # lat
        else:
            centroids[i] = [affine.c + (pc + 0.5) * affine.a,
                            affine.f + (pr + 0.5) * affine.e]
        areas_km2[i] = counts[i] * pixel_area_km2

    return labels, centroids, areas_km2


# ── Step 3: Build river network ─────────────────────────────────────────────


def _build_network(subcatchments: dict) -> tuple[RiverGraph, list[int], Tensor]:
    """Build a directed graph from subcatchment adjacency.

    Edge: subcatchment A → B if A drains into B (upstream → downstream).
    """
    labels = subcatchments["labels"]
    fdir = subcatchments["fdir"]
    grid = subcatchments["grid"]
    n_nodes = subcatchments["n_nodes"]

    # Build adjacency from flow direction at subcatchment boundaries
    edges = set()
    rows, cols = np.where(labels > 0)

    # D8 direction offsets (pysheds convention)
    # Map flow direction values to row/col offsets
    d8_offsets = {
        1: (0, 1),    # east
        2: (1, 1),    # southeast
        4: (1, 0),    # south
        8: (1, -1),   # southwest
        16: (0, -1),  # west
        32: (-1, -1), # northwest
        64: (-1, 0),  # north
        128: (-1, 1), # northeast
    }

    fdir_arr = np.asarray(fdir)
    label_arr = np.asarray(labels)

    for r, c in zip(rows, cols):
        src_label = label_arr[r, c]
        fd = int(fdir_arr[r, c])
        if fd not in d8_offsets:
            continue
        dr, dc = d8_offsets[fd]
        nr, nc = r + dr, c + dc
        if 0 <= nr < label_arr.shape[0] and 0 <= nc < label_arr.shape[1]:
            dst_label = label_arr[nr, nc]
            if dst_label > 0 and dst_label != src_label:
                edges.add((src_label - 1, dst_label - 1))  # 0-indexed

    # Build edge_index
    if edges:
        edge_list = sorted(edges)
        src = [e[0] for e in edge_list]
        dst = [e[1] for e in edge_list]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    n_edges = edge_index.shape[1]

    # Edge attributes (placeholder: length, width, slope — estimated later)
    edge_attr = torch.ones((n_edges, 3), dtype=torch.float32)
    # Travel time: 1 day per edge (rough default)
    travel_time_days = torch.ones(n_edges, dtype=torch.long)

    # Topological sort (Kahn's algorithm)
    topo_order = _topological_sort(edge_index, n_nodes)
    is_lake = torch.zeros(n_nodes, dtype=torch.bool)

    node_ids = list(range(1, n_nodes + 1))  # 1-indexed IDs

    graph = RiverGraph(
        edge_index=edge_index,
        edge_attr=edge_attr,
        topo_order=topo_order,
        is_lake=is_lake,
        travel_time_days=travel_time_days,
    )

    return graph, node_ids, is_lake


def _topological_sort(edge_index: Tensor, n_nodes: int) -> Tensor:
    """Kahn's algorithm for topological ordering."""
    in_degree = torch.zeros(n_nodes, dtype=torch.long)
    children: dict[int, list[int]] = collections.defaultdict(list)

    if edge_index.shape[1] > 0:
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        for s, d in zip(src, dst):
            children[s].append(d)
            in_degree[d] += 1

    queue = collections.deque()
    for i in range(n_nodes):
        if in_degree[i] == 0:
            queue.append(i)

    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for child in children[node]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    # If not all nodes reached, add remaining (disconnected components)
    visited = set(order)
    for i in range(n_nodes):
        if i not in visited:
            order.append(i)

    return torch.tensor(order, dtype=torch.long)


# ── Step 4: Zonal statistics ────────────────────────────────────────────────


def _read_resampled(
    path: Path,
    target_shape: tuple[int, int],
    target_transform,
    target_crs,
    resampling_method=None,
) -> np.ndarray:
    """Read a raster and resample it to *target_shape* / *target_transform*.

    Uses nearest-neighbour for integer/uint8 data (categorical),
    bilinear for float data.
    """
    import rasterio
    from rasterio.warp import reproject, Resampling

    with rasterio.open(str(path)) as src:
        data = src.read(1)
        src_transform = src.transform
        src_crs = src.crs or target_crs

    if data.shape == target_shape:
        return data   # already aligned

    if resampling_method is None:
        resampling_method = (
            Resampling.nearest if data.dtype.kind in ("u", "i") else Resampling.bilinear
        )

    dst = np.empty(target_shape, dtype=data.dtype)
    reproject(
        source=data,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=target_transform,
        dst_crs=target_crs,
        resampling=resampling_method,
    )
    return dst


def _compute_zonal_stats(
    subcatchments: dict,
    dem_path: Path,
    landcover_path: Path,
    soil_dir: Path,
    graph: RiverGraph,
    extra_stats: list[str] | None = None,
    water_occurrence_path: Path | None = None,
    lai_path: Path | None = None,
    nrcan_lc_path: Path | None = None,
    water_polygons_path: Path | None = None,
) -> tuple[Tensor, dict[str, Tensor], list[str]]:
    """Compute per-subcatchment zonal statistics from rasters."""
    from rasterstats import zonal_stats

    labels = subcatchments["labels"]
    n_nodes = subcatchments["n_nodes"]
    areas_km2 = subcatchments["areas_km2"]
    centroids = subcatchments["centroids"]
    grid = subcatchments["grid"]
    extra_stats = extra_stats or []

    # We'll use the label raster as zones
    affine = grid.affine

    # ── DEM statistics ──
    with rasterio.open(dem_path) as src:
        dem_data = src.read(1)
        dem_affine = src.transform
        dem_crs    = src.crs

    # Reference grid for resampling all other rasters
    _grid_shape = labels.shape
    _grid_tf    = dem_affine
    _grid_crs   = dem_crs

    # Compute slope from DEM
    slope_pct = _compute_slope(dem_data, dem_affine)

    # Build label index ONCE — all zonal helpers reuse it (O(catchment) per stat)
    _idx = _LabelIndex(labels, n_nodes)

    # Per-zone DEM stats
    elev_stats = _zonal_mean_per_label(labels, dem_data, n_nodes, _idx=_idx)
    slope_stats = _zonal_mean_per_label(labels, slope_pct, n_nodes, _idx=_idx)

    # Aspect (sin/cos for circular mean)
    aspect_rad = _compute_aspect(dem_data)
    sin_asp = _zonal_mean_per_label(labels, np.sin(aspect_rad), n_nodes, _idx=_idx)
    cos_asp = _zonal_mean_per_label(labels, np.cos(aspect_rad), n_nodes, _idx=_idx)

    # ── Land cover fractions (ESA WorldCover) ────────────────────────────────
    lc_data = _read_resampled(landcover_path, _grid_shape, _grid_tf, _grid_crs)
    lc_fracs = _landcover_fractions(labels, lc_data, n_nodes, _idx=_idx)

    # ── NRCan land cover (forest type + tourbières) ───────────────────────────
    nrcan_fracs: dict[str, np.ndarray] = {}
    if nrcan_lc_path is not None and Path(nrcan_lc_path).exists():
        print("  NRCan land cover...", flush=True)
        nrcan_data = _read_resampled(nrcan_lc_path, _grid_shape, _grid_tf, _grid_crs)
        nrcan_fracs = _landcover_fractions_nrcan(labels, nrcan_data, n_nodes, _idx=_idx)

    # ── Soil fractions ────────────────────────────────────────────────────────
    soil_fracs = _soil_fractions(labels, soil_dir, n_nodes, _idx=_idx)

    # ── Lake fraction: JRC permanent water (>75% occurrence) ∪ OSM polygons ──
    # JRC catches large permanent lakes/rivers but misses small ponds at 30 m.
    # OSM water polygons fill in small lakes, reservoirs, retention basins.
    jrc_mask = None
    if water_occurrence_path is not None and Path(water_occurrence_path).exists():
        print("  JRC surface water...", flush=True)
        water_occ = _read_resampled(
            water_occurrence_path, _grid_shape, _grid_tf, _grid_crs
        ).astype(np.float32)
        jrc_mask = water_occ > 75.0

    osm_mask = None
    if water_polygons_path is not None and Path(water_polygons_path).exists():
        try:
            import geopandas as gpd
            from rasterio.features import rasterize as _rasterize
            print("  OSM water polygons...", flush=True)
            wpoly = gpd.read_parquet(str(water_polygons_path))
            if wpoly.crs is None:
                wpoly = wpoly.set_crs("EPSG:4326")
            elif wpoly.crs.to_epsg() != 4326:
                wpoly = wpoly.to_crs("EPSG:4326")
            shapes = [(g, 1) for g in wpoly.geometry if g is not None and not g.is_empty]
            if shapes:
                osm_mask = _rasterize(
                    shapes, out_shape=_grid_shape, transform=_grid_tf,
                    fill=0, dtype="uint8", all_touched=False,
                ).astype(bool)
        except Exception as e:
            print(f"  OSM water polygons skipped ({e})")

    if jrc_mask is not None and osm_mask is not None:
        combined = jrc_mask | osm_mask
        lake_frac = _zonal_mean_per_label(
            labels, combined.astype(np.float32), n_nodes, _idx=_idx,
        )
        n_jrc_only = int((jrc_mask & ~osm_mask).sum())
        n_osm_only = int((osm_mask & ~jrc_mask).sum())
        print(f"  lake mask: JRC-only={n_jrc_only:,} OSM-only={n_osm_only:,} px")
    elif jrc_mask is not None:
        lake_frac = _zonal_mean_per_label(
            labels, jrc_mask.astype(np.float32), n_nodes, _idx=_idx,
        )
    elif osm_mask is not None:
        lake_frac = _zonal_mean_per_label(
            labels, osm_mask.astype(np.float32), n_nodes, _idx=_idx,
        )
    else:
        lake_frac = np.zeros(n_nodes, dtype=np.float32)

    # ── MODIS LAI → mean_lai ──────────────────────────────────────────────────
    if lai_path is not None and Path(lai_path).exists():
        print("  MODIS LAI...", flush=True)
        from rasterio.warp import Resampling as _RS
        lai_data = _read_resampled(
            lai_path, _grid_shape, _grid_tf, _grid_crs,
            resampling_method=_RS.bilinear,
        ).astype(np.float32)
        mean_lai = _zonal_mean_per_label(labels, lai_data, n_nodes, _idx=_idx)
    else:
        mean_lai = np.zeros(n_nodes, dtype=np.float32)

    # ── Network statistics ────────────────────────────────────────────────────
    cum_area = _cumulative_area(graph, areas_km2, n_nodes)
    strahler = _compute_strahler(graph, n_nodes)
    dist_km  = _dist_to_outlet(graph, centroids, n_nodes)

    # ── Build feature arrays ──────────────────────────────────────────────────
    feature_dict = collections.OrderedDict()
    feature_dict["drainage_area_km2"]   = cum_area
    feature_dict["strahler_order"]      = strahler.astype(np.float32)
    feature_dict["mean_slope_pct"]      = slope_stats
    feature_dict["mean_elevation_m"]    = elev_stats
    feature_dict["sin_aspect"]          = sin_asp
    feature_dict["cos_aspect"]          = cos_asp
    # Forest: NRCan if available (distinguishes conifer/deciduous), else ESA aggregate
    feature_dict["f_forest"]            = nrcan_fracs.get("forest",     lc_fracs["forest"])
    feature_dict["f_forest_conifer"]    = nrcan_fracs.get("conifer",    np.zeros(n_nodes, np.float32))
    feature_dict["f_forest_deciduous"]  = nrcan_fracs.get("deciduous",  np.zeros(n_nodes, np.float32))
    feature_dict["f_forest_mixed"]      = nrcan_fracs.get("mixed",      np.zeros(n_nodes, np.float32))
    feature_dict["f_agriculture"]       = nrcan_fracs.get("cropland",   lc_fracs["agriculture"])
    feature_dict["f_urban"]             = nrcan_fracs.get("urban",      lc_fracs["urban"])
    feature_dict["f_wetland"]           = nrcan_fracs.get("wetland",    lc_fracs["wetland"])
    feature_dict["f_peatland"]          = nrcan_fracs.get("peatland",   np.zeros(n_nodes, np.float32))
    feature_dict["f_water"]             = nrcan_fracs.get("water",      lc_fracs["water"])
    feature_dict["f_sand"]              = soil_fracs["sand"]
    feature_dict["f_silt"]              = soil_fracs["silt"]
    feature_dict["f_clay"]              = soil_fracs["clay"]

    # ── PTF (Saxton-Rawls 2006): texture → hydraulic parameters ──
    # Provides physically-grounded init for the NeRF spatial encoder
    # (porosity, FC, WP, Ksat). The NeRF can refine these from data.
    ptf = _saxton_rawls_2006(soil_fracs["sand"], soil_fracs["clay"])
    feature_dict["porosity"]            = ptf["porosity"]
    feature_dict["theta_fc"]            = ptf["theta_fc"]
    feature_dict["theta_wp"]            = ptf["theta_wp"]
    feature_dict["Ksat_m_day"]          = ptf["Ksat_m_day"]

    feature_dict["depth_to_bedrock_m"]  = np.zeros(n_nodes, dtype=np.float32)
    feature_dict["dist_to_outlet_km"]   = dist_km
    feature_dict["mean_lai"]            = mean_lai
    feature_dict["lake_fraction"]       = lake_frac

    # Extra stats
    if "elevation_std" in extra_stats:
        feature_dict["elevation_std"] = _zonal_std_per_label(
            labels, dem_data, n_nodes, _idx=_idx,
        )
    if "slope_p10" in extra_stats:
        feature_dict["slope_p10"] = _zonal_percentile_per_label(
            labels, slope_pct, n_nodes, 10, _idx=_idx,
        )
    if "slope_p90" in extra_stats:
        feature_dict["slope_p90"] = _zonal_percentile_per_label(
            labels, slope_pct, n_nodes, 90, _idx=_idx,
        )

    columns = list(feature_dict.keys())
    data = torch.tensor(
        np.stack([feature_dict[c] for c in columns], axis=-1),
        dtype=torch.float32,
    )

    # Physical columns (un-normalised)
    physical = {
        "area_km2_physical": torch.tensor(
            np.maximum(cum_area, 1e-3), dtype=torch.float32,
        ),
        "area_km2_local": torch.tensor(
            np.maximum(areas_km2, 1e-3), dtype=torch.float32,
        ),
        "slope_fraction": torch.tensor(
            np.maximum(slope_stats / 100.0, 1e-4), dtype=torch.float32,
        ),
    }

    return data, physical, columns


# ── Zonal helpers ────────────────────────────────────────────────────────────
#
# All helpers receive a _LabelIndex built ONCE in _compute_zonal_stats and
# operate only on the ~456k catchment pixels (not the full 158M raster).
# np.bincount replaces the O(n_nodes × raster_size) per-label loops.


class _LabelIndex:
    """Sparse index of catchment pixels sorted by label.

    Build once; reuse for every zonal stat so the cost is O(catchment_size)
    per stat instead of O(n_nodes × raster_size).
    """

    def __init__(self, labels: np.ndarray, n_nodes: int) -> None:
        flat = labels.ravel()
        valid = flat > 0                               # catchment pixels only
        self.flat_idx = np.where(valid)[0]             # flat indices into ravel()
        lbl = flat[valid]                              # labels of those pixels
        order = np.argsort(lbl, kind="stable")
        self.flat_idx = self.flat_idx[order]
        self.lbl0 = lbl[order] - 1                     # 0-indexed sorted labels
        self.boundaries = np.searchsorted(
            self.lbl0, np.arange(0, n_nodes + 1)
        )                                              # boundaries[i]:boundaries[i+1] → label i
        self.n_nodes = n_nodes
        self.counts = np.bincount(self.lbl0, minlength=n_nodes).astype(np.float64)

    def extract(self, arr: np.ndarray) -> np.ndarray:
        """Return catchment pixels sorted by label as a 1-D float64 array."""
        return arr.ravel()[self.flat_idx].astype(np.float64)


def _zonal_mean_per_label(
    labels: np.ndarray, values: np.ndarray, n_nodes: int,
    _idx: "_LabelIndex | None" = None,
) -> np.ndarray:
    """Mean of *values* per subcatchment label — vectorised via np.bincount."""
    if _idx is None:
        _idx = _LabelIndex(labels, n_nodes)
    v = _idx.extract(values)
    fin = np.isfinite(v)
    sums = np.bincount(_idx.lbl0, weights=np.where(fin, v, 0.0), minlength=n_nodes)
    cnts = np.bincount(_idx.lbl0, weights=fin.astype(np.float64), minlength=n_nodes)
    return np.where(cnts > 0, sums / cnts, 0.0).astype(np.float32)


def _zonal_std_per_label(
    labels: np.ndarray, values: np.ndarray, n_nodes: int,
    _idx: "_LabelIndex | None" = None,
) -> np.ndarray:
    if _idx is None:
        _idx = _LabelIndex(labels, n_nodes)
    v = _idx.extract(values)
    fin = np.isfinite(v)
    v_fin = np.where(fin, v, 0.0)
    cnts  = np.bincount(_idx.lbl0, weights=fin.astype(np.float64), minlength=n_nodes)
    means = np.where(cnts > 0,
                     np.bincount(_idx.lbl0, weights=v_fin, minlength=n_nodes) / np.maximum(cnts, 1),
                     0.0)
    dev2  = np.where(fin, (v - means[_idx.lbl0]) ** 2, 0.0)
    var   = np.where(cnts > 1,
                     np.bincount(_idx.lbl0, weights=dev2, minlength=n_nodes) / np.maximum(cnts - 1, 1),
                     0.0)
    return np.sqrt(var).astype(np.float32)


def _zonal_percentile_per_label(
    labels: np.ndarray, values: np.ndarray, n_nodes: int, pct: int,
    _idx: "_LabelIndex | None" = None,
) -> np.ndarray:
    if _idx is None:
        _idx = _LabelIndex(labels, n_nodes)
    v = _idx.extract(values)
    result = np.zeros(n_nodes, dtype=np.float32)
    for i in range(n_nodes):
        s, e = int(_idx.boundaries[i]), int(_idx.boundaries[i + 1])
        if e > s:
            chunk = v[s:e]
            fin = chunk[np.isfinite(chunk)]
            if len(fin):
                result[i] = np.percentile(fin, pct)
    return result


def _compute_slope(dem: np.ndarray, affine) -> np.ndarray:
    """Slope in percent from DEM using finite differences."""
    dy, dx = np.gradient(dem.astype(np.float64))
    # Convert pixel gradients to meters
    res_x = abs(affine.a) * 111_000  # degrees to meters
    res_y = abs(affine.e) * 111_000
    dx_m = dx / res_x if res_x > 0 else dx
    dy_m = dy / res_y if res_y > 0 else dy
    slope_rad = np.arctan(np.sqrt(dx_m**2 + dy_m**2))
    return (np.tan(slope_rad) * 100).astype(np.float32)  # percent


def _compute_aspect(dem: np.ndarray) -> np.ndarray:
    """Aspect in radians from DEM."""
    dy, dx = np.gradient(dem.astype(np.float64))
    aspect = np.arctan2(-dy, dx)  # radians, 0 = east
    return aspect.astype(np.float32)


def _landcover_fractions(
    labels: np.ndarray, lc: np.ndarray, n_nodes: int,
    _idx: "_LabelIndex | None" = None,
) -> dict[str, np.ndarray]:
    """ESA WorldCover class fractions per subcatchment — vectorised via np.bincount."""
    if _idx is None:
        _idx = _LabelIndex(labels, n_nodes)
    class_map = {
        "forest": [10],
        "agriculture": [40],
        "urban": [50],
        "wetland": [90, 95],
        "water": [80],
    }
    lc_flat = _idx.extract(lc.astype(np.int32))
    fracs: dict[str, np.ndarray] = {}
    for name, classes in class_map.items():
        hits = np.isin(lc_flat, classes).astype(np.float64)
        counts = np.bincount(_idx.lbl0, weights=hits, minlength=n_nodes)
        fracs[name] = np.where(_idx.counts > 0, counts / _idx.counts, 0.0).astype(np.float32)
    return fracs


def _landcover_fractions_nrcan(
    labels: np.ndarray, lc: np.ndarray, n_nodes: int,
    _idx: "_LabelIndex | None" = None,
) -> dict[str, np.ndarray]:
    """NRCan Annual Land Cover class fractions per subcatchment — vectorised via np.bincount."""
    if _idx is None:
        _idx = _LabelIndex(labels, n_nodes)
    class_map = {
        "conifer":   [1, 2],
        "deciduous": [5],
        "mixed":     [6],
        "shrubland": [8],
        "wetland":   [14],
        "peatland":  [14],
        "cropland":  [15],
        "urban":     [17],
        "water":     [18],
    }
    forest_classes = [1, 2, 5, 6]
    lc_flat = _idx.extract(lc.astype(np.int32))
    fracs: dict[str, np.ndarray] = {}
    for name, classes in class_map.items():
        hits = np.isin(lc_flat, classes).astype(np.float64)
        counts = np.bincount(_idx.lbl0, weights=hits, minlength=n_nodes)
        fracs[name] = np.where(_idx.counts > 0, counts / _idx.counts, 0.0).astype(np.float32)
    hits_f = np.isin(lc_flat, forest_classes).astype(np.float64)
    counts_f = np.bincount(_idx.lbl0, weights=hits_f, minlength=n_nodes)
    fracs["forest"] = np.where(_idx.counts > 0, counts_f / _idx.counts, 0.0).astype(np.float32)
    return fracs


def _zonal_fraction_threshold(
    labels: np.ndarray,
    values: np.ndarray,
    n_nodes: int,
    threshold: float,
    _idx: "_LabelIndex | None" = None,
) -> np.ndarray:
    """Fraction of pixels in each zone where *values* >= *threshold* — vectorised."""
    if _idx is None:
        _idx = _LabelIndex(labels, n_nodes)
    v = _idx.extract(values)
    hits = (v >= threshold).astype(np.float64)
    counts = np.bincount(_idx.lbl0, weights=hits, minlength=n_nodes)
    return np.where(_idx.counts > 0, counts / _idx.counts, 0.0).astype(np.float32)


def _saxton_rawls_2006(
    sand: np.ndarray,
    clay: np.ndarray,
    om_pct: float = 2.5,
) -> dict[str, np.ndarray]:
    """Pedotransfer functions of Saxton & Rawls (2006).

    Converts soil texture (sand/clay fractions) into hydraulic parameters
    used by van Genuchten and meandre's vertical column.

    Parameters
    ----------
    sand, clay :
        Sand and clay fractions by mass (0–1).
    om_pct :
        Organic matter content in % by mass (default 2.5 % — typical for
        temperate forested mineral soils).

    Returns
    -------
    Dict with keys ``porosity`` (m³/m³), ``theta_fc`` (m³/m³),
    ``theta_wp`` (m³/m³), and ``Ksat_m_day`` (m/day).

    Notes
    -----
    Reference: Saxton, K.E. & Rawls, W.J. (2006), "Soil water
    characteristic estimates by texture and organic matter for hydrologic
    solutions", *Soil Sci. Soc. Am. J.* 70(5):1569–1578.
    """
    S = sand.astype(np.float32)
    C = clay.astype(np.float32)
    OM = np.float32(om_pct)

    # Wilting point (-1500 kPa)
    theta_1500t = (-0.024 * S + 0.487 * C + 0.006 * OM
                   + 0.005 * S * OM - 0.013 * C * OM
                   + 0.068 * S * C + 0.031)
    theta_wp = theta_1500t + (0.14 * theta_1500t - 0.02)
    theta_wp = np.clip(theta_wp, 0.02, 0.40)

    # Field capacity (-33 kPa)
    theta_33t = (-0.251 * S + 0.195 * C + 0.011 * OM
                 + 0.006 * S * OM - 0.027 * C * OM
                 + 0.452 * S * C + 0.299)
    theta_33 = theta_33t + (1.283 * theta_33t ** 2 - 0.374 * theta_33t - 0.015)
    theta_fc = np.clip(theta_33, theta_wp + 0.02, 0.55)

    # Saturated water content = porosity
    theta_S33t = (0.278 * S + 0.034 * C + 0.022 * OM
                  - 0.018 * S * OM - 0.027 * C * OM
                  - 0.584 * S * C + 0.078)
    theta_S33 = theta_S33t + (0.636 * theta_S33t - 0.107)
    theta_S = theta_fc + theta_S33 - 0.097 * S + 0.043
    porosity = np.clip(theta_S, theta_fc + 0.01, 0.65)

    # Saturated K via Brooks-Corey lambda fitted on (theta_wp, theta_fc)
    log_ratio_kpa = np.float32(np.log(1500.0) - np.log(33.0))
    log_theta_diff = (
        np.log(np.maximum(theta_fc, 1e-3))
        - np.log(np.maximum(theta_wp, 1e-3))
    )
    log_theta_diff = np.where(np.abs(log_theta_diff) > 1e-3,
                              log_theta_diff,
                              np.float32(1e-3))
    B = log_ratio_kpa / log_theta_diff          # Brooks-Corey b
    lam = 1.0 / B                                # pore size index
    Ksat_mm_h = 1930.0 * np.maximum(porosity - theta_fc, 1e-4) ** (3.0 - lam)
    Ksat_m_day = Ksat_mm_h * 24.0 * 1e-3
    Ksat_m_day = np.clip(Ksat_m_day, 1e-5, 50.0)  # bound to physical range

    return {
        "porosity":   porosity.astype(np.float32),
        "theta_fc":   theta_fc.astype(np.float32),
        "theta_wp":   theta_wp.astype(np.float32),
        "Ksat_m_day": Ksat_m_day.astype(np.float32),
    }


def _soil_fractions(
    labels: np.ndarray, soil_dir: Path, n_nodes: int,
    _idx: "_LabelIndex | None" = None,
) -> dict[str, np.ndarray]:
    """Sand/silt/clay fractions from SoilGrids GeoTIFFs."""
    fracs = {}
    for name in ["sand", "silt", "clay"]:
        path = soil_dir / f"{name}.tif"
        if path.exists():
            with rasterio.open(path) as src:
                data = src.read(1).astype(np.float32)
                # SoilGrids values are in g/kg -> convert to fraction
                data = data / 1000.0
                # Resample to label grid if needed (simple nearest-neighbor)
                if data.shape != labels.shape:
                    from scipy.ndimage import zoom
                    zoom_factors = (
                        labels.shape[0] / data.shape[0],
                        labels.shape[1] / data.shape[1],
                    )
                    data = zoom(data, zoom_factors, order=0)
                fracs[name] = _zonal_mean_per_label(labels, data, n_nodes, _idx=_idx)
        else:
            fracs[name] = np.zeros(n_nodes, dtype=np.float32)
    return fracs


def _cumulative_area(
    graph: RiverGraph, local_areas: np.ndarray, n_nodes: int,
) -> np.ndarray:
    """Accumulate drainage area downstream through the graph."""
    cum = local_areas.copy().astype(np.float32)

    # Build children map
    if graph.n_edges > 0:
        ei = graph.edge_index.cpu().numpy()
        # Process in reverse topological order (upstream first)
        topo = graph.topo_order.cpu().numpy()
        for node in topo:
            # Find edges where node is source
            mask = ei[0] == node
            for dst in ei[1][mask]:
                cum[dst] += cum[node]

    return cum


def _compute_strahler(graph: RiverGraph, n_nodes: int) -> np.ndarray:
    """Strahler stream order via bottom-up propagation."""
    order = np.ones(n_nodes, dtype=np.int32)

    if graph.n_edges == 0:
        return order

    ei = graph.edge_index.cpu().numpy()
    children: dict[int, list[int]] = collections.defaultdict(list)
    for s, d in zip(ei[0], ei[1]):
        children[d].append(s)  # d's upstream is s

    # Process in reverse topological order
    topo = graph.topo_order.cpu().numpy()
    for node in reversed(topo):
        upstream = children.get(node, [])
        if not upstream:
            order[node] = 1
        else:
            max_order = max(order[u] for u in upstream)
            count_max = sum(1 for u in upstream if order[u] == max_order)
            if count_max >= 2:
                order[node] = max_order + 1
            else:
                order[node] = max_order

    return order


def _dist_to_outlet(
    graph: RiverGraph, centroids: np.ndarray, n_nodes: int,
) -> np.ndarray:
    """Euclidean distance to outlet along the river network (km)."""
    dist = np.zeros(n_nodes, dtype=np.float32)

    if graph.n_edges == 0 or n_nodes == 0:
        return dist

    ei = graph.edge_index.cpu().numpy()
    topo = graph.topo_order.cpu().numpy()

    # Process downstream (topological order, last = outlet)
    for node in reversed(topo):
        mask = ei[0] == node
        for dst in ei[1][mask]:
            # Haversine-ish distance between centroids
            dlon = centroids[node, 0] - centroids[dst, 0]
            dlat = centroids[node, 1] - centroids[dst, 1]
            d_km = math.sqrt(dlon**2 + dlat**2) * 111.0  # rough deg→km
            dist[node] = dist[dst] + d_km

    return dist
