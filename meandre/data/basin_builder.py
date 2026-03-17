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
    outlet: tuple[float, float],
    basin_db: str | Path,
    min_area_km2: float = 2.0,
    extra_stats: list[str] | None = None,
    normalise: bool = True,
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
        (lon, lat) of the basin outlet in EPSG:4326.
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
    grid_data = _condition_dem(dem_path)

    # Step 2: Delineate subcatchments
    print("[basin_builder] Step 2: Delineating subcatchments...")
    subcatchments = _delineate_subcatchments(
        grid_data, outlet, min_area_km2=min_area_km2,
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
    )

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


def _condition_dem(dem_path: Path) -> dict:
    """Fill depressions, compute flow direction and accumulation."""
    from pysheds.grid import Grid

    grid = Grid.from_raster(str(dem_path))
    dem = grid.read_raster(str(dem_path))

    # Fill depressions
    pit_filled = grid.fill_pits(dem)
    flooded = grid.fill_depressions(pit_filled)
    inflated = grid.resolve_flats(flooded)

    # D8 flow direction
    fdir = grid.flowdir(inflated)

    # Flow accumulation
    acc = grid.accumulation(fdir)

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
    outlet: tuple[float, float],
    min_area_km2: float = 2.0,
) -> dict:
    """Delineate subcatchments from flow accumulation threshold."""
    grid = grid_data["grid"]
    fdir = grid_data["fdir"]
    acc = grid_data["acc"]
    dem = grid_data["dem"]

    # Convert min_area_km2 to pixel count (approximate)
    # Copernicus DEM 30m: ~30m resolution → ~900 m² per pixel
    res_m = abs(grid.affine.a) * 111_000  # degrees to meters (approximate)
    pixel_area_km2 = (res_m ** 2) / 1e6
    min_pixels = max(int(min_area_km2 / pixel_area_km2), 100)

    # Snap outlet to nearest high-accumulation cell
    lon, lat = outlet
    x_snap, y_snap = grid.snap_to_mask(acc > min_pixels, (lon, lat))

    # Delineate full basin from outlet
    catch = grid.catchment(x=x_snap, y=y_snap, fdir=fdir, xytype="coordinate")

    # Extract stream network at threshold
    branches = grid.extract_river_network(
        fdir, acc > min_pixels,
    )

    # Build subcatchments by labeling pour points along the stream network
    # Use stream junction/confluence points as pour points
    pour_points = _find_pour_points(branches, grid, acc, min_pixels)

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
        "branches": branches,
        "pixel_area_km2": pixel_area_km2,
    }


def _find_pour_points(branches, grid, acc, min_pixels: int) -> list[tuple[float, float]]:
    """Extract stream junction / confluence points from the river network."""
    if not branches or "features" not in branches:
        return []

    # Collect all coordinate endpoints
    endpoints = []
    for feature in branches["features"]:
        coords = feature["geometry"]["coordinates"]
        if coords:
            endpoints.append(coords[0])   # start
            endpoints.append(coords[-1])   # end

    # Count occurrences — junctions appear multiple times
    from collections import Counter
    point_counts = Counter()
    for pt in endpoints:
        # Round to grid resolution to handle floating-point noise
        key = (round(pt[0], 6), round(pt[1], 6))
        point_counts[key] += 1

    # Junctions: points that appear more than once (confluences)
    # Plus all endpoints (to ensure coverage)
    pour_points = list(point_counts.keys())

    return pour_points


def _label_subcatchments(
    grid, fdir, acc, catch_mask, pour_points, pixel_area_km2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Label each cell with the nearest downstream pour point.

    Simple approach: compute individual catchments for each pour point,
    then assign cells to the smallest catchment they belong to (most local).
    """
    n_pour = len(pour_points)

    # Compute catchment for each pour point
    catchments = []
    valid_points = []
    for i, (px, py) in enumerate(pour_points):
        try:
            c = grid.catchment(x=px, y=py, fdir=fdir, xytype="coordinate")
            c_arr = c.astype(bool) & catch_mask.astype(bool)
            area = c_arr.sum()
            if area > 0:
                catchments.append(c_arr)
                valid_points.append(i)
        except Exception:
            continue

    if not catchments:
        # Fallback: single catchment
        labels = catch_mask.astype(np.int32)
        labels[labels > 0] = 1
        centroid = np.array(pour_points[0:1]) if pour_points else np.array([[0.0, 0.0]])
        area = np.array([catch_mask.sum() * pixel_area_km2])
        return labels, centroid, area

    # Assign each cell to the smallest containing catchment (most local)
    # Sort catchments by area (smallest first)
    areas = [c.sum() for c in catchments]
    order = np.argsort(areas)

    # Labels array: 0 = no catchment, 1..N = subcatchment ID
    labels = np.zeros(catchments[0].shape, dtype=np.int32)
    for rank, idx in enumerate(order):
        mask = catchments[idx]
        labels[mask] = rank + 1  # 1-indexed

    n_nodes = len(catchments)

    # Compute centroids and areas
    # Get affine transform for coordinate conversion
    affine = grid.affine
    centroids = np.zeros((n_nodes, 2))  # [lon, lat]
    areas_km2 = np.zeros(n_nodes)

    for rank, idx in enumerate(order):
        node_mask = labels == (rank + 1)
        rows, cols = np.where(node_mask)
        if len(rows) == 0:
            continue
        # Convert pixel coords to geographic coords
        lons = affine.c + (cols + 0.5) * affine.a
        lats = affine.f + (rows + 0.5) * affine.e
        centroids[rank] = [lons.mean(), lats.mean()]
        areas_km2[rank] = len(rows) * pixel_area_km2

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


def _compute_zonal_stats(
    subcatchments: dict,
    dem_path: Path,
    landcover_path: Path,
    soil_dir: Path,
    graph: RiverGraph,
    extra_stats: list[str] | None = None,
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

    # Compute slope from DEM
    slope_pct = _compute_slope(dem_data, dem_affine)

    # Per-zone DEM stats
    elev_stats = _zonal_mean_per_label(labels, dem_data, n_nodes)
    slope_stats = _zonal_mean_per_label(labels, slope_pct, n_nodes)

    # Aspect (sin/cos for circular mean)
    aspect_rad = _compute_aspect(dem_data)
    sin_asp = _zonal_mean_per_label(labels, np.sin(aspect_rad), n_nodes)
    cos_asp = _zonal_mean_per_label(labels, np.cos(aspect_rad), n_nodes)

    # ── Land cover fractions ──
    with rasterio.open(landcover_path) as src:
        lc_data = src.read(1)

    # ESA WorldCover classes → fractions
    lc_fracs = _landcover_fractions(labels, lc_data, n_nodes)

    # ── Soil fractions ──
    soil_fracs = _soil_fractions(labels, soil_dir, n_nodes)

    # ── Network statistics ──
    cum_area = _cumulative_area(graph, areas_km2, n_nodes)
    strahler = _compute_strahler(graph, n_nodes)
    dist_km = _dist_to_outlet(graph, centroids, n_nodes)
    lake_frac = np.zeros(n_nodes)  # no lake detection from DEM alone

    # ── Build feature arrays ──
    feature_dict = collections.OrderedDict()
    feature_dict["drainage_area_km2"] = cum_area
    feature_dict["strahler_order"] = strahler.astype(np.float32)
    feature_dict["mean_slope_pct"] = slope_stats
    feature_dict["mean_elevation_m"] = elev_stats
    feature_dict["sin_aspect"] = sin_asp
    feature_dict["cos_aspect"] = cos_asp
    feature_dict["f_forest"] = lc_fracs["forest"]
    feature_dict["f_agriculture"] = lc_fracs["agriculture"]
    feature_dict["f_urban"] = lc_fracs["urban"]
    feature_dict["f_wetland"] = lc_fracs["wetland"]
    feature_dict["f_water"] = lc_fracs["water"]
    feature_dict["f_sand"] = soil_fracs["sand"]
    feature_dict["f_silt"] = soil_fracs["silt"]
    feature_dict["f_clay"] = soil_fracs["clay"]
    feature_dict["depth_to_bedrock_m"] = np.zeros(n_nodes, dtype=np.float32)
    feature_dict["dist_to_outlet_km"] = dist_km
    feature_dict["lake_fraction"] = lake_frac.astype(np.float32)

    # Extra stats
    if "elevation_std" in extra_stats:
        feature_dict["elevation_std"] = _zonal_std_per_label(
            labels, dem_data, n_nodes,
        )
    if "slope_p10" in extra_stats:
        feature_dict["slope_p10"] = _zonal_percentile_per_label(
            labels, slope_pct, n_nodes, 10,
        )
    if "slope_p90" in extra_stats:
        feature_dict["slope_p90"] = _zonal_percentile_per_label(
            labels, slope_pct, n_nodes, 90,
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


def _zonal_mean_per_label(
    labels: np.ndarray, values: np.ndarray, n_nodes: int,
) -> np.ndarray:
    """Mean of *values* per subcatchment label (1-indexed)."""
    result = np.zeros(n_nodes, dtype=np.float32)
    for i in range(n_nodes):
        mask = labels == (i + 1)
        if mask.any():
            vals = values[mask]
            valid = np.isfinite(vals)
            if valid.any():
                result[i] = vals[valid].mean()
    return result


def _zonal_std_per_label(
    labels: np.ndarray, values: np.ndarray, n_nodes: int,
) -> np.ndarray:
    """Standard deviation of *values* per subcatchment label."""
    result = np.zeros(n_nodes, dtype=np.float32)
    for i in range(n_nodes):
        mask = labels == (i + 1)
        if mask.any():
            vals = values[mask]
            valid = np.isfinite(vals)
            if valid.sum() > 1:
                result[i] = vals[valid].std()
    return result


def _zonal_percentile_per_label(
    labels: np.ndarray, values: np.ndarray, n_nodes: int, pct: int,
) -> np.ndarray:
    """Percentile of *values* per subcatchment label."""
    result = np.zeros(n_nodes, dtype=np.float32)
    for i in range(n_nodes):
        mask = labels == (i + 1)
        if mask.any():
            vals = values[mask]
            valid = np.isfinite(vals)
            if valid.any():
                result[i] = np.percentile(vals[valid], pct)
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
) -> dict[str, np.ndarray]:
    """ESA WorldCover class fractions per subcatchment."""
    # ESA WorldCover classes
    class_map = {
        "forest": [10],         # Tree cover
        "agriculture": [40],    # Cropland
        "urban": [50],          # Built-up
        "wetland": [90, 95],    # Herbaceous wetland + Mangroves
        "water": [80],          # Permanent water
    }

    fracs = {name: np.zeros(n_nodes, dtype=np.float32) for name in class_map}

    for i in range(n_nodes):
        mask = labels == (i + 1)
        total = mask.sum()
        if total == 0:
            continue
        lc_zone = lc[mask]
        for name, classes in class_map.items():
            count = sum((lc_zone == c).sum() for c in classes)
            fracs[name][i] = count / total

    return fracs


def _soil_fractions(
    labels: np.ndarray, soil_dir: Path, n_nodes: int,
) -> dict[str, np.ndarray]:
    """Sand/silt/clay fractions from SoilGrids GeoTIFFs."""
    fracs = {}
    for name in ["sand", "silt", "clay"]:
        path = soil_dir / f"{name}.tif"
        if path.exists():
            with rasterio.open(path) as src:
                data = src.read(1).astype(np.float32)
                # SoilGrids values are in g/kg → convert to fraction
                data = data / 1000.0
                # Resample to label grid if needed (simple nearest-neighbor)
                if data.shape != labels.shape:
                    from scipy.ndimage import zoom
                    zoom_factors = (
                        labels.shape[0] / data.shape[0],
                        labels.shape[1] / data.shape[1],
                    )
                    data = zoom(data, zoom_factors, order=0)
                fracs[name] = _zonal_mean_per_label(labels, data, n_nodes)
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
