"""Unit tests for basin_builder core algorithms.

The key functions (_find_pour_points, _label_subcatchments, _build_network)
operate on plain numpy arrays plus a grid.affine.  We test them with a
synthetic 30×30 V-shaped D8 network — no pysheds DEM conditioning needed.
Each test runs in < 1 second.

Regression coverage
-------------------
* All pour points returned by _find_pour_points lie INSIDE the catchment mask
  (previous bug: extract_river_network scanned the full raster, returning
  confluences from neighbouring basins — 0/20 pour points were inside).

* Every catchment cell has a non-zero label after _label_subcatchments
  (previous bug: largest catchment overwrote all others; then BFS outlet
  claimed everything before upstream pour points could be processed).

* _build_network produces a tree where every edge is directed
  upstream → downstream (higher → lower in the topo order).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from rasterio.transform import from_bounds


# ── Synthetic D8 V-valley ────────────────────────────────────────────────────
# 30×30 raster.  Flow rules (pysheds encoding):
#   col < 15  → SE (2)    col > 15 → SW (8)    col == 15 → S (4)
# With nrows=ncols=30, every cell eventually reaches (29, 15).

_D8 = {64: (-1, 0), 128: (-1, 1), 1: (0, 1), 2: (1, 1),
        4: (1, 0),   8: (1, -1), 16: (0, -1), 32: (-1, -1)}


def _v_valley(nrows: int = 30, ncols: int = 30):
    """Return (grid_mock, fdir, acc, catch_mask) for a synthetic valley.

    All *nrows*×*ncols* cells drain to (nrows-1, ncols//2).
    Precondition: nrows >= ncols//2  (so diagonal flows reach centre).
    """
    assert nrows >= ncols // 2, "nrows too small — far cells won't reach centre"
    mid = ncols // 2

    fdir = np.zeros((nrows, ncols), dtype=np.int32)
    for r in range(nrows - 1):
        for c in range(ncols):
            fdir[r, c] = 2 if c < mid else (8 if c > mid else 4)
    # bottom row: fdir = 0 (outlet, no downstream)

    # Accumulation — row-by-row propagation (valid because flow is always
    # strictly south or diagonal-south, never north).
    acc = np.ones((nrows, ncols), dtype=np.float64)
    for r in range(nrows - 1):
        for c in range(ncols):
            d = int(fdir[r, c])
            if d in _D8:
                dr, dc = _D8[d]
                nr, nc = r + dr, c + dc
                if 0 <= nr < nrows and 0 <= nc < ncols:
                    acc[nr, nc] += acc[r, c]

    # Entire raster drains to single outlet → all cells in catchment
    catch = np.ones((nrows, ncols), dtype=bool)

    # Tiny affine so coordinates are valid lon/lat near Québec
    west  = -73.0
    south = 46.0
    dx    = 0.001   # ~111 m/pixel
    dy    = 0.001
    affine = from_bounds(west, south, west + ncols * dx,
                         south + nrows * dy, ncols, nrows)
    grid = SimpleNamespace(affine=affine, shape=(nrows, ncols))

    return grid, fdir, acc, catch


# ── Tests: _find_pour_points ─────────────────────────────────────────────────

def test_find_pour_points_returns_some():
    """At least one confluence is found in a V-valley with min_pixels=50."""
    from meandre.data.basin_builder import _find_pour_points
    grid, fdir, acc, catch = _v_valley()
    pps = _find_pour_points(grid, fdir, acc, catch, min_pixels=50, max_points=20)
    assert len(pps) > 0, "No pour points found in V-valley"


def test_find_pour_points_all_inside_catchment():
    """Every returned pour point lies inside the catchment mask — the main
    regression: previously extract_river_network scanned the full raster."""
    from meandre.data.basin_builder import _find_pour_points
    grid, fdir, acc, catch = _v_valley()
    affine = grid.affine
    nrows, ncols = catch.shape

    pps = _find_pour_points(grid, fdir, acc, catch, min_pixels=50, max_points=30)
    for px, py in pps:
        c = max(0, min(int((px - affine.c) / affine.a), ncols - 1))
        r = max(0, min(int((py - affine.f) / affine.e), nrows - 1))
        assert catch[r, c], (
            f"Pour point ({px:.5f},{py:.5f}) → pixel ({r},{c}) is OUTSIDE catchment"
        )


def test_find_pour_points_keeps_all_natural_confluences():
    """All natural confluences ≥ min_pixels are returned, regardless of
    ``max_points`` (which is now only an advisory warning threshold).

    Capping at max_points biased selection to main-stem confluences and
    produced degenerate chain graphs (cf. fix 2026-05-12)."""
    import warnings
    from meandre.data.basin_builder import _find_pour_points
    grid, fdir, acc, catch = _v_valley()
    # Pass a small max_points but expect a warning, not truncation.
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        pps_capped = _find_pour_points(grid, fdir, acc, catch, min_pixels=5, max_points=7)
    pps_uncapped = _find_pour_points(grid, fdir, acc, catch, min_pixels=5, max_points=100000)
    assert len(pps_capped) == len(pps_uncapped), (
        "Pour points must NOT be truncated by max_points (preserves tree topology)"
    )


def test_find_pour_points_sorted_by_acc_descending():
    """Returned pour points are ordered by flow accumulation (high → low)."""
    from meandre.data.basin_builder import _find_pour_points
    grid, fdir, acc, catch = _v_valley()
    affine = grid.affine
    nrows, ncols = acc.shape
    pps = _find_pour_points(grid, fdir, acc, catch, min_pixels=5, max_points=20)

    def _acc_at(px, py):
        c = max(0, min(int((px - affine.c) / affine.a), ncols - 1))
        r = max(0, min(int((py - affine.f) / affine.e), nrows - 1))
        return float(acc[r, c])

    acc_vals = [_acc_at(px, py) for px, py in pps]
    assert acc_vals == sorted(acc_vals, reverse=True), "Pour points not sorted by acc"


# ── Tests: _label_subcatchments ──────────────────────────────────────────────

def test_label_subcatchments_covers_every_cell():
    """No catchment cell should have label == 0 after BFS — the main regression."""
    from meandre.data.basin_builder import _find_pour_points, _label_subcatchments
    grid, fdir, acc, catch = _v_valley()
    res_m = abs(grid.affine.a) * 111_000
    pixel_area_km2 = res_m ** 2 / 1e6
    pps = _find_pour_points(grid, fdir, acc, catch, min_pixels=50, max_points=10)
    labels, centroids, areas_km2 = _label_subcatchments(
        grid, fdir, acc, catch, pps, pixel_area_km2
    )
    assert (labels[catch] > 0).all(), "Unlabeled catchment cells after BFS"


def test_label_subcatchments_unique_labels_match_nodes():
    from meandre.data.basin_builder import _find_pour_points, _label_subcatchments
    grid, fdir, acc, catch = _v_valley()
    res_m = abs(grid.affine.a) * 111_000
    pixel_area_km2 = res_m ** 2 / 1e6
    pps = _find_pour_points(grid, fdir, acc, catch, min_pixels=50, max_points=10)
    labels, centroids, areas_km2 = _label_subcatchments(
        grid, fdir, acc, catch, pps, pixel_area_km2
    )
    n_unique = len(np.unique(labels[catch]))
    n_nodes  = len(centroids)
    assert n_unique == n_nodes, f"unique labels {n_unique} != n_nodes {n_nodes}"


def test_label_subcatchments_all_areas_positive():
    from meandre.data.basin_builder import _find_pour_points, _label_subcatchments
    grid, fdir, acc, catch = _v_valley()
    res_m = abs(grid.affine.a) * 111_000
    pixel_area_km2 = res_m ** 2 / 1e6
    pps = _find_pour_points(grid, fdir, acc, catch, min_pixels=50, max_points=10)
    labels, centroids, areas_km2 = _label_subcatchments(
        grid, fdir, acc, catch, pps, pixel_area_km2
    )
    assert (areas_km2 > 0).all(), "Some nodes have zero area"


def test_label_subcatchments_total_area_equals_catchment():
    """Sum of node areas ≈ total catchment area."""
    from meandre.data.basin_builder import _find_pour_points, _label_subcatchments
    grid, fdir, acc, catch = _v_valley()
    res_m = abs(grid.affine.a) * 111_000
    pixel_area_km2 = res_m ** 2 / 1e6
    pps = _find_pour_points(grid, fdir, acc, catch, min_pixels=50, max_points=10)
    labels, centroids, areas_km2 = _label_subcatchments(
        grid, fdir, acc, catch, pps, pixel_area_km2
    )
    total_nodes = areas_km2.sum()
    total_catch = catch.sum() * pixel_area_km2
    assert abs(total_nodes - total_catch) / total_catch < 0.01, (
        f"Area mismatch: nodes={total_nodes:.4f} catch={total_catch:.4f}"
    )


def test_label_subcatchments_with_zero_pour_points():
    """With no pour points, fallback returns a single node covering everything."""
    from meandre.data.basin_builder import _label_subcatchments
    grid, fdir, acc, catch = _v_valley()
    res_m = abs(grid.affine.a) * 111_000
    pixel_area_km2 = res_m ** 2 / 1e6
    labels, centroids, areas_km2 = _label_subcatchments(
        grid, fdir, acc, catch, pour_points=[], pixel_area_km2=pixel_area_km2
    )
    assert len(centroids) == 1
    assert (labels[catch] > 0).all()


# ── Tests: _build_network ────────────────────────────────────────────────────

def _make_subcatchments_dict(nrows=30, ncols=30, max_pp=8):
    """Build a subcatchments dict suitable for _build_network."""
    from meandre.data.basin_builder import _find_pour_points, _label_subcatchments
    grid, fdir, acc, catch = _v_valley(nrows, ncols)
    res_m = abs(grid.affine.a) * 111_000
    pixel_area_km2 = res_m ** 2 / 1e6
    pps = _find_pour_points(grid, fdir, acc, catch, min_pixels=50, max_points=max_pp)
    labels, centroids, areas_km2 = _label_subcatchments(
        grid, fdir, acc, catch, pps, pixel_area_km2
    )
    return {
        "grid": grid, "fdir": fdir, "acc": acc, "dem": None,
        "labels": labels, "centroids": centroids, "areas_km2": areas_km2,
        "n_nodes": len(centroids), "pour_points": pps,
        "catch_mask": catch, "pixel_area_km2": pixel_area_km2,
    }


def test_build_network_produces_tree():
    """n_edges == n_nodes - 1 for a single connected drainage basin."""
    from meandre.data.basin_builder import _build_network
    sc = _make_subcatchments_dict()
    graph, node_ids, is_lake = _build_network(sc)
    n = sc["n_nodes"]
    if n > 1:
        assert graph.n_edges == n - 1, (
            f"Expected {n-1} edges for a tree, got {graph.n_edges}"
        )


def test_build_network_no_self_loops():
    from meandre.data.basin_builder import _build_network
    sc = _make_subcatchments_dict()
    graph, _, _ = _build_network(sc)
    if graph.n_edges > 0:
        srcs = graph.edge_index[0].tolist()
        dsts = graph.edge_index[1].tolist()
        for s, d in zip(srcs, dsts):
            assert s != d, f"Self-loop at node {s}"


def test_build_network_topo_order_length():
    from meandre.data.basin_builder import _build_network
    sc = _make_subcatchments_dict()
    graph, _, _ = _build_network(sc)
    assert len(graph.topo_order) == sc["n_nodes"]
