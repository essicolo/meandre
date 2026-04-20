"""Load withdrawal/discharge data from a GeoParquet file.

Expected parquet schema (long format, one row per point x date):
    withdrawal_id: str        — unique point identifier
    type: str                 — WITHDRAWAL | REJECTION (or legacy HYDROTEL types)
    date: date                — monthly or daily
    flow_m3s: float           — rate in m³/s (always positive)
    geometry: geometry         — point location (shapely Point)

Legacy HYDROTEL types (GPE, PR, ELEVAGE, CULTURE) are aggregated into
``withdrawal`` using consumption coefficients.  EFFLUENT becomes ``rejection``.

Optional columns: source (observed/prophet/permit), uncertainty_pct.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch

from meandre.routing.withdrawals import WithdrawalData

logger = logging.getLogger(__name__)

# Legacy HYDROTEL types → consumption coefficients
_LEGACY_TYPES = {"GPE": 0.8, "PR": 0.8, "ELEVAGE": 0.9, "CULTURE": 0.7}


def _haversine_km(lon1: float, lat1: float, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    """Haversine distance in km between a point and arrays of points."""
    R = 6371.0
    dlon = np.radians(lon2 - lon1)
    dlat = np.radians(lat2 - lat1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def load_withdrawals_parquet(
    parquet_path: str | Path,
    basin_db: str | Path,
    date_start: str,
    date_end: str,
    device: torch.device | None = None,
    max_snap_distance_km: float = 10.0,
    consumption_coeff: dict[str, float] | None = None,
) -> WithdrawalData:
    """Load withdrawals from a GeoParquet and snap points to model reaches.

    Supports two column conventions:

    * **Simple**: ``type`` is ``WITHDRAWAL`` or ``REJECTION``.
    * **Legacy HYDROTEL**: ``type`` is ``GPE | PR | ELEVAGE | CULTURE | EFFLUENT``.
      Consumptive types are multiplied by their alpha and summed; EFFLUENT
      becomes rejection.

    Returns:
        WithdrawalData with tensors (n_timesteps, n_nodes).
    """
    import geopandas as gpd

    parquet_path = Path(parquet_path)
    basin_db = Path(basin_db)

    alphas = dict(_LEGACY_TYPES)
    if consumption_coeff:
        alphas.update(consumption_coeff)

    # ── Load node coordinates from DuckDB ────────────────────────────────
    con = duckdb.connect(str(basin_db), read_only=True)
    nodes_df = con.execute("SELECT node_idx, node_id, lon, lat FROM nodes ORDER BY node_idx").fetchdf()
    con.close()

    n_nodes = len(nodes_df)
    node_lons = nodes_df["lon"].values
    node_lats = nodes_df["lat"].values
    logger.info(f"[withdrawal_parquet] {n_nodes} nodes loaded from {basin_db}")

    # ── Load parquet ─────────────────────────────────────────────────────
    gdf = gpd.read_parquet(parquet_path)
    required_cols = {"withdrawal_id", "type", "date", "flow_m3s"}
    missing = required_cols - set(gdf.columns)
    if missing:
        raise ValueError(f"Missing columns in parquet: {missing}")

    gdf["type"] = gdf["type"].str.upper()

    logger.info(f"[withdrawal_parquet] {len(gdf)} rows, {gdf['withdrawal_id'].nunique()} points, "
                f"types: {dict(gdf['type'].value_counts())}")

    # ── Snap points to nearest reach ─────────────────────────────────────
    points = gdf.drop_duplicates("withdrawal_id")[["withdrawal_id", "geometry"]].copy()
    snap_map = {}  # withdrawal_id → node_idx
    snap_distances = {}

    for _, row in points.iterrows():
        wid = row["withdrawal_id"]
        pt = row["geometry"]
        dists = _haversine_km(pt.x, pt.y, node_lons, node_lats)
        best_idx = int(np.argmin(dists))
        best_dist = dists[best_idx]
        snap_map[wid] = best_idx
        snap_distances[wid] = best_dist
        if best_dist > max_snap_distance_km:
            logger.warning(f"[withdrawal_parquet] {wid} snapped to node {best_idx} "
                           f"at {best_dist:.1f} km (> {max_snap_distance_km} km)")

    n_snapped = len(snap_map)
    avg_dist = np.mean(list(snap_distances.values()))
    max_dist = max(snap_distances.values())
    logger.info(f"[withdrawal_parquet] Snapped {n_snapped} points. "
                f"Avg distance: {avg_dist:.2f} km, Max: {max_dist:.2f} km")

    gdf["node_idx"] = gdf["withdrawal_id"].map(snap_map)

    # ── Build daily date range ───────────────────────────────────────────
    daily_dates = pd.date_range(date_start, date_end, freq="D")
    n_timesteps = len(daily_dates)

    # ── Temporal disaggregation (monthly → daily) ────────────────────────
    gdf["date"] = pd.to_datetime(gdf["date"])
    unique_dates = gdf["date"].unique()
    is_monthly = all(pd.Timestamp(d).day == 1 for d in unique_dates)

    if is_monthly:
        logger.info("[withdrawal_parquet] Monthly data — disaggregating to daily")
        expanded = []
        for _, row in gdf.iterrows():
            month_start = row["date"]
            month_end = month_start + pd.offsets.MonthEnd(0)
            month_days = pd.date_range(month_start, month_end, freq="D")
            month_days = month_days[(month_days >= daily_dates[0]) & (month_days <= daily_dates[-1])]
            for d in month_days:
                expanded.append({
                    "date": d, "type": row["type"],
                    "node_idx": row["node_idx"], "flow_m3s": row["flow_m3s"],
                })
        daily_df = pd.DataFrame(expanded)
    else:
        logger.info("[withdrawal_parquet] Daily data detected")
        daily_df = gdf[["date", "type", "node_idx", "flow_m3s"]].copy()
        daily_df = daily_df[(daily_df["date"] >= daily_dates[0]) & (daily_df["date"] <= daily_dates[-1])]

    # ── Aggregate into net withdrawal ────────────────────────────────────
    date_to_idx = {d: i for i, d in enumerate(daily_dates)}

    net = torch.zeros(n_timesteps, n_nodes)
    net_gw = torch.zeros(n_timesteps, n_nodes)

    has_source = "source" in daily_df.columns
    group_cols = ["date", "type", "node_idx"] + (["source"] if has_source else [])
    agg = daily_df.groupby(group_cols)["flow_m3s"].sum().reset_index()

    gw_set = {"Souterrain", "SOUTERRAIN", "GW", "groundwater"}

    for _, row in agg.iterrows():
        tidx = date_to_idx.get(row["date"])
        if tidx is None:
            continue
        nidx = int(row["node_idx"])
        wtype = row["type"]
        val = float(row["flow_m3s"])
        is_gw = has_source and str(row["source"]) in gw_set
        target = net_gw if is_gw else net

        if wtype == "WITHDRAWAL":
            target[tidx, nidx] -= val  # removal → negative
        elif wtype == "REJECTION" or wtype == "EFFLUENT":
            target[tidx, nidx] += val  # return flow → positive
        elif wtype in alphas:
            target[tidx, nidx] -= val * alphas[wtype]  # consumptive removal → negative
        else:
            logger.warning(f"[withdrawal_parquet] Unknown type '{wtype}', treating as withdrawal")
            target[tidx, nidx] -= val

    if device is not None:
        net = net.to(device)
        net_gw = net_gw.to(device)

    total_net = net.sum().item() / n_timesteps
    total_gw = net_gw.sum().item() / n_timesteps
    logger.info(f"[withdrawal_parquet] Net surface: {total_net:.3f} m³/s, "
                f"net groundwater: {total_gw:.3f} m³/s")

    return WithdrawalData(net=net, net_gw=net_gw)
