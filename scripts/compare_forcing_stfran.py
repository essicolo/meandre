"""Compare ERA5-Land (open-data) vs quebec.zarr (PHYSITEL) forcing on stfran nodes.

Diagnostic pour identifier biais P/T qui pourraient expliquer le plateau
val_kge=0.637 sur stfran open-data vs slso-sub PHYSITEL.
"""
from __future__ import annotations
import duckdb
import numpy as np
import pandas as pd
import xarray as xr

BASIN_DB = ".models/stfran/data/basin.duckdb"
ERA5_NC = ".models/stfran/data/geo_cache/forcing_era5_land.nc"
QUEBEC_ZARR = "C:/Users/parse01/documents-locaux/rqh-local/io_2026-04/data/03_imputation/quebec.zarr"
T0 = "2015-01-01"
T1 = "2024-12-31"


def extract_at_nodes(ds: xr.Dataset, lons, lats) -> pd.DataFrame:
    """Nearest-neighbour extraction. Returns DataFrame indexed by (time, node)."""
    pts = ds.sel(
        longitude=xr.DataArray(lons, dims="node"),
        latitude=xr.DataArray(lats, dims="node"),
        method="nearest",
    )
    out = pd.DataFrame({
        "time": np.repeat(pts.time.values, len(lons)),
        "node": np.tile(np.arange(len(lons)), len(pts.time)),
        "pr": pts.pr.values.ravel(),
        "tasmin": pts.tasmin.values.ravel(),
        "tasmax": pts.tasmax.values.ravel(),
    })
    return out


def main():
    con = duckdb.connect(BASIN_DB, read_only=True)
    nodes = con.execute("SELECT node_idx, lon, lat FROM nodes ORDER BY node_idx").df()
    lons = nodes.lon.values
    lats = nodes.lat.values
    print(f"stfran : {len(nodes)} nodes, lon [{lons.min():.2f}, {lons.max():.2f}], "
          f"lat [{lats.min():.2f}, {lats.max():.2f}]")

    print("\n=== ERA5-Land (open-data) ===")
    era5 = xr.open_dataset(ERA5_NC).sel(time=slice(T0, T1))
    era5_df = extract_at_nodes(era5, lons, lats)
    print(f"  {len(era5.time)} days, time aligned 00 UTC")

    print("\n=== quebec.zarr (PHYSITEL) ===")
    qc = xr.open_zarr(QUEBEC_ZARR).sel(time=slice(T0, T1))
    if "number" in qc.dims:
        qc = qc.isel(number=0)
    qc_df = extract_at_nodes(qc, lons, lats)
    print(f"  {len(qc.time)} days, time aligned {pd.Timestamp(qc.time.values[0]).hour:02d} UTC")

    qc_df["date"] = pd.to_datetime(qc_df["time"]).dt.normalize()
    era5_df["date"] = pd.to_datetime(era5_df["time"]).dt.normalize()

    merged = pd.merge(
        era5_df[["date", "node", "pr", "tasmin", "tasmax"]].rename(
            columns={"pr": "pr_era5", "tasmin": "tmin_era5", "tasmax": "tmax_era5"}
        ),
        qc_df[["date", "node", "pr", "tasmin", "tasmax"]].rename(
            columns={"pr": "pr_qc", "tasmin": "tmin_qc", "tasmax": "tmax_qc"}
        ),
        on=["date", "node"],
        how="inner",
    )
    print(f"\nMerged days × nodes: {len(merged):,}  ({merged.date.nunique()} days)")

    print("\n=== Stats journaliers basin-mean (moyenne nodes) ===")
    basin = merged.groupby("date").agg({
        "pr_era5": "mean", "pr_qc": "mean",
        "tmin_era5": "mean", "tmin_qc": "mean",
        "tmax_era5": "mean", "tmax_qc": "mean",
    })

    # Annual totals/means
    basin["year"] = basin.index.year
    annual = basin.groupby("year").agg({
        "pr_era5": "sum", "pr_qc": "sum",
        "tmin_era5": "mean", "tmin_qc": "mean",
        "tmax_era5": "mean", "tmax_qc": "mean",
    })
    annual["pr_bias_pct"] = 100.0 * (annual.pr_era5 - annual.pr_qc) / annual.pr_qc
    annual["tmin_bias_C"] = annual.tmin_era5 - annual.tmin_qc
    annual["tmax_bias_C"] = annual.tmax_era5 - annual.tmax_qc
    print(annual.round(2).to_string())

    print("\n=== Résumé biais (moyenne sur années) ===")
    print(f"  P total annuel  : ERA5 = {annual.pr_era5.mean():.1f} mm/an,  "
          f"QC = {annual.pr_qc.mean():.1f} mm/an,  "
          f"biais ERA5-QC = {annual.pr_bias_pct.mean():+.1f}%")
    print(f"  T_min annuel    : ERA5 = {annual.tmin_era5.mean():.2f}°C,  "
          f"QC = {annual.tmin_qc.mean():.2f}°C,  "
          f"biais ERA5-QC = {annual.tmin_bias_C.mean():+.2f}°C")
    print(f"  T_max annuel    : ERA5 = {annual.tmax_era5.mean():.2f}°C,  "
          f"QC = {annual.tmax_qc.mean():.2f}°C,  "
          f"biais ERA5-QC = {annual.tmax_bias_C.mean():+.2f}°C")

    # Correlations journalières basin-mean
    print("\n=== Corrélation journalière basin-mean ===")
    print(f"  pr   : r = {basin.pr_era5.corr(basin.pr_qc):.3f}")
    print(f"  tmin : r = {basin.tmin_era5.corr(basin.tmin_qc):.3f}")
    print(f"  tmax : r = {basin.tmax_era5.corr(basin.tmax_qc):.3f}")

    # Per-node biases for spatial pattern
    print("\n=== Biais par node (annuel moyen) ===")
    pernode = merged.copy()
    pernode["year"] = pernode["date"].dt.year
    pn = pernode.groupby(["node", "year"]).agg({
        "pr_era5": "sum", "pr_qc": "sum",
        "tmin_era5": "mean", "tmin_qc": "mean",
    }).groupby("node").mean()
    pn["pr_bias_pct"] = 100.0 * (pn.pr_era5 - pn.pr_qc) / pn.pr_qc
    pn["tmin_bias_C"] = pn.tmin_era5 - pn.tmin_qc
    print(f"  P  bias range : {pn.pr_bias_pct.min():+.1f}% to {pn.pr_bias_pct.max():+.1f}%  "
          f"(std={pn.pr_bias_pct.std():.1f}%)")
    print(f"  Tmin range    : {pn.tmin_bias_C.min():+.2f}°C to {pn.tmin_bias_C.max():+.2f}°C  "
          f"(std={pn.tmin_bias_C.std():.2f}°C)")

    out_path = "scripts/compare_forcing_stfran_results.csv"
    pn.to_csv(out_path)
    print(f"\nPer-node bias saved : {out_path}")


if __name__ == "__main__":
    main()
