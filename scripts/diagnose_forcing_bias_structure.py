"""Diagnostique la structure du biais ERA5 - quebec.zarr.

Question : le biais +18% est-il un offset uniforme (easy fix) ou
structure spatio-temporelle (besoin d'un vrai modèle de correction) ?
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


def extract_at_nodes(ds, lons, lats):
    pts = ds.sel(
        longitude=xr.DataArray(lons, dims="node"),
        latitude=xr.DataArray(lats, dims="node"),
        method="nearest",
    )
    df = pd.DataFrame({
        "time": np.repeat(pts.time.values, len(lons)),
        "node": np.tile(np.arange(len(lons)), len(pts.time)),
        "pr": pts.pr.values.ravel(),
        "tasmin": pts.tasmin.values.ravel(),
        "tasmax": pts.tasmax.values.ravel(),
    })
    return df


def main():
    con = duckdb.connect(BASIN_DB, read_only=True)
    nodes = con.execute(
        "SELECT n.node_idx, n.lon, n.lat, t.mean_elevation_m FROM nodes n "
        "LEFT JOIN territorial t ON n.node_idx = t.node_idx ORDER BY n.node_idx"
    ).df()
    lons = nodes.lon.values
    lats = nodes.lat.values

    era5 = xr.open_dataset(ERA5_NC).sel(time=slice(T0, T1))
    qc = xr.open_zarr(QUEBEC_ZARR).sel(time=slice(T0, T1))
    if "number" in qc.dims:
        qc = qc.isel(number=0)

    e = extract_at_nodes(era5, lons, lats)
    q = extract_at_nodes(qc, lons, lats)
    e["date"] = pd.to_datetime(e["time"]).dt.normalize()
    q["date"] = pd.to_datetime(q["time"]).dt.normalize()

    df = pd.merge(
        e[["date", "node", "pr", "tasmin", "tasmax"]].rename(
            columns={"pr": "pr_e", "tasmin": "tmin_e", "tasmax": "tmax_e"}),
        q[["date", "node", "pr", "tasmin", "tasmax"]].rename(
            columns={"pr": "pr_q", "tasmin": "tmin_q", "tasmax": "tmax_q"}),
        on=["date", "node"],
    )
    df["month"] = df.date.dt.month
    df["season"] = df.month.map(lambda m: "DJF" if m in [12,1,2] else
                                          "MAM" if m in [3,4,5] else
                                          "JJA" if m in [6,7,8] else "SON")
    # Solid vs liquid (proxy: tasmin < 0)
    df["is_snow"] = (df.tmin_e < 0) & (df.tmax_e < 2)

    print("=" * 70)
    print("1. SAISONNALITÉ DU BIAIS (basin-mean par saison)")
    print("=" * 70)
    bs = df.groupby(["date", "season"]).agg({"pr_e": "mean", "pr_q": "mean"}).reset_index()
    seas = bs.groupby("season").agg(
        pr_e=("pr_e", "mean"), pr_q=("pr_q", "mean")
    )
    seas["mm_per_day_e"] = seas.pr_e
    seas["mm_per_day_q"] = seas.pr_q
    seas["bias_mm_day"] = seas.pr_e - seas.pr_q
    seas["bias_pct"] = 100 * (seas.pr_e - seas.pr_q) / seas.pr_q
    seas["ratio"] = seas.pr_e / seas.pr_q
    print(seas[["mm_per_day_e", "mm_per_day_q", "bias_mm_day", "bias_pct", "ratio"]].round(3))

    print()
    print("=" * 70)
    print("2. BIAIS SOLIDE vs LIQUIDE (T_min<0 & T_max<2 = neige)")
    print("=" * 70)
    for snow_flag in [True, False]:
        sub = df[df.is_snow == snow_flag]
        e_mean = sub.pr_e.mean()
        q_mean = sub.pr_q.mean()
        bias = 100 * (e_mean - q_mean) / max(q_mean, 1e-6)
        label = "NEIGE" if snow_flag else "PLUIE"
        print(f"  {label:6s} : ERA5={e_mean:.3f} mm/j, QC={q_mean:.3f} mm/j, "
              f"biais={bias:+.1f}%, ratio={e_mean/max(q_mean,1e-6):.3f}, "
              f"n_obs={len(sub):,}")

    print()
    print("=" * 70)
    print("3. BIAIS PAR INTENSITÉ (quantiles de pr_q)")
    print("=" * 70)
    df["q_bin"] = pd.qcut(df.pr_q, q=[0, 0.5, 0.8, 0.95, 0.99, 1.0],
                          labels=["P0-50", "P50-80", "P80-95", "P95-99", "P99+"],
                          duplicates="drop")
    by_int = df.groupby("q_bin", observed=True).agg(
        pr_e=("pr_e", "mean"), pr_q=("pr_q", "mean"), n=("pr_e", "count")
    )
    by_int["bias_pct"] = 100 * (by_int.pr_e - by_int.pr_q) / by_int.pr_q.replace(0, np.nan)
    by_int["ratio"] = by_int.pr_e / by_int.pr_q.replace(0, np.nan)
    print(by_int.round(2))

    print()
    print("=" * 70)
    print("4. STRUCTURE SPATIALE — biais P vs elevation/lat/lon")
    print("=" * 70)
    pernode = df.groupby("node").agg(
        pr_e=("pr_e", "mean"), pr_q=("pr_q", "mean"),
        tmin_e=("tmin_e", "mean"), tmin_q=("tmin_q", "mean"),
    )
    pernode["bias_pct"] = 100 * (pernode.pr_e - pernode.pr_q) / pernode.pr_q
    pernode["tmin_bias"] = pernode.tmin_e - pernode.tmin_q
    pernode = pernode.join(nodes.set_index("node_idx")[["lon", "lat", "mean_elevation_m"]])

    print("Corrélations du biais P (%) avec covariables :")
    for col in ["lon", "lat", "mean_elevation_m"]:
        r = pernode["bias_pct"].corr(pernode[col])
        print(f"  r(bias_P, {col:18s}) = {r:+.3f}")

    print()
    print("Ajustement linéaire bias_pct ~ lon + lat + elev :")
    from sklearn.linear_model import LinearRegression
    X = pernode[["lon", "lat", "mean_elevation_m"]].fillna(pernode.mean_elevation_m.mean()).values
    y = pernode["bias_pct"].values
    lr = LinearRegression().fit(X, y)
    yhat = lr.predict(X)
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    print(f"  R² = {r2:.3f}")
    print(f"  Résiduel std après correction linéaire : {(y - yhat).std():.2f}%")

    print()
    print("=" * 70)
    print("5. INTERANNUEL — le biais est-il stable d'une année à l'autre ?")
    print("=" * 70)
    df["year"] = df.date.dt.year
    yearly = df.groupby("year").agg(pr_e=("pr_e", "mean"), pr_q=("pr_q", "mean"))
    yearly["bias_pct"] = 100 * (yearly.pr_e - yearly.pr_q) / yearly.pr_q
    print(yearly.round(2))
    print(f"\n  std interannuelle du biais : {yearly.bias_pct.std():.2f}%")

    print()
    print("=" * 70)
    print("6. PEUT-ON BÉNÉFICIER D'UN SIMPLE SCIKIT-LEARN ?")
    print("=" * 70)
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    # Sample to keep it fast
    sample = df.sample(n=min(200_000, len(df)), random_state=0).copy()
    sample = sample.merge(nodes[["node_idx", "mean_elevation_m"]],
                          left_on="node", right_on="node_idx", how="left")
    sample["doy"] = sample.date.dt.dayofyear
    feats = ["pr_e", "tmin_e", "tmax_e", "lon", "lat", "mean_elevation_m", "doy"]
    sample = sample.dropna(subset=feats + ["pr_q"])
    X = sample[feats].values
    y = sample["pr_q"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    base_bias = 100 * (sample.pr_e.mean() - sample.pr_q.mean()) / sample.pr_q.mean()
    baseline_rmse = float(np.sqrt(((sample.pr_e - sample.pr_q) ** 2).mean()))

    gbr = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                    random_state=0)
    gbr.fit(Xtr, ytr)
    yhat_te = gbr.predict(Xte)
    rmse_corr = float(np.sqrt(((yhat_te - yte) ** 2).mean()))
    print(f"  Baseline (ERA5 brut)       : RMSE journalier = {baseline_rmse:.3f} mm")
    print(f"  GBR (200 trees, depth=4)   : RMSE journalier = {rmse_corr:.3f} mm")
    print(f"  Réduction RMSE             : {100*(1-rmse_corr/baseline_rmse):+.1f}%")
    importances = dict(zip(feats, gbr.feature_importances_))
    print(f"  Importances : " + ", ".join(f"{k}={v:.2f}" for k, v in importances.items()))


if __name__ == "__main__":
    main()
