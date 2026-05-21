"""Entraîne un modèle xgboost de correction de biais ERA5-Land → quebec.zarr.

Inputs (inférence runtime) : ERA5-Land + features géo + temporelles uniquement
                              (quebec.zarr non requis).
Targets : pr_q, tmin_q, tmax_q (3 modèles séparés).

Split temporel : train 2015-2022, val 2023-2024 (honnête, pas de leakage).
"""
from __future__ import annotations
import duckdb
import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb

BASIN_DB = ".models/stfran/data/basin.duckdb"
ERA5_NC = ".models/stfran/data/geo_cache/forcing_era5_land.nc"
QUEBEC_ZARR = "C:/Users/parse01/documents-locaux/rqh-local/io_2026-04/data/03_imputation/quebec.zarr"
T_TRAIN_START = "2015-01-01"
T_TRAIN_END = "2022-12-31"
T_VAL_START = "2023-01-01"
T_VAL_END = "2024-12-31"


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


def build_features(df: pd.DataFrame, nodes_geo: pd.DataFrame) -> pd.DataFrame:
    """Features uniquement dérivables d'ERA5 + géo."""
    df = df.merge(nodes_geo, left_on="node", right_on="node_idx", how="left")
    df["doy"] = df.date.dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * df.doy / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df.doy / 365)
    df["month"] = df.date.dt.month
    # Lags ERA5 (par node)
    df = df.sort_values(["node", "date"]).reset_index(drop=True)
    for lag in [1, 3, 7]:
        df[f"pr_e_lag{lag}"] = df.groupby("node")["pr_e"].shift(lag).fillna(0)
        df[f"tmin_e_lag{lag}"] = df.groupby("node")["tmin_e"].shift(lag).bfill()
    # Rolling sums (humidité antécédente)
    df["pr_e_sum7"] = (df.groupby("node")["pr_e"]
                      .rolling(7, min_periods=1).sum()
                      .reset_index(level=0, drop=True))
    df["pr_e_sum30"] = (df.groupby("node")["pr_e"]
                       .rolling(30, min_periods=1).sum()
                       .reset_index(level=0, drop=True))
    # Diurnal range
    df["trange_e"] = df.tmax_e - df.tmin_e
    return df


def evaluate_by_quantile(y_true, y_pred_uncorrected, y_pred_corrected, name="pr"):
    df = pd.DataFrame({"true": y_true, "raw": y_pred_uncorrected, "corr": y_pred_corrected})
    qcols = ["P0-50", "P50-80", "P80-95", "P95-99", "P99+"]
    df["band"] = pd.qcut(df["true"], q=[0, 0.5, 0.8, 0.95, 0.99, 1.0],
                         labels=qcols, duplicates="drop")
    print(f"\n  Biais (%) {name} par quantile (target = quebec.zarr) :")
    out = []
    for band in qcols:
        sub = df[df.band == band]
        if len(sub) == 0:
            continue
        t_mean = sub.true.mean()
        r_mean = sub.raw.mean()
        c_mean = sub["corr"].mean()
        if t_mean > 0.01:
            b_raw = 100 * (r_mean - t_mean) / t_mean
            b_cor = 100 * (c_mean - t_mean) / t_mean
        else:
            b_raw = (r_mean - t_mean)
            b_cor = (c_mean - t_mean)
        print(f"    {band:8s}  true={t_mean:7.3f}  raw={r_mean:7.3f} ({b_raw:+6.1f}%)  "
              f"corr={c_mean:7.3f} ({b_cor:+6.1f}%)")
        out.append({"band": band, "bias_raw_pct": b_raw, "bias_corr_pct": b_cor})
    return pd.DataFrame(out)


def main():
    con = duckdb.connect(BASIN_DB, read_only=True)
    nodes = con.execute(
        "SELECT n.node_idx, n.lon, n.lat, t.mean_elevation_m FROM nodes n "
        "LEFT JOIN territorial t ON n.node_idx = t.node_idx ORDER BY n.node_idx"
    ).df()
    lons = nodes.lon.values
    lats = nodes.lat.values
    nodes["mean_elevation_m"] = nodes["mean_elevation_m"].fillna(
        nodes["mean_elevation_m"].mean())

    era5 = xr.open_dataset(ERA5_NC).sel(time=slice(T_TRAIN_START, T_VAL_END))
    qc = xr.open_zarr(QUEBEC_ZARR).sel(time=slice(T_TRAIN_START, T_VAL_END))
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
    df = build_features(df, nodes[["node_idx", "lon", "lat", "mean_elevation_m"]])

    feats = [
        "pr_e", "tmin_e", "tmax_e", "trange_e",
        "pr_e_lag1", "pr_e_lag3", "pr_e_lag7",
        "tmin_e_lag1", "tmin_e_lag3", "tmin_e_lag7",
        "pr_e_sum7", "pr_e_sum30",
        "lon", "lat", "mean_elevation_m",
        "doy_sin", "doy_cos", "month",
    ]

    train_mask = (df.date >= T_TRAIN_START) & (df.date <= T_TRAIN_END)
    val_mask = (df.date >= T_VAL_START) & (df.date <= T_VAL_END)
    print(f"Train : {train_mask.sum():,} obs ({df[train_mask].date.min()} → {df[train_mask].date.max()})")
    print(f"Val   : {val_mask.sum():,} obs ({df[val_mask].date.min()} → {df[val_mask].date.max()})")

    common = dict(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=0,
        n_jobs=-1,
        tree_method="hist",
    )

    results = {}
    for target_col, era5_col, label in [
        ("pr_q",   "pr_e",   "pr"),
        ("tmin_q", "tmin_e", "tmin"),
        ("tmax_q", "tmax_e", "tmax"),
    ]:
        print("\n" + "=" * 70)
        print(f"  Correction : {era5_col}  →  {target_col}")
        print("=" * 70)
        Xtr = df.loc[train_mask, feats].values
        ytr = df.loc[train_mask, target_col].values
        Xva = df.loc[val_mask, feats].values
        yva = df.loc[val_mask, target_col].values
        raw_va = df.loc[val_mask, era5_col].values

        model = xgb.XGBRegressor(**common)
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        yhat = model.predict(Xva)

        rmse_raw = float(np.sqrt(((raw_va - yva) ** 2).mean()))
        rmse_corr = float(np.sqrt(((yhat - yva) ** 2).mean()))
        bias_raw = 100 * (raw_va.mean() - yva.mean()) / max(yva.mean(), 1e-6)
        bias_corr = 100 * (yhat.mean() - yva.mean()) / max(yva.mean(), 1e-6)
        print(f"  Baseline (ERA5 brut)  : RMSE = {rmse_raw:.3f}, biais moyen = {bias_raw:+.2f}%"
              if label == "pr" else
              f"  Baseline (ERA5 brut)  : RMSE = {rmse_raw:.3f}°C, biais moyen = {bias_raw - 100:+.2f}°C")
        if label == "pr":
            print(f"  Corrigé (XGB)         : RMSE = {rmse_corr:.3f}, biais moyen = {bias_corr:+.2f}%")
            print(f"  Réduction RMSE        : {100*(1 - rmse_corr/rmse_raw):+.1f}%")
        else:
            print(f"  Baseline (ERA5 brut)  : RMSE = {rmse_raw:.3f}°C, "
                  f"biais moyen = {(raw_va.mean()-yva.mean()):+.2f}°C")
            print(f"  Corrigé (XGB)         : RMSE = {rmse_corr:.3f}°C, "
                  f"biais moyen = {(yhat.mean()-yva.mean()):+.2f}°C")
            print(f"  Réduction RMSE        : {100*(1 - rmse_corr/rmse_raw):+.1f}%")

        # Importance
        imp = sorted(zip(feats, model.feature_importances_), key=lambda x: -x[1])
        print(f"  Top features : " + ", ".join(f"{k}={v:.2f}" for k, v in imp[:6]))

        if label == "pr":
            evaluate_by_quantile(yva, raw_va, yhat, name="pr")

        results[label] = {
            "model": model,
            "rmse_raw": rmse_raw, "rmse_corr": rmse_corr,
            "bias_raw": bias_raw if label == "pr" else (raw_va.mean()-yva.mean()),
            "bias_corr": bias_corr if label == "pr" else (yhat.mean()-yva.mean()),
        }

    # Save the trained pr model for use later
    import os
    out_dir = "scripts/forcing_correction_models"
    os.makedirs(out_dir, exist_ok=True)
    for label in ["pr", "tmin", "tmax"]:
        results[label]["model"].save_model(f"{out_dir}/{label}_xgb.json")
    print(f"\nModèles sauvegardés dans {out_dir}/")


if __name__ == "__main__":
    main()
