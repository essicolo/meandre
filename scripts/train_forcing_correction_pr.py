"""Compare 3 approches XGB pour corriger ERA5-Land precip vers quebec.zarr.

(1) Baseline      : XGB regression MSE sur P brut
(2) Log-space     : XGB regression MSE sur log(P+1) (cible+features)
(3) 2-stage       : classifier wet/dry + regression log-space conditionnelle
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
WET_THRESHOLD = 0.5   # mm/day


def extract_at_nodes(ds, lons, lats):
    pts = ds.sel(
        longitude=xr.DataArray(lons, dims="node"),
        latitude=xr.DataArray(lats, dims="node"),
        method="nearest",
    )
    return pd.DataFrame({
        "time": np.repeat(pts.time.values, len(lons)),
        "node": np.tile(np.arange(len(lons)), len(pts.time)),
        "pr": pts.pr.values.ravel(),
        "tasmin": pts.tasmin.values.ravel(),
        "tasmax": pts.tasmax.values.ravel(),
    })


def build_features(df, nodes_geo):
    df = df.merge(nodes_geo, left_on="node", right_on="node_idx", how="left")
    df["doy"] = df.date.dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * df.doy / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df.doy / 365)
    df["month"] = df.date.dt.month
    df = df.sort_values(["node", "date"]).reset_index(drop=True)
    for lag in [1, 3, 7]:
        df[f"pr_e_lag{lag}"] = df.groupby("node")["pr_e"].shift(lag).fillna(0)
    df["pr_e_sum7"] = (df.groupby("node")["pr_e"]
                       .rolling(7, min_periods=1).sum()
                       .reset_index(level=0, drop=True))
    df["pr_e_sum30"] = (df.groupby("node")["pr_e"]
                        .rolling(30, min_periods=1).sum()
                        .reset_index(level=0, drop=True))
    df["trange_e"] = df.tmax_e - df.tmin_e
    df["log_pr_e"] = np.log1p(df.pr_e)
    df["log_pr_e_sum7"] = np.log1p(df.pr_e_sum7)
    df["log_pr_e_sum30"] = np.log1p(df.pr_e_sum30)
    return df


def bias_by_quantile(y_true, y_pred, label=""):
    df = pd.DataFrame({"true": y_true, "pred": y_pred})
    qcols = ["P0-50", "P50-80", "P80-95", "P95-99", "P99+"]
    df["band"] = pd.qcut(df["true"], q=[0, 0.5, 0.8, 0.95, 0.99, 1.0],
                         labels=qcols, duplicates="drop")
    print(f"\n  Biais (%) {label} par quantile :")
    for band in qcols:
        sub = df[df.band == band]
        if len(sub) == 0:
            continue
        t = sub.true.mean()
        p = sub.pred.mean()
        b = 100 * (p - t) / max(t, 1e-3)
        print(f"    {band:8s}  true={t:7.3f}  pred={p:7.3f}  bias={b:+7.1f}%")
    rmse = float(np.sqrt(((y_pred - y_true) ** 2).mean()))
    bias_mean = 100 * (y_pred.mean() - y_true.mean()) / max(y_true.mean(), 1e-3)
    print(f"  Global RMSE = {rmse:.3f}, biais moyen = {bias_mean:+.2f}%")


def main():
    con = duckdb.connect(BASIN_DB, read_only=True)
    nodes = con.execute(
        "SELECT n.node_idx, n.lon, n.lat, t.mean_elevation_m FROM nodes n "
        "LEFT JOIN territorial t ON n.node_idx = t.node_idx ORDER BY n.node_idx"
    ).df()
    nodes["mean_elevation_m"] = nodes["mean_elevation_m"].fillna(
        nodes["mean_elevation_m"].mean())
    lons, lats = nodes.lon.values, nodes.lat.values

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
    df["log_pr_q"] = np.log1p(df.pr_q)
    df["is_wet_q"] = (df.pr_q >= WET_THRESHOLD).astype(int)

    feats = [
        "log_pr_e", "tmin_e", "tmax_e", "trange_e",
        "pr_e_lag1", "pr_e_lag3", "pr_e_lag7",
        "log_pr_e_sum7", "log_pr_e_sum30",
        "lon", "lat", "mean_elevation_m",
        "doy_sin", "doy_cos", "month",
    ]
    train = (df.date >= T_TRAIN_START) & (df.date <= T_TRAIN_END)
    val = (df.date >= T_VAL_START) & (df.date <= T_VAL_END)

    raw_va = df.loc[val, "pr_e"].values
    y_va = df.loc[val, "pr_q"].values

    # ── (0) Baseline ERA5 brut ──
    print("=" * 70)
    print("(0) BASELINE  —  ERA5 brut, aucune correction")
    print("=" * 70)
    bias_by_quantile(y_va, raw_va, label="pr (raw)")

    # ── (1) MSE direct (déjà testé, pour rappel) ──
    print("\n" + "=" * 70)
    print("(1) XGB MSE direct sur pr_q")
    print("=" * 70)
    Xtr = df.loc[train, feats].values
    Xva = df.loc[val, feats].values
    m1 = xgb.XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                          random_state=0, n_jobs=-1)
    m1.fit(Xtr, df.loc[train, "pr_q"].values)
    pred1 = m1.predict(Xva)
    bias_by_quantile(y_va, pred1, label="pr (MSE direct)")

    # ── (2) MSE en log-space ──
    print("\n" + "=" * 70)
    print("(2) XGB MSE sur log(pr_q+1)  [hydro-standard]")
    print("=" * 70)
    m2 = xgb.XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                          random_state=0, n_jobs=-1)
    m2.fit(Xtr, df.loc[train, "log_pr_q"].values)
    pred2 = np.expm1(m2.predict(Xva))
    pred2 = np.clip(pred2, 0, None)
    bias_by_quantile(y_va, pred2, label="pr (log-MSE)")

    # ── (3) 2-stage : classifier wet/dry + log-MSE amount conditionnel ──
    print("\n" + "=" * 70)
    print("(3) 2-STAGE  : classifier wet/dry  +  log-MSE amount conditionnel")
    print("=" * 70)
    clf = xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                            random_state=0, n_jobs=-1, eval_metric="logloss")
    clf.fit(Xtr, df.loc[train, "is_wet_q"].values)
    p_wet_va = clf.predict_proba(Xva)[:, 1]

    train_wet = train & (df.is_wet_q == 1)
    Xtrw = df.loc[train_wet, feats].values
    m3 = xgb.XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                          random_state=0, n_jobs=-1)
    m3.fit(Xtrw, df.loc[train_wet, "log_pr_q"].values)
    amount_va = np.expm1(m3.predict(Xva))
    amount_va = np.clip(amount_va, 0, None)

    # Décision : wet si p > 0.5, sinon 0. Alternative : multiplier amount*p_wet.
    pred3 = np.where(p_wet_va > 0.5, amount_va, 0.0)
    bias_by_quantile(y_va, pred3, label="pr (2-stage, hard wet=0.5)")

    pred3b = amount_va * p_wet_va
    print("\n  variante soft (E[P] = p_wet * amount) :")
    bias_by_quantile(y_va, pred3b, label="pr (2-stage soft)")

    # ── Comparaison synthèse ──
    print("\n" + "=" * 70)
    print("SYNTHÈSE — RMSE et biais moyen")
    print("=" * 70)
    for name, pred in [("raw ERA5",      raw_va),
                       ("(1) MSE direct", pred1),
                       ("(2) log-MSE",    pred2),
                       ("(3) 2-stage hard", pred3),
                       ("(3b) 2-stage soft", pred3b)]:
        rmse = float(np.sqrt(((pred - y_va) ** 2).mean()))
        bm = 100 * (pred.mean() - y_va.mean()) / max(y_va.mean(), 1e-3)
        # MAE
        mae = float(np.abs(pred - y_va).mean())
        # KS distance
        from scipy.stats import ks_2samp
        ks = ks_2samp(pred, y_va).statistic
        print(f"  {name:22s}  RMSE={rmse:5.2f}  MAE={mae:5.2f}  "
              f"bias_moy={bm:+6.1f}%  KS={ks:.3f}")


if __name__ == "__main__":
    main()
