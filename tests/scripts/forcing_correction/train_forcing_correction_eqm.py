"""Quantile mapping empirique (EQM) ERA5-Land → quebec.zarr.

Par node × par mois : empirical CDF mapping. Préserve la distribution
quebec.zarr par construction (les quantiles matchent). Gère le drizzle
via wet-day threshold différencié source/target.

Pipeline EQM :
  1. Pour chaque (node, month) sur train : trier ERA5 et QC séparément.
  2. À l'inférence, pour une valeur ERA5_x : rang r = rank(ERA5_x) /
     n_train. Sortie = quantile_QC(r).
  3. Wet-day matching : remplace les valeurs ERA5 < seuil par 0 si la
     fréquence de jours secs QC > ERA5 (corrige le drizzle).

Comparaison vs MSE direct / log-MSE / 2-stage XGB.
"""
from __future__ import annotations
import duckdb
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import ks_2samp

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
    return pd.DataFrame({
        "time": np.repeat(pts.time.values, len(lons)),
        "node": np.tile(np.arange(len(lons)), len(pts.time)),
        "pr": pts.pr.values.ravel(),
    })


def build_eqm_table(era5_train, qc_train):
    """Pour chaque (node, month) : retourne (sorted_era5, sorted_qc, dry_freq_e, dry_freq_q).

    Retourne dict { (node, month) : (e_sorted, q_sorted, dry_e, dry_q) }.
    """
    table = {}
    g = era5_train.groupby(["node", "month"])
    for (node, month), sub in g:
        qsub = qc_train[(qc_train.node == node) & (qc_train.month == month)]
        if len(sub) < 30 or len(qsub) < 30:
            continue
        e_sorted = np.sort(sub.pr.values)
        q_sorted = np.sort(qsub.pr.values)
        # Wet-day fractions
        dry_e = (e_sorted < 0.1).mean()
        dry_q = (q_sorted < 0.1).mean()
        table[(node, month)] = (e_sorted, q_sorted, dry_e, dry_q)
    return table


def apply_eqm(era5_val_df: pd.DataFrame, table) -> np.ndarray:
    """Applique EQM par (node, month). era5_val_df doit avoir node, month, pr."""
    out = np.zeros(len(era5_val_df), dtype=np.float64)
    for (node, month), grp in era5_val_df.groupby(["node", "month"]):
        if (node, month) not in table:
            out[grp.index] = grp.pr.values
            continue
        e_sorted, q_sorted, dry_e, dry_q = table[(node, month)]
        vals = grp.pr.values
        # Rank ERA5 values via interp into ECDF
        ranks = np.searchsorted(e_sorted, vals, side="right") / len(e_sorted)
        ranks = np.clip(ranks, 1e-6, 1.0 - 1e-6)
        # Map to QC quantile
        idx = np.clip((ranks * len(q_sorted)).astype(int), 0, len(q_sorted) - 1)
        mapped = q_sorted[idx]
        # Wet-day correction : si QC plus sec qu'ERA5, zéroter les valeurs
        # dont le rang est sous la fréquence sèche QC.
        if dry_q > dry_e:
            mapped = np.where(ranks < dry_q, 0.0, mapped)
        out[grp.index] = mapped
    return out


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
        t, p = sub.true.mean(), sub.pred.mean()
        b = 100 * (p - t) / max(t, 1e-3)
        print(f"    {band:8s}  true={t:7.3f}  pred={p:7.3f}  bias={b:+7.1f}%")
    rmse = float(np.sqrt(((y_pred - y_true) ** 2).mean()))
    bias_mean = 100 * (y_pred.mean() - y_true.mean()) / max(y_true.mean(), 1e-3)
    print(f"  Global RMSE = {rmse:.3f}, biais moyen = {bias_mean:+.2f}%")


def main():
    con = duckdb.connect(BASIN_DB, read_only=True)
    nodes = con.execute("SELECT node_idx, lon, lat FROM nodes ORDER BY node_idx").df()
    lons, lats = nodes.lon.values, nodes.lat.values

    era5 = xr.open_dataset(ERA5_NC).sel(time=slice(T_TRAIN_START, T_VAL_END))
    qc = xr.open_zarr(QUEBEC_ZARR).sel(time=slice(T_TRAIN_START, T_VAL_END))
    if "number" in qc.dims:
        qc = qc.isel(number=0)

    e = extract_at_nodes(era5, lons, lats)
    q = extract_at_nodes(qc, lons, lats)
    e["date"] = pd.to_datetime(e["time"]).dt.normalize()
    q["date"] = pd.to_datetime(q["time"]).dt.normalize()
    e["month"] = e.date.dt.month
    q["month"] = q.date.dt.month

    e_train = e[(e.date >= T_TRAIN_START) & (e.date <= T_TRAIN_END)].reset_index(drop=True)
    q_train = q[(q.date >= T_TRAIN_START) & (q.date <= T_TRAIN_END)].reset_index(drop=True)
    e_val = e[(e.date >= T_VAL_START) & (e.date <= T_VAL_END)].reset_index(drop=True)
    q_val = q[(q.date >= T_VAL_START) & (q.date <= T_VAL_END)].reset_index(drop=True)

    print(f"Train : {len(e_train):,} obs ; Val : {len(e_val):,} obs")
    print(f"Construction EQM table par (node, month) sur train...")
    table = build_eqm_table(e_train, q_train)
    print(f"  {len(table)} cellules (node × month) ; "
          f"attendu {len(nodes) * 12} = {len(nodes) * 12}")

    print(f"\nApplication EQM sur val...")
    pred_eqm = apply_eqm(e_val, table)
    raw_va = e_val.pr.values
    y_va = q_val.pr.values

    # Sanity check : raw and y_va doivent être alignés
    assert len(raw_va) == len(y_va) == len(pred_eqm)

    print("\n" + "=" * 70)
    print("BASELINE — ERA5 brut")
    print("=" * 70)
    bias_by_quantile(y_va, raw_va, label="raw")

    print("\n" + "=" * 70)
    print("EQM — quantile mapping empirique (node × month)")
    print("=" * 70)
    bias_by_quantile(y_va, pred_eqm, label="EQM")

    print("\n" + "=" * 70)
    print("SYNTHÈSE")
    print("=" * 70)
    for name, pred in [("raw ERA5", raw_va), ("EQM", pred_eqm)]:
        rmse = float(np.sqrt(((pred - y_va) ** 2).mean()))
        bm = 100 * (pred.mean() - y_va.mean()) / max(y_va.mean(), 1e-3)
        mae = float(np.abs(pred - y_va).mean())
        ks = ks_2samp(pred, y_va).statistic
        print(f"  {name:18s}  RMSE={rmse:5.2f}  MAE={mae:5.2f}  "
              f"bias_moy={bm:+6.1f}%  KS={ks:.3f}")

    # Save the table
    import pickle, os
    out_dir = "scripts/forcing_correction_models"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/eqm_pr_stfran.pkl", "wb") as f:
        pickle.dump(table, f)
    print(f"\nEQM table sauvegardée : {out_dir}/eqm_pr_stfran.pkl")


if __name__ == "__main__":
    main()
