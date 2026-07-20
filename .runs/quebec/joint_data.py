"""Chargement d'UNE région pour l'entraînement conjoint (extrait le prep de slso.py).
Retourne train_data/val_data (TrainingData), loss_fn régionale, et les métadonnées
(n_nodes, node_ids, n_gauges) pour le MultiBasinTrainer. Forçage tronqué à 6 canaux
(P, Tmin, Tmax, R_n, u2, e_a) pour homogénéité inter-régions.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import torch
import duckdb
import xarray as xr
from meandre.data.basin_cache import BasinCache
from meandre.routing.withdrawals import WithdrawalData
from meandre.training.trainer import TrainingData
from meandre.training.loss import HydroLoss

DATE_START, DATE_END = "2000-01-01", "2024-12-31"
TRAIN_END, VAL_START, VAL_END = "2018-12-31", "2019-01-01", "2021-12-31"

FORCINGS = {
    "slso": "D:/meandre-data/slso/forcing-casr-corr.nc",
}
DBS = {"slso": ".runs/slso/data/slso.duckdb"}


def _paths(reg):
    db = DBS.get(reg, f"D:/meandre-data/quebec/{reg}.duckdb")
    fx = FORCINGS.get(reg, f"D:/meandre-data/quebec/forcing-{reg}-budyko.nc")
    return db, fx


def load_region(reg: str, lcfg: dict, device: str = "cuda"):
    """lcfg = section [loss] du TOML de base (poids identiques pour toutes les régions)."""
    reg = reg.lower()
    db_path, fx_path = _paths(reg)
    cache = BasinCache(db_path)
    h = cache.load(device=device)
    graph, territorial = h["graph"], h["territorial"]
    node_coords, n_nodes, node_ids = h["node_coords"], h["n_nodes"], h["node_ids"]

    d = xr.open_dataset(fx_path)
    F = d["forcing"].values[:, :, :6]  # (T, N, 6) — homogénéité (SLSO a 7 canaux)
    times = pd.to_datetime(d["time"].values); d.close()
    assert str(times[0])[:10] == DATE_START and str(times[-1])[:10] == DATE_END, f"{reg}: fenêtre forçage"
    forcing = torch.tensor(F, dtype=torch.float32, device=device)
    doy = torch.tensor(times.dayofyear.values, dtype=torch.float32, device=device)

    obs = cache.load_observations(date_start=DATE_START, date_end=DATE_END, min_valid_days=365)
    station_node_map = obs["station_node_map"]
    station_indices = sorted(set(station_node_map.values()))
    n_stations = len(station_indices)
    station_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
    for ni in station_indices:
        station_mask[ni] = True
    q_obs = torch.from_numpy(obs["discharge"][:, station_indices]).to(device) if n_stations else \
        torch.full((len(times), 0), float("nan"), device=device)

    withdrawals = cache.load_withdrawals(DATE_START, DATE_END, device=device)

    # multi-obj : ET 8 jours + GRACE (présence vérifiée par l'audit 2026-07-19)
    et_obs = cache.load_modis_et(DATE_START, DATE_END, device=device)
    tws_obs = None
    con = duckdb.connect(db_path, read_only=True)
    if "grace_tws" in [t[0] for t in con.execute("show tables").fetchall()]:
        g = con.execute("select date, tws_mm from grace_tws where quality_ok = true order by date").fetchdf()
        tws = torch.full((len(times),), float("nan"), device=device)
        ad = times.normalize()
        for dt, val in zip(pd.to_datetime(g["date"]), g["tws_mm"].values):
            target = pd.Timestamp(year=dt.year, month=dt.month, day=15)
            dd = np.abs((ad - target).days.values)
            i = int(dd.argmin())
            if dd[i] <= 20:
                tws[i] = float(val)
        tws_obs = tws
    con.close()

    # slices temporels (mêmes conventions que slso.py)
    def sl(d0, d1):
        i0 = int(np.searchsorted(times.values, np.datetime64(d0)))
        i1 = int(np.searchsorted(times.values, np.datetime64(d1))) + 1
        return slice(i0, i1)
    train_sl = sl(DATE_START, TRAIN_END)
    val_sl = sl(VAL_START, VAL_END)

    # loss régionale : poids partagés, stats stations locales
    station_var = torch.ones(n_stations, dtype=torch.float32, device=device)
    peak_thr = torch.full((n_stations,), float("inf"), dtype=torch.float32, device=device)
    q_train = q_obs[train_sl]
    for i in range(n_stations):
        m = ~torch.isnan(q_train[:, i])
        if m.sum() > 100:
            station_var[i] = q_train[m, i].var()
            peak_thr[i] = torch.quantile(q_train[m, i], 0.75)
    loss_fn = HydroLoss(
        w_nse=lcfg.get("w_nse", 0.0), w_kge=lcfg.get("w_kge", 0.0), w_pbias=lcfg.get("w_pbias", 0.0),
        w_mse=lcfg.get("w_mse", 0.0), w_nrmse=lcfg.get("w_nrmse", 0.0),
        w_log_nse=lcfg.get("w_log_nse", 0.0), w_log_mse=lcfg.get("w_log_mse", 0.0),
        w_et=lcfg.get("w_et", 0.0), w_tws=lcfg.get("w_tws", 0.0),
        w_peak=lcfg.get("w_peak", 0.0),
        w_physics=lcfg.get("w_physics", 0.0), w_residual=lcfg.get("w_residual", 0.0),
        per_station=True, station_weights=None, station_var=station_var,
        peak_threshold=peak_thr if lcfg.get("w_peak", 0.0) > 0 else None,
    )

    def mk(sl_):
        return TrainingData(
            forcing=forcing, q_obs=q_obs[sl_.start:],
            station_mask=station_mask,
            station_idx=torch.tensor(station_indices, device=device, dtype=torch.long),
            graph=graph, node_coords=node_coords, territorial=territorial,
            withdrawals=withdrawals, day_of_year=doy,
            train_slice=sl_, val_slice=sl_,
            et_obs=et_obs[sl_.start:] if et_obs is not None else None,
            tws_obs=tws_obs[sl_.start:] if tws_obs is not None else None,
        )
    return dict(name=reg, n_nodes=n_nodes, node_ids=node_ids, n_gauges=n_stations,
                train_data=mk(train_sl), val_data=mk(val_sl), loss_fn=loss_fn,
                node_coords=node_coords, territorial=territorial, times=times)
