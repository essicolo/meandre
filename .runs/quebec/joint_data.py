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
from meandre.spatial.territorial import TerritorialFeatures
from meandre.utils import paths as _mpaths

_QC_RAW = f"{_mpaths.DATA_ROOT}/quebec/territorial-raw-QC.parquet"
_QC_STATS = "reports/territorial_stats_QC.csv"


def _territorial_global(reg, fallback, device):
    """Attributs normalisés sur les stats PROVINCIALES (et non par région).
    Sans ça, le nœud médian de CHAQUE région arrive au NeRF avec des attributs ~0 :
    le modèle ne peut pas différencier les régions (diagnostic 2026-07-31, cause des
    5 échecs conjoints). Retombe sur la normalisation locale si le brut est absent."""
    if not (os.path.exists(_QC_RAW) and os.path.exists(_QC_STATS)):
        return fallback
    try:
        raw = pd.read_parquet(_QC_RAW)
        raw = raw[raw["region"] == reg]
        if len(raw) != fallback.data.shape[0]:
            return fallback
        stats = pd.read_csv(_QC_STATS, index_col=0)
        cols = list(fallback.columns)
        arr = np.stack([((raw[c].values - stats.loc[c, "mean"]) / (stats.loc[c, "std"] + 1e-9))
                        for c in cols], axis=1).astype(np.float32)
        return TerritorialFeatures(data=torch.tensor(arr, device=device), columns=cols,
                                   physical=fallback.physical)
    except Exception:
        return fallback
from meandre.routing.withdrawals import WithdrawalData
from meandre.training.trainer import TrainingData
from meandre.training.loss import HydroLoss

DATE_START, DATE_END = "2000-01-01", "2024-12-31"
# DÉCOUPAGE TEMPOREL, configurable par JOINT_SPLIT="fin_train,debut_val,fin_val".
# Mesuré le 2026-08-12 : le découpage historique sélectionne sur la période la plus
# SÈCHE en été (val 2019-2021 : 1062 mm de pluie JJA) et teste sur la plus HUMIDE
# (2022-2024 : 1384 mm, +30 %), avec en prime +1.0 °C de moyenne annuelle et des pics
# observés PLUS BAS malgré la pluie. Le juge est donc biaisé, et pas dans un sens
# anodin : c'est exactement le régime où méandre est le plus faible (excès d'été de
# 25-40 % contre Hydrotel). Pouvoir déplacer les trois fenêtres est la condition pour
# vérifier ce que le tenu de côté doit à la période plutôt qu'au modèle.
_SPLIT = os.environ.get("JOINT_SPLIT")
if _SPLIT:
    TRAIN_END, VAL_START, VAL_END = (x.strip() for x in _SPLIT.split(","))
    print(f"[joint] découpage NON STANDARD : train -> {TRAIN_END} | val {VAL_START}..{VAL_END}")
else:
    TRAIN_END, VAL_START, VAL_END = "2018-12-31", "2019-01-01", "2021-12-31"

FORCINGS = {
    "slso": f"{_mpaths.DATA_ROOT}/slso/forcing-casr-corr.nc",
}
DBS = {"slso": ".runs/slso/data/slso.duckdb"}


def _paths(reg):
    """Résout (base, forçage). JOINT_FX_SUFFIX = "-none" signifie CaSR BRUT, donc
    forcing-<reg>.nc — et non le repli -budyko, qui faisait passer la variante corrigée
    pour du brut dans toute la carte provinciale du 2 août (bug d'étiquette relevé par
    Essi). Le fichier retenu est imprimé : on doit toujours savoir ce qu'on mesure."""
    db = DBS.get(reg, f"{_mpaths.DATA_ROOT}/quebec/{reg}.duckdb")
    sfx = os.environ.get("JOINT_FX_SUFFIX")
    if sfx == "-none":
        fx = f"{_mpaths.DATA_ROOT}/quebec/forcing-{reg}.nc"
        if not os.path.exists(fx):
            raise FileNotFoundError(f"{reg}: CaSR brut demandé mais {fx} absent")
        return db, fx
    if sfx:
        fx = f"{_mpaths.DATA_ROOT}/quebec/forcing-{reg}{sfx}.nc"
        if os.path.exists(fx):
            return db, fx
    fx = FORCINGS.get(reg, f"{_mpaths.DATA_ROOT}/quebec/forcing-{reg}-budyko.nc")
    return db, fx


def load_region(reg: str, lcfg: dict, device: str = "cuda"):
    """lcfg = section [loss] du TOML de base (poids identiques pour toutes les régions)."""
    reg = reg.lower()
    db_path, fx_path = _paths(reg)
    cache = BasinCache(db_path)
    h = cache.load(device=device)
    graph, territorial = h["graph"], h["territorial"]
    if os.environ.get("JOINT_GLOBAL_NORM", "0") == "1":
        territorial = _territorial_global(reg, territorial, device)
    node_coords, n_nodes, node_ids = h["node_coords"], h["n_nodes"], h["node_ids"]

    print(f"[forcage] {reg}: {os.path.basename(fx_path)}")
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
    # couvert nival MOD10 (fenêtre fonte mars-juin, ingéré 2026-07-22) : supervise la
    # fonte de la colonne DANS sa structure via w_snow (fraction simulée 1-exp(-SWE/ref))
    swe_obs = cache.load_modis_snow(DATE_START, DATE_END, device=device) if lcfg.get("w_snow", 0.0) > 0 else None
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
        w_et=lcfg.get("w_et", 0.0), et_mode=lcfg.get("et_mode", "level"),
        w_tws=lcfg.get("w_tws", 0.0),
        w_snow=lcfg.get("w_snow", 0.0),
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
            swe_obs=swe_obs[sl_.start:] if swe_obs is not None else None,
        )
    return dict(name=reg, n_nodes=n_nodes, node_ids=node_ids, n_gauges=n_stations,
                train_data=mk(train_sl), val_data=mk(val_sl), loss_fn=loss_fn,
                node_coords=node_coords, territorial=territorial, times=times)
