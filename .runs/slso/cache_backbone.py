"""Cache backbone forward outputs pour itération rapide sur les têtes proba.

Forward backbone gelé UNE SEULE FOIS sur full range (2000-2024) et sauvegarde
(Q_sim, spatial_params, q_obs, day-of-year, masques periode) en .npz. Les
scripts fit_head.py et pit_diag.py consomment ce cache sans jamais re-simuler.

Usage :
  python cache_backbone.py --ckpt .runs/slso/checkpoints/best-phase1-grace.pt \
                           --out .runs/slso/data/cache_backbone.npz
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
import argparse
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import xarray as xr
import duckdb

from meandre.data.basin_cache import BasinCache
from meandre.model import HydroModel
from meandre.utils.state import HydroState

DB = ".runs/slso/data/slso.duckdb"
FORCING = ".runs/slso/data/forcing.nc"


def main(ckpt_path: str, out_path: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Loading {ckpt_path}...", flush=True)
    init = torch.load(ckpt_path, map_location="cpu", weights_only=False)["init_kwargs"]
    m = HydroModel(**init).to(device)
    m.load(ckpt_path)
    m.eval()

    cache = BasinCache(DB)
    h = cache.load(device=device)
    ds = xr.open_dataset(FORCING)
    fc_all = ds["forcing"].values.astype(np.float32)
    dt_all = pd.to_datetime(ds["time"].values).normalize()
    ds.close()
    print(f"Forcing : {fc_all.shape}, dates {dt_all[0]} → {dt_all[-1]}", flush=True)
    fc = torch.from_numpy(fc_all).to(device)
    doy = torch.tensor(dt_all.dayofyear.values, dtype=torch.long, device=device)
    wd = cache.load_withdrawals(date_start="2000-01-01", date_end="2024-12-31", device=device)

    print("Forward backbone full 25 ans (1 pass)...", flush=True)
    with torch.no_grad():
        Q, _ = m.simulate(
            forcing=fc, initial_state=HydroState.zeros(h["n_nodes"], device=device),
            graph=h["graph"], node_coords=h["node_coords"], territorial=h["territorial"],
            withdrawals=wd, day_of_year=doy,
        )
        sp = m.spatial_encoder(h["node_coords"], h["territorial"].to_tensor())
        sp_tensor = sp.to_tensor()

    # GRU context (16D latent par (t, n)) — extrait séparément pour le sauver.
    # simulate() le calcule mais ne l'expose pas dans la signature publique.
    gru_context = None
    if m.use_temporal and m.temporal_encoder is not None:
        print("Forward GRU context séparé...", flush=True)
        with torch.no_grad():
            gru_context, _ = m.temporal_encoder.encode_sequence(
                fc, doy, h0=None,
            )                                                            # (T, N, n_ctx)
        print(f"  GRU context shape : {gru_context.shape}", flush=True)

    # Indices hydrométéorologiques interprétables (IHI)
    print("Compute indices IHI (GDD, API, SPI, frost number, SWE proxy)...", flush=True)
    from meandre.temporal.indices import compute_all_indices
    with torch.no_grad():
        indices = compute_all_indices(fc, doy)
    print(f"  Indices : {list(indices.keys())}", flush=True)
    for k, v in indices.items():
        print(f"    {k:18s} shape={tuple(v.shape)}  range=[{v.min().item():.2f}, {v.max().item():.2f}]", flush=True)

    # Stations
    con = duckdb.connect(DB, read_only=True)
    st = con.execute("select node_idx, station_id, drainage_area_km2 from stations order by node_idx").fetchdf()
    ob = con.execute("select date, station_id, discharge as q from observations").fetchdf()
    con.close()
    sn = st["node_idx"].values.astype(int)
    n_st = len(sn)
    s2c = {s: i for i, s in enumerate(st["station_id"].values)}
    d2t = {d: i for i, d in enumerate(dt_all)}
    qo = np.full((len(dt_all), n_st), np.nan, dtype=np.float32)
    for _, r in ob.iterrows():
        d = pd.Timestamp(r["date"]).normalize()
        if d in d2t and r["station_id"] in s2c:
            qo[d2t[d], s2c[r["station_id"]]] = float(r["q"])

    # Extract station-level
    Q_st = Q[:, sn].cpu().numpy()                                      # (T, n_st)
    sp_st = sp_tensor[sn].cpu().numpy()                                # (n_st, F)
    gru_ctx_st = None
    if gru_context is not None:
        gru_ctx_st = gru_context[:, sn].cpu().numpy().astype(np.float32)  # (T, n_st, n_ctx)
    print(f"Q_sim_station : {Q_st.shape}, sp_station : {sp_st.shape}", flush=True)
    if gru_ctx_st is not None:
        print(f"GRU context_station : {gru_ctx_st.shape}", flush=True)

    # Period masks (1er janv pour entamer une saison hydro complète)
    train_mask = (dt_all >= pd.Timestamp("2001-01-01")) & (dt_all <= pd.Timestamp("2018-12-31"))
    val_mask = (dt_all >= pd.Timestamp("2019-01-01")) & (dt_all <= pd.Timestamp("2021-12-31"))
    test_mask = (dt_all >= pd.Timestamp("2022-01-01")) & (dt_all <= pd.Timestamp("2024-12-31"))

    # Stats utiles : station_var (depuis train, pour normalisation NSE-style)
    station_var = np.zeros(n_st, dtype=np.float32)
    for i in range(n_st):
        m_ = train_mask & ~np.isnan(qo[:, i])
        if m_.sum() > 30: station_var[i] = qo[m_, i].var()
        else: station_var[i] = 1.0

    # Surface drainée par station (pour pondération sqrt_area)
    area_per_st = np.full(n_st, np.nan, dtype=np.float32)
    s2a = dict(zip(st["station_id"], st["drainage_area_km2"]))
    for s_id, idx in s2c.items():
        if s_id in s2a and s2a[s_id] is not None and not np.isnan(s2a[s_id]):
            area_per_st[idx] = s2a[s_id]

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving cache to {out}...", flush=True)
    payload = dict(
        Q_sim=Q_st.astype(np.float32),                                  # (T, n_st)
        spatial_params=sp_st.astype(np.float32),                        # (n_st, F)
        q_obs=qo.astype(np.float32),                                    # (T, n_st)
        day_of_year=dt_all.dayofyear.values.astype(np.int16),           # (T,)
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        station_node_idx=sn.astype(np.int32),
        station_ids=np.array(st["station_id"].tolist(), dtype="U16"),
        station_var=station_var,
        station_area_km2=area_per_st,
        ckpt_path=np.array(ckpt_path, dtype="U256"),
    )
    if gru_ctx_st is not None:
        payload["gru_context"] = gru_ctx_st                              # (T, n_st, 16)
    # Indices IHI — slice aux stations (les indices sont (T, N) ou (T, 2) pour DOY)
    for k, v in indices.items():
        if v.ndim == 2 and v.shape[1] == h["n_nodes"]:
            payload[f"idx_{k}"] = v[:, sn].cpu().numpy().astype(np.float32)
        elif v.ndim == 2 and v.shape[1] == 2:           # doy_phase (T, 2) commun à toutes stations
            payload[f"idx_{k}"] = v.cpu().numpy().astype(np.float32)
    np.savez_compressed(out, **payload)
    size_mb = out.stat().st_size / 1e6
    print(f"Cache saved : {size_mb:.1f} MB", flush=True)
    print(f"Validation samples : train = {(~np.isnan(qo[train_mask])).sum()}, "
          f"val = {(~np.isnan(qo[val_mask])).sum()}, "
          f"test = {(~np.isnan(qo[test_mask])).sum()}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=".runs/slso/checkpoints/best-phase1-grace.pt")
    p.add_argument("--out", default=".runs/slso/data/cache_backbone.npz")
    args = p.parse_args()
    main(args.ckpt, args.out)
