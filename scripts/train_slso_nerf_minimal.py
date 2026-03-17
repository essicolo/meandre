"""Minimal training script using NERF mode without temporal encoder."""
import argparse
import logging
import os
import tomllib

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
import xarray as xr

from meandre.data.basin_cache import BasinCache
from meandre.data.gridded_forcing import extract_forcing
from meandre.model import YHydro
from meandre.training.loss import CompositeKGELoss
from meandre.training.run_logger import RunLogger
from meandre.training.trainer import Trainer, TrainingConfig, TrainingData
from meandre.utils.metrics import kge as compute_kge, nse as compute_nse
from meandre.utils.state import HydroState


def main():
    parser = argparse.ArgumentParser(description="Train YHydro with NERF mode")
    parser.add_argument("--config", type=str, default="notebooks/slso/config/slso.toml")
    args = parser.parse_args()

    # Load config
    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)

    paths = cfg["paths"]
    BASIN_DB      = Path(paths["basin_db"])
    ZARR_PATH     = Path(paths["weather_grid"])
    FORCING_CACHE = Path(paths["forcing_cache"])
    CHECKPOINT    = Path("notebooks/slso/checkpoints/nerf_minimal.pt")
    RUNS_DB       = Path("notebooks/slso/runs_nerf.duckdb")

    # Use short periods for fast training
    DATE_START  = "2000-01-01"
    DATE_END    = "2003-12-31"
    TRAIN_START = "2001-01-01"
    TRAIN_END   = "2002-12-31"
    VAL_START   = "2003-01-01"
    VAL_END     = "2003-12-31"
    SPINUP_END  = "2000-12-31"

    mcfg = cfg["model"]
    tcfg = cfg["training"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Training NERF mode with minimal features")
    print(f"  Train: {TRAIN_START} – {TRAIN_END}")
    print(f"  Val:   {VAL_START} – {VAL_END}")

    # Basin data
    cache = BasinCache(BASIN_DB)
    hydro = cache.load(device=device)

    graph         = hydro["graph"]
    territorial   = hydro["territorial"]
    node_coords   = hydro["node_coords"]
    n_nodes       = hydro["n_nodes"]
    print(f"Nodes: {n_nodes}, Edges: {graph.n_edges}")

    # Forcing
    forcing = extract_forcing(
        zarr_path=ZARR_PATH,
        node_coords=node_coords,
        node_elev=None,
        date_start=DATE_START,
        date_end=DATE_END,
        cache_nc=FORCING_CACHE,
        device=device,
    )

    ds_time = xr.open_dataset(FORCING_CACHE)
    all_dates = ds_time.time.sel(time=slice(DATE_START, DATE_END)).values
    ds_time.close()

    # Observations
    obs = cache.load_observations(
        date_start=DATE_START,
        date_end=DATE_END,
        min_valid_days=365,
    )
    station_node_map = obs["station_node_map"]
    station_indices = sorted(set(station_node_map.values()))
    n_stations = len(station_indices)

    station_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
    for ni in station_indices:
        station_mask[ni] = True

    discharge_np = obs["discharge"]
    q_obs_tensor = torch.from_numpy(discharge_np[:, station_indices]).to(device)
    sids = list(station_node_map.keys())

    # Get drainage areas
    _con = duckdb.connect(str(BASIN_DB), read_only=True)
    areas = []
    for s in sids:
        row = _con.execute(
            "SELECT drainage_area_km2 FROM stations WHERE station_id = ?", [s]
        ).fetchone()
        areas.append(float(row[0]) if row else 0.0)
    _con.close()

    print(f"Stations: {n_stations}")

    # Temporal slicing
    def dates_to_slice(dates, start, end):
        days = dates.astype("datetime64[D]")
        s = int(np.searchsorted(days, np.datetime64(start, "D")))
        e = int(np.searchsorted(days, np.datetime64(end, "D"), side="right"))
        return slice(s, e)

    spinup_sl = dates_to_slice(all_dates, DATE_START, SPINUP_END)
    train_sl = dates_to_slice(all_dates, TRAIN_START, TRAIN_END)
    val_sl = dates_to_slice(all_dates, VAL_START, VAL_END)
    spinup_steps = spinup_sl.stop

    doy = torch.tensor(
        [int(pd.Timestamp(d).day_of_year) for d in all_dates],
        dtype=torch.long, device=device,
    )

    # Model - NERF mode with minimal features for fast training
    model = YHydro(
        n_nodes=n_nodes,
        n_territorial=territorial.n_features,
        n_forcing=6,
        context_window=30,
        residual_history=14,
        max_travel_time=20,
        use_temporal=False,  # DISABLE temporal encoder to avoid slowness
        use_residual=False,  # DISABLE residual corrector initially
        use_travel_time_attn=False,  # DISABLE travel time attention initially
        use_temperature=True,
        dropout=0.0,
        param_mode="nerf",  # USE NERF for smooth spatial parameter fields
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,} (NERF mode)")

    # Training setup
    withdrawals = cache.load_withdrawals(
        date_start=DATE_START, date_end=DATE_END, device=device,
    )

    def make_data(period_sl):
        return TrainingData(
            forcing=forcing,
            q_obs=q_obs_tensor[period_sl.start:],
            station_mask=station_mask,
            station_idx=torch.tensor(station_indices, device=device),
            graph=graph,
            node_coords=node_coords,
            territorial=territorial,
            withdrawals=withdrawals,
            day_of_year=doy,
            train_slice=period_sl,
            val_slice=period_sl,
        )

    train_data = make_data(train_sl)
    val_data = make_data(val_sl)

    # Station weights
    idx_to_sid = {v: k for k, v in station_node_map.items()}
    sids_ordered = [idx_to_sid[ni] for ni in station_indices]
    areas_ordered = [areas[sids.index(s)] for s in sids_ordered]
    station_areas = torch.sqrt(torch.clamp(
        torch.tensor(areas_ordered, dtype=torch.float32, device=device),
        min=50.0, max=500.0,
    ))

    # Simple loss function
    loss_fn = CompositeKGELoss(
        alpha=0.5,
        eps=1.0,
        per_station=True,
        station_weights=station_areas,
        w_physics=0.0,  # No physics penalties initially
        w_residual=0.0,
    )

    # Fast training config
    train_cfg = TrainingConfig(
        lr=5e-4,  # Higher learning rate for NERF
        weight_decay=1e-5,
        grad_clip=1.0,
        n_epochs=100,
        spinup_steps=spinup_steps,
        warm_spinup_steps=30,
        tbptt_steps=60,  # Longer TBPTT since no temporal encoder
        val_every=5,
        enable_temporal_context_epoch=999,  # Never enable
        enable_residual_corrector_epoch=999,  # Never enable
        enable_travel_time_attn_epoch=999,  # Never enable
        patience=20,
    )

    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)

    # Clean old runs DB
    for f in [RUNS_DB, Path(str(RUNS_DB) + ".wal")]:
        if f.exists():
            os.remove(f)

    run_logger = RunLogger(RUNS_DB)

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        train_data=train_data,
        val_data=val_data,
        config=train_cfg,
        run_name="nerf_minimal",
        run_logger=run_logger,
        checkpoint_path=str(CHECKPOINT),
    )

    # Remove old checkpoint
    if CHECKPOINT.exists():
        CHECKPOINT.unlink()

    print("\n=== Starting NERF training ===")
    print("- Using NERF mode for smooth spatial fields")
    print("- Disabled temporal encoder for speed")
    print("- Disabled residual/TTA for simplicity")
    print("- Using realistic initial states")

    trainer.fit()

    # Post-training evaluation
    print("\n=== Evaluating NERF model ===")
    model.load(str(CHECKPOINT))
    model.eval()

    initial_state = HydroState.default_warm(n_nodes, device=device)

    with torch.no_grad():
        Q_sim, _ = model.simulate(
            forcing=forcing,
            initial_state=initial_state,
            graph=graph,
            node_coords=node_coords,
            territorial=territorial,
            withdrawals=withdrawals,
            day_of_year=doy,
        )

    # Per-station metrics
    q_sim_stations = Q_sim[:, station_mask].cpu()

    print(f"\n--- Validation period metrics ---")
    kge_vals = []
    for i, sid in enumerate(sids_ordered):
        q_o = q_obs_tensor[val_sl, i].cpu()
        q_s = q_sim_stations[val_sl, i]
        valid = ~torch.isnan(q_o)
        if valid.sum() < 30:
            continue
        q_o_v, q_s_v = q_o[valid], q_s[valid]
        kge_v = float(compute_kge(q_o_v, q_s_v))
        nse_v = float(compute_nse(q_o_v, q_s_v))
        kge_vals.append(kge_v)
        print(f"  {sid}: KGE={kge_v:.3f}, NSE={nse_v:.3f}")

    if kge_vals:
        med_kge = sorted(kge_vals)[len(kge_vals) // 2]
        print(f"\n  >> Median KGE: {med_kge:.3f}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()