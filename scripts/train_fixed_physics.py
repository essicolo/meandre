"""Training script with all physics fixes applied."""
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
from meandre.utils.metrics import kge as compute_kge, nse as compute_nse, pbias as compute_pbias
from meandre.utils.state import HydroState


def normalize_forcing(forcing: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Normalize forcing variables to reasonable ranges."""
    # forcing shape: (n_timesteps, n_nodes, n_features)
    # Features: P, T_min, T_max, R_n, u2, e_a

    forcing_norm = forcing.clone()

    # Precipitation: already in mm/day, clip to [0, 300]
    forcing_norm[:, :, 0] = torch.clamp(forcing[:, :, 0], 0, 300)

    # Temperature: already in °C, clip to [-50, 50]
    forcing_norm[:, :, 1] = torch.clamp(forcing[:, :, 1], -50, 50)
    forcing_norm[:, :, 2] = torch.clamp(forcing[:, :, 2], -50, 50)

    # Radiation: MJ/m²/day, clip to [0, 40]
    forcing_norm[:, :, 3] = torch.clamp(forcing[:, :, 3], 0, 40)

    # Wind speed: m/s, clip to [0, 20]
    forcing_norm[:, :, 4] = torch.clamp(forcing[:, :, 4], 0, 20)

    # Vapor pressure: kPa, clip to [0, 5]
    forcing_norm[:, :, 5] = torch.clamp(forcing[:, :, 5], 0, 5)

    return forcing_norm


def main():
    parser = argparse.ArgumentParser(description="Train YHydro with physics fixes")
    parser.add_argument("--config", type=str, default="notebooks/slso/config/slso.toml")
    parser.add_argument("--fast", action="store_true", help="Fast mode for testing")
    args = parser.parse_args()

    # Load config
    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)

    paths = cfg["paths"]
    BASIN_DB      = Path(paths["basin_db"])
    ZARR_PATH     = Path(paths["weather_grid"])
    FORCING_CACHE = Path(paths["forcing_cache"])
    CHECKPOINT    = Path(paths["checkpoint"])
    RUNS_DB       = Path(paths["runs_db"])

    temporal = cfg["temporal"]
    DATE_START  = temporal["date_start"]
    DATE_END    = temporal["date_end"]
    TRAIN_START = temporal["train_start"]
    TRAIN_END   = temporal["train_end"]
    VAL_START   = temporal["val_start"]
    VAL_END     = temporal["val_end"]
    SPINUP_END  = temporal["spinup_end"]

    # Fast mode: use shorter periods
    if args.fast:
        TRAIN_END = "2002-12-31"
        VAL_START = "2003-01-01"
        VAL_END = "2004-12-31"
        DATE_END = VAL_END

    mcfg = cfg["model"]
    tcfg = cfg["training"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {args.config}")
    print(f"  Train: {TRAIN_START} – {TRAIN_END}")
    print(f"  Val:   {VAL_START} – {VAL_END}")

    # Basin data
    cache = BasinCache(BASIN_DB)
    hydro = cache.load(device=device)

    graph         = hydro["graph"]
    territorial   = hydro["territorial"]
    node_coords   = hydro["node_coords"]
    node_ids      = hydro["node_ids"]
    n_nodes       = hydro["n_nodes"]
    print(f"Nodes: {n_nodes}, Edges: {graph.n_edges}")

    # Forcing
    FORCING_CACHE.parent.mkdir(parents=True, exist_ok=True)
    forcing = extract_forcing(
        zarr_path=ZARR_PATH,
        node_coords=node_coords,
        node_elev=None,
        date_start=DATE_START,
        date_end=DATE_END,
        cache_nc=FORCING_CACHE,
        device=device,
    )

    # PHYSICS FIX 1: Normalize forcing values
    forcing = normalize_forcing(forcing, device)
    print(f"Forcing normalized: {tuple(forcing.shape)}")

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

    print(f"Spinup: {DATE_START}–{SPINUP_END} ({spinup_steps} steps)")
    print(f"Train:  {TRAIN_START}–{TRAIN_END} ({train_sl.start}:{train_sl.stop})")
    print(f"Val:    {VAL_START}–{VAL_END} ({val_sl.start}:{val_sl.stop})")

    # Model - USE STATIC MODE for 28 parameters like Hydrotel
    model = YHydro(
        n_nodes=n_nodes,
        n_territorial=territorial.n_features,
        n_forcing=mcfg["n_forcing"],
        context_window=mcfg["context_window"],
        residual_history=mcfg["residual_history"],
        max_travel_time=mcfg["max_travel_days"],
        use_temporal=True,
        use_residual=False,  # Disable residual initially for stability
        use_travel_time_attn=False,  # Disable travel time attention initially
        use_temperature=True,
        dropout=0.0,  # No dropout for deterministic behavior
        param_mode="static",  # Use 28 global parameters like Hydrotel!
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Training setup
    withdrawals = cache.load_withdrawals(
        date_start=DATE_START, date_end=DATE_END, device=device,
    )
    n_wd_active = (withdrawals.net.abs() > 0).any(dim=0).sum().item()
    print(f"Withdrawals: {n_wd_active} active nodes")

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
    val_data = TrainingData(
        forcing=forcing,
        q_obs=q_obs_tensor[val_sl.start:],
        station_mask=station_mask,
        station_idx=torch.tensor(station_indices, device=device),
        graph=graph,
        node_coords=node_coords,
        territorial=territorial,
        withdrawals=withdrawals,
        day_of_year=doy,
        train_slice=val_sl,
        val_slice=val_sl,
    )

    # Station weights
    idx_to_sid = {v: k for k, v in station_node_map.items()}
    sids_ordered = [idx_to_sid[ni] for ni in station_indices]
    areas_ordered = [areas[sids.index(s)] for s in sids_ordered]
    station_areas = torch.sqrt(torch.clamp(
        torch.tensor(areas_ordered, dtype=torch.float32, device=device),
        min=50.0, max=500.0,
    ))

    # Use simpler loss with less physics penalties initially
    loss_fn = CompositeKGELoss(
        alpha=0.5,
        eps=1.0,
        per_station=True,
        station_weights=station_areas,
        w_physics=0.001,  # Reduced physics weight
        w_residual=0.0,   # No residual loss initially
    )

    # Training config with slower learning rate for stability
    n_epochs = 50 if args.fast else tcfg["n_epochs"]

    train_cfg = TrainingConfig(
        lr=1e-4,  # Reduced learning rate for stability
        weight_decay=1e-4,
        grad_clip=0.5,  # Tighter gradient clipping
        n_epochs=n_epochs,
        spinup_steps=spinup_steps,
        warm_spinup_steps=30,
        tbptt_steps=30,  # Shorter TBPTT for stability
        val_every=2,
        enable_temporal_context_epoch=999,  # Disable initially
        enable_residual_corrector_epoch=999,  # Disable initially
        enable_travel_time_attn_epoch=999,  # Disable initially
        patience=15,
    )

    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT = CHECKPOINT.parent / "physics_fixed.pt"

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
        run_name="physics_fixed_static",
        run_logger=run_logger,
        checkpoint_path=str(CHECKPOINT),
    )

    # Remove old checkpoint
    if CHECKPOINT.exists():
        print(f"Removing old checkpoint {CHECKPOINT}")
        CHECKPOINT.unlink()

    print("\n=== Starting training with physics fixes ===")
    print("- Using static parameter mode (28 params)")
    print("- Using realistic initial states")
    print("- Fixed Darcy flux (allows capillary rise)")
    print("- Normalized forcing values")
    print("- Disabled residual/TTA initially for stability")

    trainer.fit()

    # Post-training diagnostics
    print("\n=== Post-training diagnostics ===")
    model.load(str(CHECKPOINT))
    model.eval()

    # PHYSICS FIX 2: Use realistic initial states, not zeros!
    initial_state = HydroState.default_warm(n_nodes, device=device)

    with torch.no_grad():
        Q_sim, _ = model.simulate(
            forcing=forcing,
            initial_state=initial_state,  # Use realistic initial states!
            graph=graph,
            node_coords=node_coords,
            territorial=territorial,
            withdrawals=withdrawals,
            day_of_year=doy,
        )

    # Per-station metrics
    q_sim_stations = Q_sim[:, station_mask].cpu()

    def print_metrics(label, period_sl):
        print(f"\n--- {label} ---")
        nse_vals, kge_vals = [], []
        for i, sid in enumerate(sids_ordered):
            q_o = q_obs_tensor[period_sl, i].cpu()
            q_s = q_sim_stations[period_sl, i]
            valid = ~torch.isnan(q_o)
            if valid.sum() < 30:
                continue
            q_o_v, q_s_v = q_o[valid], q_s[valid]
            nse_v = float(compute_nse(q_o_v, q_s_v))
            kge_v = float(compute_kge(q_o_v, q_s_v))
            pbias_v = float(compute_pbias(q_o_v, q_s_v))
            nse_vals.append(nse_v)
            kge_vals.append(kge_v)
            print(f"  {sid}: NSE={nse_v:.3f}, KGE={kge_v:.3f}, PBIAS={pbias_v:.1f}%, "
                  f"area={areas_ordered[i]:.0f} km²")
        if nse_vals:
            med_nse = sorted(nse_vals)[len(nse_vals) // 2]
            med_kge = sorted(kge_vals)[len(kge_vals) // 2]
            print(f"  >> Median NSE={med_nse:.3f}, KGE={med_kge:.3f}")

    print_metrics("Train period", train_sl)
    print_metrics("Validation period", val_sl)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()