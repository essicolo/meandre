"""Training with temporal modules only - lighter memory footprint."""
import argparse
import gc
import logging
import os
import tomllib
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xarray as xr

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from meandre.data.basin_cache import BasinCache
from meandre.data.gridded_forcing import extract_forcing
from meandre.model import YHydro
from meandre.training.loss import CompositeKGELoss
from meandre.utils.metrics import kge as compute_kge, nse as compute_nse
from meandre.utils.state import HydroState


def train_with_temporal(model, forcing, q_obs, station_mask, graph, node_coords, territorial,
                        withdrawals, doy, train_sl, val_sl, spinup_steps, n_epochs=50):
    """Training loop with temporal modules enabled."""

    device = next(model.parameters()).device

    # Start with lower learning rate for stability
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)

    # Learning rate scheduler - reduce on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.7, patience=5
    )

    # Station weights
    n_stations = station_mask.sum().item()
    station_areas = torch.ones(n_stations, device=device) * 100.0

    # Modified loss with more focus on regular KGE
    loss_fn = CompositeKGELoss(
        alpha=0.6,  # More weight on regular KGE vs log KGE
        eps=1.0,
        per_station=True,
        station_weights=station_areas,
        w_physics=0.005,  # Low physics constraint
        w_residual=0.0,  # No residual
    )

    best_val_kge = -float('inf')
    best_val_nse = -float('inf')
    no_improve_count = 0

    for epoch in range(n_epochs):
        # Training
        model.train()
        optimizer.zero_grad()

        # Run spinup
        if spinup_steps > 0:
            with torch.no_grad():
                _, spinup_state = model.simulate(
                    forcing=forcing[:spinup_steps],
                    initial_state=HydroState.zeros(model.n_nodes, device=device),
                    graph=graph,
                    node_coords=node_coords,
                    territorial=territorial,
                    withdrawals=withdrawals,
                    day_of_year=doy[:spinup_steps],
                )
        else:
            spinup_state = HydroState.zeros(model.n_nodes, device=device)

        # Training forward pass
        Q_sim, _ = model.simulate(
            forcing=forcing[train_sl],
            initial_state=spinup_state.detach(),
            graph=graph,
            node_coords=node_coords,
            territorial=territorial,
            withdrawals=withdrawals,
            day_of_year=doy[train_sl],
        )

        # Compute loss
        n_train = train_sl.stop - train_sl.start
        q_obs_train = q_obs[:n_train]

        # Add small regularization
        reg_loss = 0.0001 * Q_sim.abs().mean()

        loss, components = loss_fn(
            q_obs=q_obs_train,
            q_sim=Q_sim,
            station_mask=station_mask,
        )

        total_loss = loss + reg_loss

        if not torch.isnan(total_loss):
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validation every 2 epochs
        if epoch % 2 == 0:
            model.eval()
            with torch.no_grad():
                Q_val, _ = model.simulate(
                    forcing=forcing[val_sl],
                    initial_state=spinup_state.detach(),
                    graph=graph,
                    node_coords=node_coords,
                    territorial=territorial,
                    withdrawals=withdrawals,
                    day_of_year=doy[val_sl],
                )

                # Compute validation metrics
                n_val = val_sl.stop - val_sl.start
                q_obs_val = q_obs[val_sl.start - train_sl.start:][:n_val]
                q_sim_stations_val = Q_val[:, station_mask]

                valid_mask_val = ~torch.isnan(q_obs_val)
                if valid_mask_val.sum() > 0:
                    # Compute real KGE and NSE for each station
                    kge_vals = []
                    nse_vals = []
                    for i in range(q_obs_val.shape[1]):
                        mask = valid_mask_val[:, i]
                        if mask.sum() > 30:
                            # These are the real KGE/NSE metrics, not the modified loss
                            kge_val = compute_kge(q_obs_val[mask, i], q_sim_stations_val[mask, i])
                            nse_val = compute_nse(q_obs_val[mask, i], q_sim_stations_val[mask, i])
                            kge_vals.append(float(kge_val))
                            nse_vals.append(float(nse_val))

                    if kge_vals:
                        mean_kge = np.mean(kge_vals)
                        mean_nse = np.mean(nse_vals)
                        median_kge = np.median(kge_vals)
                        max_kge = np.max(kge_vals)

                        # Update learning rate based on KGE
                        scheduler.step(mean_kge)

                        if mean_kge > best_val_kge:
                            best_val_kge = mean_kge
                            best_val_nse = mean_nse
                            no_improve_count = 0
                            # Save checkpoint
                            Path("notebooks/slso/checkpoints").mkdir(parents=True, exist_ok=True)
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'val_kge': mean_kge,
                                'val_nse': mean_nse,
                            }, "notebooks/slso/checkpoints/best_temporal.pt")
                            print(f"  ✓ New best model! Mean KGE: {mean_kge:.3f}, Max KGE: {max_kge:.3f}")
                        else:
                            no_improve_count += 1
                    else:
                        mean_kge = -999
                        mean_nse = -999
                        median_kge = -999
                        max_kge = -999
                else:
                    mean_kge = -999
                    mean_nse = -999
                    median_kge = -999
                    max_kge = -999

        # Print progress - show both loss KGE and real validation KGE
        current_lr = optimizer.param_groups[0]['lr']
        loss_kge = components.get('kge', -999)  # This is the modified KGE from loss
        print(f"Epoch {epoch:3d}/{n_epochs} | Loss: {total_loss.item():.4f} | "
              f"Loss KGE: {loss_kge:.3f} | LR: {current_lr:.1e}", end="")
        if epoch % 2 == 0:
            print(f"\n  Val KGE: mean={mean_kge:.3f}, median={median_kge:.3f}, max={max_kge:.3f} | "
                  f"Val NSE: {mean_nse:.3f}")
        else:
            print()

        # Early stopping
        if no_improve_count >= 15:
            print(f"Early stopping at epoch {epoch} - no improvement for 15 validations")
            break

        # Clear cache periodically
        if epoch % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return best_val_kge, best_val_nse


def main():
    parser = argparse.ArgumentParser(description="Temporal-only training for SLSO basin")
    parser.add_argument("--config", type=str, default="notebooks/slso/config/slso.toml")
    parser.add_argument("--fast", action="store_true", help="Fast mode")
    args = parser.parse_args()

    # Load config
    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)

    paths = cfg["paths"]
    BASIN_DB      = Path(paths["basin_db"])
    ZARR_PATH     = Path(paths["weather_grid"])
    FORCING_CACHE = Path(paths["forcing_cache"])

    temporal = cfg["temporal"]
    if args.fast:
        DATE_START  = "2000-01-01"
        DATE_END    = "2003-12-31"
        TRAIN_START = "2001-01-01"
        TRAIN_END   = "2002-12-31"
        VAL_START   = "2003-01-01"
        VAL_END     = "2003-12-31"
        SPINUP_END  = "2000-12-31"
        N_EPOCHS = 80
    else:
        DATE_START  = temporal["date_start"]
        DATE_END    = temporal["date_end"]
        TRAIN_START = temporal["train_start"]
        TRAIN_END   = temporal["train_end"]
        VAL_START   = temporal["val_start"]
        VAL_END     = temporal["val_end"]
        SPINUP_END  = temporal["spinup_end"]
        N_EPOCHS = 150

    mcfg = cfg["model"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Mode: {'Fast' if args.fast else 'Full'}")
    print(f"Training with temporal modules ONLY (lighter memory)")
    print(f"  Train: {TRAIN_START} – {TRAIN_END}")
    print(f"  Val:   {VAL_START} – {VAL_END}")

    # Clear GPU memory first
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Basin data
    cache = BasinCache(BASIN_DB)
    hydro = cache.load(device=device)

    graph         = hydro["graph"]
    territorial   = hydro["territorial"]
    node_coords   = hydro["node_coords"]
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

    ds_time = xr.open_dataset(FORCING_CACHE)
    all_dates = ds_time.time.sel(time=slice(DATE_START, DATE_END)).values
    ds_time.close()
    print(f"Forcing: {tuple(forcing.shape)}")

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
    spinup_steps = min(spinup_sl.stop, 180) if args.fast else min(spinup_sl.stop, 365)

    doy = torch.tensor(
        [int(pd.Timestamp(d).day_of_year) for d in all_dates],
        dtype=torch.long, device=device,
    )

    print(f"Spinup: {spinup_steps} steps")
    print(f"Train:  {train_sl.start}:{train_sl.stop}")
    print(f"Val:    {val_sl.start}:{val_sl.stop}")

    # Model with only temporal modules enabled (lighter memory)
    model = YHydro(
        n_nodes=n_nodes,
        n_territorial=territorial.n_features,
        n_forcing=mcfg["n_forcing"],
        context_window=mcfg["context_window"] if not args.fast else 20,
        residual_history=mcfg["residual_history"] if not args.fast else 10,
        max_travel_time=mcfg["max_travel_days"] if not args.fast else 15,
        use_temporal=True,  # ENABLED - crucial for dynamics
        use_residual=False,  # DISABLED to save memory
        use_travel_time_attn=False,  # DISABLED to save memory
        use_temperature=True,
        dropout=0.0,  # No dropout for stability
    ).to(device)

    # Initialize weights properly
    for name, param in model.named_parameters():
        if 'weight' in name:
            if len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.normal_(param, mean=0, std=0.02)
        elif 'bias' in name:
            nn.init.constant_(param, 0)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print("✓ Temporal modules: ENABLED")
    print("✗ Residual: DISABLED (memory)")
    print("✗ Travel time attn: DISABLED (memory)")

    # Withdrawals
    withdrawals = cache.load_withdrawals(
        date_start=DATE_START, date_end=DATE_END, device=device,
    )

    print("\n=== Starting training with temporal dynamics ===")
    print("NOTE: Loss KGE is modified (combines regular + log KGE)")
    print("      Validation KGE is the real metric we care about\n")

    # Run training
    final_kge, final_nse = train_with_temporal(
        model=model,
        forcing=forcing,
        q_obs=q_obs_tensor,
        station_mask=station_mask,
        graph=graph,
        node_coords=node_coords,
        territorial=territorial,
        withdrawals=withdrawals,
        doy=doy,
        train_sl=train_sl,
        val_sl=val_sl,
        spinup_steps=spinup_steps,
        n_epochs=N_EPOCHS,
    )

    print(f"\n=== Training complete ===")
    print(f"Best validation KGE: {final_kge:.3f}")
    print(f"Best validation NSE: {final_nse:.3f}")

    # Load best model for final evaluation
    checkpoint_path = Path("notebooks/slso/checkpoints/best_temporal.pt")
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"\nLoaded best model from epoch {checkpoint['epoch']}")
        print(f"Best checkpoint KGE: {checkpoint['val_kge']:.3f}")

        # Full evaluation
        with torch.no_grad():
            Q_sim, _ = model.simulate(
                forcing=forcing,
                initial_state=HydroState.zeros(n_nodes, device=device),
                graph=graph,
                node_coords=node_coords,
                territorial=territorial,
                withdrawals=withdrawals,
                day_of_year=doy,
            )

        q_sim_stations = Q_sim[:, station_mask].cpu()

        print("\n--- Final detailed metrics ---")
        for period_name, period_sl in [("Train", train_sl), ("Val", val_sl)]:
            nse_vals, kge_vals = [], []
            for i, ni in enumerate(station_indices):
                q_o = q_obs_tensor[period_sl, i].cpu()
                q_s = q_sim_stations[period_sl, i]
                valid = ~torch.isnan(q_o)
                if valid.sum() < 30:
                    continue
                q_o_v, q_s_v = q_o[valid], q_s[valid]
                nse_v = float(compute_nse(q_o_v, q_s_v))
                kge_v = float(compute_kge(q_o_v, q_s_v))
                nse_vals.append(nse_v)
                kge_vals.append(kge_v)

            if nse_vals:
                print(f"\n{period_name} Period:")
                print(f"  KGE: Mean={np.mean(kge_vals):.3f}, Median={np.median(kge_vals):.3f}, "
                      f"Max={np.max(kge_vals):.3f}, Min={np.min(kge_vals):.3f}")
                print(f"  NSE: Mean={np.mean(nse_vals):.3f}, Median={np.median(nse_vals):.3f}, "
                      f"Max={np.max(nse_vals):.3f}")

                # Show distribution of good stations
                good_kge = [k for k in kge_vals if k > 0.5]
                if good_kge:
                    print(f"  Stations with KGE > 0.5: {len(good_kge)}/{len(kge_vals)}")
                    print(f"  Mean KGE of good stations: {np.mean(good_kge):.3f}")


if __name__ == "__main__":
    main()