"""Training with NERF mode, FDC loss and warm initialization."""
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

import numpy as np
import torch
import xarray as xr
import pandas as pd

from meandre.data.basin_cache import BasinCache
from meandre.data.gridded_forcing import extract_forcing
from meandre.model import YHydro
from meandre.training.loss import (
    differentiable_composite_kge_loss,
    differentiable_fdc_loss,
    differentiable_mse_loss,
)
from meandre.utils.metrics import kge as compute_kge
from meandre.utils.state import HydroState


def main():
    parser = argparse.ArgumentParser(description="Train with NERF and FDC loss")
    parser.add_argument("--config", type=str, default="notebooks/slso/config/slso.toml")
    args = parser.parse_args()

    # Load config
    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)

    paths = cfg["paths"]
    BASIN_DB = Path(paths["basin_db"])
    ZARR_PATH = Path(paths["weather_grid"])
    FORCING_CACHE = Path(paths["forcing_cache"])
    CHECKPOINT = Path("notebooks/slso/checkpoints/nerf_fdc.pt")

    # Use short period for testing
    DATE_START = "2000-01-01"
    DATE_END = "2003-12-31"
    TRAIN_START = "2001-01-01"
    TRAIN_END = "2002-12-31"
    VAL_START = "2003-01-01"
    VAL_END = "2003-12-31"
    SPINUP_END = "2000-12-31"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Training with NERF mode, FDC loss and warm initialization")

    # Basin data
    cache = BasinCache(BASIN_DB)
    hydro = cache.load(device=device)

    graph = hydro["graph"]
    territorial = hydro["territorial"]
    node_coords = hydro["node_coords"]
    n_nodes = hydro["n_nodes"]
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

    # Model - NERF mode for spatial variability
    model = YHydro(
        n_nodes=n_nodes,
        n_territorial=territorial.n_features,
        n_forcing=6,
        context_window=30,
        residual_history=14,
        max_travel_time=20,
        use_temporal=False,  # Start simple
        use_residual=False,
        use_travel_time_attn=False,
        use_temperature=True,
        dropout=0.0,
        param_mode="nerf",  # NERF for spatial variation
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Withdrawals
    withdrawals = cache.load_withdrawals(
        date_start=DATE_START,
        date_end=DATE_END,
        device=device,
    )

    # Optimizer - higher learning rate for NERF
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)

    print("\n=== Starting NERF training with FDC loss ===")
    print("- Using NERF mode for spatial parameter fields")
    print("- Using WARM initialization (theta=0.3)")
    print("- Using FDC loss for better low-flow matching")
    print(f"- {n_params:,} trainable parameters")

    best_val_kge = -999

    for epoch in range(100):
        model.train()

        # CRITICAL: Use warm initialization, NOT zeros!
        initial_state = HydroState.default_warm(n_nodes, device=device)

        # Forward pass
        Q_sim_all, _ = model.simulate(
            forcing=forcing,
            initial_state=initial_state,
            graph=graph,
            node_coords=node_coords,
            territorial=territorial,
            withdrawals=withdrawals,
            day_of_year=doy,
        )

        # Get simulated discharge at stations for training period
        Q_sim = Q_sim_all[train_sl, :][:, station_mask]
        Q_obs = q_obs_tensor[train_sl]

        # Compute losses
        losses = []
        kge_values = []

        for i in range(n_stations):
            q_o = Q_obs[:, i]
            q_s = Q_sim[:, i]

            # Skip if too many NaNs
            valid = ~torch.isnan(q_o)
            if valid.sum() < 100:
                continue

            q_o_v = q_o[valid]
            q_s_v = q_s[valid]

            # Composite KGE loss (regular + log-transformed)
            kge_loss, kge_info = differentiable_composite_kge_loss(q_o_v, q_s_v, alpha=0.5)

            # FDC loss for low flows
            fdc_loss = differentiable_fdc_loss(q_o_v, q_s_v)

            # Log-MSE for better low-flow sensitivity
            log_mse = differentiable_mse_loss(
                torch.log(q_o_v + 1.0),
                torch.log(q_s_v + 1.0)
            )

            # Combined loss
            total_loss = 0.4 * kge_loss + 0.3 * fdc_loss + 0.3 * log_mse

            losses.append(total_loss)
            kge_values.append(kge_info["kge"].item())

        if len(losses) == 0:
            print(f"Epoch {epoch}: No valid stations")
            continue

        loss = torch.stack(losses).mean()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Validation every 5 epochs
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                Q_sim_val = Q_sim_all[val_sl, :][:, station_mask]
                Q_obs_val = q_obs_tensor[val_sl]

                val_kge_values = []
                for i in range(n_stations):
                    q_o = Q_obs_val[:, i].cpu()
                    q_s = Q_sim_val[:, i].cpu()

                    valid = ~torch.isnan(q_o)
                    if valid.sum() < 100:
                        continue

                    kge_val = compute_kge(q_o[valid], q_s[valid])
                    val_kge_values.append(float(kge_val))

                mean_val_kge = np.mean(val_kge_values) if val_kge_values else -999

                print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Train KGE: {np.mean(kge_values):.3f} | Val KGE: {mean_val_kge:.3f}")

                if mean_val_kge > best_val_kge:
                    best_val_kge = mean_val_kge
                    torch.save(model.state_dict(), CHECKPOINT)
                    print(f"  -> Best model saved (KGE: {best_val_kge:.3f})")

                # Early stopping if we reach good performance
                if mean_val_kge > 0.5:
                    print(f"\n=== Good KGE reached: {mean_val_kge:.3f} ===")
                    break

    print(f"\n=== Training complete ===")
    print(f"Best validation KGE: {best_val_kge:.3f}")

    # Final evaluation
    if CHECKPOINT.exists():
        model.load_state_dict(torch.load(CHECKPOINT))
        model.eval()

        with torch.no_grad():
            Q_final, _ = model.simulate(
                forcing=forcing,
                initial_state=HydroState.default_warm(n_nodes, device=device),
                graph=graph,
                node_coords=node_coords,
                territorial=territorial,
                withdrawals=withdrawals,
                day_of_year=doy,
            )

        print("\n=== Final per-station metrics (validation) ===")
        for i, node_idx in enumerate(station_indices):
            q_o = q_obs_tensor[val_sl, i].cpu()
            q_s = Q_final[val_sl, node_idx].cpu()

            valid = ~torch.isnan(q_o)
            if valid.sum() < 100:
                continue

            kge_val = compute_kge(q_o[valid], q_s[valid])

            # Check low-flow performance
            q_o_sorted = torch.sort(q_o[valid])[0]
            q_s_sorted = torch.sort(q_s[valid])[0]

            # Q95 (flow exceeded 95% of time - low flow)
            idx_95 = int(0.95 * len(q_o_sorted))
            q95_obs = q_o_sorted[idx_95]
            q95_sim = q_s_sorted[idx_95]

            print(f"  Station {node_idx}: KGE={kge_val:.3f}, Q95_obs={q95_obs:.2f}, Q95_sim={q95_sim:.2f}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()