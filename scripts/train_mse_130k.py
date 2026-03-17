"""Training with MSE loss and full 130k parameters like successful run."""
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
from meandre.utils.metrics import kge as compute_kge
from meandre.utils.state import HydroState


def main():
    parser = argparse.ArgumentParser(description="MSE training with 130k params")
    parser.add_argument("--config", type=str, default="notebooks/slso/config/slso.toml")
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)

    paths = cfg["paths"]
    BASIN_DB = Path(paths["basin_db"])
    ZARR_PATH = Path(paths["weather_grid"])
    FORCING_CACHE = Path(paths["forcing_cache"])
    CHECKPOINT = Path("notebooks/slso/checkpoints/mse_130k.pt")

    # Use longer period like original successful run
    DATE_START = "2000-01-01"
    DATE_END = "2003-12-31"
    TRAIN_START = "2001-01-01"
    TRAIN_END = "2002-12-31"
    VAL_START = "2003-01-01"
    VAL_END = "2003-12-31"
    SPINUP_END = "2000-12-31"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Training with MSE loss and 130k parameters (like successful run)")

    # Load data
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

    # Date handling
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

    # CRITICAL: Full NERF model like successful 130k run
    model = YHydro(
        n_nodes=n_nodes,
        n_territorial=territorial.n_features,
        n_forcing=6,
        context_window=30,
        residual_history=14,
        max_travel_time=20,
        use_temporal=True,  # ENABLE temporal like original
        use_residual=True,  # ENABLE residual like original
        use_travel_time_attn=True,  # ENABLE TTA like original
        use_temperature=True,
        dropout=0.1,  # Some regularization
        param_mode="nerf",  # Full 256-unit NERF (130k params)
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    withdrawals = cache.load_withdrawals(
        date_start=DATE_START,
        date_end=DATE_END,
        device=device,
    )

    # Lower learning rate for stability with 130k params
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    print("\n=== MSE Training (130k params) ===")
    print("- Using MSE loss (not KGE)")
    print("- Using warm initialization")
    print("- Full model features enabled")
    print("- Monitor KGE separately from loss")

    CHUNK_SIZE = 60  # 2 months
    WARMUP = 30

    best_val_kge = -999
    kge_not_improving = 0

    for epoch in range(200):
        model.train()
        epoch_losses = []
        epoch_kges = []

        # Warm initialization
        state = HydroState.default_warm(n_nodes, device=device)

        # Train on chunks
        train_days = train_sl.stop - train_sl.start
        n_chunks = max(1, (train_days - WARMUP) // CHUNK_SIZE)

        for chunk_i in range(n_chunks):
            start_idx = chunk_i * CHUNK_SIZE
            end_idx = min(start_idx + CHUNK_SIZE + WARMUP, train_sl.stop)

            chunk_forcing = forcing[start_idx:end_idx]
            chunk_doy = doy[start_idx:end_idx]

            # Simulate chunk
            Q_sim, state = model.simulate(
                forcing=chunk_forcing,
                initial_state=state,
                graph=graph,
                node_coords=node_coords,
                territorial=territorial,
                withdrawals=withdrawals,
                day_of_year=chunk_doy,
            )

            # Use non-warmup period for loss
            Q_sim_train = Q_sim[WARMUP:, station_mask]
            Q_obs_train = q_obs_tensor[start_idx + WARMUP:end_idx, :]

            # MSE loss (like successful run)
            chunk_losses = []
            for i in range(n_stations):
                q_o = Q_obs_train[:, i]
                q_s = Q_sim_train[:, i]

                valid = ~torch.isnan(q_o)
                if valid.sum() < 10:
                    continue

                q_o_v = q_o[valid]
                q_s_v = q_s[valid]

                # Simple MSE loss
                mse_loss = torch.mean((q_o_v - q_s_v) ** 2)
                chunk_losses.append(mse_loss)

                # Track KGE separately
                with torch.no_grad():
                    kge_val = compute_kge(q_o_v.cpu(), q_s_v.cpu())
                    epoch_kges.append(float(kge_val))

            if len(chunk_losses) > 0:
                loss = torch.stack(chunk_losses).mean()
                epoch_losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            state = state.detach()

        # Validation every 5 epochs
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_state = HydroState.default_warm(n_nodes, device=device)
                Q_val, _ = model.simulate(
                    forcing=forcing[val_sl],
                    initial_state=val_state,
                    graph=graph,
                    node_coords=node_coords,
                    territorial=territorial,
                    withdrawals=withdrawals,
                    day_of_year=doy[val_sl],
                )

                val_kges = []
                for i in range(n_stations):
                    q_o = q_obs_tensor[val_sl, i].cpu()
                    q_s = Q_val[:, station_mask][:, i].cpu()

                    valid = ~torch.isnan(q_o)
                    if valid.sum() < 30:
                        continue

                    kge_val = compute_kge(q_o[valid], q_s[valid])
                    val_kges.append(float(kge_val))

                mean_val_kge = np.mean(val_kges) if val_kges else -999
                mean_train_kge = np.mean(epoch_kges) if epoch_kges else -999
                mean_loss = np.mean(epoch_losses) if epoch_losses else 999

                print(f"Epoch {epoch:3d} | MSE: {mean_loss:.6f} | Train KGE: {mean_train_kge:.3f} | Val KGE: {mean_val_kge:.3f}")

                # Save best KGE (not best loss!)
                if mean_val_kge > best_val_kge:
                    best_val_kge = mean_val_kge
                    torch.save(model.state_dict(), CHECKPOINT)
                    print(f"  -> Best model saved (KGE: {best_val_kge:.3f})")
                    kge_not_improving = 0
                else:
                    kge_not_improving += 1

                # Early stop if KGE starts degrading (like original issue)
                if mean_val_kge > 0.5:
                    print(f"\n=== Good KGE reached: {mean_val_kge:.3f} ===")
                    break

                if kge_not_improving > 6:  # 30 epochs without improvement
                    print(f"\n=== KGE not improving, stopping ===")
                    break

    print(f"\n=== Training complete ===")
    print(f"Best validation KGE: {best_val_kge:.3f}")


if __name__ == "__main__":
    main()