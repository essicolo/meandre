"""Regularized chunked training to reduce overfitting."""
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
from meandre.training.loss import differentiable_composite_kge_loss
from meandre.utils.metrics import kge as compute_kge
from meandre.utils.state import HydroState


def main():
    parser = argparse.ArgumentParser(description="Regularized training")
    parser.add_argument("--config", type=str, default="notebooks/slso/config/slso.toml")
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)

    paths = cfg["paths"]
    BASIN_DB = Path(paths["basin_db"])
    ZARR_PATH = Path(paths["weather_grid"])
    FORCING_CACHE = Path(paths["forcing_cache"])
    CHECKPOINT = Path("notebooks/slso/checkpoints/regularized.pt")

    # Shorter training period
    DATE_START = "2001-01-01"
    DATE_END = "2002-12-31"
    TRAIN_END = "2002-06-30"
    VAL_START = "2002-07-01"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Regularized training to combat overfitting")

    # Load basin data
    cache = BasinCache(BASIN_DB)
    hydro = cache.load(device=device)

    graph = hydro["graph"]
    territorial = hydro["territorial"]
    node_coords = hydro["node_coords"]
    n_nodes = hydro["n_nodes"]
    print(f"Nodes: {n_nodes}, Edges: {graph.n_edges}")

    # Load forcing
    forcing = extract_forcing(
        zarr_path=ZARR_PATH,
        node_coords=node_coords,
        node_elev=None,
        date_start=DATE_START,
        date_end=DATE_END,
        cache_nc=Path("/tmp/forcing_reg.nc"),
        device=device,
    )

    ds_time = xr.open_dataset(Path("/tmp/forcing_reg.nc"))
    all_dates = ds_time.time.sel(time=slice(DATE_START, DATE_END)).values
    ds_time.close()

    # Load observations
    obs = cache.load_observations(
        date_start=DATE_START,
        date_end=DATE_END,
        min_valid_days=100,
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

    train_sl = dates_to_slice(all_dates, DATE_START, TRAIN_END)
    val_sl = dates_to_slice(all_dates, VAL_START, DATE_END)

    doy = torch.tensor(
        [int(pd.Timestamp(d).day_of_year) for d in all_dates],
        dtype=torch.long, device=device,
    )

    # Model with MORE regularization to reduce overfitting
    model = YHydro(
        n_nodes=n_nodes,
        n_territorial=territorial.n_features,
        n_forcing=6,
        context_window=30,
        residual_history=14,
        max_travel_time=20,
        use_temporal=False,
        use_residual=False,
        use_travel_time_attn=False,
        use_temperature=True,
        dropout=0.3,  # INCREASED dropout to reduce overfitting
        param_mode="nerf",
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    withdrawals = cache.load_withdrawals(
        date_start=DATE_START,
        date_end=DATE_END,
        device=device,
    )

    # Lower learning rate + higher weight decay for regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

    print("\n=== Regularized Training ===")
    print("- Increased dropout (0.3)")
    print("- Higher weight decay (1e-4)")
    print("- Lower learning rate (5e-4)")
    print("- Warm initialization")

    CHUNK_SIZE = 60  # Smaller chunks
    WARMUP = 30

    best_val_kge = -999
    no_improvement = 0

    for epoch in range(150):
        model.train()
        epoch_losses = []
        epoch_kges = []

        # Use warm initialization
        state = HydroState.default_warm(n_nodes, device=device)

        # Train on chunks
        train_days = train_sl.stop - train_sl.start
        n_chunks = (train_days - WARMUP) // CHUNK_SIZE

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

            # Compute loss for valid stations
            chunk_losses = []
            for i in range(n_stations):
                q_o = Q_obs_train[:, i]
                q_s = Q_sim_train[:, i]

                valid = ~torch.isnan(q_o)
                if valid.sum() < 10:
                    continue

                q_o_v = q_o[valid]
                q_s_v = q_s[valid]

                # Simple KGE loss
                kge_loss, kge_info = differentiable_composite_kge_loss(q_o_v, q_s_v, alpha=0.7)
                chunk_losses.append(kge_loss)
                epoch_kges.append(kge_info["kge"].item())

            if len(chunk_losses) > 0:
                loss = torch.stack(chunk_losses).mean()
                epoch_losses.append(loss.item())

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Tighter clipping
                optimizer.step()

            # Detach state for next chunk
            state = state.detach()

        # Validation every 5 epochs
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                # Run validation
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

                print(f"Epoch {epoch:3d} | Loss: {mean_loss:.4f} | Train KGE: {mean_train_kge:.3f} | Val KGE: {mean_val_kge:.3f}")

                if mean_val_kge > best_val_kge:
                    best_val_kge = mean_val_kge
                    torch.save(model.state_dict(), CHECKPOINT)
                    print(f"  -> Best model saved (KGE: {best_val_kge:.3f})")
                    no_improvement = 0
                else:
                    no_improvement += 1

                # Early stopping if val KGE is good or no improvement
                if mean_val_kge > 0.3:
                    print(f"\n=== Validation KGE reached: {mean_val_kge:.3f} ===")
                    break

                if no_improvement > 8:  # 40 epochs without improvement
                    print(f"\n=== No improvement for {no_improvement*5} epochs, stopping ===")
                    break

    print(f"\n=== Training complete ===")
    print(f"Best validation KGE: {best_val_kge:.3f}")


if __name__ == "__main__":
    main()