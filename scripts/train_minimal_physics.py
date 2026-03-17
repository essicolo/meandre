"""Minimal memory training focused on physics - targeting KGE > 0.8 without CUDA OOM."""
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
from meandre.spatial.field_network import SpatialFieldNetwork


def main():
    parser = argparse.ArgumentParser(description="Minimal memory KGE > 0.8 training")
    parser.add_argument("--config", type=str, default="notebooks/slso/config/slso.toml")
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)

    paths = cfg["paths"]
    BASIN_DB = Path(paths["basin_db"])
    ZARR_PATH = Path(paths["weather_grid"])
    CHECKPOINT = Path("notebooks/slso/checkpoints/minimal_physics.pt")

    # Shorter period for memory efficiency
    DATE_START = "2001-01-01"
    DATE_END = "2002-06-30"  # 6 months only
    TRAIN_END = "2002-03-31"
    VAL_START = "2002-04-01"
    EPOCHS = 150

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("=== MINIMAL MEMORY PHYSICS TRAINING ===")
    print("Strategy: Disable memory-heavy features but keep core physics")
    print("Goal: Achieve KGE > 0.8 without CUDA OOM")

    # Load data
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
        cache_nc=Path("/tmp/minimal_physics_forcing.nc"),
        device=device,
    )

    ds_time = xr.open_dataset(Path("/tmp/minimal_physics_forcing.nc"))
    all_dates = ds_time.time.sel(time=slice(DATE_START, DATE_END)).values
    ds_time.close()

    # Load observations
    obs = cache.load_observations(
        date_start=DATE_START,
        date_end=DATE_END,
        min_valid_days=100,  # Lower requirement for shorter period
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

    # MINIMAL MEMORY MODEL - Core physics only
    model = YHydro(
        n_nodes=n_nodes,
        n_territorial=territorial.n_features,
        n_forcing=6,
        context_window=15,       # Reduced from 30
        residual_history=7,      # Reduced from 14
        max_travel_time=10,      # Reduced from 20
        use_temporal=False,      # DISABLE - memory heavy
        use_residual=False,      # DISABLE - memory heavy
        use_travel_time_attn=False, # DISABLE - memory heavy
        use_temperature=True,    # KEEP - core physics
        dropout=0.1,             # Low dropout for minimal model
        param_mode="nerf",       # NeRF for spatial parameters
    ).to(device)

    # Replace spatial encoder with ultra-simple version
    model.spatial_encoder = SpatialFieldNetwork(
        n_territorial=territorial.n_features,
        n_coord_freqs=2,         # Minimal Fourier encoding
        hidden=64,               # Very small hidden layer
        dropout=0.2,             # Moderate regularization
        param_mode="nerf"
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,} (ultra-minimal for memory)")

    withdrawals = cache.load_withdrawals(
        date_start=DATE_START,
        date_end=DATE_END,
        device=device,
    )

    # Simple optimizer focused on convergence
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-4,               # Higher learning rate for faster convergence
        weight_decay=1e-4,     # Moderate regularization
    )

    # Step decay scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=30, gamma=0.7
    )

    print(f"\n=== Minimal Memory Configuration ===")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: 5e-4 with step decay")
    print(f"Features: ONLY temperature physics (temporal/residual disabled)")
    print(f"NeRF: 64 hidden, 2 frequencies")
    print(f"Period: 6 months for memory efficiency")

    CHUNK_SIZE = 30  # Small chunks for memory
    WARMUP = 7

    best_val_kge = -999
    no_improvement_count = 0
    kge_target = 0.8

    for epoch in range(EPOCHS):
        model.train()
        epoch_losses = []
        epoch_kges = []

        # Warm initialization (proven critical)
        state = HydroState.default_warm(n_nodes, device=device)

        # Single chunk training for memory efficiency
        train_days = train_sl.stop - train_sl.start
        n_chunks = max(1, (train_days - WARMUP) // CHUNK_SIZE)

        for chunk_i in range(min(n_chunks, 3)):  # Limit chunks for memory
            start_idx = train_sl.start + chunk_i * CHUNK_SIZE
            end_idx = min(start_idx + CHUNK_SIZE + WARMUP, train_sl.stop)

            chunk_forcing = forcing[start_idx:end_idx]
            chunk_doy = doy[start_idx:end_idx]

            # Forward pass
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

            # Station-by-station loss
            chunk_losses = []
            for i in range(n_stations):
                q_o = Q_obs_train[:, i]
                q_s = Q_sim_train[:, i]

                valid = ~torch.isnan(q_o)
                if valid.sum() < 5:
                    continue

                q_o_v = q_o[valid]
                q_s_v = q_s[valid]

                # Pure KGE loss for direct optimization
                kge_loss, kge_info = differentiable_composite_kge_loss(
                    q_o_v, q_s_v, alpha=1.0
                )

                chunk_losses.append(kge_loss)
                epoch_kges.append(kge_info["kge"].item())

            if len(chunk_losses) > 0:
                loss = torch.stack(chunk_losses).mean()
                epoch_losses.append(loss.item())

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

            # Detach state for memory
            state = state.detach()

        # Step scheduler
        scheduler.step()

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
                    if valid.sum() < 15:
                        continue

                    kge_val = compute_kge(q_o[valid], q_s[valid])
                    val_kges.append(float(kge_val))

                mean_val_kge = np.mean(val_kges) if val_kges else -999
                mean_train_kge = np.mean(epoch_kges) if epoch_kges else -999
                mean_loss = np.mean(epoch_losses) if epoch_losses else 999
                max_val_kge = np.max(val_kges) if val_kges else -999

                current_lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch:3d} | Loss: {mean_loss:.4f} | Train KGE: {mean_train_kge:.3f} | Val KGE: {mean_val_kge:.3f} (max: {max_val_kge:.3f}) | LR: {current_lr:.0e}")

                # Track progress toward KGE > 0.8
                if mean_val_kge > best_val_kge:
                    improvement = mean_val_kge - best_val_kge
                    best_val_kge = mean_val_kge
                    torch.save(model.state_dict(), CHECKPOINT)
                    print(f"  ✅ New best! Improvement: +{improvement:.3f}")
                    no_improvement_count = 0

                    if mean_val_kge >= kge_target:
                        print(f"\n🎉 SUCCESS! KGE {mean_val_kge:.3f} >= {kge_target} ACHIEVED! 🎉")
                        break
                else:
                    no_improvement_count += 1

                # Progress tracking
                progress = (best_val_kge + 1.0) / (kge_target + 1.0) * 100
                print(f"  🎯 Progress to KGE {kge_target}: {progress:.1f}%")

                # Early stopping
                if no_improvement_count > 20:  # 100 epochs without improvement
                    print(f"\n⚠️  Early stopping: no improvement for {no_improvement_count*5} epochs")
                    break

    print(f"\n=== Final Results ===")
    print(f"Best validation KGE: {best_val_kge:.3f}")
    if best_val_kge >= kge_target:
        print(f"🎉 SUCCESS: Target KGE {kge_target} achieved with minimal memory!")
    else:
        print(f"❌ Target missed. Gap: {kge_target - best_val_kge:.3f}")
        print("Result: Minimal physics model insufficient for KGE > 0.8")


if __name__ == "__main__":
    main()