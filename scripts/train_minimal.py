"""Minimal training script to isolate the issue."""
import logging
import os
import time
import tomllib
from pathlib import Path

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import numpy as np
import pandas as pd
import xarray as xr

from meandre.data.basin_cache import BasinCache
from meandre.data.gridded_forcing import extract_forcing
from meandre.model import YHydro
from meandre.training.loss import CompositeKGELoss
from meandre.utils.state import HydroState


def main():
    logging.info("Starting minimal training")

    # Load config
    with open("notebooks/slso/config/slso.toml", "rb") as f:
        cfg = tomllib.load(f)

    paths = cfg["paths"]
    BASIN_DB = Path(paths["basin_db"])
    FORCING_CACHE = Path(paths["forcing_cache"])

    # Use minimal time period
    DATE_START = "2000-01-01"
    DATE_END = "2000-03-31"  # Just 3 months
    TRAIN_START = "2000-02-01"
    TRAIN_END = "2000-03-31"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    # Basin data
    logging.info("Loading basin data...")
    cache = BasinCache(BASIN_DB)
    hydro = cache.load(device=device)

    graph = hydro["graph"]
    territorial = hydro["territorial"]
    node_coords = hydro["node_coords"]
    n_nodes = hydro["n_nodes"]
    logging.info(f"Nodes: {n_nodes}, Edges: {graph.n_edges}")

    # Load minimal forcing from cache
    logging.info("Loading forcing...")
    from meandre.data.gridded_forcing import extract_forcing
    forcing = extract_forcing(
        zarr_path=Path(cfg["paths"]["weather_grid"]),
        node_coords=node_coords,
        node_elev=None,
        date_start=DATE_START,
        date_end=DATE_END,
        cache_nc=FORCING_CACHE,
        device=device,
    )

    ds = xr.open_dataset(FORCING_CACHE)
    all_dates = ds.time.sel(time=slice(DATE_START, DATE_END)).values
    ds.close()
    logging.info(f"Forcing shape: {forcing.shape}")

    # Simple observations
    n_timesteps = len(all_dates)
    q_obs = torch.randn(n_timesteps, 10, device=device).abs() * 10  # Fake observations
    station_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
    station_mask[:10] = True  # First 10 nodes have observations

    # Minimal withdrawals
    from meandre.routing.withdrawals import WithdrawalData
    withdrawals = WithdrawalData(net=torch.zeros(n_timesteps, n_nodes, device=device))

    # Day of year
    doy = torch.tensor(
        [int(pd.Timestamp(d).day_of_year) for d in all_dates],
        dtype=torch.long, device=device,
    )

    # Model
    logging.info("Creating model...")
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
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Parameters: {n_params:,}")

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = CompositeKGELoss(alpha=0.5, eps=1.0, per_station=False)

    # Training slices
    days = all_dates.astype("datetime64[D]")
    train_start_idx = int(np.searchsorted(days, np.datetime64(TRAIN_START, "D")))
    train_end_idx = int(np.searchsorted(days, np.datetime64(TRAIN_END, "D"), side="right"))

    logging.info(f"Training from index {train_start_idx} to {train_end_idx}")

    # Single training step
    logging.info("Running single training iteration...")
    model.train()
    optimizer.zero_grad()

    # Simulate with timing
    start_time = time.time()
    logging.info("Starting simulation...")

    Q_sim, _ = model.simulate(
        forcing=forcing[train_start_idx:train_end_idx],
        initial_state=HydroState.zeros(n_nodes, device=device),
        graph=graph,
        node_coords=node_coords,
        territorial=territorial,
        withdrawals=WithdrawalData(net=withdrawals.net[train_start_idx:train_end_idx]),
        day_of_year=doy[train_start_idx:train_end_idx],
    )

    sim_time = time.time() - start_time
    logging.info(f"Simulation completed in {sim_time:.2f}s")
    logging.info(f"Q_sim shape: {Q_sim.shape}, mean: {Q_sim.mean():.3f}")

    # Compute loss
    q_obs_train = q_obs[train_start_idx:train_end_idx]
    loss, components = loss_fn(
        q_obs=q_obs_train,
        q_sim=Q_sim,
        station_mask=station_mask,
    )

    logging.info(f"Loss: {loss.item():.4f}")
    logging.info(f"Components: {components}")

    # Backward pass
    start_time = time.time()
    loss.backward()
    optimizer.step()
    backward_time = time.time() - start_time
    logging.info(f"Backward pass completed in {backward_time:.2f}s")

    # Memory usage
    if torch.cuda.is_available():
        mem_gb = torch.cuda.memory_allocated() / 1e9
        logging.info(f"GPU memory used: {mem_gb:.2f} GB")

    logging.info("✓ Training step completed successfully!")


if __name__ == "__main__":
    main()