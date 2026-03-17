"""Test script to diagnose training bottleneck."""
import os
import time
import tomllib
from pathlib import Path

import torch
import numpy as np
import pandas as pd

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from meandre.data.basin_cache import BasinCache
from meandre.model import YHydro
from meandre.utils.state import HydroState

def main():
    # Load config
    with open("notebooks/slso/config/slso.toml", "rb") as f:
        cfg = tomllib.load(f)

    BASIN_DB = Path(cfg["paths"]["basin_db"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Basin data
    cache = BasinCache(BASIN_DB)
    hydro = cache.load(device=device)

    graph = hydro["graph"]
    territorial = hydro["territorial"]
    node_coords = hydro["node_coords"]
    n_nodes = hydro["n_nodes"]
    print(f"Nodes: {n_nodes}, Edges: {graph.n_edges}")

    # Create a minimal model
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
    print(f"Parameters: {n_params:,}")

    # Create dummy forcing for testing
    TEST_STEPS = 100
    forcing = torch.randn(TEST_STEPS, n_nodes, 6, device=device) * 0.1
    forcing[:, :, 0] = torch.abs(forcing[:, :, 0]) * 10  # Precip positive

    # Dummy withdrawals
    from meandre.routing.withdrawals import WithdrawalData
    withdrawals = WithdrawalData(
        net=torch.zeros(TEST_STEPS, n_nodes, device=device),
    )

    # Day of year
    doy = torch.tensor([i % 365 + 1 for i in range(TEST_STEPS)], device=device)

    print(f"\nRunning test simulation with {TEST_STEPS} timesteps...")

    # Time the forward pass
    model.eval()
    with torch.no_grad():
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()

        Q_sim, states = model.simulate(
            forcing=forcing,
            initial_state=HydroState.zeros(n_nodes, device=device),
            graph=graph,
            node_coords=node_coords,
            territorial=territorial,
            withdrawals=withdrawals,
            day_of_year=doy,
        )

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - start

    print(f"Simulation completed in {elapsed:.2f} seconds")
    print(f"Time per timestep: {elapsed/TEST_STEPS*1000:.1f} ms")
    print(f"Output shape: {Q_sim.shape}")
    print(f"Max Q: {Q_sim.max().item():.2f}, Mean Q: {Q_sim.mean().item():.2f}")

    # Test backward pass
    print("\nTesting backward pass...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Smaller batch for backward
    TRAIN_STEPS = 30
    forcing_train = forcing[:TRAIN_STEPS]
    withdrawals_train = WithdrawalData(
        net=withdrawals.net[:TRAIN_STEPS],
    )
    doy_train = doy[:TRAIN_STEPS]

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()

    Q_sim, states = model.simulate(
        forcing=forcing_train,
        initial_state=HydroState.zeros(n_nodes, device=device),
        graph=graph,
        node_coords=node_coords,
        territorial=territorial,
        withdrawals=withdrawals_train,
        day_of_year=doy_train,
    )

    # Simple loss
    loss = Q_sim.mean()
    loss.backward()
    optimizer.step()

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.time() - start

    print(f"Backward pass completed in {elapsed:.2f} seconds")
    print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB" if torch.cuda.is_available() else "N/A")

    print("\n✓ Test completed successfully!")


if __name__ == "__main__":
    main()