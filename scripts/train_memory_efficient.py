"""Ultra memory-efficient training with minimal features but proper physics."""
import tomllib
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from meandre.data.basin_cache import BasinCache
from meandre.data.gridded_forcing import extract_forcing
from meandre.model import YHydro
from meandre.utils.metrics import kge as compute_kge
from meandre.training.trainer import Trainer
from meandre.training.loss import CompositeKGELoss


def main():
    config_path = "notebooks/slso/config/slso.toml"
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    paths = cfg["paths"]
    BASIN_DB = Path(paths["basin_db"])
    ZARR_PATH = Path(paths["weather_grid"])

    # Short period for memory efficiency
    DATE_START = "2002-01-01"
    DATE_END = "2002-04-30"  # 4 months only

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== ULTRA MEMORY EFFICIENT TRAINING ===")
    print("Strategy: Absolute minimal features with physics")

    cache = BasinCache(BASIN_DB)
    hydro = cache.load(device=device)

    graph = hydro["graph"]
    territorial = hydro["territorial"]
    node_coords = hydro["node_coords"]
    n_nodes = hydro["n_nodes"]

    print(f"Basin: {n_nodes} nodes")

    forcing = extract_forcing(
        zarr_path=ZARR_PATH,
        node_coords=node_coords,
        node_elev=None,
        date_start=DATE_START,
        date_end=DATE_END,
        cache_nc=Path("/tmp/memory_efficient_forcing.nc"),
        device=device,
    )

    obs = cache.load_observations(
        date_start=DATE_START,
        date_end=DATE_END,
        min_valid_days=60,  # Lower threshold for short period
    )
    station_node_map = obs["station_node_map"]
    station_indices = sorted(set(station_node_map.values()))
    n_stations = len(station_indices)

    print(f"Observations: {n_stations} stations")

    station_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
    for ni in station_indices:
        station_mask[ni] = True

    q_obs_tensor = torch.from_numpy(obs["discharge"][:, station_indices]).to(device)
    doy = torch.tensor([i % 365 + 1 for i in range(len(forcing))], dtype=torch.long, device=device)

    withdrawals = cache.load_withdrawals(
        date_start=DATE_START,
        date_end=DATE_END,
        device=device,
    )

    # Ultra minimal model for memory efficiency
    model = YHydro(
        n_nodes=n_nodes,
        n_territorial=territorial.n_features,
        n_forcing=6,
        context_window=15,        # Reduced from 30
        residual_history=7,       # Reduced from 14
        max_travel_time=10,       # Reduced from 20
        use_temporal=False,       # Disabled
        use_residual=False,       # Disabled
        use_travel_time_attn=False,  # Disabled
        use_temperature=True,     # Keep core physics
        dropout=0.1,             # Light regularization
        param_mode="static",      # Simplest parameters
    ).to(device)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.9, patience=15)
    criterion = CompositeKGELoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        run_name="memory_efficient",
        db_path=Path("notebooks/slso/runs.duckdb"),
    )

    print("\n=== TRAINING MEMORY EFFICIENT MODEL ===")
    trainer.train(
        forcing=forcing,
        graph=graph,
        node_coords=node_coords,
        territorial=territorial,
        withdrawals=withdrawals,
        day_of_year=doy,
        q_obs=q_obs_tensor,
        station_mask=station_mask,
        epochs=500,  # More epochs for static parameters
        print_every=20,
    )

    print("\n=== MEMORY EFFICIENT TRAINING COMPLETE ===")


if __name__ == "__main__":
    main()