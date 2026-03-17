"""Ultra-simple training without withdrawal test - straight to training."""
import argparse
import tomllib
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from meandre.data.basin_cache import BasinCache
from meandre.data.gridded_forcing import extract_forcing
from meandre.model import YHydro
from meandre.utils.metrics import kge as compute_kge
from meandre.utils.state import HydroState
from meandre.training.trainer import Trainer
from meandre.training.loss import CompositeKGELoss


def physics_init_static_params(model):
    """Initialize static parameters to physics values that should work."""
    with torch.no_grad():
        # These are physically reasonable values that should give proper discharge
        params = torch.tensor([
            # Hydraulic conductivity (m/day) - critical for flow
            1.0,   # K_sat_1 (surface) - higher for good infiltration
            0.4,   # K_sat_2 (subsurface)
            0.15,   # K_sat_3 (deep) - for baseflow

            # Porosity (0.4-0.5 typical for soil)
            0.45, 0.45, 0.45,

            # Field capacity (typically 0.25-0.35)
            0.32, 0.30, 0.28,

            # Wilting point (typically 0.1-0.2)
            0.16, 0.15, 0.14,

            # Soil depths (mm) - critical for storage
            300.0,  # surface layer
            500.0,  # subsurface layer
            1000.0, # deep layer

            # Snow parameters
            7.0,    # C_f - strong melt rate
            0.0,    # T_melt - 0°C
            1.5,    # T_snow - snow threshold

            # Surface parameters
            1.2,    # interception (mm)
            0.04,   # manning_n - moderate roughness
            0.7,    # frost_alpha

            # Flow parameters
            0.2,    # f_wetland
            1.2,    # slope_factor
            0.025,  # krec - good recession

            # Groundwater
            0.01,   # k_gw - moderate GW recession

            # Temperature
            12.0,   # T_gw - warmer groundwater
            0.4,    # K_atm - good heat exchange
            0.1,    # alpha_T - thermal damping
        ], device=model.spatial_encoder.static_params.device)

        # Directly set the static parameters
        model.spatial_encoder.static_params.data = params

        print("=== PHYSICS INITIALIZATION (NO TEST) ===")
        print(f"K_sat values: {params[0:3].cpu().numpy()} m/day")
        print(f"Soil depths: {params[12:15].cpu().numpy()} mm")
        print(f"Snow melt factor: {params[15]:.1f} mm/°C/day")


def main():
    config_path = "notebooks/slso/config/slso.toml"
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    paths = cfg["paths"]
    BASIN_DB = Path(paths["basin_db"])
    ZARR_PATH = Path(paths["weather_grid"])

    # 6 months for good training
    DATE_START = "2002-01-01"
    DATE_END = "2002-06-30"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== NO-TEST PHYSICS TRAINING ===")
    print("Strategy: Skip testing, straight to physics training")

    # Load data
    cache = BasinCache(BASIN_DB)
    hydro = cache.load(device=device)

    graph = hydro["graph"]
    territorial = hydro["territorial"]
    node_coords = hydro["node_coords"]
    n_nodes = hydro["n_nodes"]

    print(f"Basin: {n_nodes} nodes")

    # Load forcing
    forcing = extract_forcing(
        zarr_path=ZARR_PATH,
        node_coords=node_coords,
        node_elev=None,
        date_start=DATE_START,
        date_end=DATE_END,
        cache_nc=Path("/tmp/no_test_forcing.nc"),
        device=device,
    )

    # Load observations
    obs = cache.load_observations(
        date_start=DATE_START,
        date_end=DATE_END,
        min_valid_days=100,
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

    # Create ultra-simple model - static parameters only
    print("\n=== CREATING STATIC PHYSICS MODEL ===")
    model = YHydro(
        n_nodes=n_nodes,
        n_territorial=territorial.n_features,
        n_forcing=6,
        context_window=30,
        residual_history=14,
        max_travel_time=20,
        use_temporal=False,     # No temporal complexity
        use_residual=False,     # No residual complexity
        use_travel_time_attn=False,  # No attention complexity
        use_temperature=True,   # Keep temperature physics
        dropout=0.0,           # No regularization complexity
        param_mode="static",   # Static parameters - simplest possible
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Apply physics initialization
    physics_init_static_params(model)

    # Ultra-simple training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=12)
    criterion = CompositeKGELoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        run_name="no_test",
        db_path=Path("notebooks/slso/runs.duckdb"),
    )

    print("\n=== STARTING NO-TEST TRAINING ===")
    print("Physics: Static parameters with direct initialization")
    print("Target: KGE > 0.8")

    # Train for extended epochs
    trainer.train(
        forcing=forcing,
        graph=graph,
        node_coords=node_coords,
        territorial=territorial,
        withdrawals=withdrawals,
        day_of_year=doy,
        q_obs=q_obs_tensor,
        station_mask=station_mask,
        epochs=400,  # Extended training
        print_every=15,
    )

    print("\n=== NO-TEST TRAINING COMPLETE ===")


if __name__ == "__main__":
    main()