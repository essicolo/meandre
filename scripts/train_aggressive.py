"""Aggressive learning approach with higher learning rates and targeted physics."""
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


def aggressive_physics_init(model):
    """Aggressive physics initialization targeting discharge scale."""
    with torch.no_grad():
        # Targeting higher discharge by optimizing key flow parameters
        params = torch.tensor([
            # Hydraulic conductivity - AGGRESSIVE for high flow
            2.5,   # K_sat_1 - very high surface flow
            1.0,   # K_sat_2 - high subsurface
            0.3,   # K_sat_3 - moderate baseflow

            # Porosity - high capacity
            0.50, 0.48, 0.46,

            # Field capacity - efficient storage
            0.35, 0.33, 0.30,

            # Wilting point - reasonable extraction
            0.18, 0.16, 0.14,

            # Soil depths - DEEP for big storage
            400.0,  # surface layer
            600.0,  # subsurface layer
            1200.0, # deep layer - massive storage

            # Snow parameters - FAST melting
            10.0,   # C_f - aggressive melt rate
            0.0,    # T_melt - 0°C
            1.0,    # T_snow - quick snow accumulation

            # Surface parameters
            2.0,    # interception - high capacity
            0.05,   # manning_n - fast flow
            0.8,    # frost_alpha

            # Flow parameters - AGGRESSIVE
            0.3,    # f_wetland - high wetland fraction
            1.5,    # slope_factor - amplify slopes
            0.04,   # krec - fast baseflow recession

            # Groundwater - ACTIVE
            0.02,   # k_gw - fast GW response

            # Temperature - WARM
            15.0,   # T_gw - warm groundwater
            0.6,    # K_atm - strong heat exchange
            0.15,   # alpha_T - thermal damping
        ], device=model.spatial_encoder.static_params.device)

        model.spatial_encoder.static_params.data = params

        print("=== AGGRESSIVE PHYSICS INITIALIZATION ===")
        print(f"K_sat (aggressive): {params[0:3].cpu().numpy()} m/day")
        print(f"Deep soil depths: {params[12:15].cpu().numpy()} mm")
        print(f"Fast snow melt: {params[15]:.1f} mm/°C/day")
        print(f"Manning's n (fast): {params[17]:.3f}")


def main():
    config_path = "notebooks/slso/config/slso.toml"
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    paths = cfg["paths"]
    BASIN_DB = Path(paths["basin_db"])
    ZARR_PATH = Path(paths["weather_grid"])

    DATE_START = "2002-01-01"
    DATE_END = "2002-06-30"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== AGGRESSIVE LEARNING TRAINING ===")
    print("Strategy: High learning rates + aggressive physics")

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
        cache_nc=Path("/tmp/aggressive_forcing.nc"),
        device=device,
    )

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

    # Create static model for aggressive training
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
        dropout=0.0,
        param_mode="static",  # Static for focused learning
    ).to(device)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Apply aggressive initialization
    aggressive_physics_init(model)

    # AGGRESSIVE training setup - high learning rates
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-3,  # HIGH learning rate
        weight_decay=2e-3,  # HIGH weight decay for regularization
        betas=(0.9, 0.999),  # Default momentum
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=8)
    criterion = CompositeKGELoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        run_name="aggressive",
        db_path=Path("notebooks/slso/runs.duckdb"),
    )

    print("\n=== STARTING AGGRESSIVE TRAINING ===")
    print("Physics: Aggressive flow parameters + high learning rate")
    print("Target: KGE > 0.8 via aggressive optimization")

    trainer.train(
        forcing=forcing,
        graph=graph,
        node_coords=node_coords,
        territorial=territorial,
        withdrawals=withdrawals,
        day_of_year=doy,
        q_obs=q_obs_tensor,
        station_mask=station_mask,
        epochs=350,  # Extended training
        print_every=12,
    )

    print("\n=== AGGRESSIVE TRAINING COMPLETE ===")


if __name__ == "__main__":
    main()