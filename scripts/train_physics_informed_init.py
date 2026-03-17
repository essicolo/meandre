"""Training with physics-informed parameter initialization."""
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


def physics_informed_init(model):
    """Initialize static parameters to physically reasonable values."""
    with torch.no_grad():
        # Define target physics values (middle of reasonable ranges)
        target_values = torch.tensor([
            # K_sat (m/day) - hydraulic conductivity
            0.5,  # K_sat_1 (surface) - moderate permeability
            0.2,  # K_sat_2 (subsurface) - lower permeability
            0.1,  # K_sat_3 (deep) - lowest permeability

            # Porosity (dimensionless) - void fraction
            0.45, 0.45, 0.45,  # porosity_1,2,3 - typical soil porosity

            # Water content at field capacity
            0.30, 0.30, 0.30,  # theta_fc_1,2,3 - field capacity

            # Water content at wilting point
            0.15, 0.15, 0.15,  # theta_wp_1,2,3 - wilting point

            # Soil layer depths (mm)
            200.0, 300.0, 500.0,  # depths - surface, subsurface, deep

            # Snow parameters
            5.0,   # C_f - melt factor (mm/°C/day)
            0.0,   # T_melt - melt temperature (°C)
            2.0,   # T_snow - snow temperature threshold (°C)

            # Interception and surface
            0.5,   # interception capacity (mm)
            0.03,  # manning_n - Manning's roughness
            0.5,   # frost_alpha - frost effect factor

            # Subsurface flow
            0.1,   # f_wetland - wetland fraction
            0.5,   # slope_factor - topographic effect
            0.01,  # krec - baseflow recession (1/day)

            # Groundwater
            0.005, # k_gw - groundwater recession (1/day)

            # Temperature
            8.0,   # T_gw - groundwater temperature (°C)
            0.2,   # K_atm - atmospheric heat exchange (1/day)
            0.05,  # alpha_T - thermal damping (1/day)
        ], device=model.spatial_encoder.static_params.device)

        print("=== PHYSICS-INFORMED INITIALIZATION ===")
        print("Target physical values:")
        print(f"K_sat: {target_values[0:3].cpu().numpy()} m/day")
        print(f"Porosity: {target_values[3:6].cpu().numpy()}")
        print(f"Field capacity: {target_values[6:9].cpu().numpy()}")
        print(f"Wilting point: {target_values[9:12].cpu().numpy()}")
        print(f"Depths: {target_values[12:15].cpu().numpy()} mm")
        print(f"Snow C_f: {target_values[15]:.1f} mm/°C/day")
        print(f"Manning n: {target_values[17]:.3f}")
        print(f"GW temp: {target_values[23]:.1f} °C")

        # Now we need to find raw values that will produce these targets
        # after applying constraints. This requires inverting the constraint functions.

        # For now, let's set the raw parameters to produce reasonable outputs
        # This is approximate - ideally we'd invert the exact constraint functions
        raw_init = torch.zeros_like(model.spatial_encoder.static_params)

        # For tanh constraints: tanh(x) = target => x = atanh(target_normalized)
        # For sigmoid constraints: sigmoid(x) = target => x = logit(target)

        # Approximation: start with small positive values that will produce
        # reasonable outputs after sigmoid/tanh transforms
        raw_init[0:3] = torch.logit(torch.tensor([0.4, 0.2, 0.1]))  # K_sat
        raw_init[3:6] = torch.logit(torch.tensor([0.45, 0.45, 0.45]))  # porosity
        raw_init[6:9] = torch.logit(torch.tensor([0.3, 0.3, 0.3]))  # theta_fc
        raw_init[9:12] = torch.logit(torch.tensor([0.15, 0.15, 0.15]))  # theta_wp
        raw_init[12:15] = torch.tensor([0.5, 1.0, 1.5])  # depths (will be scaled)
        raw_init[15] = torch.tensor(0.5)  # C_f
        raw_init[16] = torch.tensor(-1.0)  # T_melt (around 0°C)
        raw_init[17] = torch.tensor(0.2)  # T_snow
        raw_init[18] = torch.tensor(0.0)  # interception
        raw_init[19] = torch.logit(torch.tensor(0.03))  # manning_n
        raw_init[20:] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0])  # remaining

        model.spatial_encoder.static_params.data = raw_init

        # Test the initialization by checking actual output values
        test_coords = torch.zeros(1, 2, device=raw_init.device)
        test_territorial = torch.zeros(1, 17, device=raw_init.device)
        params = model.spatial_encoder(test_coords, test_territorial)

        print("\nActual initialized values:")
        print(f"K_sat: {torch.stack([params.K_sat_1, params.K_sat_2, params.K_sat_3])[0].cpu().numpy():.3f}")
        print(f"Porosity: {torch.stack([params.porosity_1, params.porosity_2, params.porosity_3])[0].cpu().numpy():.3f}")
        print(f"Field capacity: {torch.stack([params.theta_fc_1, params.theta_fc_2, params.theta_fc_3])[0].cpu().numpy():.3f}")
        print(f"C_f: {params.C_f[0]:.1f}")
        print(f"T_gw: {params.T_gw[0]:.1f}")


def main():
    config_path = "notebooks/slso/config/slso.toml"
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    paths = cfg["paths"]
    BASIN_DB = Path(paths["basin_db"])
    ZARR_PATH = Path(paths["weather_grid"])

    # 6-month training period for reasonable performance test
    DATE_START = "2002-01-01"
    DATE_END = "2002-06-30"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== PHYSICS-INFORMED INITIALIZATION TRAINING ===")
    print("Strategy: Initialize parameters to physically reasonable values")
    print("Goal: Achieve KGE > 0.8 with better parameter starting point")

    # Load data
    cache = BasinCache(BASIN_DB)
    hydro = cache.load(device=device)

    graph = hydro["graph"]
    territorial = hydro["territorial"]
    node_coords = hydro["node_coords"]
    n_nodes = hydro["n_nodes"]

    print(f"Nodes: {n_nodes}")

    # Load forcing
    forcing = extract_forcing(
        zarr_path=ZARR_PATH,
        node_coords=node_coords,
        node_elev=None,
        date_start=DATE_START,
        date_end=DATE_END,
        cache_nc=Path("/tmp/physics_init_forcing.nc"),
        device=device,
    )

    print(f"Forcing: {forcing.shape} (days, nodes, features)")

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

    # Create station mask
    station_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
    for ni in station_indices:
        station_mask[ni] = True

    q_obs_tensor = torch.from_numpy(obs["discharge"][:, station_indices]).to(device)
    print(f"Observed discharge range: {q_obs_tensor[~torch.isnan(q_obs_tensor)].min():.2f} - {q_obs_tensor[~torch.isnan(q_obs_tensor)].max():.1f} m³/s")

    doy = torch.tensor([i % 365 + 1 for i in range(len(forcing))], dtype=torch.long, device=device)

    withdrawals = cache.load_withdrawals(
        date_start=DATE_START,
        date_end=DATE_END,
        device=device,
    )

    # Create model with STATIC parameters
    print("\nCreating model with physics-informed initialization...")
    model = YHydro(
        n_nodes=n_nodes,
        n_territorial=territorial.n_features,
        n_forcing=6,
        context_window=30,
        residual_history=14,
        max_travel_time=20,
        use_temporal=False,    # Keep minimal for memory
        use_residual=False,    # Keep minimal for memory
        use_travel_time_attn=False,  # Keep minimal for memory
        use_temperature=True,  # Core physics
        dropout=0.0,
        param_mode="static",   # Global parameters
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Apply physics-informed initialization
    physics_informed_init(model)

    # Test initial simulation
    print("\n=== INITIAL SIMULATION TEST ===")
    model.eval()
    with torch.no_grad():
        initial_state = HydroState.default_warm(n_nodes, device=device)

        Q_sim, _ = model.simulate(
            forcing=forcing[:7],  # Just 1 week test
            initial_state=initial_state,
            graph=graph,
            node_coords=node_coords,
            territorial=territorial,
            withdrawals=withdrawals[:7],
            day_of_year=doy[:7],
        )

    outlet_Q = Q_sim[:, -1].mean().item()
    print(f"Initial outlet discharge: {outlet_Q:.2f} m³/s")

    expected_range_min = q_obs_tensor[~torch.isnan(q_obs_tensor)].min().item()
    expected_range_max = q_obs_tensor[~torch.isnan(q_obs_tensor)].max().item()

    if expected_range_min <= outlet_Q <= expected_range_max * 2:
        print("✅ Initial discharge in reasonable range!")
    else:
        print(f"⚠️ Initial discharge outside expected range [{expected_range_min:.1f}, {expected_range_max:.1f}] m³/s")

    # Training configuration
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=10, verbose=True)
    criterion = CompositeKGELoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        run_name="physics_informed_init",
        db_path=Path("notebooks/slso/runs.duckdb"),
    )

    # Train
    print("\n=== TRAINING WITH PHYSICS-INFORMED INIT ===")
    trainer.train(
        forcing=forcing,
        graph=graph,
        node_coords=node_coords,
        territorial=territorial,
        withdrawals=withdrawals,
        day_of_year=doy,
        q_obs=q_obs_tensor,
        station_mask=station_mask,
        epochs=200,
        print_every=5,
    )

    print("\n=== TRAINING COMPLETE ===")


if __name__ == "__main__":
    main()