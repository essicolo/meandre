"""Smart physics initialization by reverse-engineering the constraint functions."""
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


def smart_initialize_from_constraints(model):
    """Initialize by working backward from desired physics values."""
    with torch.no_grad():
        # Target physical values we want (from constraint analysis)
        targets = {
            'K_sat_1': 0.8,    # m/day
            'K_sat_2': 0.3,    # m/day
            'K_sat_3': 0.1,    # m/day
            'porosity_1': 0.45,
            'porosity_2': 0.45,
            'porosity_3': 0.45,
            'theta_fc_1': 0.30,
            'theta_fc_2': 0.30,
            'theta_fc_3': 0.30,
            'theta_wp_1': 0.15,
            'theta_wp_2': 0.15,
            'theta_wp_3': 0.15,
            'depth_1': 250.0,  # mm
            'depth_2': 400.0,  # mm
            'depth_3': 800.0,  # mm
            'C_f': 6.0,        # mm/°C/day
            'T_melt': 0.0,     # °C
            'T_snow': 2.0,     # °C
            'interception': 1.0,  # mm
            'manning_n': 0.035,
            'frost_alpha': 0.6,
            'f_wetland': 0.15,
            'slope_factor': 1.0,
            'krec': 0.02,      # 1/day
            'k_gw': 0.008,     # 1/day
            'T_gw': 10.0,      # °C
            'K_atm': 0.3,      # 1/day
            'alpha_T': 0.08,   # 1/day
        }

        # Now we need to work backward from these constraints to raw values
        # Looking at the constraint functions, we need to invert them

        # For sigmoid constraints: target = a + b * sigmoid(raw) => raw = logit((target - a) / b)
        # For tanh constraints: target = a + b * tanh(raw) => raw = atanh((target - a) / b)
        # For additive constraints: target = a + b * raw => raw = (target - a) / b

        raw_values = torch.zeros(28, device=model.spatial_encoder.static_params.device)

        # K_sat values: 0.1 + 2.9 * sigmoid(raw) for K_sat_1
        raw_values[0] = torch.logit(torch.tensor((targets['K_sat_1'] - 0.1) / 2.9))  # K_sat_1
        raw_values[1] = torch.logit(torch.tensor((targets['K_sat_2'] - 0.05) / 1.45))  # K_sat_2
        raw_values[2] = torch.logit(torch.tensor((targets['K_sat_3'] - 0.01) / 0.49))  # K_sat_3

        # Porosity: 0.2 + 0.4 * sigmoid(raw)
        for i in range(3):
            raw_values[3 + i] = torch.logit(torch.tensor((targets[f'porosity_{i+1}'] - 0.2) / 0.4))

        # Field capacity: 0.1 + 0.4 * sigmoid(raw)
        for i in range(3):
            raw_values[6 + i] = torch.logit(torch.tensor((targets[f'theta_fc_{i+1}'] - 0.1) / 0.4))

        # Wilting point: 0.05 + 0.25 * sigmoid(raw)
        for i in range(3):
            raw_values[9 + i] = torch.logit(torch.tensor((targets[f'theta_wp_{i+1}'] - 0.05) / 0.25))

        # Depths: direct scaling (need to check exact constraint)
        # From constraint code: 50.0 + 450.0 * sigmoid(raw) for depth_1
        raw_values[12] = torch.logit(torch.tensor((targets['depth_1'] - 50.0) / 450.0))   # depth_1
        raw_values[13] = torch.logit(torch.tensor((targets['depth_2'] - 100.0) / 400.0))  # depth_2
        raw_values[14] = torch.logit(torch.tensor((targets['depth_3'] - 200.0) / 800.0))  # depth_3

        # Snow parameters
        # C_f: 2.0 + 8.0 * sigmoid(raw)
        raw_values[15] = torch.logit(torch.tensor((targets['C_f'] - 2.0) / 8.0))

        # T_melt: -2.0 + 4.0 * tanh(raw)
        raw_values[16] = torch.atanh(torch.tensor((targets['T_melt'] - (-2.0)) / 4.0))

        # T_snow: 0.0 + 4.0 * sigmoid(raw)
        raw_values[17] = torch.logit(torch.tensor((targets['T_snow'] - 0.0) / 4.0))

        # Interception: 0.1 + 1.9 * sigmoid(raw)
        raw_values[18] = torch.logit(torch.tensor((targets['interception'] - 0.1) / 1.9))

        # Manning's n: 0.01 + 0.09 * sigmoid(raw)
        raw_values[19] = torch.logit(torch.tensor((targets['manning_n'] - 0.01) / 0.09))

        # Frost alpha: 0.1 + 0.8 * sigmoid(raw)
        raw_values[20] = torch.logit(torch.tensor((targets['frost_alpha'] - 0.1) / 0.8))

        # f_wetland: 0.05 + 0.45 * sigmoid(raw)
        raw_values[21] = torch.logit(torch.tensor((targets['f_wetland'] - 0.05) / 0.45))

        # slope_factor: 0.1 + 1.9 * sigmoid(raw)
        raw_values[22] = torch.logit(torch.tensor((targets['slope_factor'] - 0.1) / 1.9))

        # krec: 0.001 + 0.199 * sigmoid(raw)
        raw_values[23] = torch.logit(torch.tensor((targets['krec'] - 0.001) / 0.199))

        # k_gw: 0.001 + 0.049 * sigmoid(raw)
        raw_values[24] = torch.logit(torch.tensor((targets['k_gw'] - 0.001) / 0.049))

        # T_gw: 3.0 + 10.0 * sigmoid(raw)
        raw_values[25] = torch.logit(torch.tensor((targets['T_gw'] - 3.0) / 10.0))

        # K_atm: 0.05 + 0.5 * sigmoid(raw)
        raw_values[26] = torch.logit(torch.tensor((targets['K_atm'] - 0.05) / 0.5))

        # alpha_T: 0.01 + 0.14 * sigmoid(raw)
        raw_values[27] = torch.logit(torch.tensor((targets['alpha_T'] - 0.01) / 0.14))

        # Set the parameters
        model.spatial_encoder.static_params.data = raw_values

        print("=== SMART CONSTRAINT-BASED INITIALIZATION ===")
        print("Computed raw values to produce target physics parameters")

        # Test what we actually get
        test_coords = torch.zeros(1, 2, device=raw_values.device)
        test_territorial = torch.zeros(1, 17, device=raw_values.device)
        params = model.spatial_encoder(test_coords, test_territorial)

        print("Verification - actual vs target:")
        print(f"K_sat_1: {params.K_sat_1[0]:.3f} vs {targets['K_sat_1']:.3f}")
        print(f"K_sat_2: {params.K_sat_2[0]:.3f} vs {targets['K_sat_2']:.3f}")
        print(f"K_sat_3: {params.K_sat_3[0]:.3f} vs {targets['K_sat_3']:.3f}")
        # Depth parameters - check actual attribute names available
        print(f"Available attributes: {[attr for attr in dir(params) if 'depth' in attr]}")
        print(f"C_f: {params.C_f[0]:.1f} vs {targets['C_f']:.1f}")
        print(f"T_gw: {params.T_gw[0]:.1f} vs {targets['T_gw']:.1f}")


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
    print("=== SMART CONSTRAINT INITIALIZATION TRAINING ===")
    print("Strategy: Reverse-engineer constraint functions for exact physics values")

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
        cache_nc=Path("/tmp/smart_init_forcing.nc"),
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

    # Create model with static parameters
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
        param_mode="static",
    ).to(device)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Apply smart constraint-based initialization
    smart_initialize_from_constraints(model)

    # Test initial simulation
    print("\n=== INITIAL PHYSICS VALIDATION ===")
    model.eval()
    with torch.no_grad():
        initial_state = HydroState.default_warm(n_nodes, device=device)
        Q_sim, _ = model.simulate(
            forcing=forcing[:14],
            initial_state=initial_state,
            graph=graph,
            node_coords=node_coords,
            territorial=territorial,
            withdrawals=withdrawals.slice_time(0, 14),
            day_of_year=doy[:14],
        )

    outlet_Q = Q_sim[:, -1].mean().item()
    expected_Q = q_obs_tensor[~torch.isnan(q_obs_tensor)].median().item()

    print(f"Initial outlet Q: {outlet_Q:.2f} m³/s")
    print(f"Expected outlet Q: {expected_Q:.2f} m³/s")
    print(f"Scale ratio: {outlet_Q/expected_Q:.2f}")

    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=8e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.75, patience=10)
    criterion = CompositeKGELoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        run_name="smart_init",
        db_path=Path("notebooks/slso/runs.duckdb"),
    )

    print("\n=== TRAINING WITH SMART INITIALIZATION ===")
    trainer.train(
        forcing=forcing,
        graph=graph,
        node_coords=node_coords,
        territorial=territorial,
        withdrawals=withdrawals,
        day_of_year=doy,
        q_obs=q_obs_tensor,
        station_mask=station_mask,
        epochs=250,
        print_every=8,
    )

    print("\n=== SMART INITIALIZATION TRAINING COMPLETE ===")


if __name__ == "__main__":
    main()