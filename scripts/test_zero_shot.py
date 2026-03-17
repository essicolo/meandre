"""Test model performance with ZERO training - just physics."""
import argparse
import tomllib
from pathlib import Path
import numpy as np
import torch

from meandre.data.basin_cache import BasinCache
from meandre.data.gridded_forcing import extract_forcing
from meandre.model import YHydro
from meandre.utils.metrics import kge as compute_kge
from meandre.utils.state import HydroState


def main():
    config_path = "notebooks/slso/config/slso.toml"
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    paths = cfg["paths"]
    BASIN_DB = Path(paths["basin_db"])
    ZARR_PATH = Path(paths["weather_grid"])

    # Test period
    DATE_START = "2002-01-01"
    DATE_END = "2002-06-30"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== ZERO-SHOT PHYSICS MODEL TEST ===")
    print("Testing if physics alone can achieve reasonable performance")
    print("NO TRAINING - just reasonable parameter initialization\n")

    # Load data
    cache = BasinCache(BASIN_DB)
    hydro = cache.load(device=device)

    graph = hydro["graph"]
    territorial = hydro["territorial"]
    node_coords = hydro["node_coords"]
    n_nodes = hydro["n_nodes"]

    # Load forcing
    forcing = extract_forcing(
        zarr_path=ZARR_PATH,
        node_coords=node_coords,
        node_elev=None,
        date_start=DATE_START,
        date_end=DATE_END,
        cache_nc=Path("/tmp/zero_shot_forcing.nc"),
        device=device,
    )

    # Load observations
    obs = cache.load_observations(
        date_start=DATE_START,
        date_end=DATE_END,
        min_valid_days=50,
    )
    station_indices = sorted(set(obs["station_node_map"].values()))
    n_stations = len(station_indices)

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

    # Create model with STATIC parameters (no NeRF complexity)
    print("Creating model with static physics parameters...")
    model = YHydro(
        n_nodes=n_nodes,
        n_territorial=territorial.n_features,
        n_forcing=6,
        context_window=30,
        residual_history=14,
        max_travel_time=20,
        use_temporal=False,  # Disable for simplicity
        use_residual=False,  # Disable for simplicity
        use_travel_time_attn=False,
        use_temperature=True,
        dropout=0.0,
        param_mode="static",  # USE STATIC PARAMETERS
    ).to(device)

    # Initialize with BETTER default parameters
    with torch.no_grad():
        # Override random initialization with reasonable defaults
        better_params = torch.tensor([
            0.5, 0.2, 0.1,     # K_sat (m/day) - reasonable for each layer
            0.45, 0.45, 0.45,  # porosity - typical soil
            0.30, 0.30, 0.30,  # theta_fc - field capacity
            0.15, 0.15, 0.15,  # theta_wp - wilting point
            200, 300, 500,     # depths (mm) - reasonable soil layers
            5.0,               # C_f - melt factor
            0.0,               # T_melt
            2.0,               # T_snow
            0.5,               # interception capacity
            0.03,              # manning_n
            0.5,               # frost_alpha
            0.1,               # f_wetland
            0.5,               # slope_factor
            0.01,              # krec
            0.005,             # k_gw
            8.0,               # T_gw
            0.2,               # K_atm
            0.05,              # alpha_T
        ], device=device)

        # Set these as the static parameters
        model.spatial_encoder.static_params.data = better_params

    print("Parameters initialized with physically reasonable values")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Run simulation WITH NO TRAINING
    print("\nRunning simulation with zero-shot physics model...")
    model.eval()
    with torch.no_grad():
        initial_state = HydroState.default_warm(n_nodes, device=device)

        Q_sim, final_state = model.simulate(
            forcing=forcing,
            initial_state=initial_state,
            graph=graph,
            node_coords=node_coords,
            territorial=territorial,
            withdrawals=withdrawals,
            day_of_year=doy,
        )

    # Evaluate performance
    print("\n=== ZERO-SHOT PERFORMANCE ===")

    # Check discharge magnitudes
    outlet_Q = Q_sim[:, -1].mean().item()
    print(f"Outlet discharge: {outlet_Q:.2f} m³/s")
    print(f"Observed range: {q_obs_tensor.min():.1f} - {q_obs_tensor.max():.1f} m³/s")

    # Calculate KGE for each station
    kges = []
    for i in range(min(10, n_stations)):  # Check first 10 stations
        q_o = q_obs_tensor[:, i].cpu()
        q_s = Q_sim[:, station_mask][:, i].cpu()

        valid = ~torch.isnan(q_o)
        if valid.sum() < 20:
            continue

        kge_val = compute_kge(q_o[valid], q_s[valid])
        kges.append(float(kge_val))
        print(f"Station {i}: KGE = {kge_val:.3f}")

    if kges:
        mean_kge = np.mean(kges)
        max_kge = np.max(kges)
        print(f"\nMean KGE: {mean_kge:.3f}")
        print(f"Max KGE: {max_kge:.3f}")

        if mean_kge > 0.3:
            print("✅ Physics model works! Just needs parameter tuning.")
        elif mean_kge > 0:
            print("⚠️ Physics produces reasonable patterns but needs calibration")
        else:
            print("❌ Physics model has fundamental issues")
    else:
        print("❌ No valid stations for evaluation")


if __name__ == "__main__":
    main()