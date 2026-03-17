"""Debug parameter values to understand the physics issue."""
import argparse
import tomllib
from pathlib import Path
import torch
import xarray as xr
import pandas as pd

from meandre.data.basin_cache import BasinCache
from meandre.data.gridded_forcing import extract_forcing
from meandre.model import YHydro
from meandre.utils.state import HydroState

def main():
    config_path = "notebooks/slso/config/slso.toml"

    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    paths = cfg["paths"]
    BASIN_DB = Path(paths["basin_db"])
    ZARR_PATH = Path(paths["weather_grid"])

    # Quick test period
    DATE_START = "2002-01-01"
    DATE_END = "2002-01-03"  # Just 2 days

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("=== DEBUGGING PARAMETER VALUES ===")

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
        cache_nc=Path("/tmp/debug_forcing.nc"),
        device=device,
    )

    doy = torch.tensor([1, 2], dtype=torch.long, device=device)  # Jan 1-2

    # Test both parameter modes
    for param_mode in ["static", "nerf"]:
        print(f"\n=== Testing {param_mode.upper()} parameter mode ===")

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
            param_mode=param_mode,
        ).to(device)

        # Extract spatial parameters for first 5 nodes
        coords_norm = 2.0 * (node_coords - node_coords.mean(0)) / (node_coords.std(0) + 1e-6)
        sample_indices = torch.arange(5, device=device)
        sample_coords = coords_norm[sample_indices]

        if hasattr(territorial, 'features'):
            sample_features = territorial.features[sample_indices]
        elif hasattr(territorial, 'normalized'):
            sample_features = territorial.normalized[sample_indices]
        else:
            print(f"ERROR: Cannot access territorial features for {param_mode}")
            continue

        with torch.no_grad():
            spatial_params = model.spatial_encoder(sample_coords, sample_features)

        # Check K_sat values (first 3 parameters)
        print(f"K_sat Layer 1: {spatial_params[:, 0].min():.4f} - {spatial_params[:, 0].max():.4f} m/day")
        print(f"K_sat Layer 2: {spatial_params[:, 1].min():.4f} - {spatial_params[:, 1].max():.4f} m/day")
        print(f"K_sat Layer 3: {spatial_params[:, 2].min():.4f} - {spatial_params[:, 2].max():.4f} m/day")

        # Check theta_fc (field capacity) - params 3-5
        print(f"theta_fc Layer 1: {spatial_params[:, 3].min():.3f} - {spatial_params[:, 3].max():.3f}")
        print(f"theta_fc Layer 2: {spatial_params[:, 4].min():.3f} - {spatial_params[:, 4].max():.3f}")
        print(f"theta_fc Layer 3: {spatial_params[:, 5].min():.3f} - {spatial_params[:, 5].max():.3f}")

        # Check depths - params 9-11
        print(f"Depth Layer 1: {spatial_params[:, 9].min():.0f} - {spatial_params[:, 9].max():.0f} mm")
        print(f"Depth Layer 2: {spatial_params[:, 10].min():.0f} - {spatial_params[:, 10].max():.0f} mm")
        print(f"Depth Layer 3: {spatial_params[:, 11].min():.0f} - {spatial_params[:, 11].max():.0f} mm")

        # Run quick simulation to get discharge
        withdrawals = cache.load_withdrawals(
            date_start=DATE_START,
            date_end=DATE_END,
            device=device,
        )

        initial_state = HydroState.default_warm(n_nodes, device=device)

        Q_sim, _ = model.simulate(
            forcing=forcing,
            initial_state=initial_state,
            graph=graph,
            node_coords=node_coords,
            territorial=territorial,
            withdrawals=withdrawals,
            day_of_year=doy,
        )

        outlet_Q = Q_sim[:, -1].mean()
        print(f"Outlet discharge: {outlet_Q:.6f} m³/s")

if __name__ == "__main__":
    main()