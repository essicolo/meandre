"""Debug lateral inflow magnitudes - check if they are physically reasonable."""
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

    # Very short test period
    DATE_START = "2002-01-01"
    DATE_END = "2002-01-02"  # Just 1 day

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("=== DEBUGGING LATERAL INFLOW MAGNITUDES ===")

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
        cache_nc=Path("/tmp/debug_lateral_forcing.nc"),
        device=device,
    )

    doy = torch.tensor([1], dtype=torch.long, device=device)  # Jan 1

    withdrawals = cache.load_withdrawals(
        date_start=DATE_START,
        date_end=DATE_END,
        device=device,
    )

    # Test different parameter modes
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

        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Use warm initialization
        initial_state = HydroState.default_warm(n_nodes, device=device)

        with torch.no_grad():
            Q_sim, _ = model.simulate(
                forcing=forcing,
                initial_state=initial_state,
                graph=graph,
                node_coords=node_coords,
                territorial=territorial,
                withdrawals=withdrawals,
                day_of_year=doy,
            )

        outlet_Q = Q_sim[0, -1]  # First day, last node
        print(f"Outlet discharge: {outlet_Q:.6f} m³/s")

        # CRITICAL: Check lateral inflow values directly from vertical column
        # We need to access the lateral inflow before units conversion

        # Get first few nodes for testing
        test_nodes = slice(0, 5)

        # Extract spatial parameters
        coords_norm = 2.0 * (node_coords - node_coords.mean(0)) / (node_coords.std(0) + 1e-6)

        # Run vertical column directly to get lateral inflow in mm/day
        from meandre.vertical.column import VerticalColumn
        from meandre.spatial.field_network import SpatialParams

        column = VerticalColumn()

        # Get spatial parameters for test nodes
        if hasattr(territorial, 'features'):
            features = territorial.features[test_nodes]
        elif hasattr(territorial, 'data'):
            features = territorial.data[test_nodes]
        else:
            print(f"  Cannot access territorial features for {param_mode}")
            continue

        test_coords = coords_norm[test_nodes]
        spatial_params = model.spatial_encoder(test_coords, features)
        params = SpatialParams.from_tensor(spatial_params)

        # Run column for just these test nodes
        test_forcing = forcing[:, test_nodes]
        test_state = initial_state.slice(test_nodes)

        output = column(
            test_forcing,
            test_state,
            params,
            doy,
            return_diagnostics=True
        )

        lateral_mm_day = output.lateral_inflow[0]  # First day
        print(f"  Lateral inflow (mm/day): min={lateral_mm_day.min():.3f}, max={lateral_mm_day.max():.3f}, mean={lateral_mm_day.mean():.3f}")

        # Show precipitation input for reference
        precip = test_forcing[0, :, 0]  # First day, all test nodes, precipitation
        print(f"  Precipitation (mm/day): min={precip.min():.1f}, max={precip.max():.1f}, mean={precip.mean():.1f}")

        # Check if lateral inflow is physically reasonable
        runoff_ratio = lateral_mm_day.mean() / precip.mean()
        print(f"  Runoff ratio: {runoff_ratio:.3f} (0.1-0.3 typical for winter)")

        # Check units conversion
        if hasattr(territorial, 'area_km2_local') and territorial.area_km2_local is not None:
            area_km2 = territorial.area_km2_local[test_nodes].mean()
            lateral_m3s = lateral_mm_day.mean() * 1e-3 * area_km2 * 1e6 / 86400.0
            print(f"  After units conversion: {lateral_m3s:.6f} m³/s per node")
            print(f"  Total basin (×{n_nodes}): {lateral_m3s * n_nodes:.3f} m³/s")
        else:
            print(f"  ERROR: No area data for units conversion!")

        print(f"  Physical assessment:")
        if lateral_mm_day.mean() < 0.1:
            print(f"    ❌ Lateral inflow too low ({lateral_mm_day.mean():.3f} mm/day)")
        elif lateral_mm_day.mean() > 20:
            print(f"    ❌ Lateral inflow too high ({lateral_mm_day.mean():.3f} mm/day)")
        else:
            print(f"    ✅ Lateral inflow reasonable ({lateral_mm_day.mean():.3f} mm/day)")

if __name__ == "__main__":
    main()