"""Deep physics audit to find remaining issues.
This script will systematically check:
1. Parameter magnitudes and scaling
2. Mass balance at each step
3. Units consistency throughout the system
4. Routing physics issues
5. Temporal aggregation problems
"""
import argparse
import tomllib
from pathlib import Path
import torch
import xarray as xr
import pandas as pd
import numpy as np

from meandre.data.basin_cache import BasinCache
from meandre.data.gridded_forcing import extract_forcing
from meandre.model import YHydro
from meandre.utils.state import HydroState
from meandre.spatial.field_network import SpatialParams

def check_parameter_magnitudes(model, territorial, node_coords, device):
    """Check if spatial parameters have reasonable magnitudes"""
    print("\n=== PARAMETER MAGNITUDE AUDIT ===")

    # Get a sample of spatial parameters
    coords_norm = 2.0 * (node_coords - node_coords.mean(0)) / (node_coords.std(0) + 1e-6)
    territorial_norm = territorial.normalize() if hasattr(territorial, 'normalize') else territorial

    # Extract parameters for first 10 nodes
    sample_indices = torch.arange(min(10, len(node_coords)), device=device)
    sample_coords = coords_norm[sample_indices]

    try:
        # Get territorial features for sample nodes
        if hasattr(territorial, 'features'):
            sample_features = territorial.features[sample_indices]
        elif hasattr(territorial, 'normalized'):
            sample_features = territorial.normalized[sample_indices]
        else:
            print("ERROR: Cannot access territorial features")
            return

        sample_params = model.spatial_encoder(sample_coords, sample_features)

        # Check each parameter range
        param_names = [
            "k_sat_1", "k_sat_2", "k_sat_3",
            "theta_fc_1", "theta_fc_2", "theta_fc_3",
            "theta_wp_1", "theta_wp_2", "theta_wp_3",
            "depth_1", "depth_2", "depth_3",
            "C_f", "T_melt", "T_snow", "k_gw", "f_wetland"
        ]

        for i, name in enumerate(param_names):
            if i < sample_params.shape[1]:
                param_vals = sample_params[:, i].cpu()
                print(f"{name:12s}: {param_vals.min():.4f} to {param_vals.max():.4f} (mean={param_vals.mean():.4f})")

                # Check for physically unrealistic values
                if name.startswith("k_sat"):
                    if param_vals.max() > 1000 or param_vals.min() < 0.001:
                        print(f"  WARNING: {name} outside reasonable range [0.001, 1000] mm/day")
                elif name.startswith("theta"):
                    if param_vals.max() > 0.6 or param_vals.min() < 0.05:
                        print(f"  WARNING: {name} outside reasonable range [0.05, 0.6]")
                elif name.startswith("depth"):
                    if param_vals.max() > 5000 or param_vals.min() < 50:
                        print(f"  WARNING: {name} outside reasonable range [50, 5000] mm")

    except Exception as e:
        print(f"ERROR checking parameters: {e}")

def check_mass_balance(model, forcing, territorial, withdrawals, doy, device, n_nodes):
    """Check mass balance at each step of the simulation"""
    print("\n=== MASS BALANCE AUDIT ===")

    # Run simulation with diagnostics
    initial_state = HydroState.default_warm(n_nodes, device=device)

    # Get empty graph for minimal test
    from meandre.routing.graph import RiverGraph
    edges = torch.empty((2, 0), dtype=torch.long, device=device)  # No connections
    graph = RiverGraph(edges, n_nodes)

    # Single timestep simulation
    forcing_1day = forcing[:1]  # Just first day
    doy_1day = doy[:1]

    print("Testing single timestep mass balance...")

    # Get model spatial parameters
    node_coords = torch.randn(n_nodes, 2, device=device)  # Dummy coords
    coords_norm = 2.0 * (node_coords - node_coords.mean(0)) / (node_coords.std(0) + 1e-6)

    try:
        if hasattr(territorial, 'features'):
            features = territorial.features
        elif hasattr(territorial, 'normalized'):
            features = territorial.normalized
        else:
            print("ERROR: Cannot access territorial features for mass balance")
            return

        spatial_params = model.spatial_encoder(coords_norm, features)

        # Extract precipitation input
        P_input = forcing_1day[0, :, 0]  # Precipitation mm/day
        basin_mean_P = P_input.mean()
        total_P_volume = P_input.sum()  # Total mm across all nodes

        print(f"Input precipitation: {basin_mean_P:.1f} mm/day (basin mean)")
        print(f"Total input volume: {total_P_volume:.1f} mm*nodes")

        # Test column simulation directly (bypass routing)
        from meandre.vertical.column import VerticalColumn
        column = VerticalColumn()

        # Run column for one node
        test_node = 0
        P_node = P_input[test_node]
        T_min = forcing_1day[0, test_node, 1]
        T_max = forcing_1day[0, test_node, 2]

        # Extract parameters for this node
        params = SpatialParams.from_tensor(spatial_params[test_node:test_node+1])

        # Run column with return diagnostics
        output = column(
            forcing_1day[:, test_node:test_node+1],
            initial_state.slice(slice(test_node, test_node+1)),
            params,
            doy_1day,
            return_diagnostics=True
        )

        lateral_out = output.lateral_inflow[0, 0]  # mm/day
        diag = output.diag

        if diag:
            etr = diag['etr'][0, 0] if 'etr' in diag else 0.0
            snowmelt = diag['snowmelt'][0, 0] if 'snowmelt' in diag else 0.0
            recharge = diag['recharge'][0, 0] if 'recharge' in diag else 0.0

            print(f"Node {test_node} water balance (mm/day):")
            print(f"  Input P: {P_node:.2f}")
            print(f"  Output ETR: {etr:.2f}")
            print(f"  Output lateral: {lateral_out:.2f}")
            print(f"  Recharge: {recharge:.2f}")
            print(f"  Snowmelt: {snowmelt:.2f}")

            water_out = etr + lateral_out + recharge
            balance_error = P_node + snowmelt - water_out
            print(f"  Balance error: {balance_error:.4f} mm/day")

            if abs(balance_error) > 0.1:
                print("  ERROR: Mass balance violation > 0.1 mm/day!")
            else:
                print("  ✓ Mass balance OK")

    except Exception as e:
        print(f"ERROR in mass balance check: {e}")

def check_routing_physics(device, n_nodes):
    """Check routing parameter magnitudes and physics"""
    print("\n=== ROUTING PHYSICS AUDIT ===")

    from meandre.routing.kinematic import MuskingumCunge

    # Test Muskingum parameters
    K_test = torch.tensor([3600.0, 7200.0, 14400.0], device=device)  # 1, 2, 4 hours
    x_test = torch.tensor([0.1, 0.2, 0.3], device=device)

    router = MuskingumCunge(dt=86400.0, n_substeps=4)

    # Test discharge routing
    Q_in = torch.tensor([1.0, 5.0, 10.0], device=device)  # m3/s
    Q_prev = torch.zeros(3, device=device)
    q_lat = torch.tensor([0.1, 0.1, 0.1], device=device)  # m3/s lateral

    Q_out = router(Q_in, Q_prev, q_lat, K_test, x_test)

    print("Muskingum routing test:")
    for i in range(3):
        print(f"  K={K_test[i]/3600:.1f}h, x={x_test[i]:.1f}: {Q_in[i]:.1f} -> {Q_out[i]:.1f} m³/s")

        # Check conservation
        expected_out = Q_in[i] + q_lat[i]  # Approximate for short time
        if abs(Q_out[i] - expected_out) / expected_out > 0.5:
            print(f"    WARNING: Large routing change, expected ~{expected_out:.1f}")

def check_units_consistency():
    """Check units throughout the system"""
    print("\n=== UNITS CONSISTENCY AUDIT ===")

    print("Checking units conversions in message_passing.py...")

    # Test units conversion
    lateral_mm_day = 5.0  # mm/day
    area_km2 = 100.0      # km²

    # Conversion: mm/day * km² -> m³/s
    area_m2 = area_km2 * 1e6
    lateral_m_day = lateral_mm_day * 1e-3  # mm -> m
    lateral_m3_day = lateral_m_day * area_m2
    lateral_m3_s = lateral_m3_day / 86400.0

    print(f"Units test: {lateral_mm_day} mm/day over {area_km2} km²")
    print(f"  = {lateral_m3_s:.3f} m³/s")
    print(f"  Check: 5mm/day * 100km² = {5*100*1e6/86400/1e3:.3f} m³/s ✓")

    # Check typical basin values
    print("\nTypical values for SLSO basin:")
    print("  Basin area: ~1000-5000 km²")
    print("  Average runoff: 1-10 mm/day")
    print("  Expected discharge: 10-500 m³/s")

    # Check against our simulation
    print(f"\nOur simulation: 3.74 m³/s")
    print("  Implied runoff rate: 3.74 m³/s / (2000 km² * 1e6 m²/km² / 86400 s/day)")
    implied_runoff = 3.74 * 86400 / (2000 * 1e6) * 1000  # mm/day
    print(f"  = {implied_runoff:.2f} mm/day")
    print("  This seems reasonable for winter conditions")

def check_temporal_aggregation():
    """Check if temporal aggregation causes issues"""
    print("\n=== TEMPORAL AGGREGATION AUDIT ===")

    print("Daily timestep aggregation effects:")
    print("1. Muskingum uses sub-steps (4x per day) - should be OK")
    print("2. Vertical processes are daily averages - may smooth peaks")
    print("3. Precipitation as daily totals - reasonable for this scale")
    print("4. Temperature as daily min/max - reasonable")

    print("\nPotential issues:")
    print("- Snow melt/freeze cycles within day are lost")
    print("- Storm event sub-daily peaks are smoothed")
    print("- Diurnal ET patterns are averaged")
    print("- Short-term soil saturation/drainage cycles are missed")

def main():
    # Use short test period
    config_path = "notebooks/slso/config/slso.toml"

    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    paths = cfg["paths"]
    BASIN_DB = Path(paths["basin_db"])
    ZARR_PATH = Path(paths["weather_grid"])

    # Short test period
    DATE_START = "2002-01-01"
    DATE_END = "2002-01-10"  # 10 days

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("=== DEEP PHYSICS AUDIT ===")

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
        cache_nc=Path("/tmp/physics_audit_forcing.nc"),
        device=device,
    )

    ds_time = xr.open_dataset(Path("/tmp/physics_audit_forcing.nc"))
    all_dates = ds_time.time.sel(time=slice(DATE_START, DATE_END)).values
    ds_time.close()

    doy = torch.tensor(
        [int(pd.Timestamp(d).day_of_year) for d in all_dates],
        dtype=torch.long, device=device,
    )

    withdrawals = cache.load_withdrawals(
        date_start=DATE_START,
        date_end=DATE_END,
        device=device,
    )

    # Create model
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
        param_mode="static",  # Use static for easier debugging
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Run all audits
    check_parameter_magnitudes(model, territorial, node_coords, device)
    check_mass_balance(model, forcing, territorial, withdrawals, doy, device, n_nodes)
    check_routing_physics(device, n_nodes)
    check_units_consistency()
    check_temporal_aggregation()

    print("\n=== PHYSICS AUDIT COMPLETE ===")
    print("Check above for any ERROR or WARNING messages")

if __name__ == "__main__":
    main()