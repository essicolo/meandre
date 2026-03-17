"""Diagnose the 1000x scale error in discharge calculation."""
import tomllib
from pathlib import Path
import torch
import numpy as np

from meandre.data.basin_cache import BasinCache
from meandre.data.gridded_forcing import extract_forcing
from meandre.model import YHydro
from meandre.utils.state import HydroState
from meandre.vertical.column import VerticalColumn
from meandre.spatial.field_network import SpatialParams

def main():
    config_path = "notebooks/slso/config/slso.toml"
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    paths = cfg["paths"]
    BASIN_DB = Path(paths["basin_db"])
    ZARR_PATH = Path(paths["weather_grid"])

    DATE_START = "2002-01-01"
    DATE_END = "2002-01-31"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== SCALE ERROR DIAGNOSIS ===")

    # Load data
    cache = BasinCache(BASIN_DB)
    hydro = cache.load(device=device)

    graph = hydro["graph"]
    territorial = hydro["territorial"]
    node_coords = hydro["node_coords"]
    n_nodes = hydro["n_nodes"]

    print(f"\nBasin info:")
    print(f"  Nodes: {n_nodes}")
    print(f"  Graph edges: {graph.n_edges}")

    # Check area values
    print("\n=== AREA ANALYSIS ===")

    # Check territorial features
    if hasattr(territorial, 'features'):
        terr_data = territorial.features
    elif hasattr(territorial, 'data'):
        terr_data = territorial.data
    else:
        terr_data = territorial

    print(f"Territorial shape: {terr_data.shape}")
    print(f"Territorial features: {territorial.n_features}")

    # Look for area in graph attributes
    if hasattr(graph, 'area_km2'):
        areas = graph.area_km2
        print(f"\nCumulative areas (km²):")
        print(f"  Min: {areas.min():.1f}")
        print(f"  Max: {areas.max():.1f}")
        print(f"  Mean: {areas.mean():.1f}")
        print(f"  Outlet (last node): {areas[-1]:.1f}")

    if hasattr(graph, 'area_km2_local'):
        local_areas = graph.area_km2_local
        print(f"\nLocal areas (km²):")
        print(f"  Min: {local_areas.min():.1f}")
        print(f"  Max: {local_areas.max():.1f}")
        print(f"  Mean: {local_areas.mean():.1f}")
        print(f"  Sum: {local_areas.sum():.1f}")

    # Load forcing for one day
    forcing = extract_forcing(
        zarr_path=ZARR_PATH,
        node_coords=node_coords,
        node_elev=None,
        date_start=DATE_START,
        date_end=DATE_END,
        cache_nc=Path("/tmp/scale_diagnosis.nc"),
        device=device,
    )

    print(f"\n=== WATER BALANCE ANALYSIS ===")

    # Check precipitation
    precip = forcing[0, :, 0]  # First day, all nodes, precip channel
    print(f"Precipitation (mm/day):")
    print(f"  Min: {precip.min():.2f}")
    print(f"  Max: {precip.max():.2f}")
    print(f"  Mean: {precip.mean():.2f}")

    # Test vertical column to get lateral inflow
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

    # Set reasonable physics parameters
    with torch.no_grad():
        better_params = torch.tensor([
            0.5, 0.2, 0.1,     # K_sat
            0.45, 0.45, 0.45,  # porosity
            0.30, 0.30, 0.30,  # theta_fc
            0.15, 0.15, 0.15,  # theta_wp
            200, 300, 500,     # depths
            5.0,               # C_f
            0.0,               # T_melt
            2.0,               # T_snow
            0.5,               # interception
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
        model.spatial_encoder.static_params.data = better_params

    # Get spatial parameters for a few nodes
    sample_nodes = slice(0, 5)
    coords_norm = 2.0 * (node_coords - node_coords.mean(0)) / (node_coords.std(0) + 1e-6)
    sample_coords = coords_norm[sample_nodes]

    if hasattr(territorial, 'features'):
        sample_features = territorial.features[sample_nodes]
    elif hasattr(territorial, 'data'):
        sample_features = territorial.data[sample_nodes]
    else:
        sample_features = terr_data[sample_nodes]

    with torch.no_grad():
        spatial_params = model.spatial_encoder(sample_coords, sample_features)
        params = SpatialParams.from_tensor(spatial_params)

    # Run vertical column
    column = VerticalColumn()
    test_forcing = forcing[:1, sample_nodes]
    test_state = HydroState.default_warm(5, device=device)
    doy = torch.tensor([1], dtype=torch.long, device=device)

    output = column(test_forcing, test_state, params, doy, return_diagnostics=True)
    lateral_inflow_mm = output.lateral_inflow[0]  # mm/day

    print(f"\nLateral inflow (mm/day):")
    print(f"  Min: {lateral_inflow_mm.min():.4f}")
    print(f"  Max: {lateral_inflow_mm.max():.4f}")
    print(f"  Mean: {lateral_inflow_mm.mean():.4f}")

    # Now convert to m³/s using area
    if hasattr(graph, 'area_km2_local'):
        # Use local areas for these nodes
        local_areas_sample = graph.area_km2_local[sample_nodes]

        # Manual conversion matching routing code
        q_m3s = lateral_inflow_mm * 1e-3 * local_areas_sample * 1e6 / 86400.0

        print(f"\nConverted to m³/s (using local areas):")
        print(f"  Areas (km²): {local_areas_sample.cpu().numpy()}")
        print(f"  Discharge (m³/s): {q_m3s.cpu().numpy()}")
        print(f"  Mean: {q_m3s.mean():.6f} m³/s")

        # Expected rough calculation
        # 1 mm/day over 1000 km² = 1e-3 m × 1e9 m² / 86400 s ≈ 11.6 m³/s
        expected_per_1000km2 = 1.0 * 1e-3 * 1000 * 1e6 / 86400
        print(f"\nExpected discharge for 1mm/day over 1000km²: {expected_per_1000km2:.1f} m³/s")

        # Check if areas might be wrong units
        if local_areas_sample.mean() < 10:
            print("\n⚠️ WARNING: Local areas seem too small! Might be in wrong units.")
            print("  If areas are in m² instead of km², that's a 1e6 error!")

            # Test with corrected areas
            corrected_areas = local_areas_sample * 1e6  # If stored as m² not km²
            q_corrected = lateral_inflow_mm * 1e-3 * corrected_areas * 1e6 / 86400.0
            print(f"\nIf areas were m² not km²:")
            print(f"  Corrected discharge: {q_corrected.mean():.2f} m³/s")

    # Also check cumulative routing
    if hasattr(graph, 'area_km2'):
        outlet_area = graph.area_km2[-1]
        total_precip_volume = precip.mean() * 1e-3 * outlet_area * 1e6 / 86400.0
        print(f"\n=== OUTLET SCALE CHECK ===")
        print(f"Outlet drainage area: {outlet_area:.1f} km²")
        print(f"Expected outlet discharge from {precip.mean():.1f}mm/day precip:")
        print(f"  Direct calculation: {total_precip_volume:.1f} m³/s")
        print(f"  (This assumes 100% runoff, real should be ~10-30% of this)")

    print("\n=== DIAGNOSIS COMPLETE ===")


if __name__ == "__main__":
    main()