"""Test model with FIXED area issue - use proper local areas."""
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
    print("=== FIXED AREA TEST ===")
    print("Testing if fixing area_km2_local solves the scale problem")

    # Load data
    cache = BasinCache(BASIN_DB)
    hydro = cache.load(device=device)

    graph = hydro["graph"]
    territorial = hydro["territorial"]
    node_coords = hydro["node_coords"]
    n_nodes = hydro["n_nodes"]

    # CRITICAL FIX: Calculate proper local areas
    print("\nArea Fix Strategy:")
    print("Original area_km2_local mean:", territorial.area_km2_local.mean().item(), "km²")

    # Use a more reasonable scaling factor
    # The local areas are too small - scale them up by ratio of physical to local
    scale_factor = territorial.area_km2_physical.mean() / territorial.area_km2_local.mean()
    print(f"Scale factor needed: {scale_factor:.1f}x")

    # Override the local areas with scaled values
    territorial.area_km2_local = territorial.area_km2_local * scale_factor
    print("Fixed area_km2_local mean:", territorial.area_km2_local.mean().item(), "km²")

    # Load forcing
    forcing = extract_forcing(
        zarr_path=ZARR_PATH,
        node_coords=node_coords,
        node_elev=None,
        date_start=DATE_START,
        date_end=DATE_END,
        cache_nc=Path("/tmp/fixed_area_forcing.nc"),
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

    # Create model with STATIC parameters
    print("\nCreating model with static physics parameters...")
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

    # Initialize with reasonable parameters
    with torch.no_grad():
        better_params = torch.tensor([
            0.5, 0.2, 0.1,     # K_sat (m/day)
            0.45, 0.45, 0.45,  # porosity
            0.30, 0.30, 0.30,  # theta_fc
            0.15, 0.15, 0.15,  # theta_wp
            200, 300, 500,     # depths (mm)
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
        model.spatial_encoder.static_params.data = better_params

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Run simulation with FIXED areas
    print("\nRunning simulation with FIXED area values...")
    model.eval()
    with torch.no_grad():
        initial_state = HydroState.default_warm(n_nodes, device=device)

        Q_sim, final_state = model.simulate(
            forcing=forcing,
            initial_state=initial_state,
            graph=graph,
            node_coords=node_coords,
            territorial=territorial,  # Now has fixed area_km2_local
            withdrawals=withdrawals,
            day_of_year=doy,
        )

    # Evaluate performance
    print("\n=== RESULTS WITH FIXED AREAS ===")

    # Check discharge magnitudes
    outlet_Q = Q_sim[:, -1].mean().item()
    print(f"Outlet discharge: {outlet_Q:.2f} m³/s")
    print(f"Observed range: {q_obs_tensor.min():.1f} - {q_obs_tensor.max():.1f} m³/s")

    # Expected rough scale
    expected = territorial.area_km2_physical[-1] * 0.001 * 2  # ~2mm/day runoff over basin
    print(f"Expected outlet order of magnitude: ~{expected:.0f} m³/s")

    # Calculate KGE for each station
    kges = []
    for i in range(min(10, n_stations)):
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
            print("✅ AREA FIX WORKS! Physics model improved significantly!")
            print("   The problem was incorrect local area values")
        elif mean_kge > 0.1:
            print("⚠️ Partial improvement but still needs calibration")
        else:
            print("❌ Area fix alone doesn't solve the problem")
    else:
        print("❌ No valid stations for evaluation")


if __name__ == "__main__":
    main()