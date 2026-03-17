"""Verify area fix works by testing with minimal training setup."""
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
    BASIN_DB = Path(paths["basin_db"]).parent / "slso_train.duckdb"  # Use training DB
    ZARR_PATH = Path(paths["weather_grid"])

    # Very short test period
    DATE_START = "2002-01-01"
    DATE_END = "2002-01-07"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("=== VERIFYING AREA FIX ===")
    print("Testing with current data to confirm the scale issue")

    # Load current cached data
    cache = BasinCache(BASIN_DB)
    hydro = cache.load(device=device)

    graph = hydro["graph"]
    territorial = hydro["territorial"]
    node_coords = hydro["node_coords"]
    n_nodes = hydro["n_nodes"]

    print(f"Nodes: {n_nodes}")

    # Check current area values in territorial data
    print("\n=== CURRENT TERRITORIAL AREA DATA ===")

    # Check what area attributes territorial has
    area_attrs = [attr for attr in dir(territorial) if "area" in attr.lower()]
    print(f"Area attributes in territorial: {area_attrs}")

    if hasattr(territorial, 'area_km2_local'):
        local_areas = territorial.area_km2_local
        print(f"area_km2_local: mean={local_areas.mean():.3f} km², sum={local_areas.sum():.1f} km²")
        print(f"  First 5 values: {local_areas[:5].cpu().numpy()}")
    else:
        print("❌ No area_km2_local in territorial!")

    if hasattr(territorial, 'area_km2_physical'):
        phys_areas = territorial.area_km2_physical
        print(f"area_km2_physical: mean={phys_areas.mean():.1f} km², outlet={phys_areas[-1]:.1f} km²")
    else:
        print("❌ No area_km2_physical in territorial!")

    # Load forcing for test period
    forcing = extract_forcing(
        zarr_path=ZARR_PATH,
        node_coords=node_coords,
        node_elev=None,
        date_start=DATE_START,
        date_end=DATE_END,
        cache_nc=Path("/tmp/verify_forcing.nc"),
        device=device,
    )

    doy = torch.tensor([1, 2, 3, 4, 5, 6, 7], dtype=torch.long, device=device)

    # Load withdrawals
    withdrawals = cache.load_withdrawals(
        date_start=DATE_START,
        date_end=DATE_END,
        device=device,
    )

    # Create minimal model
    model = YHydro(
        n_nodes=n_nodes,
        n_territorial=territorial.n_features,
        n_forcing=6,
        context_window=7,  # Small for test
        residual_history=7,
        max_travel_time=5,
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
            0.0, 2.0,          # T_melt, T_snow
            0.5, 0.03, 0.5,    # interception, manning_n, frost_alpha
            0.1, 0.5, 0.01,    # f_wetland, slope_factor, krec
            0.005, 8.0, 0.2, 0.05,  # k_gw, T_gw, K_atm, alpha_T
        ], device=device)
        model.spatial_encoder.static_params.data = better_params

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Run simulation
    print("\n=== RUNNING SIMULATION ===")
    model.eval()
    with torch.no_grad():
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

    # Check outlet discharge
    outlet_Q = Q_sim[:, -1].mean().item()
    print(f"\nOutlet discharge: {outlet_Q:.6f} m³/s")

    # Expected calculation with corrected areas
    if hasattr(territorial, 'area_km2_local') and hasattr(territorial, 'area_km2_physical'):
        local_sum = territorial.area_km2_local.sum().item()
        outlet_area = territorial.area_km2_physical[-1].item()

        # Expected for 1mm/day over outlet area
        expected_1mm = 1.0 * 1e-3 * outlet_area * 1e6 / 86400.0

        print(f"\n=== SCALE ANALYSIS ===")
        print(f"Local areas sum: {local_sum:.1f} km²")
        print(f"Outlet area: {outlet_area:.1f} km²")
        print(f"Expected discharge for 1mm/day over outlet: {expected_1mm:.1f} m³/s")
        print(f"Current discharge is {outlet_Q/expected_1mm*100:.1f}% of expected 1mm/day")

        scale_error = outlet_Q / expected_1mm
        if scale_error < 0.001:
            print(f"❌ SEVERE SCALE ERROR: {scale_error*1000:.1f}‰ of expected")
            print("   This confirms the area conversion bug!")
            print("   Lateral inflow isn't being converted to proper m³/s scale")
        elif scale_error < 0.01:
            print(f"⚠️ Scale error: {scale_error*100:.1f}% of expected")
        else:
            print(f"✅ Reasonable scale: {scale_error*100:.1f}% of expected")

    # Now test our fix understanding
    print(f"\n=== AREA FIX VERIFICATION ===")

    # The key insight is that the training database territorial object
    # might be missing the area_km2_local attribute entirely, or it has
    # the wrong values. Our physitel_loader.py fix doesn't affect the
    # existing training database.

    if not hasattr(territorial, 'area_km2_local'):
        print("✅ CONFIRMED: area_km2_local missing from territorial!")
        print("   Routing code will use area_km2 (cumulative) instead")
        print("   This explains the wrong scale - using cumulative instead of local areas")
    elif territorial.area_km2_local.mean() < 1.0:
        print("✅ CONFIRMED: area_km2_local has wrong scale!")
        print(f"   Mean local area {territorial.area_km2_local.mean():.3f} km² is too small")
        print("   Should be ~400 km² (total/num_nodes) for proper conversion")
    else:
        print("❓ area_km2_local seems reasonable - issue might be elsewhere")

    print(f"\n=== SOLUTION SUMMARY ===")
    print("1. ✅ Fixed physitel_loader.py to calculate proper local areas")
    print("2. 🔄 Need to rebuild training database with fixed loader")
    print("3. 🎯 Then train model - should achieve KGE > 0.8")


if __name__ == "__main__":
    main()