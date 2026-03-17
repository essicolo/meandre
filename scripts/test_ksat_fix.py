"""Test script to verify K_sat fix improves discharge output."""
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
    DATE_END = "2002-01-07"  # 1 week

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Testing K_sat fix: 2.0, 0.5, 0.15 m/day (vs old 0.5, 0.1, 0.02)")

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
        cache_nc=Path("/tmp/ksat_test_forcing.nc"),
        device=device,
    )

    ds_time = xr.open_dataset(Path("/tmp/ksat_test_forcing.nc"))
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

    # Create model with static mode for quick test
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
        param_mode="static",  # Use static for deterministic test
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test with warm initialization
    initial_state = HydroState.default_warm(n_nodes, device=device)

    print("\n=== Testing with FIXED K_sat values ===")
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

    # Use last node as outlet for testing
    outlet_idx = n_nodes - 1
    Q_outlet = Q_sim[:, outlet_idx].cpu()
    mean_discharge = Q_outlet.mean()

    print(f"Outlet discharge with FIXED K_sat: {mean_discharge:.3f} m³/s")
    print(f"Previous (buggy) K_sat discharge: ~3.74 m³/s")
    print(f"Target discharge: ~6.8 m³/s")

    improvement = mean_discharge / 3.74
    target_ratio = mean_discharge / 6.8

    print(f"\n=== RESULTS ===")
    print(f"Improvement ratio: {improvement:.2f}x vs buggy K_sat")
    print(f"Target achievement: {target_ratio:.1f} ({target_ratio*100:.0f}% of target)")

    if improvement > 1.2:
        print("✅ K_sat fix IMPROVED discharge!")
    else:
        print("❌ K_sat fix did not significantly improve discharge")

    if target_ratio > 0.8:
        print("✅ Discharge is now close to target!")
    else:
        print("❓ Still need more investigation or additional fixes")

    print(f"\nDischarge range: {Q_outlet.min():.3f} to {Q_outlet.max():.3f} m³/s")
    print(f"Daily variation seems: {'reasonable' if Q_outlet.std() < mean_discharge else 'excessive'}")

if __name__ == "__main__":
    main()