"""Check physics computation for sanity."""
import torch
import numpy as np
from pathlib import Path
from meandre.data.basin_cache import BasinCache
from meandre.data.gridded_forcing import extract_forcing
from meandre.model import YHydro
from meandre.utils.state import HydroState
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
BASIN_DB = Path("notebooks/slso/data/slso.duckdb")
ZARR_PATH = Path("/home/essi/Documents/quebec.zarr")
FORCING_CACHE = Path("notebooks/slso/data/forcing.nc")

cache = BasinCache(BASIN_DB)
hydro = cache.load(device=device)

graph = hydro["graph"]
territorial = hydro["territorial"]
node_coords = hydro["node_coords"]
n_nodes = hydro["n_nodes"]

print(f"Nodes: {n_nodes}")

# Just load 30 days of forcing for quick test
forcing = extract_forcing(
    zarr_path=ZARR_PATH,
    node_coords=node_coords,
    node_elev=None,
    date_start="2001-01-01",
    date_end="2001-01-30",
    cache_nc=Path("/tmp/forcing_test.nc"),
    device=device,
)

print(f"Forcing shape: {forcing.shape}")
print(f"Forcing ranges:")
print(f"  Precip (mm/day): {forcing[:,:,0].min():.1f} to {forcing[:,:,0].max():.1f}")
print(f"  T_min (C): {forcing[:,:,1].min():.1f} to {forcing[:,:,1].max():.1f}")
print(f"  T_max (C): {forcing[:,:,2].min():.1f} to {forcing[:,:,2].max():.1f}")
print(f"  Radiation (MJ/m2/day): {forcing[:,:,3].min():.1f} to {forcing[:,:,3].max():.1f}")

# Create a minimal model
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
    param_mode="static",  # Use static for simplicity
).to(device)

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

# Get withdrawals
withdrawals = cache.load_withdrawals(
    date_start="2001-01-01",
    date_end="2001-01-30",
    device=device,
)

# Day of year
doy = torch.tensor([i for i in range(1, 31)], dtype=torch.long, device=device)

# Run simulation with default initialization
print("\n=== Testing with ZERO initial states ===")
initial_state = HydroState.zeros(n_nodes, device=device)
print(f"Initial theta1: {initial_state.theta1.mean():.3f}")

with torch.no_grad():
    Q_sim, final_state = model.simulate(
        forcing=forcing,
        initial_state=initial_state,
        graph=graph,
        node_coords=node_coords,
        territorial=territorial,
        withdrawals=withdrawals,
        day_of_year=doy,
    )

print(f"\nSimulated discharge (m3/s):")
print(f"  Mean: {Q_sim.mean():.2f}")
print(f"  Min: {Q_sim.min():.2f}")
print(f"  Max: {Q_sim.max():.2f}")
print(f"  Negative values: {(Q_sim < 0).sum().item()}")
print(f"  NaN values: {torch.isnan(Q_sim).sum().item()}")
print(f"  Inf values: {torch.isinf(Q_sim).sum().item()}")

# Check at a few specific nodes
sample_nodes = [0, 100, 500, 1000, 2000]
for node in sample_nodes:
    print(f"\nNode {node}:")
    print(f"  Discharge: {Q_sim[:5, node].cpu().numpy()}")
    print(f"  Mean Q: {Q_sim[:, node].mean():.2f} m3/s")

print("\nFinal state:")
print(f"  theta1: {final_state.theta1.mean():.3f} (should be ~0.2-0.4)")
print(f"  theta2: {final_state.theta2.mean():.3f}")
print(f"  theta3: {final_state.theta3.mean():.3f}")
print(f"  SWE: {final_state.swe.mean():.1f} mm")
print(f"  S_gw: {final_state.S_gw.mean():.1f} mm")

# Test with realistic initial states
print("\n=== Testing with REALISTIC initial states ===")
initial_state = HydroState.default_warm(n_nodes, device=device)
print(f"Initial theta1: {initial_state.theta1.mean():.3f}")

with torch.no_grad():
    Q_sim2, final_state2 = model.simulate(
        forcing=forcing,
        initial_state=initial_state,
        graph=graph,
        node_coords=node_coords,
        territorial=territorial,
        withdrawals=withdrawals,
        day_of_year=doy,
    )

print(f"\nSimulated discharge (m3/s):")
print(f"  Mean: {Q_sim2.mean():.2f}")
print(f"  Min: {Q_sim2.min():.2f}")
print(f"  Max: {Q_sim2.max():.2f}")
print(f"  Negative values: {(Q_sim2 < 0).sum().item()}")
print(f"  NaN values: {torch.isnan(Q_sim2).sum().item()}")

# Load observations for comparison
obs = cache.load_observations(
    date_start="2001-01-01",
    date_end="2001-01-30",
    min_valid_days=10,
)

if obs["discharge"].size > 0:
    q_obs = obs["discharge"]
    valid = ~np.isnan(q_obs)
    if valid.any():
        print(f"\nObserved discharge (m3/s):")
        print(f"  Mean: {np.nanmean(q_obs):.2f}")
        print(f"  Min: {np.nanmin(q_obs):.2f}")
        print(f"  Max: {np.nanmax(q_obs):.2f}")

        # Compare magnitudes
        sim_order = np.log10(Q_sim2.mean().cpu().item() + 1e-6)
        obs_order = np.log10(np.nanmean(q_obs) + 1e-6)
        print(f"\nOrder of magnitude comparison:")
        print(f"  Simulated: 10^{sim_order:.1f}")
        print(f"  Observed: 10^{obs_order:.1f}")
        print(f"  Ratio sim/obs: {Q_sim2.mean().cpu().item() / (np.nanmean(q_obs) + 1e-6):.2f}")

# Check parameter values
with torch.no_grad():
    coords_norm = (node_coords - node_coords.mean(dim=0)) / node_coords.std(dim=0)
    spatial_params = model.spatial_encoder(coords_norm, territorial.features)

print(f"\n=== Spatial parameters (sample) ===")
print(f"K_sat_1 (m/day): {spatial_params.K_sat_1.mean():.3f} (typical: 0.1-2.0)")
print(f"Porosity_1: {spatial_params.porosity_1.mean():.3f} (typical: 0.3-0.5)")
print(f"theta_fc_1: {spatial_params.theta_fc_1.mean():.3f} (typical: 0.2-0.3)")
print(f"Manning_n: {spatial_params.manning_n.mean():.3f} (typical: 0.01-0.1)")
print(f"C_f (melt factor): {spatial_params.C_f.mean():.3f} (typical: 2-6)")

# Check if parameters are within physical bounds
issues = []
if (spatial_params.porosity_1 < 0.1).any() or (spatial_params.porosity_1 > 0.8).any():
    issues.append("Porosity out of range")
if (spatial_params.theta_fc_1 > spatial_params.porosity_1).any():
    issues.append("Field capacity > porosity!")
if (spatial_params.theta_wp_1 > spatial_params.theta_fc_1).any():
    issues.append("Wilting point > field capacity!")
if (spatial_params.K_sat_1 < 0).any():
    issues.append("Negative K_sat!")
if (spatial_params.manning_n < 0).any():
    issues.append("Negative Manning's n!")

if issues:
    print(f"\n⚠️ PHYSICS ISSUES FOUND:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print(f"\n✓ Parameters within physical bounds")

print("\n=== Checking unit conversions ===")
# Check a simple water balance
P_total = forcing[:, 0, 0].sum() * 1e-3  # mm -> m
print(f"Total precip at node 0: {P_total:.3f} m")
print(f"Total outflow at node 0: {Q_sim2[:, 0].sum():.1f} m3/s * days")
print("(These units don't match - need area conversion!)")