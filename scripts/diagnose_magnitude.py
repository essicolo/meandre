"""Diagnose Q_sim magnitude: where is the volume lost?

Traces water from precipitation → lateral_inflow → Q_out at the outlet.
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import numpy as np
import xarray as xr
from pathlib import Path
import pandas as pd

from meandre.data.basin_cache import BasinCache
from meandre.data.gridded_forcing import extract_forcing
from meandre.model import YHydro
from meandre.routing.withdrawals import WithdrawalData

NB_DIR = Path("notebooks")
BASIN_DB = NB_DIR / "data/slso.duckdb"
FORCING_CACHE = NB_DIR / "data/slso/forcing.nc"
CHECKPOINT = NB_DIR / "checkpoints/slso/best.pt"

DATE_START, DATE_END = "2000-01-01", "2001-12-31"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cache = BasinCache(BASIN_DB)
hydro = cache.load(device=device)
graph = hydro["graph"]
territorial = hydro["territorial"]
node_coords = hydro["node_coords"]
initial_state = hydro["initial_state"]
node_ids = hydro["node_ids"]
n_nodes = hydro["n_nodes"]

forcing = extract_forcing(
    zarr_path=FORCING_CACHE, node_coords=node_coords,
    node_elev=territorial.mean_elevation_m,
    date_start=DATE_START, date_end=DATE_END,
    cache_nc=FORCING_CACHE, device=device,
)

ds_time = xr.open_dataset(FORCING_CACHE)
all_dates = ds_time.time.sel(time=slice(DATE_START, DATE_END)).values
ds_time.close()

doy = torch.tensor(
    [pd.Timestamp(d).day_of_year for d in all_dates],
    dtype=torch.long, device=device,
)

withdrawals = WithdrawalData.zeros(len(all_dates), n_nodes, device=device)

model = YHydro(
    n_nodes=n_nodes, n_forcing=6, context_window=90,
    residual_history=14, max_travel_time=20,
    use_temporal=True, use_residual=True, use_travel_time_attn=True,
).to(device)

if CHECKPOINT.exists():
    model.load_state_dict(torch.load(str(CHECKPOINT), map_location=device))
    print("Loaded trained checkpoint")
else:
    print("NO CHECKPOINT")

model.eval()

# Disable untrained modules (curriculum had epoch=9999 for all three)
model.use_temporal = False
model.use_residual = False
model.routing.use_tta = False
print(f"Disabled untrained modules: temporal={model.use_temporal}, residual={model.use_residual}, tta={model.routing.use_tta}")

# Find the outlet node (troncon 24) — largest drainage area
outlet_idx = node_ids.index(24) if 24 in node_ids else 0
print(f"Outlet node index: {outlet_idx} (troncon {node_ids[outlet_idx]})")

# Check area_km2_local
area_km2_local = (territorial.area_km2_local.to(device)
                  if territorial.area_km2_local is not None else None)
area_km2 = (territorial.area_km2_physical.to(device)
            if territorial.area_km2_physical is not None
            else torch.ones(n_nodes, device=device))

print(f"\n── AREA DIAGNOSTICS ──")
if area_km2_local is not None:
    print(f"  area_km2_local: mean={area_km2_local.mean():.2f} sum={area_km2_local.sum():.0f} "
          f"[{area_km2_local.min():.3f}, {area_km2_local.max():.0f}]")
    print(f"  area_km2_local[outlet]={area_km2_local[outlet_idx]:.2f}")
else:
    print("  area_km2_local: NONE!")
print(f"  area_km2_cumul: mean={area_km2.mean():.2f} max={area_km2.max():.0f}")
print(f"  area_km2_cumul[outlet]={area_km2[outlet_idx]:.0f}")

# Run full simulation with diagnostics
print(f"\n── RUNNING FULL SIMULATION ({len(all_dates)} days) ──")
with torch.no_grad():
    Q_sim, final_state, diagnostics = model.simulate(
        forcing=forcing,
        initial_state=initial_state,
        graph=graph,
        node_coords=node_coords,
        territorial=territorial,
        withdrawals=withdrawals,
        day_of_year=doy,
        return_diagnostics=True,
    )

print(f"Q_sim shape: {Q_sim.shape}")

# 1. Lateral inflow magnitude
lat_mm = diagnostics.lateral_mm  # (T, N) mm/day
q_lat = diagnostics.q_lateral    # (T, N) m3/s
q_up = diagnostics.q_upstream    # (T, N) m3/s

print(f"\n── LATERAL INFLOW (mm/day) ──")
print(f"  Basin mean: {lat_mm.mean():.3f} mm/day")
print(f"  At outlet:  {lat_mm[:, outlet_idx].mean():.3f} mm/day")
print(f"  Max across basin: {lat_mm.max():.1f}")
print(f"  > 1mm/day nodes per day: {(lat_mm > 1).float().mean()*n_nodes:.0f}")

# 2. Expected Q at outlet from simple area conversion
total_area_km2 = area_km2_local.sum() if area_km2_local is not None else area_km2.sum()
basin_mean_lat = lat_mm.mean(dim=1)  # (T,) mean mm/day per day
expected_Q = basin_mean_lat * 1e-3 * total_area_km2 * 1e6 / 86400.0
actual_Q_outlet = Q_sim[:, outlet_idx]

print(f"\n── Q AT OUTLET (troncon {node_ids[outlet_idx]}) ──")
print(f"  Total local area sum: {total_area_km2:.0f} km2")
print(f"  Expected Q from lateral (simple sum): mean={expected_Q.mean():.1f}, max={expected_Q.max():.1f} m3/s")
print(f"  Actual Q_sim at outlet:               mean={actual_Q_outlet.mean():.1f}, max={actual_Q_outlet.max():.1f} m3/s")
print(f"  Ratio actual/expected: {actual_Q_outlet.mean() / (expected_Q.mean() + 1e-6):.3f}")

# 3. Check routing accumulation — does Q increase downstream?
print(f"\n── ROUTING ACCUMULATION ──")
# Mean Q per node
q_mean = Q_sim.mean(dim=0)
print(f"  Q_sim mean across all nodes: {q_mean.mean():.3f} m3/s")
print(f"  Q_sim max across all nodes:  {q_mean.max():.3f} m3/s (node {q_mean.argmax().item()})")
print(f"  Q_sim at outlet:             {q_mean[outlet_idx]:.3f} m3/s")

# q_upstream (aggregated upstream Q) at outlet
q_up_outlet = q_up[:, outlet_idx]
print(f"  q_upstream at outlet: mean={q_up_outlet.mean():.3f}, max={q_up_outlet.max():.3f} m3/s")

# 4. Check q_lateral conversion at outlet
q_lat_outlet = q_lat[:, outlet_idx]
print(f"  q_lateral at outlet:  mean={q_lat_outlet.mean():.3f}, max={q_lat_outlet.max():.3f} m3/s")

# 5. Graph connectivity check
print(f"\n── GRAPH TOPOLOGY ──")
print(f"  n_nodes={graph.n_nodes}  n_edges={graph.n_edges}")
if graph.n_edges > 0:
    src = graph.edge_index[0]
    dst = graph.edge_index[1]
    # How many nodes have upstream connections?
    has_upstream = torch.zeros(n_nodes, dtype=torch.bool)
    has_upstream[dst.unique()] = True
    # How many nodes are headwaters (no upstream)?
    headwaters = ~has_upstream
    print(f"  Headwater nodes: {headwaters.sum().item()}")
    print(f"  Nodes with upstream: {has_upstream.sum().item()}")
    # Outlet: how many upstream edges?
    outlet_upstream = (dst == outlet_idx).sum().item()
    print(f"  Outlet (idx={outlet_idx}) has {outlet_upstream} direct upstream edges")
    # Check if outlet is actually in topo_order
    if hasattr(graph, 'topo_order') and graph.topo_order is not None:
        outlet_topo_pos = (graph.topo_order == outlet_idx).nonzero()
        if len(outlet_topo_pos) > 0:
            print(f"  Outlet position in topo_order: {outlet_topo_pos[0].item()} / {len(graph.topo_order)}")
        else:
            print(f"  WARNING: outlet NOT in topo_order!")

# 6. Muskingum params after training
K_musk = model.K_musk
x_musk = model.x_musk
print(f"\n── MUSKINGUM PARAMS ──")
print(f"  K (hours): {(K_musk/3600).mean():.1f} ± {(K_musk/3600).std():.1f} [{(K_musk/3600).min():.1f}, {(K_musk/3600).max():.1f}]")
print(f"  x: {x_musk.mean():.4f} ± {x_musk.std():.4f}")

# 7. Spatial params from trained model
spatial_params = model.spatial_encoder(node_coords, territorial.to_tensor())
print(f"\n── TRAINED SPATIAL PARAMS ──")
for name in ['K_sat_1', 'K_sat_2', 'K_sat_3', 'porosity_1', 'theta_fc_1', 'theta_wp_1',
             'f_root_1', 'f_root_2', 'f_root_3', 'C_f', 'T_melt', 'T_snow', 'f_wetland']:
    v = getattr(spatial_params, name)
    print(f"  {name:25s}: {v.mean():.4f} ± {v.std():.4f}")

# 8. Check Muskingum double-counting
print(f"\n── MUSKINGUM LATERAL DOUBLE-COUNTING CHECK ──")
print(f"  In _route_vectorized:")
print(f"    Q_in_all = Q_agg + q_lat_m3s  (lateral included)")
print(f"    muskingum(Q_in=Q_in_all, q_lateral=q_lat_m3s)  (lateral passed AGAIN)")
print(f"    -> Q_out = (c0+c1)*(Q_agg + q_lat) + c2*Q_prev + (c0+c1)*q_lat")
print(f"    -> lateral is DOUBLE-COUNTED")
dt = 86400.0
K = K_musk.mean().item()
x = x_musk.mean().item()
denom = 2*K*(1-x) + dt
c0c1 = 2*dt / denom
print(f"    c0+c1={c0c1:.4f}  -> each q_lat gets weight {2*c0c1:.4f} instead of {c0c1:.4f}")

# 9. Time series at outlet for a few days
print(f"\n── Q_SIM AT OUTLET — first 30 summer days ──")
t_start = 152
for t in range(t_start, t_start + 10):
    d = str(all_dates[t])[:10]
    print(f"  {d}: Q_sim={Q_sim[t, outlet_idx]:.2f}  q_lat={q_lat[t, outlet_idx]:.3f}  "
          f"q_upstream={q_up[t, outlet_idx]:.2f}  lat_mm={lat_mm[t, outlet_idx]:.2f}")
