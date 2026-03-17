"""Diagnostic: instrument one forward pass to find where the signal dies.

Run from repo root:
    python scripts/diagnose_forward.py
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import numpy as np
import xarray as xr
from pathlib import Path

from meandre.data.basin_cache import BasinCache
from meandre.data.station_obs import load_hydrometric_stations
from meandre.model import YHydro
from meandre.routing.withdrawals import WithdrawalData
from meandre.spatial.field_network import SpatialFieldNetwork

# ── Paths ──
NB_DIR = Path("notebooks")
BASIN_DB = NB_DIR / "data/slso.duckdb"
FORCING_CACHE = NB_DIR / "data/slso/forcing.nc"
CHECKPOINT = NB_DIR / "checkpoints/slso/best.pt"

DATE_START, DATE_END = "2000-01-01", "2001-12-31"
TRAIN_START, TRAIN_END = "2000-06-01", "2000-12-31"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Load data ──
cache = BasinCache(BASIN_DB)
hydro = cache.load(device=device)
graph = hydro["graph"]
territorial = hydro["territorial"]
node_coords = hydro["node_coords"]
initial_state = hydro["initial_state"]
node_ids = hydro["node_ids"]
n_nodes = hydro["n_nodes"]

from meandre.data.gridded_forcing import extract_forcing
forcing = extract_forcing(
    zarr_path=FORCING_CACHE, node_coords=node_coords,
    node_elev=territorial.mean_elevation_m,
    date_start=DATE_START, date_end=DATE_END,
    cache_nc=FORCING_CACHE, device=device,
)

ds_time = xr.open_dataset(FORCING_CACHE)
all_dates = ds_time.time.sel(time=slice(DATE_START, DATE_END)).values
ds_time.close()

import pandas as pd
doy = torch.tensor(
    [pd.Timestamp(d).day_of_year for d in all_dates],
    dtype=torch.long, device=device,
)

withdrawals = WithdrawalData.zeros(len(all_dates), n_nodes, device=device)

# ── Build model & load checkpoint ──
model = YHydro(
    n_nodes=n_nodes, n_forcing=6, context_window=90,
    residual_history=14, max_travel_time=20,
    use_temporal=True, use_residual=True, use_travel_time_attn=True,
).to(device)

if CHECKPOINT.exists():
    model.load_state_dict(torch.load(str(CHECKPOINT), map_location=device))
    print("Loaded checkpoint")
else:
    print("NO CHECKPOINT — using random init (fresh constraints)")

model.eval()

# ── Diagnostic: step through 30 days of summer (should have rain events) ──
# Pick June-July 2000
t_start = 152  # ~June 1
t_end = t_start + 30

print(f"\n{'='*70}")
print(f"DIAGNOSTIC: days {t_start}-{t_end} ({all_dates[t_start]} to {all_dates[t_end-1]})")
print(f"{'='*70}")

# 1. Spatial params
spatial_params = model.spatial_encoder(node_coords, territorial.to_tensor())
print(f"\n── SPATIAL PARAMS (mean ± std across {n_nodes} nodes) ──")
for name in ['K_sat_1', 'K_sat_2', 'K_sat_3',
             'porosity_1', 'porosity_2', 'porosity_3',
             'theta_fc_1', 'theta_fc_2', 'theta_fc_3',
             'theta_wp_1', 'theta_wp_2', 'theta_wp_3',
             'f_root_1', 'f_root_2', 'f_root_3',
             'C_f', 'T_melt', 'T_snow',
             'interception_capacity', 'manning_n', 'frost_alpha',
             'f_wetland']:
    v = getattr(spatial_params, name)
    print(f"  {name:25s}: {v.mean():.4f} ± {v.std():.4f}  [{v.min():.4f}, {v.max():.4f}]")

# 2. Check initial state
print(f"\n── INITIAL STATE ──")
state = initial_state
for name in ['theta1', 'theta2', 'theta3', 'swe', 't_soil', 'canopy_storage', 'wetland_storage']:
    v = getattr(state, name)
    print(f"  {name:20s}: {v.mean():.4f} ± {v.std():.4f}  [{v.min():.4f}, {v.max():.4f}]")

# 3. Step through days, printing fluxes
print(f"\n── STEPPING THROUGH {t_end - t_start} DAYS ──")
print(f"{'day':>4} {'P_mm':>7} {'T_air':>6} "
      f"{'P_thru':>7} "
      f"{'ET1':>5} {'ET2':>5} {'ET3':>5} "
      f"{'R_sur':>6} {'interf':>6} {'basef':>7} "
      f"{'lat_mm':>7} "
      f"{'theta1':>8} {'theta2':>7} {'theta3':>7}")

from meandre.temporal.ring_buffer import OutflowRingBuffer
outflow_buffer = OutflowRingBuffer(n_nodes, depth=20, device=device)
Q_out_prev = torch.zeros(n_nodes, device=device)
lake_storage = torch.zeros(n_nodes, device=device) if graph.is_lake.any() else None

area_km2 = (territorial.area_km2_physical.to(device)
            if territorial.area_km2_physical is not None
            else torch.ones(n_nodes, device=device))
area_km2_local = (territorial.area_km2_local.to(device)
                  if territorial.area_km2_local is not None
                  else None)

# Run spinup first (days 0 to t_start)
with torch.no_grad():
    for t in range(t_start):
        enriched = forcing[t]
        lateral_inflow, state = model.vertical_column(enriched, state, spatial_params)
        Q_out, lake_storage = model.routing(
            lateral_inflow, graph, Q_out_prev, outflow_buffer, withdrawals, t,
            model.K_musk, model.x_musk,
            lake_storage=lake_storage, area_km2=area_km2,
            area_km2_local=area_km2_local,
        )
        outflow_buffer.push(Q_out)
        Q_out_prev = Q_out

print(f"\n── POST-SPINUP STATE (after {t_start} days) ──")
for name in ['theta1', 'theta2', 'theta3', 'swe', 't_soil']:
    v = getattr(state, name)
    print(f"  {name:20s}: {v.mean():.4f} ± {v.std():.4f}  [{v.min():.4f}, {v.max():.4f}]")

# Now step through diagnostic days with detailed output
with torch.no_grad():
    for t in range(t_start, t_end):
        state_before_t1 = state.theta1.mean().item()

        P = forcing[t, :, 0]
        T_min = forcing[t, :, 1]
        T_max = forcing[t, :, 2]
        T_air = 0.5 * (T_min + T_max)

        # Vertical column with diagnostics
        lateral_inflow, new_state, vc_diag = model.vertical_column(
            forcing[t], state, spatial_params, return_diagnostics=True,
        )

        # Decompose lateral: we need soil output separately
        # Re-run soil step to get components
        from meandre.vertical.snow import SnowModule
        from meandre.vertical.interception import InterceptionModule
        from meandre.vertical.evapotranspiration import ETModule

        # Get soil outputs directly
        snow = model.vertical_column.snow
        frost = model.vertical_column.frost
        interception = model.vertical_column.interception
        et = model.vertical_column.et
        soil = model.vertical_column.soil

        P_eff, _ = snow(P, T_air, state.swe, spatial_params.C_f, spatial_params.T_melt, spatial_params.T_snow)
        K_sat_1_eff, _ = frost(T_air, state.t_soil, spatial_params.K_sat_1, spatial_params.frost_alpha)
        ETP = et.penman_monteith(T_min, T_max, forcing[t,:,3], forcing[t,:,4], forcing[t,:,5])
        P_thru, E_canopy, _ = interception(P_eff, ETP, state.canopy_storage, spatial_params.interception_capacity)
        ET1, ET2, ET3, _ = et(
            T_min, T_max, forcing[t,:,3], forcing[t,:,4], forcing[t,:,5],
            state.theta1, state.theta2, state.theta3,
            spatial_params.theta_wp_1, spatial_params.theta_wp_2, spatial_params.theta_wp_3,
            spatial_params.theta_fc_1, spatial_params.theta_fc_2, spatial_params.theta_fc_3,
            spatial_params.f_root_1, spatial_params.f_root_2, spatial_params.f_root_3,
            E_canopy,
        )
        th1_new, th2_new, th3_new, R_surf, interflow, baseflow = soil(
            P_thru, ET1, ET2, ET3,
            state.theta1, state.theta2, state.theta3,
            K_sat_1_eff, spatial_params.K_sat_2, spatial_params.K_sat_3,
            spatial_params.porosity_1, spatial_params.porosity_2, spatial_params.porosity_3,
            spatial_params.theta_fc_1, spatial_params.theta_fc_2, spatial_params.theta_fc_3,
            spatial_params.theta_wp_1, spatial_params.theta_wp_2, spatial_params.theta_wp_3,
        )

        # Routing
        Q_out, lake_storage = model.routing(
            lateral_inflow, graph, Q_out_prev, outflow_buffer, withdrawals, t,
            model.K_musk, model.x_musk,
            lake_storage=lake_storage, area_km2=area_km2,
            area_km2_local=area_km2_local,
        )
        outflow_buffer.push(Q_out)
        Q_out_prev = Q_out
        state = new_state

        # Print basin-mean values
        print(f"{t:4d} {P.mean():7.2f} {T_air.mean():6.1f} "
              f"{P_thru.mean():7.2f} "
              f"{ET1.mean():5.2f} {ET2.mean():5.2f} {ET3.mean():5.2f} "
              f"{R_surf.mean():6.2f} {interflow.mean():6.3f} {baseflow.mean():7.2f} "
              f"{lateral_inflow.mean():7.2f} "
              f"{new_state.theta1.mean():.5f} {new_state.theta2.mean():.4f} {new_state.theta3.mean():.4f}")

        # Detailed trace for first rainy day > 10mm
        if t == t_start or (P.mean() > 10 and t == 162):
            print(f"    --- DETAILED TRACE (basin mean) ---")
            print(f"    P_eff={P_eff.mean():.4f} P_thru={P_thru.mean():.4f} E_canopy={E_canopy.mean():.4f}")
            print(f"    theta1_in={state_before_t1:.6f}  porosity_1={spatial_params.porosity_1.mean():.4f}")
            print(f"    theta_fc_1={spatial_params.theta_fc_1.mean():.4f}  theta_wp_1={spatial_params.theta_wp_1.mean():.4f}")
            dz12 = (0.3 + 0.7) / 2
            q12_val = soil._darcy_flux(
                state.theta1 if t > t_start else torch.zeros_like(state.theta1),
                state.theta2, K_sat_1_eff, spatial_params.theta_wp_1, spatial_params.porosity_1, dz12
            )
            print(f"    q12={q12_val.mean()*1e3:.4f} mm/day  q23={0:.4f}  q_recharge(baseflow)={baseflow.mean():.4f}")
            print(f"    q_inter_1={soil._interflow(state.theta1 if t>t_start else torch.zeros_like(state.theta1), spatial_params.theta_fc_1, spatial_params.porosity_1, 0.3).mean()*1e3:.4f} mm/day")
            print(f"    excess_1={R_surf.mean():.4f} mm (R_surface)")
            # Check mass balance
            P_m = P_thru.mean() * 1e-3
            dth = (new_state.theta1.mean() - (state_before_t1)).item()
            print(f"    dtheta1={dth:.6f}  expected_from_P_only={P_m/0.3:.6f}")

# 4. Routing diagnostics
print(f"\n── ROUTING PARAMS ──")
K_musk = model.K_musk
x_musk = model.x_musk
print(f"  K_musk (hours): {(K_musk/3600).mean():.1f} ± {(K_musk/3600).std():.1f}  [{(K_musk/3600).min():.1f}, {(K_musk/3600).max():.1f}]")
print(f"  x_musk:         {x_musk.mean():.4f} ± {x_musk.std():.4f}  [{x_musk.min():.4f}, {x_musk.max():.4f}]")

# 5. Area diagnostics
print(f"\n── AREA CONVERSION ──")
if area_km2_local is not None:
    print(f"  area_km2_local: {area_km2_local.mean():.2f} ± {area_km2_local.std():.2f}  [{area_km2_local.min():.2f}, {area_km2_local.max():.2f}]")
else:
    print(f"  area_km2_local: NONE (using area_km2)")
print(f"  area_km2 (cumul): {area_km2.mean():.2f} ± {area_km2.std():.2f}  [{area_km2.min():.2f}, {area_km2.max():.2f}]")

# 6. Check if conversion factor makes sense
lat_mean = lateral_inflow.mean().item()
if area_km2_local is not None:
    conv = area_km2_local
else:
    conv = area_km2
q_m3s = lat_mean * 1e-3 * conv.mean().item() * 1e6 / 86400.0
print(f"  lateral_mm={lat_mean:.3f} mm/day -> q_lat_m3s={q_m3s:.4f} m3/s (using mean area)")

# 7. Soil module learnable params
print(f"\n── SOIL MODULE LEARNED PARAMS ──")
print(f"  k_interflow: {soil.k_interflow.item():.6f} 1/day")
print(f"  log_k_interflow: {soil.log_k_interflow.item():.4f}")

# 8. Muskingum coefficient check
print(f"\n── MUSKINGUM COEFFICIENTS (mean values) ──")
dt = 86400.0
K = K_musk.mean().item()
x = x_musk.mean().item()
denom = 2*K*(1-x) + dt
c0 = (dt - 2*K*x) / denom
c1 = (dt + 2*K*x) / denom
c2 = (2*K*(1-x) - dt) / denom
print(f"  c0={c0:.4f}  c1={c1:.4f}  c2={c2:.4f}  sum={c0+c1+c2:.4f}")
print(f"  (c0+c1)={c0+c1:.4f}  -- weight on lateral inflow")
print(f"  c2={c2:.4f}  -- memory of previous Q_out")

# 9. Check the Muskingum formula itself
print(f"\n── MUSKINGUM FORMULA CHECK ──")
print(f"  Q_out = c0*Q_in + c1*Q_in + c2*Q_out_prev + (c0+c1)*q_lateral")
print(f"  NOTE: c0*Q_in + c1*Q_in = (c0+c1)*Q_in -- both terms use SAME Q_in!")
print(f"  This means: Q_out = (c0+c1)*Q_in + c2*Q_out_prev + (c0+c1)*q_lateral")
print(f"            = (c0+c1)*(Q_in + q_lateral) + c2*Q_out_prev")
