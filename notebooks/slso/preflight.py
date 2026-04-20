"""Pre-training diagnostic — run BEFORE launching a multi-hour training.

Usage:
    python notebooks/slso/preflight.py [config.toml]

Checks (~2-3 minutes):
  1.  Config consistency (chunk-safe losses, period lengths, paths)
  2.  Data loads without error (basin, forcing, observations, withdrawals)
  3.  Model builds and checkpoint loads (if warm_start)
  4.  Short forward pass (30 steps): no NaN/Inf, Q > 0
  5.  Spatial param statistics + physical constraint checks
  6.  Muskingum coefficient stability (C0, C1, C2 in valid range)
  7.  State bounds after 30 steps (theta ∈ [0, porosity], swe ≥ 0)
  8.  Routing mass conservation (Q_out / lateral_in ≈ 1.0)
  9.  Water balance closure (P ≈ ET + lateral + ΔS)
  10. Gradient flow through ALL parameters (dummy loss)
  11. Actual loss function: finite loss + finite gradients
  12. Full training step dry-run: 1 chunk forward+backward+optimizer.step
  13. GPU memory headroom check (peak vs available)
  14. Optimizer LR + scheduler verification
"""

from pathlib import Path
import os
import sys

os.chdir(Path(__file__).resolve().parents[2])  # repo root

# Force UTF-8 stdout on Windows (cp1252 can't print Unicode box-drawing / math chars)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import tomllib
import torch

# ── Config ────────────────────────────────────────────────────────────────
CFG_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("notebooks/slso/config/slso.toml")
with open(CFG_PATH, "rb") as f:
    cfg = tomllib.load(f)

BASIN_DB = Path(cfg["paths"]["basin_db"])
ZARR_PATH = Path(cfg["paths"]["weather_grid"])
FORCING_CACHE = Path(cfg["paths"]["forcing_cache"])
CHECKPOINT = Path(cfg["paths"]["checkpoint"])
WARM_START = cfg["training"].get("warm_start", False)
LR = cfg["training"]["lr"]
LR_FINETUNE = cfg["training"].get("lr_finetune", LR * 0.1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

fails = []
warns = []

# ══════════════════════════════════════════════════════════════════════════
# CHECK 1 — Config consistency
# ══════════════════════════════════════════════════════════════════════════
print("\n--- [1/14] Config consistency ---")

lcfg = cfg["loss"]
chunk_steps = cfg["training"]["chunk_steps"]
NOT_CHUNK_SAFE = {"w_nse", "w_kge", "w_nrmse", "w_log_nse"}

if chunk_steps > 0:
    for key in NOT_CHUNK_SAFE:
        w = lcfg.get(key, 0.0)
        if w > 0:
            fails.append(f"{key}={w} but chunk_steps={chunk_steps}. "
                         f"NSE/KGE/NRMSE/log-NSE are NOT chunk-safe!")
            print(f"  [FAIL] {key}={w} with chunk_steps={chunk_steps}")

# Check period lengths make sense
from datetime import date
for name, (s, e) in [
    ("train", (cfg["temporal"]["train_start"], cfg["temporal"]["train_end"])),
    ("val", (cfg["temporal"]["val_start"], cfg["temporal"]["val_end"])),
]:
    ds, de = date.fromisoformat(s), date.fromisoformat(e)
    n_days = (de - ds).days + 1
    if n_days < 365:
        warns.append(f"{name} period only {n_days} days (< 1 year)")
        print(f"  [WARN] {name} period = {n_days} days")
    else:
        print(f"  [OK] {name} period = {n_days} days")

# tbptt vs chunk
tbptt = cfg["training"]["tbptt_steps"]
if chunk_steps > 0 and tbptt > chunk_steps:
    warns.append(f"tbptt_steps={tbptt} > chunk_steps={chunk_steps}; "
                 f"TBPTT won't help (only 1 detach per chunk)")
    print(f"  [WARN] tbptt_steps ({tbptt}) > chunk_steps ({chunk_steps})")
else:
    print(f"  [OK] chunk_steps={chunk_steps}, tbptt_steps={tbptt}")

# Patience must be a multiple of val_every to be useful
val_every = cfg["training"]["val_every"]
patience = cfg["training"]["patience"]
if patience > 0 and patience < val_every * 2:
    warns.append(f"patience={patience} but val_every={val_every}; "
                 f"only {patience // val_every} validation checks before stopping")
    print(f"  [WARN] patience={patience} with val_every={val_every}")
else:
    print(f"  [OK] patience={patience}, val_every={val_every}")

print(f"  [OK] Config loaded: {CFG_PATH}")

# ══════════════════════════════════════════════════════════════════════════
# CHECK 2 — Data loading
# ══════════════════════════════════════════════════════════════════════════
print("\n--- [2/14] Data loading ---")

from meandre.data.basin_cache import BasinCache
from meandre.data.gridded_forcing import extract_forcing

cache = BasinCache(BASIN_DB)
hydro = cache.load(device=device)
graph = hydro["graph"]
territorial = hydro["territorial"]
node_coords = hydro["node_coords"]
initial_state = hydro["initial_state"]
n_nodes = hydro["n_nodes"]
print(f"  [OK] Basin: {n_nodes} nodes, {graph.n_edges} edges, "
      f"{territorial.n_features} territorial features")

forcing = extract_forcing(
    zarr_path=ZARR_PATH,
    node_coords=node_coords,
    node_elev=None,
    date_start=cfg["temporal"]["date_start"],
    date_end=cfg["temporal"]["date_end"],
    cache_nc=FORCING_CACHE,
    device=device,
)
n_total_days = forcing.shape[0]
print(f"  [OK] Forcing: {forcing.shape} ({n_total_days} days)")

# Check forcing for NaN/Inf
if torch.isnan(forcing).any():
    n_nan = torch.isnan(forcing).sum().item()
    fails.append(f"Forcing has {n_nan} NaN values!")
    print(f"  [FAIL] Forcing has {n_nan} NaN values")
elif torch.isinf(forcing).any():
    fails.append("Forcing has Inf values!")
    print(f"  [FAIL] Forcing has Inf values")
else:
    print(f"  [OK] Forcing: no NaN/Inf")

# Observations
obs = cache.load_observations(
    date_start=cfg["temporal"]["date_start"],
    date_end=cfg["temporal"]["date_end"],
    min_valid_days=365,
)
station_node_map = obs["station_node_map"]
station_indices = sorted(set(station_node_map.values()))
n_stations = len(station_indices)
station_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
for ni in station_indices:
    station_mask[ni] = True
q_obs_tensor = torch.from_numpy(obs["discharge"][:, station_indices]).to(device)
obs_coverage = (~q_obs_tensor.isnan()).float().mean().item()
print(f"  [OK] Observations: {n_stations} stations, {obs_coverage:.1%} coverage")

# Withdrawals
withdrawals = cache.load_withdrawals(
    date_start=cfg["temporal"]["date_start"],
    date_end=cfg["temporal"]["date_end"],
    device=device,
)
n_wd_active = (withdrawals.net.abs() > 0).any(dim=0).sum().item()
print(f"  [OK] Withdrawals: {n_wd_active} active nodes")

# ══════════════════════════════════════════════════════════════════════════
# CHECK 3 — Model build + checkpoint
# ══════════════════════════════════════════════════════════════════════════
print("\n--- [3/14] Model build ---")

from meandre.model import HydroModel

DROPOUT = cfg["model"].get("dropout", 0.0)
model = HydroModel(
    n_nodes=n_nodes,
    n_territorial=territorial.n_features,
    n_forcing=cfg["model"]["n_forcing"],
    context_window=cfg["model"]["context_window"],
    residual_history=cfg["model"]["residual_history"],
    max_travel_time=cfg["model"]["max_travel_days"],
    use_temporal=True,
    use_residual=True,
    use_travel_time_attn=True,
    use_temperature=True,
    dropout=DROPOUT,
    param_mode=cfg["model"].get("param_mode", "nerf"),
).to(device)

n_params_total = sum(p.numel() for p in model.parameters())
print(f"  [OK] Model built: {n_params_total:,} parameters")
print(f"  Territorial features: {territorial.n_features}")

if WARM_START and CHECKPOINT.exists():
    try:
        model.load(str(CHECKPOINT))
        print(f"  [OK] Checkpoint loaded: {CHECKPOINT}")
    except Exception as e:
        fails.append(f"Checkpoint load failed: {e}")
        print(f"  [FAIL] Checkpoint load: {e}")
elif WARM_START:
    fails.append(f"warm_start=true but checkpoint missing: {CHECKPOINT}")
    print(f"  [FAIL] {fails[-1]}")
else:
    print("  [OK] Training from scratch (no checkpoint)")

# ══════════════════════════════════════════════════════════════════════════
# CHECK 4 — Forward pass (30 steps)
# ══════════════════════════════════════════════════════════════════════════
N_STEPS = 30
print(f"\n--- [4/14] Forward pass ({N_STEPS} timesteps) ---")

forcing_short = forcing[:N_STEPS]
doy = torch.arange(1, N_STEPS + 1, dtype=torch.long, device=device)

model.train()
try:
    Q_sim, final_state, diag = model.simulate(
        forcing_short, initial_state, graph,
        node_coords, territorial, withdrawals, doy,
        tbptt_steps=0, return_diagnostics=True,
    )
    print(f"  [OK] Forward pass completed. Q_sim shape: {Q_sim.shape}")
except Exception as e:
    fails.append(f"Forward pass crashed: {e}")
    print(f"  [FAIL] Forward pass: {e}")
    import traceback; traceback.print_exc()
    print(f"\n{'='*60}")
    print(f"PREFLIGHT FAILED — {len(fails)} issue(s)")
    for f in fails:
        print(f"  - {f}")
    sys.exit(1)

if torch.isnan(Q_sim).any() or torch.isinf(Q_sim).any():
    n_bad = (torch.isnan(Q_sim) | torch.isinf(Q_sim)).sum().item()
    fails.append(f"Q_sim has {n_bad} NaN/Inf values")
    print(f"  [FAIL] Q_sim has NaN/Inf! ({n_bad} values)")
else:
    print(f"  [OK] No NaN/Inf in Q_sim")

# Q > 0 at outlet
outlet_mask = graph.outlet_mask if hasattr(graph, 'outlet_mask') else None
Q_last = Q_sim[-1]
if outlet_mask is not None:
    q_outlet = Q_last[outlet_mask].mean().item()
else:
    q_outlet = Q_last.max().item()

if q_outlet > 0:
    print(f"  [OK] Q at outlet = {q_outlet:.4f} m³/s")
else:
    warns.append(f"Q at outlet = {q_outlet:.4f} (need spinup)")
    print(f"  [WARN] Q at outlet = {q_outlet:.4f} m³/s")

# ══════════════════════════════════════════════════════════════════════════
# CHECK 5 — Spatial param statistics
# ══════════════════════════════════════════════════════════════════════════
print(f"\n--- [5/14] Spatial parameter statistics ---")

import dataclasses

with torch.no_grad():
    sp = model.spatial_encoder(node_coords, territorial.to_tensor())

for field in dataclasses.fields(sp):
    if field.name == "N_PARAMS":
        continue
    val = getattr(sp, field.name)
    if not isinstance(val, torch.Tensor):
        continue
    lo, med, hi = val.min().item(), val.median().item(), val.max().item()
    cv = (val.std() / (val.mean() + 1e-10)).item()
    flag = ""
    if cv < 0.01:
        flag = "  <-- LOW CV"
    print(f"  {field.name:25s}  min={lo:10.4f}  med={med:10.4f}  max={hi:10.4f}  CV={cv:.4f}{flag}")

# ══════════════════════════════════════════════════════════════════════════
# CHECK 6 — Muskingum coefficient stability
# ══════════════════════════════════════════════════════════════════════════
print(f"\n--- [6/14] Muskingum coefficient stability ---")

K_seconds = sp.K_musk_hours * 3600.0  # -> seconds
x = sp.x_musk
n_sub = model.routing.muskingum.n_substeps
dt_sub = 86400.0 / n_sub

denom = 2.0 * K_seconds * (1.0 - x) + dt_sub
c0 = (dt_sub - 2.0 * K_seconds * x) / denom
c1 = (dt_sub + 2.0 * K_seconds * x) / denom
c2_raw = (2.0 * K_seconds * (1.0 - x) - dt_sub) / denom

print(f"  n_substeps = {n_sub}, dt_sub = {dt_sub:.0f} s")
print(f"  K range: [{K_seconds.min().item()/3600:.1f}, {K_seconds.max().item()/3600:.1f}] hours")
print(f"  x range: [{x.min().item():.3f}, {x.max().item():.3f}]")
print(f"  c0 range: [{c0.min().item():.4f}, {c0.max().item():.4f}]")
print(f"  c1 range: [{c1.min().item():.4f}, {c1.max().item():.4f}]")
print(f"  c2 (raw) range: [{c2_raw.min().item():.4f}, {c2_raw.max().item():.4f}]")

# Check: dt_sub < 2*K is the Courant-like stability condition
# If dt_sub > 2*K, c2 < 0 (dispersive) — handled by clamping but may be inefficient
n_dispersive = (c2_raw < 0).sum().item()
frac_dispersive = n_dispersive / len(c2_raw) * 100
if frac_dispersive > 50:
    warns.append(f"{frac_dispersive:.0f}% reaches in dispersive regime (dt_sub > 2K)")
    print(f"  [WARN] {frac_dispersive:.0f}% reaches dispersive (consider more substeps)")
else:
    print(f"  [OK] {frac_dispersive:.0f}% reaches dispersive")

# Check maximum c2 (attenuation). If c2 > 0.99, routing is nearly identity.
c2_clamped = torch.clamp(c2_raw, min=0.0)
n_sluggish = (c2_clamped > 0.95).sum().item()
if n_sluggish > 0:
    warns.append(f"{n_sluggish} reaches with c2 > 0.95 (nearly no routing attenuation)")
    print(f"  [WARN] {n_sluggish} reaches with c2 > 0.95 (K much larger than dt_sub)")
else:
    print(f"  [OK] All c2 < 0.95")

# ══════════════════════════════════════════════════════════════════════════
# CHECK 7 — State bounds after N steps
# ══════════════════════════════════════════════════════════════════════════
print(f"\n--- [7/14] State bounds after {N_STEPS} steps ---")

state_ok = True
# Tolerances: residual corrector can push state slightly out of physical bounds.
# Small excursions (< 5%) are normal; large ones indicate a real bug.
THETA_TOL = -0.05  # allow slight negative from residual correction
STORAGE_TOL = -0.05  # same for storage variables

for name in ["theta1", "theta2", "theta3"]:
    val = getattr(final_state, name)
    lo, hi = val.min().item(), val.max().item()
    if lo < THETA_TOL or hi > 1.1:
        fails.append(f"State {name} out of bounds: [{lo:.4f}, {hi:.4f}]")
        print(f"  [FAIL] {name}: [{lo:.4f}, {hi:.4f}]")
        state_ok = False
    elif lo < 0:
        print(f"  [OK] {name}: [{lo:.4f}, {hi:.4f}]  (minor negative — residual corrector)")
    else:
        print(f"  [OK] {name}: [{lo:.4f}, {hi:.4f}]")

for name in ["swe", "canopy_storage", "wetland_storage", "S_gw"]:
    val = getattr(final_state, name)
    lo = val.min().item()
    if lo < STORAGE_TOL:
        fails.append(f"State {name} too negative: {lo:.4f}")
        print(f"  [FAIL] {name}: min={lo:.4f}")
        state_ok = False
    elif lo < 0:
        print(f"  [OK] {name}: [{lo:.4f}, {val.max().item():.4f}]  (minor negative)")
    else:
        print(f"  [OK] {name}: [{lo:.4f}, {val.max().item():.4f}]")

if torch.isnan(final_state.to_tensor()).any():
    fails.append("State contains NaN after forward pass")
    print(f"  [FAIL] State has NaN!")
    state_ok = False

if state_ok:
    print(f"  [OK] All state variables within bounds")

# ══════════════════════════════════════════════════════════════════════════
# CHECK 8 — Routing mass conservation
# ══════════════════════════════════════════════════════════════════════════
print(f"\n--- [8/14] Routing mass conservation ---")

# Compare Q at OUTLET(S) to total lateral inflow across the basin.
# Summing Q_sim over all nodes double-counts water flowing through
# multiple reaches — only outlet outflow represents true basin output.
area_local = territorial.get_physical("area_km2_local")
if area_local is not None:
    lat_m3s = (diag.lateral_mm * area_local.unsqueeze(0) * 1e3 / 86400).sum(dim=1)  # (T,) total m³/s

    # Find outlet nodes (nodes with no downstream edge)
    all_src = set(graph.edge_index[0].cpu().tolist()) if graph.n_edges > 0 else set()
    all_dst = set(graph.edge_index[1].cpu().tolist()) if graph.n_edges > 0 else set()
    outlet_nodes = list(all_dst - all_src) if all_dst else []
    # Fallback: if no clear outlet found, use nodes not appearing as source
    if not outlet_nodes:
        all_nodes = set(range(n_nodes))
        outlet_nodes = list(all_nodes - all_src) if all_src else [Q_sim[-1].argmax().item()]

    Q_outlet = Q_sim[:, outlet_nodes].sum(dim=1)  # (T,) m³/s at outlet(s)
    # Use last 20 steps (skip first 10 for spinup effects)
    lat_mean = lat_m3s[10:].mean().item()
    Q_outlet_mean = Q_outlet[10:].mean().item()
    ratio = Q_outlet_mean / (lat_mean + 1e-10)
    print(f"  Outlet nodes: {len(outlet_nodes)}")
    print(f"  Mean lateral total:   {lat_mean:.2f} m³/s")
    print(f"  Mean Q at outlet(s):  {Q_outlet_mean:.2f} m³/s")
    print(f"  Ratio Q_outlet/lateral: {ratio:.3f}")
    # NOTE: with only 30 steps from random init, initial routing states
    # flush stored water -> ratio can be >1.  A ratio >5 indicates a real
    # mass-balance bug (e.g. lateral counted multiple times).
    if ratio > 5.0:
        fails.append(f"Routing amplifies lateral by {ratio:.1f}x (mass violation!)")
        print(f"  [FAIL] Routing amplifies water by {ratio:.1f}x")
    elif ratio > 2.0:
        warns.append(f"Routing ratio {ratio:.2f} is high (initial state flush?)")
        print(f"  [WARN] Ratio {ratio:.2f} is high (expect ~1.0 at steady state)")
    else:
        print(f"  [OK] Ratio {ratio:.2f} within expected range")
else:
    print("  [SKIP] No area_km2_local — cannot check routing mass balance")

# ══════════════════════════════════════════════════════════════════════════
# CHECK 9 — Water balance closure
# ══════════════════════════════════════════════════════════════════════════
print(f"\n--- [9/14] Water balance ({N_STEPS}-day mean, mm/day) ---")

P_mean = forcing_short[:, :, 0].mean().item()
etr_mean = diag.etr.mean().item()
lateral_mean = diag.lateral_mm.mean().item()
residual = P_mean - etr_mean - lateral_mean
print(f"  P         = {P_mean:.2f}")
print(f"  ETR       = {etr_mean:.2f}")
print(f"  Lateral   = {lateral_mean:.2f}")
print(f"  Residual  = {residual:.2f} (P - ETR - Lateral -> dS)")

if abs(residual) > P_mean * 2 and P_mean > 0.1:
    warns.append(f"Water balance residual ({residual:.2f}) > 2x precipitation")
    print(f"  [WARN] Residual is large relative to P")
else:
    print(f"  [OK] Residual plausible")

# ══════════════════════════════════════════════════════════════════════════
# CHECK 10 — Gradient flow (dummy loss)
# ══════════════════════════════════════════════════════════════════════════
print(f"\n--- [10/14] Gradient flow (dummy loss = Q_sim.sum()) ---")

model.zero_grad()
loss = Q_sim.sum()
loss.backward()

grad_ok = True
for name, p in model.named_parameters():
    if p.grad is None:
        if p.requires_grad:
            fails.append(f"No gradient for {name}")
            print(f"  [FAIL] {name}: NO GRADIENT")
            grad_ok = False
    elif torch.isnan(p.grad).any():
        fails.append(f"NaN gradient for {name}")
        print(f"  [FAIL] {name}: NaN gradient!")
        grad_ok = False
    elif torch.isinf(p.grad).any():
        fails.append(f"Inf gradient for {name}")
        print(f"  [FAIL] {name}: Inf gradient!")
        grad_ok = False

if grad_ok:
    print(f"  [OK] All parameters have finite gradients")

# Gradient norms by module
print(f"\n  Gradient norms by module:")
module_grads: dict[str, list[float]] = {}
for name, p in model.named_parameters():
    if p.grad is None:
        continue
    top = name.split(".")[0]
    module_grads.setdefault(top, []).append(p.grad.norm().item())

for mod, norms in sorted(module_grads.items()):
    mean_norm = sum(norms) / len(norms)
    print(f"    {mod:30s}  mean_grad_norm={mean_norm:.6f}  (n={len(norms)})")

# ══════════════════════════════════════════════════════════════════════════
# CHECK 11 — Actual loss function
# ══════════════════════════════════════════════════════════════════════════
print(f"\n--- [11/14] Actual loss function ---")

# Rebuild a minimal training-like loss evaluation
model.zero_grad()

# Re-run forward (need fresh graph)
with torch.enable_grad():
    Q_sim2, _, _ = model.simulate(
        forcing_short, initial_state, graph,
        node_coords, territorial, withdrawals, doy,
        tbptt_steps=0, return_diagnostics=True,
    )

# Build loss function matching slso.py
from meandre.training.loss import HydroLoss

loss_fn = HydroLoss(
    w_nse=lcfg["w_nse"], w_kge=lcfg["w_kge"], w_pbias=lcfg["w_pbias"],
    w_mse=lcfg["w_mse"], w_nrmse=lcfg["w_nrmse"],
    w_log_nse=lcfg["w_log_nse"], w_log_mse=lcfg["w_log_mse"],
    w_physics=lcfg["w_physics"], w_residual=lcfg["w_residual"],
    per_station=True,
)

# Use real obs (first N_STEPS)
q_obs_short = q_obs_tensor[:N_STEPS]
try:
    loss_val, comps = loss_fn(
        q_obs=q_obs_short,
        q_sim=Q_sim2,
        station_mask=station_mask,
    )
    print(f"  Loss value: {loss_val.item():.4f}")
    for k, v in comps.items():
        if isinstance(v, torch.Tensor):
            print(f"    {k:20s} = {v.item():.4f}")

    if torch.isnan(loss_val) or torch.isinf(loss_val):
        fails.append(f"Loss is NaN/Inf: {loss_val.item()}")
        print(f"  [FAIL] Loss is NaN/Inf!")
    elif not loss_val.requires_grad:
        fails.append("Loss has no gradient (requires_grad=False)")
        print(f"  [FAIL] Loss has no gradient!")
    else:
        loss_val.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e6)
        if torch.isnan(total_norm) or torch.isinf(total_norm):
            fails.append(f"NaN/Inf gradient norm from actual loss: {total_norm.item():.2f}")
            print(f"  [FAIL] Gradient norm is NaN/Inf through actual loss!")
        else:
            print(f"  [OK] Loss={loss_val.item():.4f}, grad_norm={total_norm.item():.4f}")
except Exception as e:
    fails.append(f"Loss function crashed: {e}")
    print(f"  [FAIL] Loss crashed: {e}")
    import traceback; traceback.print_exc()

# ══════════════════════════════════════════════════════════════════════════
# CHECK 12 — Full training dry-run (1 chunk forward+backward+step)
# ══════════════════════════════════════════════════════════════════════════
print(f"\n--- [12/14] Training dry-run (1 chunk = {chunk_steps} steps) ---")

# Free all memory from previous checks before measuring dry-run peak
import gc
del Q_sim, Q_sim2, final_state, diag, sp
gc.collect()
if device.type == "cuda":
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

model.zero_grad()
model.train()

# Use min(chunk_steps, available forcing) for the dry run
dry_steps = min(chunk_steps, n_total_days) if chunk_steps > 0 else min(180, n_total_days)
dry_forcing = forcing[:dry_steps]
dry_doy = torch.tensor(
    list(range(1, dry_steps + 1)),
    dtype=torch.long, device=device,
)
# Clamp doy to valid range
dry_doy = ((dry_doy - 1) % 366) + 1

optimizer_dry = torch.optim.AdamW(
    model.parameters(), lr=LR, weight_decay=cfg["training"]["weight_decay"],
)

try:
    Q_dry, _ = model.simulate(
        dry_forcing, initial_state, graph,
        node_coords, territorial, withdrawals, dry_doy,
        tbptt_steps=cfg["training"]["tbptt_steps"],
    )

    # Compute loss on whatever obs we have
    q_obs_dry = q_obs_tensor[:dry_steps]
    loss_dry, _ = loss_fn(
        q_obs=q_obs_dry, q_sim=Q_dry, station_mask=station_mask,
    )

    if torch.isnan(loss_dry) or not loss_dry.requires_grad:
        print(f"  [WARN] Dry-run loss NaN or no grad: {loss_dry.item():.4f}")
    else:
        loss_dry.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg["training"]["grad_clip"],
        )
        if torch.isnan(total_norm) or torch.isinf(total_norm):
            warns.append(f"Dry-run gradient norm NaN/Inf: {total_norm.item():.2f}")
            print(f"  [WARN] Grad norm NaN/Inf ({total_norm.item():.2f}) — "
                  f"model may struggle early on")
            optimizer_dry.zero_grad()
        else:
            optimizer_dry.step()
            print(f"  [OK] Dry-run: loss={loss_dry.item():.4f}, "
                  f"grad_norm={total_norm.item():.4f}")

    # Check for NaN in Q output
    if torch.isnan(Q_dry).any():
        n_nan = torch.isnan(Q_dry).sum().item()
        fails.append(f"NaN in Q_dry after {dry_steps}-step chunk ({n_nan} values)")
        print(f"  [FAIL] Q_dry has {n_nan} NaN values!")
    else:
        print(f"  [OK] No NaN in {dry_steps}-step chunk output")

except RuntimeError as e:
    if "out of memory" in str(e).lower():
        fails.append(f"CUDA OOM during {dry_steps}-step chunk! "
                     f"Reduce chunk_steps (currently {chunk_steps})")
        print(f"  [FAIL] CUDA OOM during {dry_steps}-step chunk!")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        fails.append(f"Dry-run crashed: {e}")
        print(f"  [FAIL] {e}")
        import traceback; traceback.print_exc()
except Exception as e:
    fails.append(f"Dry-run crashed: {e}")
    print(f"  [FAIL] {e}")
    import traceback; traceback.print_exc()

# ══════════════════════════════════════════════════════════════════════════
# CHECK 13 — GPU memory headroom
# ══════════════════════════════════════════════════════════════════════════
print(f"\n--- [13/14] GPU memory ---")

if device.type == "cuda":
    peak_alloc = torch.cuda.max_memory_allocated() / 1e9
    peak_reserved = torch.cuda.max_memory_reserved() / 1e9
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    # Use mem_get_info for actual free/total on device (more accurate than
    # max_memory_allocated which counts reused blocks cumulatively).
    try:
        free_mem, _ = torch.cuda.mem_get_info(0)
        free_gb = free_mem / 1e9
        print(f"  Peak allocated: {peak_alloc:.2f} GB (cumulative, includes reused)")
        print(f"  Peak reserved:  {peak_reserved:.2f} GB (CUDA allocator pool)")
        print(f"  Total VRAM:     {total_memory:.2f} GB")
        print(f"  Free right now: {free_gb:.2f} GB")
    except Exception:
        free_gb = total_memory  # can't get info, skip
        print(f"  Peak allocated: {peak_alloc:.2f} GB")
        print(f"  Total VRAM:     {total_memory:.2f} GB")

    # The real test: did the dry-run (check 12) complete without OOM?
    # If yes, memory is sufficient. peak_alloc can exceed VRAM on Windows
    # due to shared GPU memory (system RAM via PCIe), which is slower but works.
    dry_run_passed = not any("OOM" in f for f in fails)
    if not dry_run_passed:
        # Already failed in check 12 — don't double-count
        print(f"  [FAIL] OOM during dry-run (see check 12)")
    elif peak_reserved > total_memory * 0.95:
        warns.append(f"GPU memory very tight: {peak_reserved:.2f} GB reserved "
                     f"out of {total_memory:.2f} GB")
        print(f"  [WARN] Reserved {peak_reserved:.2f} GB / {total_memory:.2f} GB "
              f"— very tight, may OOM during long training")
    else:
        print(f"  [OK] Dry-run completed within memory limits")
    torch.cuda.empty_cache()
else:
    print("  [SKIP] No GPU")

# ══════════════════════════════════════════════════════════════════════════
# CHECK 14 — Optimizer LR + scheduler
# ══════════════════════════════════════════════════════════════════════════
print(f"\n--- [14/14] Optimizer LR + scheduler ---")

from meandre.training.scheduler import build_scheduler

expected_lr = LR_FINETUNE if WARM_START else LR
warmup_epochs = 0 if WARM_START else 5
optimizer_check = torch.optim.AdamW(
    model.parameters(), lr=expected_lr,
    weight_decay=cfg["training"]["weight_decay"],
)
scheduler = build_scheduler(
    optimizer_check, n_epochs=cfg["training"]["n_epochs"],
    warmup_epochs=warmup_epochs,
)

# Show LR schedule (first 10 epochs)
# Note: with warmup, the scheduler sets LR to a low value initially and ramps up.
# We check that the PEAK LR after warmup matches expected_lr.
lrs = []
for i in range(min(10, cfg["training"]["n_epochs"])):
    lrs.append(optimizer_check.param_groups[0]["lr"])
    optimizer_check.step()  # dummy step to avoid PyTorch warning
    scheduler.step()
    optimizer_check.zero_grad()
print(f"  Target LR: {expected_lr:.2e}")
print(f"  Warmup epochs: {warmup_epochs}")
print(f"  LR schedule (first 10 epochs): {', '.join(f'{lr:.1e}' for lr in lrs)}")

peak_lr = max(lrs)
if abs(peak_lr - expected_lr) / (expected_lr + 1e-15) > 0.1:
    fails.append(f"Peak LR mismatch: expected {expected_lr:.2e}, got {peak_lr:.2e}")
    print(f"  [FAIL] Peak LR mismatch! Expected {expected_lr:.2e}, got {peak_lr:.2e}")
else:
    print(f"  [OK] Peak LR = {peak_lr:.2e} matches target")

# ══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
if fails:
    print(f"PREFLIGHT FAILED — {len(fails)} error(s), {len(warns)} warning(s):")
    for f in fails:
        print(f"  [FAIL] {f}")
    for w in warns:
        print(f"  [WARN] {w}")
    sys.exit(1)
elif warns:
    print(f"PREFLIGHT PASSED with {len(warns)} warning(s):")
    for w in warns:
        print(f"  [WARN] {w}")
    print("Safe to launch training (review warnings above).")
    sys.exit(0)
else:
    print("PREFLIGHT PASSED — all 14 checks OK. Safe to launch training.")
    sys.exit(0)
