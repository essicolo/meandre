"""MC Dropout ensemble for predictive uncertainty.

Replaces classical multi-model ensembles (Hydrotel + Raven + ...) by
sampling N trajectories from the SAME trained meandre model, each with a
different frozen dropout mask.  Each member is a "model configuration"
analogous to a different parameter set.

Output: NetCDF with mean, p10/p50/p90 per (time, node), plus full ensemble
tensor for further analysis.

Usage
-----
Edit the constants below or pass via CLI.  Then::

    python mc_uncertainty.py
    # → notebooks/slso/results/mc_ensemble.nc

Memory note: with N_MEMBERS=50 and 1096 days × 2889 nodes × float32, the
full ensemble tensor uses ~640 MB — fine on disk and CPU but consider
streaming to disk if RAM-limited.
"""
from __future__ import annotations

import os
import sys
import time
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parent
os.chdir(REPO)

from meandre.data.basin_cache import BasinCache
from meandre.data.gridded_forcing import extract_forcing
from meandre.model import HydroModel
from meandre.training.uncertainty import frozen_dropout, frozen_param_noise
from meandre.utils.state import HydroState

# ── Configuration ──────────────────────────────────────────────────
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_MEMBERS    = 50           # number of MC trajectories
EVAL_START   = "2022-01-01" # held-out test period
EVAL_END     = "2024-12-31"
OUTPUT       = Path("notebooks/slso/results/mc_ensemble.nc")

with open("notebooks/slso/config/slso.toml", "rb") as f:
    cfg = tomllib.load(f)
paths, mcfg, temp, sc, tcfg = cfg["paths"], cfg["model"], cfg["temporal"], cfg.get("soil",{}), cfg["training"]

# ── Basin + forcing + withdrawals ──────────────────────────────────
print(f"[mc] device={DEVICE}  N_members={N_MEMBERS}  period={EVAL_START}→{EVAL_END}")
cache = BasinCache(paths["basin_db"])
hydro = cache.load(device=DEVICE)
graph, territorial = hydro["graph"], hydro["territorial"]
node_coords, n_nodes = hydro["node_coords"], hydro["n_nodes"]
node_ids = hydro["node_ids"]

forcing = extract_forcing(
    zarr_path=paths["weather_grid"], node_coords=node_coords, node_elev=None,
    date_start=temp["date_start"], date_end=temp["date_end"],
    cache_nc=paths["forcing_cache"], device=DEVICE,
)
ds = xr.open_dataset(paths["forcing_cache"])
all_dates = ds.time.sel(time=slice(temp["date_start"], temp["date_end"])).values
ds.close()
dates_pd = pd.DatetimeIndex(all_dates)
doy = torch.tensor(dates_pd.dayofyear.values, dtype=torch.long, device=DEVICE)
withdrawals = cache.load_withdrawals(date_start=temp["date_start"], date_end=temp["date_end"], device=DEVICE)

days = all_dates.astype("datetime64[D]")
i_eval_start = int(np.searchsorted(days, np.datetime64(EVAL_START, "D")))
i_eval_end   = int(np.searchsorted(days, np.datetime64(EVAL_END,   "D"), side="right"))
n_eval       = i_eval_end - i_eval_start
print(f"[mc] eval indices [{i_eval_start}:{i_eval_end}]  ({n_eval} days)")

# ── Load model + match training curriculum ─────────────────────────
soil_bounds = {k: sc[k] for k in (
    "z2_min","z2_max","z3_min","z3_max","rain_hours_min","rain_hours_max"
) if k in sc}
model = HydroModel(
    n_nodes=n_nodes, n_territorial=territorial.n_features,
    n_forcing=mcfg["n_forcing"], context_window=mcfg["context_window"],
    residual_history=mcfg["residual_history"], max_travel_time=mcfg["max_travel_days"],
    use_temporal=True, use_residual=True, use_travel_time_attn=True, use_temperature=True,
    dropout=mcfg.get("dropout", 0.0),
    concrete_dropout=mcfg.get("concrete_dropout", False),
    concrete_init_p=mcfg.get("concrete_init_p", 0.1),
    param_mode=mcfg.get("param_mode", "nerf"),
    soil_z1=sc.get("z1", 0.30), soil_bounds=soil_bounds,
    param_noise=mcfg.get("param_noise", False),
    param_noise_init_sigma=mcfg.get("param_noise_init_sigma", 0.05),
).to(DEVICE)
model.load(paths["checkpoint"])
n_epochs = tcfg.get("n_epochs", 0)
model.use_residual = n_epochs >= tcfg.get("enable_residual_epoch", 0)
if hasattr(model.routing, "use_tta"):
    model.routing.use_tta = n_epochs >= tcfg.get("enable_travel_epoch", 0)

# Probe uncertainty sources: ParamNoise (Position B primary) + ConcreteDropout (temporal)
from meandre.spatial.concrete_dropout import ConcreteDropout

has_param_noise = getattr(model.spatial_encoder, "param_noise", False)
if has_param_noise:
    sigmas = model.spatial_encoder.param_log_sigma.exp().detach().cpu().numpy()
    print(f"[mc] ParamNoise σ : min={sigmas.min():.4f}  median={np.median(sigmas):.4f}  max={sigmas.max():.4f}")
else:
    print("[mc] ParamNoise: disabled (model.spatial_encoder.param_noise=False)")

n_drop = 0
ps: list[float] = []
for name, m in model.named_modules():
    if isinstance(m, ConcreteDropout):
        n_drop += 1
        ps.append(float(m.p.item()))
if n_drop > 0:
    print(f"[mc] ConcreteDropout layers : {n_drop}; p range [{min(ps):.3f}, {max(ps):.3f}], mean {np.mean(ps):.3f}")

if not has_param_noise and (n_drop == 0 or max(ps) < 1e-3):
    print("⚠  No active uncertainty source — all members will be identical.")
    print("   Set [model] param_noise=true and re-train.")
    sys.exit(1)

# ── Spinup ONCE under deterministic eval (no dropout) ──────────────
# All ensemble members start from the same physics-driven state.
# Stochasticity is added only over the eval period.
print(f"[mc] spinup forcing[0:{i_eval_start}] (deterministic)...")
t0 = time.time()
model.eval()
with torch.no_grad():
    _, spinup_state = model.simulate(
        forcing=forcing[:i_eval_start],
        initial_state=HydroState.zeros(n_nodes, device=DEVICE),
        graph=graph, node_coords=node_coords, territorial=territorial,
        withdrawals=type(withdrawals)(
            net=withdrawals.net[:i_eval_start],
            net_gw=withdrawals.net_gw[:i_eval_start],
        ),
        day_of_year=doy[:i_eval_start],
    )
    h_ctx_spinup = (model._last_h_context.detach()
                    if model._last_h_context is not None else None)
print(f"[mc] spinup done in {time.time()-t0:.0f}s")

# ── MC ensemble over eval period ───────────────────────────────────
print(f"[mc] running {N_MEMBERS} MC trajectories...")
ensemble = torch.empty(N_MEMBERS, n_eval, n_nodes, dtype=torch.float32)
t0 = time.time()
wd_eval = type(withdrawals)(
    net=withdrawals.net[i_eval_start:i_eval_end],
    net_gw=withdrawals.net_gw[i_eval_start:i_eval_end],
)
for m_i in range(N_MEMBERS):
    with frozen_param_noise(model, seed=m_i), frozen_dropout(model, seed=m_i):
        with torch.no_grad():
            Q_m, _ = model.simulate(
                forcing=forcing[i_eval_start:i_eval_end],
                initial_state=spinup_state,
                graph=graph, node_coords=node_coords, territorial=territorial,
                withdrawals=wd_eval,
                day_of_year=doy[i_eval_start:i_eval_end],
                h_context=h_ctx_spinup,
            )
    ensemble[m_i] = Q_m.detach().cpu().float()
    if (m_i + 1) % 5 == 0:
        elapsed = time.time() - t0
        eta = elapsed / (m_i + 1) * (N_MEMBERS - m_i - 1)
        print(f"  [{m_i+1}/{N_MEMBERS}]  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")
print(f"[mc] ensemble done in {time.time()-t0:.0f}s")

# ── Quantile statistics ────────────────────────────────────────────
print("[mc] computing quantiles...")
ens_np = ensemble.numpy()           # (N, T, n_nodes)
mean_q = ens_np.mean(axis=0)        # (T, n_nodes)
std_q  = ens_np.std(axis=0)
p10    = np.quantile(ens_np, 0.10, axis=0)
p50    = np.quantile(ens_np, 0.50, axis=0)
p90    = np.quantile(ens_np, 0.90, axis=0)

# ── Export NetCDF ──────────────────────────────────────────────────
print(f"[mc] exporting {OUTPUT}...")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
out_dates = pd.date_range(EVAL_START, EVAL_END, freq="D")[:n_eval]

ds_out = xr.Dataset(
    data_vars={
        "Q_mean": (["time", "node"], mean_q, {"units": "m3/s"}),
        "Q_std":  (["time", "node"], std_q,  {"units": "m3/s"}),
        "Q_p10":  (["time", "node"], p10,    {"units": "m3/s"}),
        "Q_p50":  (["time", "node"], p50,    {"units": "m3/s"}),
        "Q_p90":  (["time", "node"], p90,    {"units": "m3/s"}),
        "Q_ensemble": (["member", "time", "node"], ens_np, {"units": "m3/s"}),
    },
    coords={
        "time":   out_dates,
        "node":   np.array(node_ids),
        "member": np.arange(N_MEMBERS),
    },
    attrs={
        "title":      "meandre MC Dropout uncertainty ensemble",
        "checkpoint": str(paths["checkpoint"]),
        "n_members":  N_MEMBERS,
        "method":     "Frozen Concrete Dropout — 1 mask per trajectory",
    },
)
ds_out.to_netcdf(OUTPUT, engine="netcdf4")

# ── Quick sanity at the outlet ─────────────────────────────────────
outlet_idx = graph.topo_order[-1].item()
spread_outlet = (p90[:, outlet_idx] - p10[:, outlet_idx]).mean()
mean_outlet   = mean_q[:, outlet_idx].mean()
cv_outlet     = std_q[:, outlet_idx].mean() / max(mean_outlet, 1e-6)
print(f"\n[mc] outlet (node {node_ids[outlet_idx]}):")
print(f"     mean Q   = {mean_outlet:.1f} m³/s")
print(f"     mean p90-p10 spread = {spread_outlet:.1f} m³/s")
print(f"     mean CV  = {cv_outlet*100:.1f} %")
print(f"\n[mc] done.  NetCDF: {OUTPUT}")
