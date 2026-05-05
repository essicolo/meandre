"""Verify the _val_epoch fix produces honest metrics.

Loads current best.pt, runs the patched _val_epoch, and compares against
the diagnostic notebook's KGE_managed (which we know is the correct value).

Expected: val_epoch returns metrics close to diagnostic's KGE_managed median ≈ -0.30.
If it returns kge ≈ 0.78 (the phantom), the fix is incomplete.
"""
from __future__ import annotations

import os
import sys
import time
import tomllib
from pathlib import Path

import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parent
os.chdir(REPO)

from meandre.data.basin_cache import BasinCache
from meandre.data.gridded_forcing import extract_forcing
from meandre.model import HydroModel
from meandre.training.loss import HydroLoss
from meandre.training.trainer import Trainer, TrainingConfig, TrainingData
from meandre.utils.state import HydroState

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("notebooks/slso/config/slso.toml", "rb") as f:
    cfg = tomllib.load(f)
paths = cfg["paths"]
mcfg  = cfg["model"]
temp  = cfg["temporal"]
sc    = cfg.get("soil", {})
tcfg  = cfg["training"]

DATE_START, DATE_END = temp["date_start"], temp["date_end"]

print(f"[verify] device={DEVICE}")
print(f"[verify] checkpoint={paths['checkpoint']}")
print(f"[verify] period {DATE_START} → {DATE_END}")

# ── Basin + forcing ─────────────────────────────────────────────────
cache = BasinCache(paths["basin_db"])
hydro = cache.load(device=DEVICE)
graph = hydro["graph"]; territorial = hydro["territorial"]
node_coords = hydro["node_coords"]; n_nodes = hydro["n_nodes"]

t0 = time.time()
forcing = extract_forcing(
    zarr_path=paths["weather_grid"], node_coords=node_coords, node_elev=None,
    date_start=DATE_START, date_end=DATE_END,
    cache_nc=paths["forcing_cache"], device=DEVICE,
)
import xarray as xr, pandas as pd
ds = xr.open_dataset(paths["forcing_cache"])
all_dates = ds.time.sel(time=slice(DATE_START, DATE_END)).values
ds.close()
dates_pd = pd.DatetimeIndex(all_dates)
doy = torch.tensor(dates_pd.dayofyear.values, dtype=torch.long, device=DEVICE)
print(f"[verify] forcing loaded {forcing.shape} in {time.time()-t0:.0f}s")

# ── Indices ─────────────────────────────────────────────────────────
days = all_dates.astype("datetime64[D]")
def day_idx(s: str, side: str = "left") -> int:
    return int(np.searchsorted(days, np.datetime64(s, "D"), side=side))

train_sl = slice(day_idx(temp["train_start"]), day_idx(temp["train_end"], "right"))
val_sl   = slice(day_idx(temp["val_start"]),   day_idx(temp["val_end"],   "right"))
print(f"[verify] train_sl={train_sl}, val_sl={val_sl}")

# ── Withdrawals + observations ──────────────────────────────────────
withdrawals = cache.load_withdrawals(date_start=DATE_START, date_end=DATE_END, device=DEVICE)
obs = cache.load_observations(date_start=DATE_START, date_end=DATE_END, min_valid_days=365)
station_node_map = obs["station_node_map"]
station_indices = sorted(set(station_node_map.values()))
station_mask = torch.zeros(n_nodes, dtype=torch.bool, device=DEVICE)
for ni in station_indices:
    station_mask[ni] = True
discharge_np = obs["discharge"]
q_obs_tensor = torch.from_numpy(discharge_np[:, station_indices]).to(DEVICE)

# ── Build val_data exactly like slso.py does ────────────────────────
val_data = TrainingData(
    forcing=forcing,
    q_obs=q_obs_tensor[val_sl.start:],
    station_mask=station_mask,
    station_idx=torch.tensor(station_indices, device=DEVICE),
    graph=graph,
    node_coords=node_coords,
    territorial=territorial,
    withdrawals=withdrawals,
    day_of_year=doy,
    train_slice=val_sl,   # NOTE: matches slso.py construction
    val_slice=val_sl,
)

# ── Build model and load best.pt ────────────────────────────────────
soil_bounds = {k: sc[k] for k in (
    "z2_min","z2_max","z3_min","z3_max","rain_hours_min","rain_hours_max"
) if k in sc}
model = HydroModel(
    n_nodes=n_nodes, n_territorial=territorial.n_features,
    n_forcing=mcfg["n_forcing"], context_window=mcfg["context_window"],
    residual_history=mcfg["residual_history"], max_travel_time=mcfg["max_travel_days"],
    use_temporal=True, use_residual=True,
    use_travel_time_attn=True, use_temperature=True,
    dropout=mcfg.get("dropout", 0.0),
    param_mode=mcfg.get("param_mode", "nerf"),
    soil_z1=sc.get("z1", 0.30),
    soil_bounds=soil_bounds,
).to(DEVICE)
model.load(paths["checkpoint"])
model.eval()
# Match training curriculum (residual + tta disabled at epoch 9999)
n_epochs = tcfg.get("n_epochs", 0)
model.use_residual = n_epochs >= tcfg.get("enable_residual_epoch", 0)
if hasattr(model.routing, "use_tta"):
    model.routing.use_tta = n_epochs >= tcfg.get("enable_travel_epoch", 0)
print(f"[verify] model loaded; use_residual={model.use_residual}, "
      f"use_tta={getattr(model.routing,'use_tta','n/a')}")

# ── Build a minimal Trainer just to call _val_epoch ─────────────────
# Hydro loss only needed because Trainer expects it
station_areas = torch.ones(len(station_indices), device=DEVICE)
station_var = torch.ones(len(station_indices), device=DEVICE)
loss_fn = HydroLoss(
    w_nse=0.0, w_kge=0.5, w_pbias=0.1, w_mse=0.1, w_nrmse=0.0,
    w_log_nse=0.0, w_log_mse=0.5, w_physics=0.01, w_residual=0.01,
    per_station=True, station_weights=station_areas, station_var=station_var,
)
spinup_end = min(tcfg.get("spinup_end_idx", 366), val_sl.start)
train_cfg = TrainingConfig(
    lr=1e-4, weight_decay=2e-4, grad_clip=1.0, n_epochs=1,
    spinup_steps=366, warm_spinup_steps=0,
    tbptt_steps=tcfg.get("tbptt_steps", 365), chunk_steps=tcfg.get("chunk_steps", 180),
)
# Trainer expects optimizer/scheduler etc — we'll bypass init by direct attribute set
trainer = object.__new__(Trainer)
trainer.model = model
trainer.config = train_cfg
trainer.train_data = val_data
trainer.val_data = val_data
trainer.loss_fn = loss_fn
trainer._cached_spinup_state = None
trainer._amp_dtype = torch.bfloat16
trainer._use_amp = (DEVICE.type == "cuda")
trainer._last_h_context = None

print("\n[verify] running patched _val_epoch (~1-3 min CPU)...")
t0 = time.time()
metrics = trainer._val_epoch()
print(f"[verify] done in {time.time()-t0:.0f}s\n")

print("═" * 72)
print("  PATCHED _val_epoch metrics on best.pt")
print("═" * 72)
for k in ["kge", "nse", "kge_station", "kge_median", "beta", "gamma", "r", "kge_log"]:
    if k in metrics:
        print(f"  {k:14s} : {metrics[k]:.4f}")

print()
print("Reference (diagnostic notebook KGE_managed): médian ≈ -0.30, β ≈ 1.99")
print("If patched val_epoch reports kge_median ≈ -0.30, the fix is correct.")
print("If it still reports ≈ 0.78, there's another bug.")
