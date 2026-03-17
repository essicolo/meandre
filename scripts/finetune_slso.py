"""Fine-tune SLSO from MSE-pretrained checkpoint with NSE+KGE loss.

MSE pretraining gets the model to a reasonable fit (val_kge~0.50-0.55).
NSE+KGE fine-tuning at lower LR should push KGE higher by directly
optimizing the skill scores.
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from pathlib import Path
import numpy as np
import torch
import pandas as pd
import xarray as xr

# ── Paths ─────────────────────────────────────────────────────────────────
NB_DIR = Path("notebooks")
BASIN_DB = NB_DIR / "data/slso.duckdb"
ZARR_PATH = NB_DIR / "data/quebec.zarr"
FORCING_CACHE = NB_DIR / "data/slso/forcing.nc"
CHECKPOINT_IN = NB_DIR / "checkpoints/slso/best.pt"
CHECKPOINT_OUT = NB_DIR / "checkpoints/slso/best_finetuned.pt"
RUNS_DB = NB_DIR / "runs.duckdb"

# ── Temporal window ───────────────────────────────────────────────────────
DATE_START  = "2000-01-01"
SPINUP_END  = "2000-12-31"
TRAIN_START = "2001-01-01"
TRAIN_END   = "2003-12-31"
VAL_START   = "2010-01-01"
VAL_END     = "2019-12-31"
DATE_END    = "2019-12-31"

# ── Training ──────────────────────────────────────────────────────────────
N_EPOCHS = 100
LR = 5e-5           # Lower LR for fine-tuning (was 3e-4 for pretrain)
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
ENABLE_TEMPORAL_EPOCH = 9999
ENABLE_RESIDUAL_EPOCH = 9999
ENABLE_TRAVEL_EPOCH = 9999

# ── Device ────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Basin data ────────────────────────────────────────────────────────────
from meandre.data.basin_cache import BasinCache

cache = BasinCache(BASIN_DB)
hydro = cache.load(device=device)

graph         = hydro["graph"]
territorial   = hydro["territorial"]
node_coords   = hydro["node_coords"]
node_ids      = hydro["node_ids"]
n_nodes       = hydro["n_nodes"]

print(f"Nodes: {n_nodes}, Edges: {graph.n_edges}")

# ── Forcing ───────────────────────────────────────────────────────────────
from meandre.data.gridded_forcing import extract_forcing

forcing = extract_forcing(
    zarr_path   = ZARR_PATH,
    node_coords = node_coords,
    node_elev   = territorial.mean_elevation_m,
    date_start  = DATE_START,
    date_end    = DATE_END,
    cache_nc    = FORCING_CACHE,
    device      = device,
)

ds_time = xr.open_dataset(FORCING_CACHE)
all_dates = ds_time.time.sel(time=slice(DATE_START, DATE_END)).values
ds_time.close()
n_time = len(all_dates)
print(f"Forcing: {tuple(forcing.shape)}, {all_dates[0]} to {all_dates[-1]}")

# ── Observations ──────────────────────────────────────────────────────────
from meandre.routing.withdrawals import WithdrawalData

obs = cache.load_observations(
    date_start=DATE_START, date_end=DATE_END, min_valid_days=365,
)

station_node_map = obs["station_node_map"]
station_indices  = sorted(set(station_node_map.values()))
n_stations       = len(station_indices)

station_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
for ni in station_indices:
    station_mask[ni] = True

discharge_np = obs["discharge"]
q_obs_tensor = torch.from_numpy(discharge_np[:, station_indices]).to(device)
print(f"Stations: {n_stations}, Q_obs: {q_obs_tensor.shape}")

# Station areas from DuckDB
import duckdb
sids = list(station_node_map.keys())
_con = duckdb.connect(str(BASIN_DB), read_only=True)
areas = []
for s in sids:
    row = _con.execute(
        "SELECT drainage_area_km2 FROM stations WHERE station_id = ?", [s]
    ).fetchone()
    areas.append(float(row[0]) if row else 0.0)
_con.close()

# ── Time slicing ──────────────────────────────────────────────────────────
def dates_to_slice(dates, start, end):
    days = dates.astype("datetime64[D]")
    s = int(np.searchsorted(days, np.datetime64(start, "D")))
    e = int(np.searchsorted(days, np.datetime64(end, "D"), side="right"))
    return slice(s, e)

spinup_sl = dates_to_slice(all_dates, DATE_START, SPINUP_END)
train_sl  = dates_to_slice(all_dates, TRAIN_START, TRAIN_END)
val_sl    = dates_to_slice(all_dates, VAL_START, VAL_END)
spinup_steps = spinup_sl.stop

doy = torch.tensor(
    [pd.Timestamp(d).day_of_year for d in all_dates],
    dtype=torch.long, device=device,
)

print(f"Spinup: {DATE_START}–{SPINUP_END} ({spinup_sl.stop} steps)")
print(f"Train:  {TRAIN_START}–{TRAIN_END} (steps {train_sl.start}:{train_sl.stop})")
print(f"Val:    {VAL_START}–{VAL_END} (steps {val_sl.start}:{val_sl.stop})")

# ── Model (from checkpoint) ──────────────────────────────────────────────
from meandre.model import YHydro

model = YHydro.from_checkpoint(CHECKPOINT_IN).to(device)
print(f"Loaded checkpoint: {CHECKPOINT_IN}")
n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params:,}")

# ── Withdrawals ───────────────────────────────────────────────────────────
withdrawals = WithdrawalData.zeros(n_time, n_nodes, device=device)

# ── Training ──────────────────────────────────────────────────────────────
from meandre.training.loss import HydroLoss
from meandre.training.trainer import Trainer, TrainingConfig, TrainingData
from meandre.training.run_logger import RunLogger

train_data = TrainingData(
    forcing=forcing, q_obs=q_obs_tensor[train_sl.start:],
    station_mask=station_mask,
    station_idx=torch.tensor(station_indices, device=device),
    graph=graph, node_coords=node_coords,
    territorial=territorial, withdrawals=withdrawals,
    day_of_year=doy, train_slice=train_sl, val_slice=train_sl,
)

val_data = TrainingData(
    forcing=forcing, q_obs=q_obs_tensor[val_sl.start:],
    station_mask=station_mask,
    station_idx=torch.tensor(station_indices, device=device),
    graph=graph, node_coords=node_coords,
    territorial=territorial, withdrawals=withdrawals,
    day_of_year=doy, train_slice=val_sl, val_slice=val_sl,
)

idx_to_sid = {v: k for k, v in station_node_map.items()}
sids_ordered = [idx_to_sid[ni] for ni in station_indices]
areas_ordered = [areas[sids.index(s)] for s in sids_ordered]
station_areas = torch.sqrt(torch.tensor(areas_ordered, dtype=torch.float32, device=device))

# NSE+KGE loss for fine-tuning (directly optimizes skill scores)
loss_fn = HydroLoss(
    w_nse=1.0, w_pbias=0.1, w_kge=1.0,
    w_mse=0.0,
    w_physics=0.01, w_residual=0.001,
    per_station=True, station_weights=station_areas,
)

train_cfg = TrainingConfig(
    lr=LR, weight_decay=WEIGHT_DECAY, grad_clip=GRAD_CLIP,
    n_epochs=N_EPOCHS, spinup_steps=spinup_steps,
    warm_spinup_steps=30, tbptt_steps=90, val_every=5,
    chunk_steps=0,
    enable_temporal_context_epoch=ENABLE_TEMPORAL_EPOCH,
    enable_residual_corrector_epoch=ENABLE_RESIDUAL_EPOCH,
    enable_travel_time_attn_epoch=ENABLE_TRAVEL_EPOCH,
)

CHECKPOINT_OUT.parent.mkdir(parents=True, exist_ok=True)
run_logger = RunLogger(RUNS_DB)

trainer = Trainer(
    model=model, loss_fn=loss_fn,
    train_data=train_data, val_data=val_data,
    config=train_cfg,
    run_name="slso_finetune_nse_kge",
    run_logger=run_logger,
    checkpoint_path=str(CHECKPOINT_OUT),
)
trainer.fit()

print("\n=== Fine-tuning complete ===")
