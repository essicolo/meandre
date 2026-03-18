"""CLI training + diagnostics for SLSO basin.

Usage: uv run python scripts/train_and_diagnose.py
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from pathlib import Path
import torch
import numpy as np
import xarray as xr

# ── Paths ──────────────────────────────────────────────────────────────────
NB_DIR = Path("notebooks")
BASIN_DB = NB_DIR / "data/slso.duckdb"
FORCING_CACHE = NB_DIR / "data/slso/forcing.nc"
CHECKPOINT = NB_DIR / "checkpoints/slso/best.pt"
RUNS_DB = NB_DIR / "runs.duckdb"
BASIN_PREFIX = "SLSO"

# ── Temporal ───────────────────────────────────────────────────────────────
DATE_START = "2000-01-01"
SPINUP_END = "2000-06-30"
TRAIN_START = "2000-07-01"
TRAIN_END = "2001-12-31"
VAL_START = "2000-07-01"
VAL_END = "2001-12-31"
DATE_END = "2001-12-31"

# ── Model ──────────────────────────────────────────────────────────────────
N_FORCING = 6
CONTEXT_WINDOW = 90
RESIDUAL_HISTORY = 14
MAX_TRAVEL_DAYS = 20

# ── Training ───────────────────────────────────────────────────────────────
N_EPOCHS = 500
LR = 3e-4
WEIGHT_DECAY = 5e-4
GRAD_CLIP = 0.5
ENABLE_TEMPORAL_EPOCH = 0
ENABLE_RESIDUAL_EPOCH = 9999
ENABLE_TRAVEL_EPOCH = 9999

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Basin data ─────────────────────────────────────────────────────────────
from meandre.data.basin_cache import BasinCache

cache = BasinCache(BASIN_DB)
hydro = cache.load(device=device)

graph = hydro["graph"]
territorial = hydro["territorial"]
node_coords = hydro["node_coords"]
initial_state = hydro["initial_state"]
node_ids = hydro["node_ids"]
n_nodes = hydro["n_nodes"]
print(f"Nodes: {n_nodes}, Edges: {graph.n_edges}")

# ── Forcing ────────────────────────────────────────────────────────────────
from meandre.data.gridded_forcing import extract_forcing

forcing = extract_forcing(
    zarr_path=FORCING_CACHE,
    node_coords=node_coords,
    node_elev=territorial.mean_elevation_m,
    date_start=DATE_START,
    date_end=DATE_END,
    cache_nc=FORCING_CACHE,
    device=device,
)

ds_time = xr.open_dataset(FORCING_CACHE)
all_dates = ds_time.time.sel(time=slice(DATE_START, DATE_END)).values
ds_time.close()
n_time = len(all_dates)
print(f"Forcing: {tuple(forcing.shape)}, {all_dates[0]} to {all_dates[-1]}")

# ── Stations ───────────────────────────────────────────────────────────────
from meandre.routing.withdrawals import WithdrawalData
import pandas as pd

obs = cache.load_observations(
    date_start=DATE_START,
    date_end=DATE_END,
    min_valid_days=365,
)

station_node_map = obs["station_node_map"]
station_indices = sorted(set(station_node_map.values()))
n_stations = len(station_indices)

station_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
for ni in station_indices:
    station_mask[ni] = True

discharge_np = obs["discharge"]  # (T, N_all)
q_obs_tensor = torch.from_numpy(discharge_np[:, station_indices]).to(device)
sids = list(station_node_map.keys())

# Get drainage areas from stations table in DuckDB
import duckdb
_con = duckdb.connect(str(BASIN_DB), read_only=True)
areas = []
for s in sids:
    row = _con.execute(
        "SELECT drainage_area_km2 FROM gauging_stations WHERE station_id = ?", [s]
    ).fetchone()
    areas.append(float(row[0]) if row else 0.0)
_con.close()

print(f"Stations: {n_stations} — {sids}")
print(f"Areas: {areas}")

# ── Slices ─────────────────────────────────────────────────────────────────
dates_pd = pd.DatetimeIndex(all_dates)

def date_idx(d):
    return int(np.searchsorted(dates_pd, pd.Timestamp(d)))

spinup_sl = slice(0, date_idx(SPINUP_END) + 1)
train_sl = slice(date_idx(TRAIN_START), date_idx(TRAIN_END) + 1)
val_sl = slice(date_idx(VAL_START), date_idx(VAL_END) + 1)

spinup_steps = spinup_sl.stop
doy = torch.tensor(
    [int(pd.Timestamp(d).day_of_year) for d in all_dates],
    dtype=torch.long, device=device,
)

print(f"Spinup: {DATE_START}–{SPINUP_END} ({spinup_steps} steps)")
print(f"Train:  {TRAIN_START}–{TRAIN_END} ({train_sl.start}:{train_sl.stop})")
print(f"Val:    {VAL_START}–{VAL_END} ({val_sl.start}:{val_sl.stop})")

# ── Model ──────────────────────────────────────────────────────────────────
from meandre.model import YHydro
from meandre.utils.state import HydroState

model = YHydro(
    n_nodes=n_nodes,
    n_forcing=N_FORCING,
    context_window=CONTEXT_WINDOW,
    residual_history=RESIDUAL_HISTORY,
    max_travel_time=MAX_TRAVEL_DAYS,
    use_temporal=True,
    use_residual=True,
    use_travel_time_attn=True,
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params:,}")

# ── Training setup ─────────────────────────────────────────────────────────
from meandre.training.loss import HydroLoss
from meandre.training.trainer import Trainer, TrainingConfig, TrainingData
from meandre.training.run_logger import RunLogger

withdrawals = WithdrawalData.zeros(n_time, n_nodes, device=device)

def make_data(period_sl):
    return TrainingData(
        forcing=forcing,
        q_obs=q_obs_tensor[period_sl.start:],
        station_mask=station_mask,
        station_idx=torch.tensor(station_indices, device=device),
        graph=graph,
        node_coords=node_coords,
        territorial=territorial,
        withdrawals=withdrawals,
        day_of_year=doy,
        train_slice=period_sl,
        val_slice=period_sl,
    )

train_data = make_data(train_sl)
val_data = TrainingData(
    forcing=forcing,
    q_obs=q_obs_tensor[val_sl.start:],
    station_mask=station_mask,
    station_idx=torch.tensor(station_indices, device=device),
    graph=graph,
    node_coords=node_coords,
    territorial=territorial,
    withdrawals=withdrawals,
    day_of_year=doy,
    train_slice=val_sl,
    val_slice=val_sl,
)

# Reorder areas to match station_indices (sorted by node index), not sids order
idx_to_sid = {v: k for k, v in station_node_map.items()}
sids_ordered = [idx_to_sid[ni] for ni in station_indices]
areas_ordered = [areas[sids.index(s)] for s in sids_ordered]
# Mild area-based weighting: clip tiny stations up, cap large ones.
# Range: sqrt(50)=7.1 to sqrt(500)=22.4, preventing any one station from dominating.
station_areas = torch.sqrt(torch.clamp(
    torch.tensor(areas_ordered, dtype=torch.float32, device=device),
    min=50.0, max=500.0,
))
print(f"Station weights order: {list(zip(sids_ordered, areas_ordered))}")
loss_fn = HydroLoss(
    w_nse=1.0, w_pbias=0.1, w_kge=0.5, w_nrmse=0.0, w_log_nse=0.0,
    w_physics=0.01, w_residual=0.001,
    per_station=True, station_weights=station_areas,
)

train_cfg = TrainingConfig(
    lr=LR,
    weight_decay=WEIGHT_DECAY,
    grad_clip=GRAD_CLIP,
    n_epochs=N_EPOCHS,
    spinup_steps=spinup_steps,
    warm_spinup_steps=30,
    tbptt_steps=90,
    val_every=3,
    enable_temporal_context_epoch=ENABLE_TEMPORAL_EPOCH,
    enable_residual_corrector_epoch=ENABLE_RESIDUAL_EPOCH,
    enable_travel_time_attn_epoch=ENABLE_TRAVEL_EPOCH,
    patience=80,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)

# Clean old runs DB
for f in ["runs.duckdb", "runs.duckdb.wal"]:
    p = NB_DIR / f
    if p.exists():
        os.remove(p)

run_logger = RunLogger(RUNS_DB)

trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    train_data=train_data,
    val_data=val_data,
    config=train_cfg,
    run_name=f"slso_{TRAIN_START[:4]}_{TRAIN_END[:4]}",
    run_logger=run_logger,
    checkpoint_path=str(CHECKPOINT),
    optimizer=optimizer,
)


print("\n=== Starting training ===")
trainer.fit()

# ── Post-training diagnostics ──────────────────────────────────────────────
print("\n=== Post-training diagnostics ===")
model.load(str(CHECKPOINT))
model.eval()
print(f"temporal={model.use_temporal}, residual={model.use_residual}, tta={model.routing.use_tta}")

# Run full simulation from zeros (same as training spinup), NOT from initial_state
with torch.no_grad():
    Q_sim, _, diag = model.simulate(
        forcing=forcing,
        initial_state=HydroState.zeros(n_nodes, device=device),
        graph=graph,
        node_coords=node_coords,
        territorial=territorial,
        withdrawals=withdrawals,
        day_of_year=doy,
        return_diagnostics=True,
    )

with torch.no_grad():
    sp = model.spatial_encoder(node_coords, territorial.to_tensor())

# Spatial params summary
print("\n--- Spatial parameters (mean ± std over nodes) ---")
for name in ["K_sat_1", "K_sat_2", "K_sat_3", "porosity_1", "theta_fc_1", "theta_wp_1",
             "f_root_1", "f_root_2", "f_root_3", "C_f", "T_melt", "T_snow",
             "interception_capacity", "manning_n", "frost_alpha", "f_wetland",
             "slope_factor", "krec"]:
    v = getattr(sp, name)
    print(f"  {name:25s}: {v.mean():.4f} ± {v.std():.4f}  [{v.min():.4f}, {v.max():.4f}]")

# Per-station diagnostics — separate periods
print("\n--- Per-station diagnostics (train period) ---")
q_sim_stations = Q_sim[:, station_mask].cpu()

from meandre.utils.metrics import kge as compute_kge, rmse as compute_rmse, nrmse as compute_nrmse, mae as compute_mae, log_nse as compute_log_nse

def print_station_metrics(period_name, period_sl):
    """Compute and print error-based metrics per station over a slice."""
    all_metrics = {"nse": [], "kge": [], "rmse": [], "nrmse": [], "mae": [], "pbias": [], "log_nse": []}
    for i, sid in enumerate(sids_ordered):
        q_o = q_obs_tensor[period_sl, i].cpu()
        q_s = q_sim_stations[period_sl, i]
        valid = ~torch.isnan(q_o)
        q_o_v = q_o[valid]
        q_s_v = q_s[valid]
        if q_o_v.numel() == 0:
            print(f"  {sid}: no valid obs in {period_name}")
            continue

        nse_num = ((q_o_v - q_s_v) ** 2).sum()
        nse_den = ((q_o_v - q_o_v.mean()) ** 2).sum()
        nse_val = 1.0 - (nse_num / (nse_den + 1e-8))
        kge_val = compute_kge(q_o_v, q_s_v)
        rmse_val = compute_rmse(q_o_v, q_s_v)
        nrmse_val = compute_nrmse(q_o_v, q_s_v)
        mae_val = compute_mae(q_o_v, q_s_v)
        log_nse_val = compute_log_nse(q_o_v, q_s_v)
        pbias_val = 100.0 * (q_s_v - q_o_v).sum() / (q_o_v.sum() + 1e-8)
        peak_ratio = q_s_v.max() / (q_o_v.max() + 1e-8)

        for k, v in [("nse", nse_val), ("kge", kge_val), ("rmse", rmse_val),
                     ("nrmse", nrmse_val), ("mae", mae_val), ("pbias", pbias_val),
                     ("log_nse", log_nse_val)]:
            all_metrics[k].append(float(v))

        print(f"  {sid}: NSE={nse_val:.3f}, KGE={kge_val:.3f}, logNSE={log_nse_val:.3f}, "
              f"RMSE={rmse_val:.2f}, PBIAS={pbias_val:.1f}%, peak={peak_ratio:.3f}, "
              f"mean_obs={q_o_v.mean():.2f}, mean_sim={q_s_v.mean():.2f}, "
              f"area={areas_ordered[i]:.0f} km²")
    if all_metrics["rmse"]:
        def median(vals):
            s = sorted(vals)
            return s[len(s)//2]
        print(f"  >> Median NSE={median(all_metrics['nse']):.3f}, "
              f"KGE={median(all_metrics['kge']):.3f}, "
              f"logNSE={median(all_metrics['log_nse']):.3f}, "
              f"RMSE={median(all_metrics['rmse']):.2f}, "
              f"PBIAS={median(all_metrics['pbias']):.1f}%")

print_station_metrics("train", train_sl)

print("\n--- Per-station diagnostics (val period) ---")
print_station_metrics("val", val_sl)

print("\n--- Per-station diagnostics (full period incl. spinup) ---")
print_station_metrics("full", slice(0, n_time))

# Aggregate metrics across all stations for each period
for label, sl in [("train", train_sl), ("val", val_sl), ("full", slice(0, n_time))]:
    q_o_all = q_obs_tensor[sl].cpu()
    q_s_all = q_sim_stations[sl]
    valid = ~torch.isnan(q_o_all)
    q_o_v = q_o_all[valid]
    q_s_v = q_s_all[valid]
    agg_nse = 1.0 - ((q_o_v - q_s_v) ** 2).sum() / ((q_o_v - q_o_v.mean()) ** 2).sum()
    agg_kge = compute_kge(q_o_v, q_s_v)
    agg_rmse = compute_rmse(q_o_v, q_s_v)
    agg_nrmse = compute_nrmse(q_o_v, q_s_v)
    agg_mae = compute_mae(q_o_v, q_s_v)
    print(f"  Aggregate ({label}): RMSE={agg_rmse:.2f}, nRMSE={agg_nrmse:.3f}, "
          f"MAE={agg_mae:.2f}, NSE={agg_nse:.4f}, KGE={agg_kge:.4f}")

# Water balance check
print("\n--- Basin-wide water balance ---")
P_mean = forcing[:, :, 0].mean(dim=1).cpu()
lat_mm = diag.lateral_mm.cpu() if hasattr(diag, 'lateral_mm') else None

if lat_mm is not None:
    lat_mean = lat_mm.mean(dim=1)
    print(f"  P mean: {P_mean.mean():.2f} mm/day")
    print(f"  Lateral (runoff) mean: {lat_mean.mean():.4f} mm/day")
    print(f"  Runoff ratio: {lat_mean.sum() / (P_mean.sum() + 1e-8):.3f}")
    # Also show runoff ratio for train period only
    lat_train = lat_mm[train_sl].mean(dim=1)
    P_train = forcing[train_sl, :, 0].mean(dim=1).cpu()
    print(f"  Runoff ratio (train): {lat_train.sum() / (P_train.sum() + 1e-8):.3f}")

# Routing check: K_musk
with torch.no_grad():
    K_musk = model.K_musk
    print(f"\n--- Routing ---")
    print(f"  K_musk: mean={K_musk.mean():.0f}s ({K_musk.mean()/3600:.1f}h), "
          f"min={K_musk.min():.0f}s, max={K_musk.max():.0f}s")

print("\n=== Done ===")
