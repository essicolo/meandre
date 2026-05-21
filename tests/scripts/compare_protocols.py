"""Compare two simulation protocols on the SAME checkpoint:
A) ONE long simulate(forcing[0:val_end]) from zeros
B) spinup_call(forcing[0:val_start]) + diag_call(forcing[val_start:val_end], h_context=...)

Both should give IDENTICAL Q over val period.  If they don't, there's a state
propagation bug between simulate calls.

Runs on GPU for speed (~3 min).
"""
from __future__ import annotations
import os, sys, time, tomllib
from pathlib import Path
import numpy as np, torch, xarray as xr, pandas as pd
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parent
os.chdir(REPO)

from meandre.data.basin_cache import BasinCache
from meandre.data.gridded_forcing import extract_forcing
from meandre.model import HydroModel
from meandre.routing.withdrawals import WithdrawalData
from meandre.utils.state import HydroState

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("notebooks/slso/config/slso.toml", "rb") as f:
    cfg = tomllib.load(f)
paths, mcfg, temp, sc, tcfg = cfg["paths"], cfg["model"], cfg["temporal"], cfg.get("soil",{}), cfg["training"]

cache = BasinCache(paths["basin_db"])
hydro = cache.load(device=DEVICE)
graph, territorial = hydro["graph"], hydro["territorial"]
node_coords, n_nodes = hydro["node_coords"], hydro["n_nodes"]

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
val_start = int(np.searchsorted(days, np.datetime64(temp["val_start"], "D")))
val_end   = int(np.searchsorted(days, np.datetime64(temp["val_end"],   "D"), side="right"))
print(f"forcing.shape={tuple(forcing.shape)}, val_slice=[{val_start}:{val_end}]")

soil_bounds = {k: sc[k] for k in ("z2_min","z2_max","z3_min","z3_max","rain_hours_min","rain_hours_max") if k in sc}
model = HydroModel(
    n_nodes=n_nodes, n_territorial=territorial.n_features,
    n_forcing=mcfg["n_forcing"], context_window=mcfg["context_window"],
    residual_history=mcfg["residual_history"], max_travel_time=mcfg["max_travel_days"],
    use_temporal=True, use_residual=True, use_travel_time_attn=True, use_temperature=True,
    dropout=mcfg.get("dropout", 0.0), param_mode=mcfg.get("param_mode", "nerf"),
    soil_z1=sc.get("z1", 0.30), soil_bounds=soil_bounds,
).to(DEVICE)
model.load(paths["checkpoint"])
model.eval()
n_epochs = tcfg.get("n_epochs", 0)
model.use_residual = n_epochs >= tcfg.get("enable_residual_epoch", 0)
if hasattr(model.routing, "use_tta"):
    model.routing.use_tta = n_epochs >= tcfg.get("enable_travel_epoch", 0)
print(f"model: use_residual={model.use_residual}, use_tta={getattr(model.routing,'use_tta','n/a')}")

# ── Protocol A: ONE long simulate(forcing[0:val_end]) ─────────────────
print("\n[A] ONE long simulate (forcing[0:val_end]) ...")
t0 = time.time()
with torch.no_grad():
    Q_A_full, _ = model.simulate(
        forcing=forcing[:val_end],
        initial_state=HydroState.zeros(n_nodes, device=DEVICE),
        graph=graph, node_coords=node_coords, territorial=territorial,
        withdrawals=WithdrawalData(net=withdrawals.net[:val_end], net_gw=withdrawals.net_gw[:val_end]),
        day_of_year=doy[:val_end],
    )
Q_A = Q_A_full[val_start:val_end].clone()
print(f"   done in {time.time()-t0:.0f}s; Q_A.shape={tuple(Q_A.shape)}")
del Q_A_full

# ── Protocol B: spinup + diag, with h_context propagation ─────────────
print("\n[B] spinup_call + diag_call (h_context propagated) ...")
t0 = time.time()
with torch.no_grad():
    _, spinup_state = model.simulate(
        forcing=forcing[:val_start],
        initial_state=HydroState.zeros(n_nodes, device=DEVICE),
        graph=graph, node_coords=node_coords, territorial=territorial,
        withdrawals=WithdrawalData(net=withdrawals.net[:val_start], net_gw=withdrawals.net_gw[:val_start]),
        day_of_year=doy[:val_start],
    )
    h_ctx = model._last_h_context
    if h_ctx is not None:
        h_ctx = h_ctx.detach()
    Q_B, _ = model.simulate(
        forcing=forcing[val_start:val_end],
        initial_state=spinup_state,
        graph=graph, node_coords=node_coords, territorial=territorial,
        withdrawals=WithdrawalData(net=withdrawals.net[val_start:val_end], net_gw=withdrawals.net_gw[val_start:val_end]),
        day_of_year=doy[val_start:val_end],
        h_context=h_ctx,
    )
print(f"   done in {time.time()-t0:.0f}s; Q_B.shape={tuple(Q_B.shape)}")

# ── Compare ───────────────────────────────────────────────────────────
diff = (Q_A - Q_B).abs()
mean_A, mean_B = Q_A.mean().item(), Q_B.mean().item()
max_abs_diff   = diff.max().item()
mean_abs_diff  = diff.mean().item()
relative_diff  = (mean_abs_diff / max(mean_A, 1e-6)) * 100

print("\n══════════════════════════════════════════════════════════════")
print("  Protocol comparison (val period 2019-2021, 1096 days)")
print("══════════════════════════════════════════════════════════════")
print(f"  mean(Q_A) one-long-call : {mean_A:.4f} m³/s")
print(f"  mean(Q_B) split-with-h  : {mean_B:.4f} m³/s")
print(f"  max  |Q_A - Q_B|        : {max_abs_diff:.4f} m³/s")
print(f"  mean |Q_A - Q_B|        : {mean_abs_diff:.4f} m³/s")
print(f"  relative difference     : {relative_diff:.2f}%")
print()
if max_abs_diff < 1e-3:
    print("  ✓ IDENTICAL — h_context propagation is WORKING.  Diagnostic bug is elsewhere.")
elif relative_diff < 1.0:
    print("  ~ NEAR-IDENTICAL (<1% diff) — minor numerics, h_context propagation OK.")
else:
    print("  ✗ DIVERGENCE — protocol B is broken.  State or h_context not propagating.")
