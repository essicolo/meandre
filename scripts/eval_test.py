"""Held-out TEST evaluation on best.pt (CPU, doesn't conflict with GPU training).

Loads best.pt (whichever epoch was selected best on dev period),
simulates the full forcing window, then evaluates on the test period
(2023-2024 — held-out, never seen during training).

Reports:
  - Pooled KGE
  - Per-station median KGE
  - Per-station mean KGE
  - Distribution KGE > 0.5, < 0
  - Per-station table (saved as parquet)
"""
from __future__ import annotations

import os
import sys
import time
import tomllib
from pathlib import Path

import duckdb
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
from meandre.routing.withdrawals import WithdrawalData
from meandre.utils.metrics import kge, kge_components, nse
from meandre.utils.state import HydroState

DEVICE = torch.device("cpu")

with open("notebooks/slso/config/slso.toml", "rb") as f:
    cfg = tomllib.load(f)
paths = cfg["paths"]
mcfg = cfg["model"]
temp = cfg["temporal"]
sc = cfg.get("soil", {})

DATE_START, DATE_END = temp["date_start"], temp["date_end"]
TEST_START = temp.get("test_start", "2023-01-01")
TEST_END   = temp.get("test_end", "2024-12-31")

# ── Basin + forcing ─────────────────────────────────────────────
cache = BasinCache(paths["basin_db"])
hydro = cache.load(device=DEVICE)
graph = hydro["graph"]; territorial = hydro["territorial"]
node_coords = hydro["node_coords"]; n_nodes = hydro["n_nodes"]

forcing = extract_forcing(
    zarr_path=paths["weather_grid"], node_coords=node_coords, node_elev=None,
    date_start=DATE_START, date_end=DATE_END,
    cache_nc=paths["forcing_cache"], device=DEVICE,
)
ds = xr.open_dataset(paths["forcing_cache"])
all_dates = ds.time.sel(time=slice(DATE_START, DATE_END)).values
ds.close()
dates_pd = pd.DatetimeIndex(all_dates)
doy = torch.tensor(dates_pd.dayofyear.values, dtype=torch.long)

days = all_dates.astype("datetime64[D]")
i_test = int(np.searchsorted(days, np.datetime64(TEST_START, "D")))
i_end  = int(np.searchsorted(days, np.datetime64(TEST_END, "D"), side="right"))
print(f"[test] held-out period {TEST_START} → {TEST_END}  (days [{i_test}:{i_end}])")

# ── Build model + load best.pt ──────────────────────────────────
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
print(f"[model] loaded {paths['checkpoint']}")

# ── Withdrawals ──────────────────────────────────────────────────
withdrawals = cache.load_withdrawals(date_start=DATE_START, date_end=DATE_END, device=DEVICE)

# ── Simulate full forcing on CPU ────────────────────────────────
T = forcing.shape[0]
print(f"[sim] forward pass {T} days on CPU... (estimated 25-40 min)")
t0 = time.time()
with torch.no_grad():
    Q_all, _ = model.simulate(
        forcing=forcing, initial_state=HydroState.zeros(n_nodes, device=DEVICE),
        graph=graph, node_coords=node_coords, territorial=territorial,
        withdrawals=withdrawals, day_of_year=doy,
    )
print(f"[sim] done in {time.time()-t0:.0f}s")

Q_test = Q_all[i_test:i_end].cpu()

# ── Observations ────────────────────────────────────────────────
obs = cache.load_observations(date_start=DATE_START, date_end=DATE_END, min_valid_days=365)
station_node_map = obs["station_node_map"]
station_indices = sorted(set(station_node_map.values()))
mask = torch.zeros(n_nodes, dtype=torch.bool)
for ni in station_indices:
    mask[ni] = True
q_obs_full = torch.from_numpy(obs["discharge"][:, station_indices]).cpu()
q_obs_test = q_obs_full[i_test:i_end]
sids = list(station_node_map.keys())

# Station metadata
con = duckdb.connect(paths["basin_db"], read_only=True)
sta_meta = {}
for s in sids:
    row = con.execute(
        "SELECT lon, lat, drainage_area_km2 FROM stations WHERE station_id=?", [s]
    ).fetchone()
    if row:
        sta_meta[s] = dict(lon=float(row[0]), lat=float(row[1]),
                            area=float(row[2]))
con.close()

Q_test_stn = Q_test[:, mask]
n_common = min(Q_test_stn.shape[0], q_obs_test.shape[0])
Q_test_stn = Q_test_stn[:n_common]; q_obs_test = q_obs_test[:n_common]

# ── Per-station metrics ─────────────────────────────────────────
rows = []
for sid in sids:
    ni = station_node_map[sid]
    col = station_indices.index(ni)
    qo = q_obs_test[:, col]
    qs = Q_test_stn[:, col]
    v = ~torch.isnan(qo) & ~torch.isnan(qs)
    if v.sum() < 30:
        continue
    qo_v, qs_v = qo[v], qs[v]
    kc = kge_components(qo_v, qs_v)
    rows.append(dict(
        station=sid,
        area_km2=sta_meta.get(sid, {}).get("area", 0),
        n_days=int(v.sum()),
        r=float(kc["r"]),
        beta=float(kc["beta"]),
        gamma=float(kc["gamma"]),
        KGE=float(kge(qo_v, qs_v)),
        NSE=float(nse(qo_v, qs_v)),
    ))

df = pd.DataFrame(rows).sort_values("KGE")
out = Path("notebooks/slso/results/test_eval")
out.mkdir(parents=True, exist_ok=True)
df.to_parquet(out / "per_station.parquet", index=False)

# ── Pooled KGE ──────────────────────────────────────────────────
qo_flat = q_obs_test.reshape(-1)
qs_flat = Q_test_stn.reshape(-1)
v_flat = ~torch.isnan(qo_flat) & ~torch.isnan(qs_flat)
pooled_kge = float(kge(qo_flat[v_flat], qs_flat[v_flat]))
pooled_kc = kge_components(qo_flat[v_flat], qs_flat[v_flat])

print("\n" + "═" * 72)
print(f"  HELD-OUT TEST {TEST_START} → {TEST_END}")
print(f"  best.pt loaded from {paths['checkpoint']}")
print("═" * 72)
print(f"\n  POOLED metrics (all stations × all days flattened):")
print(f"    KGE     : {pooled_kge:.4f}")
print(f"    r       : {pooled_kc['r'].item():.4f}")
print(f"    beta    : {pooled_kc['beta'].item():.4f}")
print(f"    gamma   : {pooled_kc['gamma'].item():.4f}")
print(f"    kge_log : {pooled_kc['kge_log'].item():.4f}")

print(f"\n  PER-STATION KGE distribution ({len(df)} stations):")
print(f"    Median  : {df['KGE'].median():.4f}")
print(f"    Mean    : {df['KGE'].mean():.4f}")
print(f"    P25     : {df['KGE'].quantile(0.25):.4f}")
print(f"    P75     : {df['KGE'].quantile(0.75):.4f}")
print(f"    KGE > 0.7 : {(df['KGE'] > 0.7).sum()}/{len(df)}")
print(f"    KGE > 0.5 : {(df['KGE'] > 0.5).sum()}/{len(df)}")
print(f"    KGE < 0   : {(df['KGE'] < 0).sum()}/{len(df)}")

# Worst 5 + best 5 stations
print(f"\n  Worst 5 stations (KGE):")
for _, r in df.head(5).iterrows():
    print(f"    {r['station']}  area={r['area_km2']:.0f}km²  KGE={r['KGE']:.3f}  "
          f"β={r['beta']:.2f}  γ={r['gamma']:.2f}  r={r['r']:.2f}")

print(f"\n  Best 5 stations (KGE):")
for _, r in df.tail(5).iterrows():
    print(f"    {r['station']}  area={r['area_km2']:.0f}km²  KGE={r['KGE']:.3f}  "
          f"β={r['beta']:.2f}  γ={r['gamma']:.2f}  r={r['r']:.2f}")

print(f"\n[out] notebooks/slso/results/test_eval/per_station.parquet")
