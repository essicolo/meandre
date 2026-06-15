"""Scan décisif de infil_ratio (capacité d'infiltration de surface, forward only).

Question (2026-06-14) : si on FORCE infil_ratio bas (surface scellée → plus de
ruissellement de Horton), est-ce que peak_ratio monte EN GARDANT le KGE, parce
que infil_ratio est découplé du drainage ? Contraste avec le scan K_sat, où
baisser K_sat montait les pics mais cassait le volume (même bouton = drainage).
  - Si peak_ratio ↑ et kge tient → le mécanisme MARCHE, manque l'incitation.
  - Si peak_ratio ↑ mais kge/volume cassent → Horton casse aussi le volume.

  python .runs/slso-od/scan_infil_ratio.py
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
import math
import tomllib
import numpy as np
import torch
import pandas as pd
import xarray as xr

from meandre.data.basin_cache import BasinCache
from meandre.model import HydroModel
from meandre.utils.state import HydroState

CFG_PATH = ".runs/slso-od/config/slso-od-vsafull-mo-infil.toml"
CKPT = ".runs/slso-od/checkpoints/best-vsafull-mo-infil.pt"
WIN_START = "2017-01-01"
VAL_START = "2019-01-01"
VAL_END = "2021-12-31"
RATIOS = [1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05]

cfg = tomllib.load(open(CFG_PATH, "rb"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATE_START = cfg["temporal"]["date_start"]
DATE_END = cfg["temporal"]["date_end"]

cache = BasinCache(".runs/slso-od/data/basin.duckdb")
h = cache.load(device=device)
n_nodes = h["n_nodes"]

ds = xr.open_dataset(cfg["paths"]["forcing_cache"])
all_times = pd.to_datetime(ds["time"].values)
forcing_full = ds["forcing"].values.astype(np.float32)
ds.close()

w0 = int(np.searchsorted(all_times, np.datetime64(WIN_START)))
win_times = all_times[w0:]
fc = torch.from_numpy(forcing_full[w0:]).to(device)
doy = torch.tensor(win_times.dayofyear.values, dtype=torch.long, device=device)
wd = cache.load_withdrawals(str(win_times[0].date()), str(win_times[-1].date()), device=device)

obs = cache.load_observations(date_start=DATE_START, date_end=DATE_END, min_valid_days=365)
station_indices = sorted(set(obs["station_node_map"].values()))
discharge_full = obs["discharge"]
q_obs_win = torch.from_numpy(discharge_full[w0:][:, station_indices]).to(device)

val_mask = (win_times >= pd.Timestamp(VAL_START)) & (win_times <= pd.Timestamp(VAL_END))
val_idx = torch.tensor(np.where(val_mask)[0], device=device)

print(f"n_nodes={n_nodes}  val {VAL_START}..{VAL_END} ({int(val_mask.sum())} j)  "
      f"stations={len(station_indices)}  device={device}", flush=True)

_ck = torch.load(CKPT, map_location="cpu", weights_only=False)
_kw = dict(_ck["init_kwargs"])
_kw["n_coord_freqs"] = cfg["model"].get("n_coord_freqs", 6)
m = HydroModel(**_kw).to(device)
m.load(CKPT)
m.temperature = None
m.routing.routing_mode = cfg["model"].get("routing_mode", "operator-lagged")
m.eval()
soil = m.vertical_column.soil
assert getattr(soil, "use_separate_infil_capacity", False), "checkpoint sans capacité d'infiltration séparée"

LO, HI = soil._infil_bounds


def set_ratio(r):
    frac = min(max((r - LO) / (HI - LO), 1e-4), 1.0 - 1e-4)
    with torch.no_grad():
        soil.infil_ratio_raw.copy_(torch.tensor(math.log(frac / (1 - frac))))


def kge(sim, obs):
    msk = ~np.isnan(obs); s, o = sim[msk], obs[msk]
    if len(o) < 30 or o.std() < 1e-9 or s.std() < 1e-9:
        return np.nan
    r = np.corrcoef(s, o)[0, 1]; beta = s.mean()/o.mean()
    gamma = (s.std()/s.mean())/(o.std()/o.mean())
    return 1.0 - math.sqrt((r-1)**2 + (beta-1)**2 + (gamma-1)**2)


def peak_ratio(sim, obs):
    msk = ~np.isnan(obs); o = obs[msk]
    if len(o) < 50:
        return np.nan
    s = sim[msk]; hi = o >= np.quantile(o, 0.99)
    if hi.sum() < 3 or o[hi].mean() < 1e-9:
        return np.nan
    return s[hi].mean()/o[hi].mean()


@torch.no_grad()
def run(r):
    set_ratio(r)
    Q, _ = m.simulate(
        forcing=fc, initial_state=HydroState.zeros(n_nodes, device=device),
        graph=h["graph"], node_coords=h["node_coords"], territorial=h["territorial"],
        withdrawals=wd, day_of_year=doy,
    )
    sim = Q[val_idx][:, station_indices].cpu().numpy()
    o = q_obs_win[val_idx].cpu().numpy()
    kges = [kge(sim[:, j], o[:, j]) for j in range(sim.shape[1])]
    prs = [peak_ratio(sim[:, j], o[:, j]) for j in range(sim.shape[1])]
    return np.nanmedian(kges), np.nanmedian(prs), float(np.nansum(sim))


print("\n=== scan infil_ratio (forçage, forward only) ===", flush=True)
print(f"{'ratio':>6} {'kge_med':>9} {'peak_ratio':>11} {'vol_rel':>9}", flush=True)
vol0 = None
for r in RATIOS:
    km, pr, vol = run(r)
    if vol0 is None:
        vol0 = vol
    print(f"{r:6.2f} {km:9.3f} {pr:11.3f} {vol/vol0:9.3f}", flush=True)
print("SCAN_DONE", flush=True)
