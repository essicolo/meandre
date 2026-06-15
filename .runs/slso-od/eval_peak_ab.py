"""A/B tête-à-tête sur le top 1% des crues : baseline vs infil2, forward only.

Évalue kge_med ET peak_ratio (le vrai juge des pics) sur la fenêtre de
validation, pour chaque checkpoint à son réglage entraîné (pas de forçage).

  python .runs/slso-od/eval_peak_ab.py
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

WIN_START, VAL_START, VAL_END = "2017-01-01", "2019-01-01", "2021-12-31"
RUNS = [
    ("baseline (ET+GRACE, sans infil)", ".runs/slso-od/config/slso-od-vsafull-mo.toml",
     ".runs/slso-od/checkpoints/best-vsafull-mo.pt"),
    ("infil2 (capacité infiltration découplée)", ".runs/slso-od/config/slso-od-vsafull-mo-infil.toml",
     ".runs/slso-od/checkpoints/best-vsafull-mo-infil.pt"),
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cache = BasinCache(".runs/slso-od/data/basin.duckdb")
h = cache.load(device=device)
n_nodes = h["n_nodes"]

cfg0 = tomllib.load(open(RUNS[0][1], "rb"))
DATE_START, DATE_END = cfg0["temporal"]["date_start"], cfg0["temporal"]["date_end"]
ds = xr.open_dataset(cfg0["paths"]["forcing_cache"])
all_times = pd.to_datetime(ds["time"].values)
forcing_full = ds["forcing"].values.astype(np.float32)
ds.close()
w0 = int(np.searchsorted(all_times, np.datetime64(WIN_START)))
win_times = all_times[w0:]
fc = torch.from_numpy(forcing_full[w0:]).to(device)
doy = torch.tensor(win_times.dayofyear.values, dtype=torch.long, device=device)
wd = cache.load_withdrawals(str(win_times[0].date()), str(win_times[-1].date()), device=device)
obs = cache.load_observations(date_start=DATE_START, date_end=DATE_END, min_valid_days=365)
st = sorted(set(obs["station_node_map"].values()))
q_obs = torch.from_numpy(obs["discharge"][w0:][:, st]).to(device)
val_mask = (win_times >= pd.Timestamp(VAL_START)) & (win_times <= pd.Timestamp(VAL_END))
val_idx = torch.tensor(np.where(val_mask)[0], device=device)
print(f"n_nodes={n_nodes}  val ({int(val_mask.sum())} j)  stations={len(st)}  device={device}\n", flush=True)


def kge(sim, o):
    m = ~np.isnan(o); s, o = sim[m], o[m]
    if len(o) < 30 or o.std() < 1e-9 or s.std() < 1e-9:
        return np.nan
    r = np.corrcoef(s, o)[0, 1]
    return 1.0 - math.sqrt((r-1)**2 + (s.mean()/o.mean()-1)**2 + ((s.std()/s.mean())/(o.std()/o.mean())-1)**2)


def peak_ratio(sim, o):
    m = ~np.isnan(o); o = o[m]
    if len(o) < 50:
        return np.nan
    s = sim[m]; hi = o >= np.quantile(o, 0.99)
    if hi.sum() < 3 or o[hi].mean() < 1e-9:
        return np.nan
    return s[hi].mean()/o[hi].mean()


@torch.no_grad()
def evaluate(cfg_path, ckpt):
    cfg = tomllib.load(open(cfg_path, "rb"))
    _ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    kw = dict(_ck["init_kwargs"]); kw["n_coord_freqs"] = cfg["model"].get("n_coord_freqs", 6)
    m = HydroModel(**kw).to(device)
    m.load(ckpt); m.temperature = None
    m.routing.routing_mode = cfg["model"].get("routing_mode", "operator-lagged"); m.eval()
    Q, _ = m.simulate(forcing=fc, initial_state=HydroState.zeros(n_nodes, device=device),
                      graph=h["graph"], node_coords=h["node_coords"], territorial=h["territorial"],
                      withdrawals=wd, day_of_year=doy)
    sim = Q[val_idx][:, st].cpu().numpy(); o = q_obs[val_idx].cpu().numpy()
    kges = [kge(sim[:, j], o[:, j]) for j in range(sim.shape[1])]
    prs = [peak_ratio(sim[:, j], o[:, j]) for j in range(sim.shape[1])]
    return np.nanmedian(kges), np.nanmedian(prs)


print(f"{'modèle':>42} {'kge_med':>9} {'peak_ratio':>11}", flush=True)
for name, cfg_path, ckpt in RUNS:
    km, pr = evaluate(cfg_path, ckpt)
    print(f"{name:>42} {km:9.3f} {pr:11.3f}", flush=True)
print("\nréférence Hydrotel (MG24HS) : peak_ratio ≈ 0.89", flush=True)
print("AB_DONE", flush=True)
