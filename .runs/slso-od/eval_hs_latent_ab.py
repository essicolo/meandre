"""A/B réseau définitif : D8 vs HydroSHEDS, chaque modèle sur SON réseau.

kge_med + peak_ratio (top 1% crues) sur la validation, forward only. Chaque
run charge son propre bassin + forçage + obs depuis son config.

  python .runs/slso-od/eval_network_ab.py
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
import math, tomllib
import numpy as np
import torch
import pandas as pd
import xarray as xr

from meandre.data.basin_cache import BasinCache
from meandre.model import HydroModel
from meandre.utils.state import HydroState

WIN_START, VAL_START, VAL_END = "2017-01-01", "2019-01-01", "2021-12-31"
RUNS = [
    ("HS baseline (sans codes)", ".runs/slso-od/config/slso-od-hs.toml",
     ".runs/slso-od/checkpoints/best-hs.pt", ".runs/slso-od/data/basin_hydrosheds.duckdb"),
    ("HS + codes ADDITIFS", ".runs/slso-od/config/slso-od-hs-latent.toml",
     ".runs/slso-od/checkpoints/best-hs-latent.pt", ".runs/slso-od/data/basin_hydrosheds.duckdb"),
]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def kge(sim, o):
    m = ~np.isnan(o); s, o = sim[m], o[m]
    if len(o) < 30 or o.std() < 1e-9 or s.std() < 1e-9: return np.nan
    r = np.corrcoef(s, o)[0, 1]
    return 1.0 - math.sqrt((r-1)**2 + (s.mean()/o.mean()-1)**2 + ((s.std()/s.mean())/(o.std()/o.mean())-1)**2)

def peak_ratio(sim, o):
    m = ~np.isnan(o); o = o[m]
    if len(o) < 50: return np.nan
    s = sim[m]; hi = o >= np.quantile(o, 0.99)
    if hi.sum() < 3 or o[hi].mean() < 1e-9: return np.nan
    return s[hi].mean()/o[hi].mean()


@torch.no_grad()
def evaluate(cfg_path, ckpt, basin_db):
    cfg = tomllib.load(open(cfg_path, "rb"))
    DATE_START, DATE_END = cfg["temporal"]["date_start"], cfg["temporal"]["date_end"]
    cache = BasinCache(basin_db)
    h = cache.load(device=device); n_nodes = h["n_nodes"]
    ds = xr.open_dataset(cfg["paths"]["forcing_cache"])
    all_times = pd.to_datetime(ds["time"].values)
    forcing_full = ds["forcing"].values.astype(np.float32); ds.close()
    w0 = int(np.searchsorted(all_times, np.datetime64(WIN_START)))
    win = all_times[w0:]
    fc = torch.from_numpy(forcing_full[w0:]).to(device)
    doy = torch.tensor(win.dayofyear.values, dtype=torch.long, device=device)
    wd = cache.load_withdrawals(str(win[0].date()), str(win[-1].date()), device=device)
    obs = cache.load_observations(date_start=DATE_START, date_end=DATE_END, min_valid_days=365)
    st = sorted(set(obs["station_node_map"].values()))
    q_obs = torch.from_numpy(obs["discharge"][w0:][:, st]).to(device)
    vmask = (win >= pd.Timestamp(VAL_START)) & (win <= pd.Timestamp(VAL_END))
    vidx = torch.tensor(np.where(vmask)[0], device=device)

    _ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    kw = dict(_ck["init_kwargs"]); kw["n_coord_freqs"] = cfg["model"].get("n_coord_freqs", 6)
    m = HydroModel(**kw).to(device); m.load(ckpt); m.temperature = None
    m.routing.routing_mode = cfg["model"].get("routing_mode", "operator-lagged"); m.eval()
    Q, _ = m.simulate(forcing=fc, initial_state=HydroState.zeros(n_nodes, device=device),
                      graph=h["graph"], node_coords=h["node_coords"], territorial=h["territorial"],
                      withdrawals=wd, day_of_year=doy)
    sim = Q[vidx][:, st].cpu().numpy(); o = q_obs[vidx].cpu().numpy()
    kges = [kge(sim[:, j], o[:, j]) for j in range(sim.shape[1])]
    prs = [peak_ratio(sim[:, j], o[:, j]) for j in range(sim.shape[1])]
    return len(st), np.nanmedian(kges), np.nanmedian(prs)


print(f"{'réseau':>34} {'n_st':>5} {'kge_med':>9} {'peak_ratio':>11}", flush=True)
for name, cfg_path, ckpt, db in RUNS:
    n, km, pr = evaluate(cfg_path, ckpt, db)
    print(f"{name:>34} {n:>5} {km:9.3f} {pr:11.3f}", flush=True)
print("\nréférence Hydrotel (MG24HS) : peak_ratio ≈ 0.89", flush=True)
print("AB_DONE", flush=True)
