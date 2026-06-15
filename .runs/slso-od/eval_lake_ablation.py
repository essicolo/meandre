"""Ablation des lacs (forward only, best-hs) : combien rabotent-ils les pics ?

Forward avec lacs ACTIFS vs lacs BYPASSÉS (is_lake=False → routage rivière).
Δpeak_ratio = effet de lissage des lacs. Gros écart ⇒ les lacs sur-lissent,
levier = leur gestion / leur nombre (hypothèse Essi 2026-06-14).

  python .runs/slso-od/eval_lake_ablation.py
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

CFG = ".runs/slso-od/config/slso-od-hs.toml"
CKPT = ".runs/slso-od/checkpoints/best-hs.pt"
DB = ".runs/slso-od/data/basin_hydrosheds.duckdb"
WIN_START, VAL_START, VAL_END = "2017-01-01", "2019-01-01", "2021-12-31"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = tomllib.load(open(CFG, "rb"))
DS, DE = cfg["temporal"]["date_start"], cfg["temporal"]["date_end"]
cache = BasinCache(DB); h = cache.load(device=device); n_nodes = h["n_nodes"]
n_lakes = int(h["graph"].is_lake.sum())
ds = xr.open_dataset(cfg["paths"]["forcing_cache"])
times = pd.to_datetime(ds["time"].values); ff = ds["forcing"].values.astype(np.float32); ds.close()
w0 = int(np.searchsorted(times, np.datetime64(WIN_START))); win = times[w0:]
fc = torch.from_numpy(ff[w0:]).to(device)
doy = torch.tensor(win.dayofyear.values, dtype=torch.long, device=device)
wd = cache.load_withdrawals(str(win[0].date()), str(win[-1].date()), device=device)
obs = cache.load_observations(date_start=DS, date_end=DE, min_valid_days=365)
st = sorted(set(obs["station_node_map"].values()))
q_obs = torch.from_numpy(obs["discharge"][w0:][:, st]).to(device)
vmask = (win >= pd.Timestamp(VAL_START)) & (win <= pd.Timestamp(VAL_END))
vidx = torch.tensor(np.where(vmask)[0], device=device)
print(f"{n_lakes} lacs sur {n_nodes} nœuds, {len(st)} stations\n", flush=True)


def kge(sim, o):
    m = ~np.isnan(o); s, o = sim[m], o[m]
    if len(o) < 30 or o.std() < 1e-9 or s.std() < 1e-9: return np.nan
    r = np.corrcoef(s, o)[0, 1]
    return 1.0 - math.sqrt((r-1)**2 + (s.mean()/o.mean()-1)**2 + ((s.std()/s.mean())/(o.std()/o.mean())-1)**2)

def pr(sim, o):
    m = ~np.isnan(o); o = o[m]
    if len(o) < 50: return np.nan
    s = sim[m]; hi = o >= np.quantile(o, 0.99)
    if hi.sum() < 3 or o[hi].mean() < 1e-9: return np.nan
    return s[hi].mean()/o[hi].mean()


_ck = torch.load(CKPT, map_location="cpu", weights_only=False)
kw = dict(_ck["init_kwargs"]); kw["n_coord_freqs"] = cfg["model"].get("n_coord_freqs", 6)
m = HydroModel(**kw).to(device); m.load(CKPT); m.temperature = None
m.routing.routing_mode = cfg["model"].get("routing_mode", "operator-lagged"); m.eval()
o = q_obs[vidx].cpu().numpy()


@torch.no_grad()
def run(lakes_on):
    g = h["graph"]
    saved = g.is_lake.clone()
    if not lakes_on:
        g.is_lake = torch.zeros_like(saved)
    Q, _ = m.simulate(forcing=fc, initial_state=HydroState.zeros(n_nodes, device=device),
                      graph=g, node_coords=h["node_coords"], territorial=h["territorial"],
                      withdrawals=wd, day_of_year=doy)
    g.is_lake = saved
    sim = Q[vidx][:, st].cpu().numpy()
    kges = [kge(sim[:, j], o[:, j]) for j in range(sim.shape[1])]
    prs = [pr(sim[:, j], o[:, j]) for j in range(sim.shape[1])]
    return np.nanmedian(kges), np.nanmedian(prs)


print(f"{'config':>16} {'kge_med':>9} {'peak_ratio':>11}", flush=True)
for name, on in [("lacs ACTIFS", True), ("lacs BYPASSÉS", False)]:
    km, prr = run(on)
    print(f"{name:>16} {km:9.3f} {prr:11.3f}", flush=True)
print("DONE", flush=True)
