"""Effet de sélection : KGE du modèle HydroSHEDS, stations communes PHYSITEL
(CEHQ) vs HYDAT-seules. Si les communes sont bien meilleures, le gap PHYSITEL
vient en partie du jeu de stations, pas du modèle (hypothèse Essi 2026-06-14).

  python .runs/slso-od/eval_station_selection.py
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
import math, tomllib
import numpy as np
import torch
import pandas as pd
import xarray as xr
import duckdb

from meandre.data.basin_cache import BasinCache
from meandre.model import HydroModel
from meandre.utils.state import HydroState

CFG = ".runs/slso-od/config/slso-od-hs.toml"
CKPT = ".runs/slso-od/checkpoints/best-hs.pt"
DB = ".runs/slso-od/data/basin_hydrosheds.duckdb"
WIN_START, VAL_START, VAL_END = "2017-01-01", "2019-01-01", "2021-12-31"

# Jeu de stations PHYSITEL (CEHQ).
p = duckdb.connect(".runs/slso/data/slso.duckdb", read_only=True)
PHYSITEL = set(r[0] for r in p.execute("SELECT DISTINCT station_id FROM stations").fetchall())
p.close()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = tomllib.load(open(CFG, "rb"))
DS, DE = cfg["temporal"]["date_start"], cfg["temporal"]["date_end"]
cache = BasinCache(DB); h = cache.load(device=device); n_nodes = h["n_nodes"]
ds = xr.open_dataset(cfg["paths"]["forcing_cache"])
times = pd.to_datetime(ds["time"].values); ff = ds["forcing"].values.astype(np.float32); ds.close()
w0 = int(np.searchsorted(times, np.datetime64(WIN_START))); win = times[w0:]
fc = torch.from_numpy(ff[w0:]).to(device)
doy = torch.tensor(win.dayofyear.values, dtype=torch.long, device=device)
wd = cache.load_withdrawals(str(win[0].date()), str(win[-1].date()), device=device)
obs = cache.load_observations(date_start=DS, date_end=DE, min_valid_days=365)
snm = obs["station_node_map"]
node_to_sid = {}
for sid, ni in snm.items():
    node_to_sid.setdefault(ni, sid)
st_idx = sorted(set(snm.values()))
q_obs = torch.from_numpy(obs["discharge"][w0:][:, st_idx]).to(device)
vmask = (win >= pd.Timestamp(VAL_START)) & (win <= pd.Timestamp(VAL_END))
vidx = torch.tensor(np.where(vmask)[0], device=device)


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
with torch.no_grad():
    Q, _ = m.simulate(forcing=fc, initial_state=HydroState.zeros(n_nodes, device=device),
                      graph=h["graph"], node_coords=h["node_coords"], territorial=h["territorial"],
                      withdrawals=wd, day_of_year=doy)
sim = Q[vidx][:, st_idx].cpu().numpy(); o = q_obs[vidx].cpu().numpy()

groups = {"communes PHYSITEL (CEHQ)": ([], []), "HYDAT seules": ([], [])}
for j, ni in enumerate(st_idx):
    sid = node_to_sid[ni]
    g = "communes PHYSITEL (CEHQ)" if sid in PHYSITEL else "HYDAT seules"
    groups[g][0].append(kge(sim[:, j], o[:, j]))
    groups[g][1].append(pr(sim[:, j], o[:, j]))

print(f"{'groupe':>28} {'n':>4} {'kge_med':>9} {'peak_ratio':>11}", flush=True)
allk, allp = [], []
for g, (ks, ps) in groups.items():
    allk += ks; allp += ps
    print(f"{g:>28} {len(ks):>4} {np.nanmedian(ks):9.3f} {np.nanmedian(ps):11.3f}", flush=True)
print(f"{'TOUTES':>28} {len(allk):>4} {np.nanmedian(allk):9.3f} {np.nanmedian(allp):11.3f}", flush=True)
print("DONE", flush=True)
