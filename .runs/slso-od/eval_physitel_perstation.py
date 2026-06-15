"""PHYSITEL best_phaseB_kge0904 : KGE PAR STATION (pas pooled) sur val 2019-2021.

Le but : le 0.904 de PHYSITEL est-il du pooled ? Le vrai kge_med par station,
comparé à l'open-data HydroSHEDS (CEHQ communes : kge_med 0.770), dit si
PHYSITEL est réellement supérieur ou si l'écart était un artefact de métrique.

  python .runs/slso-od/eval_physitel_perstation.py
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
import math
import numpy as np
import torch
import pandas as pd
import xarray as xr
import duckdb

from meandre.data.basin_cache import BasinCache
from meandre.model import HydroModel
from meandre.utils.state import HydroState

DB = ".runs/slso/data/slso.duckdb"
FORCING = ".runs/slso/data/forcing.nc"
CKPT = ".runs/slso/checkpoints/best_phaseB_kge0904_epoch50.pt"
DS, DE = "2000-01-01", "2021-12-31"
WIN_START, VAL_START, VAL_END = "2017-01-01", "2019-01-01", "2021-12-31"

# Stations open-data déjà évaluées (pour le sous-ensemble commun).
od = duckdb.connect(".runs/slso-od/data/basin_hydrosheds.duckdb", read_only=True)
OD_STATIONS = set(r[0] for r in od.execute(
    "SELECT station_id FROM observations GROUP BY station_id HAVING COUNT(*)>=365").fetchall())
od.close()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cache = BasinCache(DB); h = cache.load(device=device); n_nodes = h["n_nodes"]
ds = xr.open_dataset(FORCING)
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

def pooled_kge(sim2d, o2d):
    s = sim2d.ravel(); o = o2d.ravel(); m = ~np.isnan(o)
    s, o = s[m], o[m]
    r = np.corrcoef(s, o)[0, 1]
    return 1.0 - math.sqrt((r-1)**2 + (s.mean()/o.mean()-1)**2 + ((s.std()/s.mean())/(o.std()/o.mean())-1)**2)


_ck = torch.load(CKPT, map_location="cpu", weights_only=False)
kw = dict(_ck["init_kwargs"])
m = HydroModel(**kw).to(device); m.load(CKPT); m.temperature = None
m.eval()
with torch.no_grad():
    Q, _ = m.simulate(forcing=fc, initial_state=HydroState.zeros(n_nodes, device=device),
                      graph=h["graph"], node_coords=h["node_coords"], territorial=h["territorial"],
                      withdrawals=wd, day_of_year=doy)
sim = Q[vidx][:, st_idx].cpu().numpy(); o = q_obs[vidx].cpu().numpy()

all_k, all_p, com_k, com_p = [], [], [], []
for j, ni in enumerate(st_idx):
    sid = node_to_sid[ni]
    k = kge(sim[:, j], o[:, j]); p = pr(sim[:, j], o[:, j])
    all_k.append(k); all_p.append(p)
    if sid in OD_STATIONS:
        com_k.append(k); com_p.append(p)

print(f"PHYSITEL best_phaseB (val 2019-2021), {len(st_idx)} stations", flush=True)
print(f"  POOLED kge (toutes stations empilées) = {pooled_kge(sim, o):.3f}", flush=True)
print(f"  PAR STATION  kge_med = {np.nanmedian(all_k):.3f}   peak_ratio_med = {np.nanmedian(all_p):.3f}", flush=True)
print(f"  sous-ensemble COMMUN open-data ({len(com_k)}) : kge_med = {np.nanmedian(com_k):.3f}   peak_ratio = {np.nanmedian(com_p):.3f}", flush=True)
print("\n=== RAPPEL open-data HydroSHEDS (CEHQ communes) : kge_med 0.770, peak_ratio 0.689 ===", flush=True)
print("DONE", flush=True)
