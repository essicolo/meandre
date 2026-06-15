"""kge_med + peak_ratio du modèle baseline, SÉPARÉS par source d'obs.

Teste l'hypothèse d'Essi (2026-06-14) : les stations HYDAT (que le Québec
n'utilise pas) plombent l'évaluation open-data. Si les CEHQ sont nettement
meilleures, le plafond était un artefact de vérité-terrain, pas du modèle.

  python .runs/slso-od/eval_by_source.py
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
import math, re, tomllib
import numpy as np
import torch
import pandas as pd
import xarray as xr

from meandre.data.basin_cache import BasinCache
from meandre.model import HydroModel
from meandre.utils.state import HydroState

CFG = ".runs/slso-od/config/slso-od-vsafull-mo.toml"
CKPT = ".runs/slso-od/checkpoints/best-vsafull-mo.pt"
WIN_START, VAL_START, VAL_END = "2017-01-01", "2019-01-01", "2021-12-31"

def is_hydat(sid):
    return bool(re.match(r"^0[0-9][A-Z]", sid))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = tomllib.load(open(CFG, "rb"))
DATE_START, DATE_END = cfg["temporal"]["date_start"], cfg["temporal"]["date_end"]
cache = BasinCache(".runs/slso-od/data/basin.duckdb")
h = cache.load(device=device); n_nodes = h["n_nodes"]
ds = xr.open_dataset(cfg["paths"]["forcing_cache"])
all_times = pd.to_datetime(ds["time"].values)
forcing_full = ds["forcing"].values.astype(np.float32); ds.close()
w0 = int(np.searchsorted(all_times, np.datetime64(WIN_START)))
win_times = all_times[w0:]
fc = torch.from_numpy(forcing_full[w0:]).to(device)
doy = torch.tensor(win_times.dayofyear.values, dtype=torch.long, device=device)
wd = cache.load_withdrawals(str(win_times[0].date()), str(win_times[-1].date()), device=device)

obs = cache.load_observations(date_start=DATE_START, date_end=DATE_END, min_valid_days=365)
snm = obs["station_node_map"]                       # {station_id: node_idx}
# node_idx → liste des station_id (au cas où collision), on garde le 1er
node_to_sid = {}
for sid, nidx in snm.items():
    node_to_sid.setdefault(nidx, sid)
station_indices = sorted(set(snm.values()))
sources = ["HYDAT" if is_hydat(node_to_sid[ni]) else "CEHQ" for ni in station_indices]
q_obs = torch.from_numpy(obs["discharge"][w0:][:, station_indices]).to(device)
val_mask = (win_times >= pd.Timestamp(VAL_START)) & (win_times <= pd.Timestamp(VAL_END))
val_idx = torch.tensor(np.where(val_mask)[0], device=device)
print(f"stations: {sources.count('HYDAT')} HYDAT, {sources.count('CEHQ')} CEHQ\n", flush=True)

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

_ck = torch.load(CKPT, map_location="cpu", weights_only=False)
kw = dict(_ck["init_kwargs"]); kw["n_coord_freqs"] = cfg["model"].get("n_coord_freqs", 6)
m = HydroModel(**kw).to(device); m.load(CKPT); m.temperature = None
m.routing.routing_mode = cfg["model"].get("routing_mode", "operator-lagged"); m.eval()
with torch.no_grad():
    Q, _ = m.simulate(forcing=fc, initial_state=HydroState.zeros(n_nodes, device=device),
                      graph=h["graph"], node_coords=h["node_coords"], territorial=h["territorial"],
                      withdrawals=wd, day_of_year=doy)
sim = Q[val_idx][:, station_indices].cpu().numpy(); o = q_obs[val_idx].cpu().numpy()

rows = {"HYDAT": ([], []), "CEHQ": ([], [])}
for j, src in enumerate(sources):
    rows[src][0].append(kge(sim[:, j], o[:, j]))
    rows[src][1].append(peak_ratio(sim[:, j], o[:, j]))

print(f"{'source':>8} {'n':>4} {'kge_med':>9} {'peak_ratio_med':>15}", flush=True)
for src in ("HYDAT", "CEHQ"):
    k, p = rows[src]
    print(f"{src:>8} {len(k):>4} {np.nanmedian(k):9.3f} {np.nanmedian(p):15.3f}", flush=True)
# ensemble
allk = rows["HYDAT"][0]+rows["CEHQ"][0]; allp = rows["HYDAT"][1]+rows["CEHQ"][1]
print(f"{'TOUS':>8} {len(allk):>4} {np.nanmedian(allk):9.3f} {np.nanmedian(allp):15.3f}", flush=True)
print("DONE", flush=True)
