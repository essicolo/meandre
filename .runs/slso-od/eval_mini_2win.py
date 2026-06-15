"""Éval per-station du mini-bassin sur DEUX fenêtres : train (2020-2021) et
val (2022). Le contraste train vs val diagnostique l'overfit des effets
aléatoires : si les codes lèvent train mais pas val, ils captent du bruit.

  python .runs/slso-od/eval_mini_2win.py <config.toml> <checkpoint.pt>
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

CFG, CKPT = sys.argv[1], sys.argv[2]
WIN_START = "2019-01-01"
TRAIN = ("2020-01-01", "2021-12-31")
VAL = ("2022-01-01", "2022-12-31")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = tomllib.load(open(CFG, "rb"))
DB = (".runs/slso-od/" + cfg["paths"]["basin_db"]) if not cfg["paths"]["basin_db"].startswith("/") else cfg["paths"]["basin_db"]
DS, DE = cfg["temporal"]["date_start"], cfg["temporal"]["date_end"]
cache = BasinCache(DB); h = cache.load(device=device); n_nodes = h["n_nodes"]
ds = xr.open_dataset(cfg["paths"]["forcing_cache"])
times = pd.to_datetime(ds["time"].values); ff = ds["forcing"].values.astype(np.float32); ds.close()
w0 = int(np.searchsorted(times, np.datetime64(WIN_START))); win = times[w0:]
fc = torch.from_numpy(ff[w0:]).to(device)
doy = torch.tensor(win.dayofyear.values, dtype=torch.long, device=device)
wd = cache.load_withdrawals(str(win[0].date()), str(win[-1].date()), device=device)
obs = cache.load_observations(date_start=DS, date_end=DE, min_valid_days=365)
st = sorted(set(obs["station_node_map"].values()))
q_obs = torch.from_numpy(obs["discharge"][w0:][:, st]).to(device)

def idx(a, b):
    m = (win >= pd.Timestamp(a)) & (win <= pd.Timestamp(b))
    return torch.tensor(np.where(m)[0], device=device)
tr_i, va_i = idx(*TRAIN), idx(*VAL)

def kge(sim, o):
    m = ~np.isnan(o); s, o = sim[m], o[m]
    if len(o) < 30 or o.std() < 1e-9 or s.std() < 1e-9: return np.nan
    r = np.corrcoef(s, o)[0, 1]
    return 1.0 - math.sqrt((r-1)**2 + (s.mean()/o.mean()-1)**2 + ((s.std()/s.mean())/(o.std()/o.mean())-1)**2)

def pr(sim, o):
    m = ~np.isnan(o); o = o[m]
    if len(o) < 40: return np.nan
    s = sim[m]; hi = o >= np.quantile(o, 0.99)
    if hi.sum() < 2 or o[hi].mean() < 1e-9: return np.nan
    return s[hi].mean()/o[hi].mean()

_ck = torch.load(CKPT, map_location="cpu", weights_only=False)
kw = dict(_ck["init_kwargs"]); kw["n_coord_freqs"] = cfg["model"].get("n_coord_freqs", 6)
m = HydroModel(**kw).to(device); m.load(CKPT); m.temperature = None
m.routing.routing_mode = cfg["model"].get("routing_mode", "operator-lagged"); m.eval()
with torch.no_grad():
    Q, _ = m.simulate(forcing=fc, initial_state=HydroState.zeros(n_nodes, device=device),
                      graph=h["graph"], node_coords=h["node_coords"], territorial=h["territorial"],
                      withdrawals=wd, day_of_year=doy)

def med(win_i):
    sim = Q[win_i][:, st].cpu().numpy(); o = q_obs[win_i].cpu().numpy()
    ks = [kge(sim[:, j], o[:, j]) for j in range(sim.shape[1])]
    ps = [pr(sim[:, j], o[:, j]) for j in range(sim.shape[1])]
    return np.nanmedian(ks), np.nanmedian(ps)

tk, tp = med(tr_i); vk, vp = med(va_i)
enc = m.spatial_encoder
zmag = float(enc.latent_codes.abs().mean()) if getattr(enc, "use_latent_codes", False) else 0.0
print(f"{CKPT.split('/')[-1]:>26} | train kge_med={tk:.3f} pr={tp:.3f} | val kge_med={vk:.3f} pr={vp:.3f} | |z|={zmag:.4f}", flush=True)
