"""Où le pic se perd-il : GÉNÉRATION ou ROUTAGE de canal ?

Compare, aux jauges, sur le top 1% des crues observées :
  - peak_ratio ROUTÉ   : Q routé du modèle / obs (ce qu'on mesure = 0.74)
  - peak_ratio INSTANT : (somme amont des apports latéraux, routage parfait
                          instantané) / obs
Si INSTANT >> ROUTÉ → le canal diffuse, levier = célérité dépendante du débit.
Si INSTANT ≈ ROUTÉ → déficit de GÉNÉRATION, levier = neige/sol/gel.

  python .runs/slso-od/diag_peak_loss.py
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

CFG = ".runs/slso-od/config/slso-od-hs-latent.toml"
CKPT = ".runs/slso-od/checkpoints/best-hs-latent.pt"
DB = ".runs/slso-od/data/basin_hydrosheds.duckdb"
WIN_START, VAL_START, VAL_END = "2017-01-01", "2019-01-01", "2021-12-31"

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
snm = obs["station_node_map"]; st = sorted(set(snm.values()))
q_obs = obs["discharge"][w0:][:, st]
vmask = (win >= pd.Timestamp(VAL_START)) & (win <= pd.Timestamp(VAL_END))
vi = np.where(vmask)[0]

# Ancêtres (amont) par nœud, depuis edges (src draine vers dst).
con = duckdb.connect(DB, read_only=True)
edges = con.execute("SELECT src, dst FROM edges").fetchall(); con.close()
children = {}  # dst -> [src...] (amont direct)
for s, d in edges:
    children.setdefault(int(d), []).append(int(s))
def ancestors(node):
    seen = set(); stack = [node]
    while stack:
        x = stack.pop()
        for c in children.get(x, []):
            if c not in seen:
                seen.add(c); stack.append(c)
    seen.add(node); return seen

# Forward avec diagnostics.
_ck = torch.load(CKPT, map_location="cpu", weights_only=False)
kw = dict(_ck["init_kwargs"]); kw["n_coord_freqs"] = cfg["model"].get("n_coord_freqs", 6)
m = HydroModel(**kw).to(device); m.load(CKPT); m.temperature = None
m.routing.routing_mode = cfg["model"].get("routing_mode", "operator-lagged"); m.eval()
with torch.no_grad():
    Q, _, diag = m.simulate(forcing=fc, initial_state=HydroState.zeros(n_nodes, device=device),
                            graph=h["graph"], node_coords=h["node_coords"], territorial=h["territorial"],
                            withdrawals=wd, day_of_year=doy, return_diagnostics=True)
qlat = diag.q_lateral.cpu().numpy()    # (T, N) m³/s, apport latéral local
Qr = Q.cpu().numpy()                    # (T, N) m³/s routé

def peak_ratio(sim, o):
    msk = ~np.isnan(o); o = o[msk]
    if len(o) < 50: return np.nan
    s = sim[msk]; hi = o >= np.quantile(o, 0.99)
    if hi.sum() < 3 or o[hi].mean() < 1e-9: return np.nan
    return s[hi].mean() / o[hi].mean()

pr_routed, pr_instant = [], []
for j, ni in enumerate(st):
    anc = list(ancestors(ni))
    q_instant = qlat[:, anc].sum(axis=1)   # routage instantané parfait
    o = q_obs[:, j]
    pr_routed.append(peak_ratio(Qr[vi, ni], o[vi]))
    pr_instant.append(peak_ratio(q_instant[vi], o[vi]))

print(f"{len(st)} jauges, top 1% crues, val {VAL_START}..{VAL_END}", flush=True)
print(f"  peak_ratio ROUTÉ (modèle)        = {np.nanmedian(pr_routed):.3f}", flush=True)
print(f"  peak_ratio INSTANT (somme amont) = {np.nanmedian(pr_instant):.3f}", flush=True)
gap = np.nanmedian(pr_instant) - np.nanmedian(pr_routed)
print(f"  perte due au ROUTAGE de canal    = {gap:.3f}", flush=True)
print("VERDICT:", "ROUTAGE diffuse (→ célérité)" if gap > 0.05 else "GÉNÉRATION limite (→ neige/sol/gel)", flush=True)
print("DONE", flush=True)
