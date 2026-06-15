"""Sensibilité aux lacs : le processus de stockage tire-t-il son poids aux jauges ?

Forward court (3 ans) sur le checkpoint PHYSITEL entraîné, deux fois :
  - lacs actifs (LakeModule, tarage Q=k·S^β) — comportement nominal
  - lacs désactivés (is_lake=False partout → tous routés en tronçon Muskingum)
Compare le KGE par station. Caveat : le modèle a été ENTRAÎNÉ avec les lacs,
donc les désactiver à l'inférence est une borne basse (les params n'ont pas
été ré-identifiés sans lacs). Si le KGE tient quand même, les lacs ne pèsent
pas aux jauges ; s'il chute, le stockage compte → le rendre appris.
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
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
CKPT = ".runs/slso/checkpoints/best-phenology-no-gru.pt"
N_STEPS = 1096

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
init = torch.load(CKPT, map_location="cpu", weights_only=False)["init_kwargs"]
m = HydroModel(**init).to(device); m.load(CKPT); m.eval()

cache = BasinCache(DB)
h = cache.load(device=device)
ds = xr.open_dataset(FORCING)
fc = torch.from_numpy(ds["forcing"].values[:N_STEPS].astype(np.float32)).to(device)
dt_all = pd.to_datetime(ds["time"].values[:N_STEPS]).normalize()
ds.close()
doy = torch.tensor(dt_all.dayofyear.values, dtype=torch.long, device=device)
wd = cache.load_withdrawals("2000-01-01", "2002-12-31", device=device)

graph = h["graph"]
is_lake_orig = graph.is_lake.clone()
n_lakes = int(is_lake_orig.sum())

def run(lakes_on: bool):
    graph.is_lake = is_lake_orig if lakes_on else torch.zeros_like(is_lake_orig)
    if hasattr(graph, "_operator_topo"):
        graph._operator_topo = None
    with torch.no_grad():
        Q, _ = m.simulate(forcing=fc, initial_state=HydroState.zeros(h["n_nodes"], device=device),
                          graph=graph, node_coords=h["node_coords"], territorial=h["territorial"],
                          withdrawals=wd, day_of_year=doy)
    return Q.cpu().numpy()

Q_on = run(True)
Q_off = run(False)
graph.is_lake = is_lake_orig

con = duckdb.connect(DB, read_only=True)
st = con.execute("SELECT node_idx, station_id FROM stations ORDER BY node_idx").fetchdf()
ob = con.execute("SELECT date, station_id, discharge AS q FROM observations").fetchdf()
con.close()
sn = st["node_idx"].values.astype(int)
s2c = {s: i for i, s in enumerate(st["station_id"])}
d2t = {d: i for i, d in enumerate(dt_all)}
qo = np.full((len(dt_all), len(sn)), np.nan, np.float32)
for _, r in ob.iterrows():
    d = pd.Timestamp(r["date"]).normalize()
    if d in d2t and r["station_id"] in s2c:
        qo[d2t[d], s2c[r["station_id"]]] = float(r["q"])
test = dt_all >= "2001-01-01"

def kge(s, o):
    msk = ~np.isnan(o) & ~np.isnan(s)
    if msk.sum() < 30: return np.nan
    s, o = s[msk], o[msk]; r = np.corrcoef(s, o)[0, 1]
    return 1 - np.sqrt((r-1)**2 + (s.std()/o.std()-1)**2 + (s.mean()/o.mean()-1)**2)

# Quelles stations sont sous influence de lac (lac strictement en amont) ?
src = graph.edge_index[0].cpu().numpy(); dst = graph.edge_index[1].cpu().numpy()
from collections import defaultdict, deque
parents = defaultdict(list)
for s, d in zip(src, dst): parents[d].append(s)
isl = is_lake_orig.cpu().numpy().astype(bool)
def has_upstream_lake(node):
    seen, q = set(), deque([node])
    while q:
        u = q.popleft()
        for p in parents[u]:
            if isl[p]: return True
            if p not in seen: seen.add(p); q.append(p)
    return False

print(f"lacs={n_lakes}, stations={len(sn)}", flush=True)
print(f"{'station':10s} {'amont_lac':9s} {'KGE_on':>8s} {'KGE_off':>8s} {'delta':>8s}")
rows = []
for i, sid in enumerate(st["station_id"]):
    k_on = kge(Q_on[test, sn[i]], qo[test, i])
    k_off = kge(Q_off[test, sn[i]], qo[test, i])
    if not (np.isfinite(k_on) and np.isfinite(k_off)): continue
    ul = has_upstream_lake(sn[i])
    rows.append((sid, ul, k_on, k_off, k_off - k_on))
rows.sort(key=lambda r: r[4])
for sid, ul, k_on, k_off, d in rows[:8] + rows[-3:]:
    print(f"{sid:10s} {'oui' if ul else 'non':9s} {k_on:8.3f} {k_off:8.3f} {d:+8.3f}")
lake_st = [r for r in rows if r[1]]
print(f"\nPoolé : KGE_on={kge(Q_on[test][:,sn].ravel(), qo[test].ravel()):.4f} "
      f"KGE_off={kge(Q_off[test][:,sn].ravel(), qo[test].ravel()):.4f}")
print(f"Stations sous influence de lac : {len(lake_st)}/{len(rows)} ; "
      f"delta KGE médian {np.median([r[4] for r in lake_st]):+.4f}" if lake_st else "aucune station sous lac")
print(f"Stations sans lac amont : delta KGE médian {np.median([r[4] for r in rows if not r[1]]):+.4f}")
print("LAKE_SENS_DONE", flush=True)
