"""Validation du routage opérateur sur le vrai bassin (checkpoint entraîné).

Forward COURT (3 ans, un cycle hydrologique complet) sur PHYSITEL, trois modes.
L'équivalence est pas-à-pas : quelques centaines de pas suffisent. Attendu : operator reproduit level (KGE identique), lagged
proche (écart quantifié), et chronos comparés.
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
import time
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
init = torch.load(CKPT, map_location="cpu", weights_only=False)["init_kwargs"]
m = HydroModel(**init).to(device)
m.load(CKPT)
m.eval()
# Thermie coupée : le chemin opérateur ne la porte pas (et elle n'entre pas
# dans la perte) ; sans ça, fallback silencieux vers le balayage.
m.temperature = None

cache = BasinCache(DB)
h = cache.load(device=device)
ds = xr.open_dataset(FORCING)
N_STEPS = 1096  # 3 ans : crue de fonte + étiage + hiver
fc = torch.from_numpy(ds["forcing"].values[:N_STEPS].astype(np.float32)).to(device)
dt_all = pd.to_datetime(ds["time"].values[:N_STEPS]).normalize()
ds.close()
doy = torch.tensor(dt_all.dayofyear.values, dtype=torch.long, device=device)
wd = cache.load_withdrawals("2000-01-01", "2002-12-31", device=device)

def run(mode):
    m.routing.routing_mode = mode
    m.routing._op_state = None
    if hasattr(h["graph"], "_operator_topo"):
        h["graph"]._operator_topo = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        Q, _ = m.simulate(forcing=fc, initial_state=HydroState.zeros(h["n_nodes"], device=device),
                          graph=h["graph"], node_coords=h["node_coords"], territorial=h["territorial"],
                          withdrawals=wd, day_of_year=doy)
    torch.cuda.synchronize()
    return Q.cpu().numpy(), time.perf_counter() - t0

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
test = (dt_all >= "2001-01-01")  # comparaison hors spinup an 1

def kge(s, o):
    msk = ~np.isnan(o) & ~np.isnan(s)
    if msk.sum() < 30: return np.nan
    s, o = s[msk], o[msk]; r = np.corrcoef(s, o)[0, 1]
    return 1 - np.sqrt((r-1)**2 + (s.std()/o.std()-1)**2 + (s.mean()/o.mean()-1)**2)

results = {}
for mode in ("level", "operator", "operator-lagged"):
    Q, wall = run(mode)
    Qs = Q[:, sn]
    kp = kge(Qs[test].ravel(), qo[test].ravel())
    results[mode] = (Q, wall, kp)
    print(f"{mode:16s} forward 3 ans: {wall:6.1f} s | KGE test poolé = {kp:.4f}", flush=True)

Q_lvl = results["level"][0]
for mode in ("operator", "operator-lagged"):
    Qm = results[mode][0]
    d = np.abs(Qm - Q_lvl)
    denom = max(np.abs(Q_lvl).max(), 1e-6)
    print(f"{mode:16s} vs level : écart max {d.max():.4f} m3/s "
          f"(relatif {d.max()/denom:.2e}), écart moyen {d.mean():.5f}")
print("VALIDATE_OP_DONE", flush=True)
