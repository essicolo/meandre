"""Validation de l'approximation translation vs Muskingum (étape 1 du bypass).

Relance le modèle entraîné deux fois sur le réseau PHYSITEL : routage Muskingum
normal, puis routage translation pure (coefficients forcés c01=1, c2=0, 1 sous-pas
=> Q_out = Q_in + apport latéral), tout le reste identique (lacs, prélèvements,
balayage topologique). Compare Q aux stations et KGE vs observé.
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
init = torch.load(CKPT, map_location="cpu", weights_only=False)["init_kwargs"]
m = HydroModel(**init).to(device); m.load(CKPT); m.eval()

cache = BasinCache(DB)
h = cache.load(device=device)
ds = xr.open_dataset(FORCING)
fc = torch.from_numpy(ds["forcing"].values.astype(np.float32)).to(device)
dt_all = pd.to_datetime(ds["time"].values).normalize()
ds.close()
doy = torch.tensor(dt_all.dayofyear.values, dtype=torch.long, device=device)
wd = cache.load_withdrawals("2000-01-01", "2024-12-31", device=device)

def run():
    with torch.no_grad():
        Q, _ = m.simulate(forcing=fc, initial_state=HydroState.zeros(h["n_nodes"], device=device),
                          graph=h["graph"], node_coords=h["node_coords"], territorial=h["territorial"],
                          withdrawals=wd, day_of_year=doy)
    return Q.cpu().numpy()

print("Forward Muskingum...", flush=True)
Q_musk = run()

print("Forward translation (c01=1, c2=0, 1 sous-pas)...", flush=True)
musk = m.routing.muskingum
orig_n = musk.n_substeps
orig_precompute = musk.precompute_coefficients
musk.n_substeps = 1
musk.precompute_coefficients = lambda K, x: (torch.ones_like(K), torch.zeros_like(K))
Q_trans = run()
musk.n_substeps = orig_n
musk.precompute_coefficients = orig_precompute

# Observations + stations
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

test = (dt_all >= "2022-01-01") & (dt_all <= "2024-12-31")
def kge(s, o):
    msk = ~np.isnan(o) & ~np.isnan(s)
    if msk.sum() < 30: return np.nan
    s, o = s[msk], o[msk]; r = np.corrcoef(s, o)[0, 1]
    return 1 - np.sqrt((r-1)**2 + (s.std()/o.std()-1)**2 + (s.mean()/o.mean()-1)**2)

Qm_s, Qt_s = Q_musk[:, sn], Q_trans[:, sn]
kge_m = np.array([kge(Qm_s[test, i], qo[test, i]) for i in range(len(sn))])
kge_t = np.array([kge(Qt_s[test, i], qo[test, i]) for i in range(len(sn))])
# corrélation entre les deux routages (sur tout)
corr = np.array([np.corrcoef(Qm_s[:, i], Qt_s[:, i])[0, 1] for i in range(len(sn))])
ratio = np.array([np.nanmean(Qt_s[test, i]) / max(np.nanmean(Qm_s[test, i]), 1e-6) for i in range(len(sn))])
v = np.isfinite(kge_m) & np.isfinite(kge_t)

print("\n=== Translation vs Muskingum (test 2022-2024, stations) ===")
print(f"KGE poolé Muskingum   : {kge(Qm_s[test].ravel(), qo[test].ravel()):.3f}")
print(f"KGE poolé Translation : {kge(Qt_s[test].ravel(), qo[test].ravel()):.3f}")
print(f"KGE médiane/station   : Musk {np.nanmedian(kge_m):.3f}  vs  Trans {np.nanmedian(kge_t):.3f}")
print(f"ΔKGE médian (T - M)   : {np.nanmedian(kge_t[v] - kge_m[v]):+.3f}")
print(f"corr(Q_musk, Q_trans) : médiane {np.median(corr):.4f}, min {corr.min():.4f}")
print(f"ratio débit moyen T/M : médiane {np.median(ratio):.3f} (1 = identique)")
print(f"stations où |ΔKGE|>0.05 : {(np.abs(kge_t[v]-kge_m[v])>0.05).sum()}/{v.sum()}")
print("VALIDATE_TRANS_DONE", flush=True)
