"""Profil léger du forward meandre (CPU-bound) : où passe le temps ?

Chronométrage par composant via accumulateurs wall-time autour de chaque appel
forward (encodeur spatial, colonne verticale, routage), sans torch.profiler
(trop lourd sur la boucle à ~2 M d'ops). Comme le goulot est CPU-bound
(GPU sous-utilisé), le temps mur autour des appels reflète bien le coût réel.

  python .runs/slso-od/profile_forward.py [n_steps]
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
import time
import tomllib
import numpy as np
import torch
import pandas as pd
import xarray as xr

from meandre.data.basin_cache import BasinCache
from meandre.model import HydroModel
from meandre.utils.state import HydroState

N_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 120
MODE = sys.argv[2] if len(sys.argv) > 2 else "level"
N_TRAIN_STEPS = 6574  # pas d'entraînement plein-période (référence extrapolation)

cfg = tomllib.load(open(".runs/slso-od/config/slso-od.toml", "rb"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cache = BasinCache(".runs/slso-od/data/basin.duckdb")
h = cache.load(device=device)
n_nodes = h["n_nodes"]

ds = xr.open_dataset(cfg["paths"]["forcing_cache"])
fc = torch.from_numpy(ds["forcing"].values[:N_STEPS].astype(np.float32)).to(device)
dt = pd.to_datetime(ds["time"].values[:N_STEPS])
ds.close()
doy = torch.tensor(dt.dayofyear.values, dtype=torch.long, device=device)
wd = cache.load_withdrawals(str(dt[0].date()), str(dt[-1].date()), device=device)

_ck = torch.load(".runs/slso-od/checkpoints/_smoke.pt", map_location="cpu", weights_only=False)
m = HydroModel(**_ck["init_kwargs"]).to(device)
m.load(".runs/slso-od/checkpoints/_smoke.pt")
m.temperature = None  # thermie inutile à la perte ; requis pour le chemin opérateur
m.routing.routing_mode = MODE
m.train()
print(f"n_nodes={n_nodes}, n_steps={N_STEPS}, device={device}, mode={MODE}", flush=True)

# Accumulateurs wall-time par composant (somme sur tous les appels).
acc = {"NERF": 0.0, "VERTICAL": 0.0, "ROUTING": 0.0}
orig = {}
def wrap(mod, name):
    o = mod.forward
    orig[name] = o
    def w(*a, **k):
        t = time.perf_counter()
        r = o(*a, **k)
        acc[name] += time.perf_counter() - t
        return r
    mod.forward = w
wrap(m.spatial_encoder, "NERF")
wrap(m.vertical_column, "VERTICAL")
wrap(m.routing, "ROUTING")

def fwd_bwd():
    Q, _ = m.simulate(
        forcing=fc, initial_state=HydroState.zeros(n_nodes, device=device),
        graph=h["graph"], node_coords=h["node_coords"], territorial=h["territorial"],
        withdrawals=wd, day_of_year=doy,
    )
    (Q.abs().mean()).backward()
    m.zero_grad(set_to_none=True)

fwd_bwd()  # warmup
if device.type == "cuda":
    torch.cuda.synchronize()
for k in acc:
    acc[k] = 0.0  # reset après warmup

t0 = time.perf_counter()
fwd_bwd()
if device.type == "cuda":
    torch.cuda.synchronize()
wall = time.perf_counter() - t0

print(f"\nfwd+bwd {N_STEPS} pas : {wall:.2f} s", flush=True)
print(f"  -> epoch ({N_TRAIN_STEPS} pas train) ≈ {wall*N_TRAIN_STEPS/N_STEPS/60:.1f} min")
tot = sum(acc.values())
print("\n=== temps forward par composant (wall, somme sur la boucle) ===")
for k in ("NERF", "VERTICAL", "ROUTING"):
    print(f"  {k:9s} {acc[k]:7.2f} s  ({100*acc[k]/max(tot,1e-9):5.1f} % du forward instrumenté)")
print(f"  (le reste = backward + ops hors composants : {wall - tot:.2f} s)")
print("PROFILE_DONE", flush=True)
