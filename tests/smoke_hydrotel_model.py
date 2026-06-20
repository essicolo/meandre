"""Smoke test Phase A seam 2 : HydroModel(column_mode="hydrotel") — la colonne
fidèle clonée branchée dans la boucle simulate complète (vertical + routage).
Vérifie qu'un forward+backward tourne sur le banc mini, sans NaN, et que le
gradient remonte end-to-end jusqu'au NeRF et aux params de la colonne.

  python tests/smoke_hydrotel_model.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tomllib
import numpy as np
import pandas as pd
import torch
import xarray as xr
from meandre.data.basin_cache import BasinCache
from meandre.model import HydroModel
from meandre.utils.state import HydroState

CFG = ".runs/slso-od/config/slso-od-mini-clone.toml"
CKPT = ".runs/slso-od/checkpoints/best-mini-clone.pt"
cfg = tomllib.load(open(CFG, "rb"))
DB = ".runs/slso-od/" + cfg["paths"]["basin_db"]
cache = BasinCache(DB); h = cache.load(device="cpu"); n = h["n_nodes"]

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
kw = dict(ck["init_kwargs"]); kw["n_coord_freqs"] = cfg["model"].get("n_coord_freqs", 6)
kw["column_mode"] = "hydrotel"; kw["et_mode"] = "mcguinness"
# Cold model : l'archi colonne diffère du checkpoint clone, on teste le branchement
# et la différentiabilité (pas la calibration).
m = HydroModel(**kw); m.routing.routing_mode = "operator-lagged"; m.temperature = None

ds = xr.open_dataset(cfg["paths"]["forcing_cache"])
times = pd.to_datetime(ds["time"].values); ff = ds["forcing"].values.astype(np.float32); ds.close()
w0 = int(np.searchsorted(times, np.datetime64("2019-01-01")))
NDAYS = 60
fc = torch.from_numpy(ff[w0:w0 + NDAYS]); win = times[w0:w0 + NDAYS]
doy = torch.tensor(win.dayofyear.values, dtype=torch.long)
wd = cache.load_withdrawals(str(win[0].date()), str(win[-1].date()), device="cpu")

Q, _ = m.simulate(
    forcing=fc, initial_state=HydroState.default_warm(n), graph=h["graph"],
    node_coords=h["node_coords"], territorial=h["territorial"], withdrawals=wd,
    day_of_year=doy, tbptt_steps=30)

assert Q.shape == (NDAYS, n), Q.shape
assert not torch.isnan(Q).any(), "Q contient des NaN"
loss = Q.mean(); loss.backward()
ng = sum(float(p.grad.abs().sum()) for p in m.spatial_encoder.parameters() if p.grad is not None)
cg = sum(float(p.grad.abs().sum()) for p in m.vertical_column.parameters() if p.grad is not None)
print(f"Q {tuple(Q.shape)}  mean {float(Q.mean()):.2f}  max {float(Q.max()):.1f}  NaN={bool(torch.isnan(Q).any())}")
print(f"grad NeRF {ng:.1f}  grad colonne {cg:.3f}")
assert ng > 0 and cg > 0, "gradient ne remonte pas end-to-end"
print("SMOKE OK (colonne Hydrotel branchee, differentiable end-to-end)")
