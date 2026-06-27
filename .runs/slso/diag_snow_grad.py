"""Le gradient de la loss SCF (MODIS snow) atteint-il sp_fonte ? ZERO entrainement.
Forward sur une fenetre enneigee, loss MSE(SCF_sim, snow_frac), backward, on lit
sp_fonte.grad + si diag.swe porte le gradient.
  python .runs/slso/diag_snow_grad.py
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, xarray as xr, torch
from meandre.data.basin_cache import BasinCache
from meandre.model import HydroModel
from meandre.utils.state import HydroState

DB = ".runs/slso/data/slso.duckdb"; FORC = ".runs/slso/data/forcing.nc"
CKPT = ".runs/slso/checkpoints/best-physitel-hydrotel-modissnow.pt"
T0, T1 = "2023-01-01", "2023-06-30"   # hiver/printemps : neige + MODIS dispo

cache = BasinCache(DB); h = cache.load(device="cpu"); n = h["n_nodes"]
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
kw = dict(ck["init_kwargs"]); kw["compile_soil"] = False; kw["compile_column"] = False
m = HydroModel(**kw); m.load_state_dict(ck["state_dict"], strict=False); m.train()

# sp_fonte dans les params ?
spf = {nm: p for nm, p in m.named_parameters() if "sp_fonte" in nm}
print("sp_fonte params:", {k: (round(float(torch.nn.functional.softplus(v)),2), v.requires_grad) for k,v in spf.items()})

ds = xr.open_dataset(FORC); times = pd.to_datetime(ds["time"].values).normalize()
w0 = int(np.searchsorted(times, np.datetime64(T0))); w1 = int(np.searchsorted(times, np.datetime64(T1)))+1
fc = torch.from_numpy(ds["forcing"].values[w0:w1].astype(np.float32)); ds.close()
win = times[w0:w1]; doy = torch.tensor(win.dayofyear.values, dtype=torch.long)
wd = cache.load_withdrawals(str(win[0].date()), str(win[-1].date()), device="cpu")
snow = cache.load_modis_snow(T0, T1, device="cpu")   # (T,n)

Q, _, diag = m.simulate(forcing=fc, initial_state=HydroState.default_warm(n), graph=h["graph"],
                        node_coords=h["node_coords"], territorial=h["territorial"], withdrawals=wd,
                        day_of_year=doy, return_diagnostics=True)
swe = diag.swe
print(f"diag.swe: shape={tuple(swe.shape)} requires_grad={swe.requires_grad} grad_fn={swe.grad_fn is not None} mean={float(swe.mean()):.2f} max={float(swe.max()):.1f}")
scf = 1.0 - torch.exp(-swe / 15.0)
valid = ~torch.isnan(snow)
print(f"snow valid: {int(valid.sum())}/{snow.numel()}  scf mean(valid)={float(scf[valid].mean()):.3f}  snow mean(valid)={float(snow[valid].mean()):.3f}")
loss = ((scf[valid] - snow[valid])**2).mean()
print(f"snow loss = {float(loss):.5f}")
m.zero_grad(); loss.backward()
for k,v in spf.items():
    g = v.grad
    print(f"  {k}.grad = {None if g is None else float(g):.3e}")
