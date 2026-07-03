"""Pourquoi CaSR fond-il trop tôt ? Compare l'évolution du manteau neigeux (SWE)
sous CaSR vs quebec.zarr, MÊMES params de neige (du checkpoint McGuinness), pour
isoler l'effet du FORÇAGE. Sort, par hiver : SWE moyen bassin, date du pic, date de
demi-fonte (onset freshet), neige cumulée, jours de fonte. Modèle-libre côté fonte
(juste le module neige DegreJourModifie, vectorisé sur les nœuds).
  python .runs/slso/diag_snowpack.py
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, xarray as xr, torch
from meandre.data.basin_cache import BasinCache
from meandre.model import HydroModel
from hydrotel_clone.snow import init_state as snow_init

torch.set_default_dtype(torch.float32)
DB = ".runs/slso/data/slso.duckdb"
CKPT = ".runs/slso/checkpoints/best-physitel-hydrotel-casr-riox.pt"
FORCS = {"quebec": ".runs/slso/data/forcing.nc", "CaSR": ".runs/slso/data/forcing-casr-riox.nc"}
Y0, Y1 = "2018-08-01", "2021-07-31"   # 3 hivers

h = BasinCache(DB).load(device="cpu"); n = h["n_nodes"]
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
kw = dict(ck["init_kwargs"]); kw["compile_soil"] = False; kw["compile_column"] = False
m = HydroModel(**kw); m.load_state_dict(ck["state_dict"], strict=False); m.eval()
col = m.vertical_column
col._node_lat = h["node_coords"][:, 1]
sp = m.spatial_encoder(h["node_coords"], h["territorial"].to_tensor())
col.params_from_nerf(sp, h["territorial"], h["node_coords"])
ps = col._static["snow"]

def run_snow(forc):
    ds = xr.open_dataset(forc); t = pd.to_datetime(ds["time"].values).normalize()
    sl = (t >= pd.Timestamp(Y0)) & (t <= pd.Timestamp(Y1))
    ff = ds["forcing"].values[sl]; tt = t[sl]; ds.close()
    P = torch.from_numpy(ff[..., 0].astype(np.float32))
    tmin = torch.from_numpy(ff[..., 1].astype(np.float32))
    tmax = torch.from_numpy(ff[..., 2].astype(np.float32))
    state = snow_init(n, dtype=torch.float32)
    swe, melt, snowfall = [], [], []
    with torch.no_grad():
        for k in range(len(tt)):
            doy = torch.tensor(float(tt[k].dayofyear)).expand(n)
            pluie, neige = col._split_precip(P[k], tmin[k], tmax[k])
            ap, state = col.snow(tmin[k], tmax[k], pluie, neige, doy, state, ps)
            swe.append(float(state["couvert_nival_mm"].mean()))
            melt.append(float(ap.mean())); snowfall.append(float(neige.mean()))
    return tt, np.array(swe), np.array(melt), np.array(snowfall)

res = {}
for k, f in FORCS.items():
    res[k] = run_snow(f)
    print(f"[ok] {k} : forçage neige simulé")

print(f"\nHiver | forçage | SWE_pic(mm) date_pic | demi-fonte | neige_cumul(mm)")
for yr in [2019, 2020, 2021]:
    for k in FORCS:
        tt, swe, melt, snf = res[k]
        win = (tt >= pd.Timestamp(yr - 1, 11, 1)) & (tt <= pd.Timestamp(yr, 7, 1))
        s = swe[win]; d = tt[win]
        if len(s) == 0 or s.max() < 1: continue
        ipk = int(s.argmax()); peak = s[ipk]
        # demi-fonte = 1er jour après le pic où SWE < 50% du pic
        after = s[ipk:]; half = ipk + int(np.argmax(after < 0.5 * peak)) if (after < 0.5 * peak).any() else ipk
        snf_cum = snf[win][:ipk + 1].sum()
        print(f"  {yr}  | {k:7s} | {peak:6.0f}  {d[ipk].strftime('%m-%d')}    | {d[half].strftime('%m-%d')}  | {snf_cum:6.0f}")
    print()
# moyenne température hivernale par forçage déjà connue (~égales). Ici on regarde SWE.
print("Lecture : si CaSR a une demi-fonte PLUS TÔT à SWE pic comparable => fonte précoce")
print("intrinsèque ; si SWE pic CaSR plus BAS (neige captée en moins ou fondue en hiver)")
print("=> le manteau part plus petit et disparaît plus vite.")
