"""Calibration HORS-LIGNE du facteur de fonte sur la DATE de disparition du manteau
neigeux (MODIS). Pour une grille de melt_scale, forward -> SWE simulé -> date de
fonte (printemps) basin-moyenne, comparée à la date MODIS (snow_frac basin-moyen
passe sous 0.5). Le scale qui matche la date observée = facteur fidèle.
Contourne saturation + dilution estivale de la loss par-jour. ZERO entrainement.
  python .runs/slso/calibrate_melt_snowdate.py
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, xarray as xr, torch
from meandre.data.basin_cache import BasinCache
from meandre.model import HydroModel
from meandre.utils.state import HydroState

DB = ".runs/slso/data/slso.duckdb"; FORC = ".runs/slso/data/forcing.nc"
CKPT = ".runs/slso/checkpoints/best-physitel-hydrotel-overnight.pt"
SPIN0, END = "2015-01-01", "2018-12-31"   # spinup 2015 + 3 ans neige 2016-2018 (MODIS dispo)
YEARS = [2016, 2017, 2018]
SCALES = [0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
SWE_GONE = 10.0   # mm : manteau "disparu" sous ce seuil

cache = BasinCache(DB); h = cache.load(device="cpu"); n = h["n_nodes"]
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
kw = dict(ck["init_kwargs"]); kw["compile_soil"] = False; kw["compile_column"] = False

ds = xr.open_dataset(FORC); times = pd.to_datetime(ds["time"].values).normalize()
w0 = int(np.searchsorted(times, np.datetime64(SPIN0))); w1 = int(np.searchsorted(times, np.datetime64(END)))+1
ff = ds["forcing"].values[w0:w1].astype(np.float32); ds.close()
win = times[w0:w1]; doy_t = torch.tensor(win.dayofyear.values, dtype=torch.long)
wd = cache.load_withdrawals(str(win[0].date()), str(win[-1].date()), device="cpu")
snow = cache.load_modis_snow(SPIN0, END, device="cpu").numpy()   # (T,n)

def gone_doy_from_series(frac_or_swe, dates, year, thresh, is_obs):
    """DOY de fonte : dernier jour (fev-juil) au-dessus du seuil, +1."""
    sel = (dates >= pd.Timestamp(year,2,1)) & (dates <= pd.Timestamp(year,7,31))
    v = frac_or_swe[sel]; dd = dates[sel]
    above = v > thresh
    if not np.any(above): return np.nan
    return dd[np.where(above)[0][-1]].dayofyear

# date MODIS basin-moyenne (sur les nœuds valides chaque jour)
snow_masked = np.where(np.isnan(snow), np.nan, snow)
snow_basin = np.nanmean(snow_masked, axis=1)   # (T,)
obs_gone = {y: gone_doy_from_series(snow_basin, win, y, 0.5, True) for y in YEARS}
print("MODIS snow-gone DOY (basin) :", {y: (None if np.isnan(v) else int(v)) for y,v in obs_gone.items()})

print(f"\n{'scale':>6} {'fonte(mm/C/j)':>14} | " + " ".join(f"sim{y}" for y in YEARS) + " | err_moy(j)")
results = []
for sc in SCALES:
    m = HydroModel(**kw); m.load_state_dict(ck["state_dict"], strict=False); m.eval()
    if sc != 1.0:
        with torch.no_grad():
            for nm in ("sp_fonte_conif","sp_fonte_feu","sp_fonte_dec"):
                p = getattr(m.vertical_column, nm)
                eff = torch.nn.functional.softplus(p) * sc
                p.copy_(torch.log(torch.expm1(eff.clamp(min=1e-4))))
    fonte = float(torch.nn.functional.softplus(m.vertical_column.sp_fonte_conif)) * 1  # conif comme repère
    with torch.no_grad():
        _, _, diag = m.simulate(forcing=torch.from_numpy(ff), initial_state=HydroState.default_warm(n),
                                graph=h["graph"], node_coords=h["node_coords"], territorial=h["territorial"],
                                withdrawals=wd, day_of_year=doy_t, return_diagnostics=True)
    swe_basin = diag.swe.mean(dim=1).numpy()   # (T,)
    errs = []
    sim_gone = {}
    for y in YEARS:
        sg = gone_doy_from_series(swe_basin, win, y, SWE_GONE, False)
        sim_gone[y] = sg
        if not np.isnan(sg) and not np.isnan(obs_gone[y]): errs.append(abs(sg - obs_gone[y]))
    err = float(np.mean(errs)) if errs else np.nan
    results.append((sc, fonte, err))
    print(f"{sc:>6.2f} {fonte:>14.1f} | " + " ".join(f"{int(sim_gone[y]) if not np.isnan(sim_gone[y]) else 0:4d}" for y in YEARS) + f" | {err:.1f}")

best = min((r for r in results if not np.isnan(r[2])), key=lambda r: r[2])
print(f"\nMEILLEUR : melt_scale={best[0]} (fonte {best[1]:.1f} mm/C/j) — erreur date {best[2]:.1f} j")
print("=> compare au sweep KGE (melt0.4). Si concordant, le fix est FIDÈLE (calé MODIS, pas KGE).")
