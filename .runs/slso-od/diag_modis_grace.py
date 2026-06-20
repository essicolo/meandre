"""Contre-vérification observationnelle de l'ET et du stockage du mini-banc
contre MODIS (MOD16A2 ETR 8-jours) et GRACE (TWS mensuel), forward seul, même
checkpoint clone, pour les deux modes d'ET (Penman-Monteith vs McGuinness).

Question : laquelle des deux ET colle le mieux à l'observation MODIS
INDÉPENDANTE (magnitude + saisonnalité + structure spatiale), et le stockage
simulé suit-il l'anomalie GRACE ? C'est la vérification demandée AVANT d'écrire
le terme de loss.

Comparaison ET sur les jours MODIS-valides uniquement (pas d'extrapolation
annuelle : les composites valides sont biaisés été). Comparaison TWS en anomalie
mensuelle centrée (références absolues sim≠GRACE).

  python .runs/slso-od/diag_modis_grace.py
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
import tomllib
import numpy as np
import torch
import pandas as pd
import xarray as xr
import duckdb

from meandre.data.basin_cache import BasinCache
from meandre.model import HydroModel
from meandre.utils.state import HydroState

CFG = ".runs/slso-od/config/slso-od-mini-clone.toml"
CKPT = ".runs/slso-od/checkpoints/best-mini-clone.pt"
WIN_START = "2019-01-01"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = tomllib.load(open(CFG, "rb"))
DB = ".runs/slso-od/" + cfg["paths"]["basin_db"]
cache = BasinCache(DB); h = cache.load(device=device); n_nodes = h["n_nodes"]
ds = xr.open_dataset(cfg["paths"]["forcing_cache"])
times = pd.to_datetime(ds["time"].values); ff = ds["forcing"].values.astype(np.float32); ds.close()
w0 = int(np.searchsorted(times, np.datetime64(WIN_START))); win = times[w0:]
fc = torch.from_numpy(ff[w0:]).to(device)
doy = torch.tensor(win.dayofyear.values, dtype=torch.long, device=device)
wd = cache.load_withdrawals(str(win[0].date()), str(win[-1].date()), device=device)
DSTR, DEND = str(win[0].date()), str(win[-1].date())

con = duckdb.connect(DB, read_only=True)
area = np.array([r[0] for r in con.execute(
    "SELECT area_km2_local FROM territorial ORDER BY node_idx").fetchall()], dtype=np.float64)
con.close()
aw = area / area.sum()  # poids domaine

# ── Observations indépendantes ──
et_obs = cache.load_modis_et(DSTR, DEND, device="cpu")   # (T, N) mm/j, NaN sparse
grace = cache.load_grace_tws(DSTR, DEND)                  # DataFrame(date, tws_mm, uncertainty)
print(f"MODIS : {int((~torch.isnan(et_obs)).sum())} obs valides ({(~torch.isnan(et_obs)).float().mean()*100:.1f}% des (t,n))")
print(f"GRACE : {len(grace)} mois {grace['date'].min().date()}..{grace['date'].max().date()}")
et_obs = et_obs.numpy()

_ck = torch.load(CKPT, map_location="cpu", weights_only=False)
kw = dict(_ck["init_kwargs"]); kw["n_coord_freqs"] = cfg["model"].get("n_coord_freqs", 6)
m = HydroModel(**kw).to(device); m.load(CKPT); m.temperature = None
m.routing.routing_mode = cfg["model"].get("routing_mode", "operator-lagged"); m.eval()
with torch.no_grad():
    sp = m.spatial_encoder(h["node_coords"], h["territorial"].to_tensor())
Z2 = sp.Z2.cpu().numpy(); Z3 = sp.Z3.cpu().numpy()  # (N,) m

def run(et_mode):
    m.vertical_column.et.et_mode = et_mode
    with torch.no_grad():
        Q, _, d = m.simulate(
            forcing=fc, initial_state=HydroState.zeros(n_nodes, device=device),
            graph=h["graph"], node_coords=h["node_coords"], territorial=h["territorial"],
            withdrawals=wd, day_of_year=doy, return_diagnostics=True)
    etr = d.etr.cpu().numpy()  # (T, N) mm/j
    # stockage total (mm), formule trainer : z1=0.30 hardcodé + Z2/Z3 NeRF
    soil = (d.theta1.cpu().numpy() * 0.30 + d.theta2.cpu().numpy() * Z2
            + d.theta3.cpu().numpy() * Z3) * 1000.0
    stor = soil + d.swe.cpu().numpy() + d.s_gw.cpu().numpy() \
        + d.canopy.cpu().numpy() + d.wetland.cpu().numpy()  # (T, N) mm
    return etr, stor

# GRACE → série mensuelle alignée sur win (anomalie centrée plus bas)
gmonth = grace.set_index("date")["tws_mm"]
win_month = pd.to_datetime(win).to_period("M")

sims = {mode: run(mode) for mode in ("penman", "mcguinness")}

# ── 1. ET vs MODIS sur jours valides ──
valid = ~np.isnan(et_obs)  # (T, N)
obs_dom_daily = (np.where(valid, et_obs, 0) * aw[None, :]).sum(1) / np.where(
    valid, aw[None, :], 0).sum(1).clip(1e-9)  # moy domaine par jour valide
print("\n=== 1. ET vs MODIS (jours/nœuds MODIS-valides) ===")
print(f"{'mode':>12} | {'ETR moy (mm/j)':>14} | {'MODIS (mm/j)':>12} | {'biais%':>7} | {'r spatial':>9}")
obs_mean = et_obs[valid].mean()
for mode, (etr, _) in sims.items():
    sim_mean = etr[valid].mean()
    # r spatial : moyenne par-nœud sur jours valides, sim vs MODIS
    pn_sim = np.array([etr[valid[:, n], n].mean() if valid[:, n].any() else np.nan
                       for n in range(n_nodes)])
    pn_obs = np.array([et_obs[valid[:, n], n].mean() if valid[:, n].any() else np.nan
                       for n in range(n_nodes)])
    ok = ~np.isnan(pn_sim) & ~np.isnan(pn_obs)
    r = np.corrcoef(pn_sim[ok], pn_obs[ok])[0, 1] if ok.sum() > 5 else np.nan
    print(f"{mode:>12} | {sim_mean:14.2f} | {obs_mean:12.2f} | {100*(sim_mean-obs_mean)/obs_mean:7.1f} | {r:9.3f}")

# ── saisonnalité mensuelle ET (domaine, jours valides) ──
print("\n  ET mensuelle domaine (mm/j) — MODIS vs PM vs McGuinness :")
mois_lbl = "  mois :" + "".join(f"{mm:>6}" for mm in range(1, 13))
print(mois_lbl)
def monthly_et(arr):
    out = []
    for mm in range(1, 13):
        sel = (pd.to_datetime(win).month == mm)[:, None] & valid
        out.append((arr[sel].mean()) if sel.any() else np.nan)
    return out
print("  MODIS :" + "".join(f"{v:6.2f}" for v in monthly_et(et_obs)))
for mode, (etr, _) in sims.items():
    print(f"  {mode[:5]:>5} :" + "".join(f"{v:6.2f}" for v in monthly_et(etr)))

# ── 2. TWS vs GRACE (anomalie mensuelle, basin-mean) ──
print("\n=== 2. stockage vs GRACE TWS (anomalie mensuelle, mm) ===")
print(f"{'mode':>12} | {'std sim':>8} | {'std GRACE':>9} | {'r':>6}")
# GRACE mensuel aligné
g_by_month = {p: v for p, v in zip(gmonth.index.to_period("M"), gmonth.values)}
months_uniq = pd.PeriodIndex(win_month).unique()
g_series = np.array([g_by_month.get(p, np.nan) for p in months_uniq])
gvalid = ~np.isnan(g_series)
g_anom = g_series - np.nanmean(g_series[gvalid])
std_g = np.nanstd(g_anom[gvalid])
for mode, (_, stor) in sims.items():
    sb = (stor * aw[None, :]).sum(1)  # basin-mean (T,)
    sm = np.array([sb[np.asarray(win_month == p)].mean() for p in months_uniq])
    sm_anom = sm - sm[gvalid].mean()
    r = np.corrcoef(sm_anom[gvalid], g_anom[gvalid])[0, 1]
    print(f"{mode:>12} | {np.std(sm_anom[gvalid]):8.1f} | {std_g:9.1f} | {r:6.3f}")
print("\nDONE")
