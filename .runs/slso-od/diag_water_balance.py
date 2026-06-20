"""Diagnostic de PARTITION du bilan hydrique de la colonne Hydrotel calée.
Où part l'eau : surf / hypo / base / ETR / Δstockage ? Le pic raboté (peak_ratio
0.67) vient-il d'un excès d'infiltration en surface pendant les orages ?

Charge best-mini-hydrotel-pm.pt (modèle calé), wrappe column_step pour capturer
prod_surf/hypo/base + apport + etr par pas, ferme le bilan, et analyse les jours
d'orage (fraction de l'apport ruisselée en surface vs infiltrée).

  python .runs/slso-od/diag_water_balance.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import tomllib
import numpy as np
import pandas as pd
import torch
import xarray as xr
from meandre.data.basin_cache import BasinCache
from meandre.model import HydroModel
from meandre.utils.state import HydroState

CFG = ".runs/slso-od/config/slso-od-mini-hydrotel-pm.toml"
CKPT = ".runs/slso-od/checkpoints/best-mini-hydrotel-pm.pt"
cfg = tomllib.load(open(CFG, "rb"))
DB = ".runs/slso-od/" + cfg["paths"]["basin_db"]
cache = BasinCache(DB); h = cache.load(device="cpu"); n = h["n_nodes"]

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
kw = dict(ck["init_kwargs"]); kw["n_coord_freqs"] = cfg["model"].get("n_coord_freqs", 6)
m = HydroModel(**kw); m.routing.routing_mode = "operator-lagged"; m.temperature = None
m.load_state_dict(ck["state_dict"]); m.eval()
print(f"modèle calé chargé : column_mode={kw.get('column_mode')} et_mode={kw.get('et_mode')} "
      f"theta_frac={kw.get('column_theta_init_frac')}")

ds = xr.open_dataset(cfg["paths"]["forcing_cache"])
times = pd.to_datetime(ds["time"].values); ff = ds["forcing"].values.astype(np.float32); ds.close()
w0 = int(np.searchsorted(times, np.datetime64("2020-01-01")))
w1 = int(np.searchsorted(times, np.datetime64("2022-12-31")))
fc = torch.from_numpy(ff[w0:w1]); win = times[w0:w1]; NT = len(win)
doy = torch.tensor(win.dayofyear.values, dtype=torch.long)
wd = cache.load_withdrawals(str(win[0].date()), str(win[-1].date()), device="cpu")
print(f"fenêtre {win[0].date()} → {win[-1].date()}  ({NT} pas, {n} nœuds)")

# ── wrap column_step : capture la partition par pas (moyenne bassin, mm) ──
orig = m.vertical_column.column_step
rec = {k: [] for k in ("apport", "prod_surf", "prod_hypo", "prod_base", "etr", "etp")}
theta_hist = []
def wrapped(enriched, state, doy=None, return_diagnostics=False, **kw):
    out = orig(enriched, state, doy=doy, return_diagnostics=True, **kw)
    d = out.diag
    for k in rec:
        rec[k].append(float(d[k].mean()))
    theta_hist.append((float(out.state.theta1.mean()), float(out.state.theta2.mean()),
                       float(out.state.theta3.mean())))
    return out
m.vertical_column.column_step = wrapped

with torch.no_grad():
    Q, _ = m.simulate(
        forcing=fc, initial_state=HydroState.default_warm(n), graph=h["graph"],
        node_coords=h["node_coords"], territorial=h["territorial"], withdrawals=wd,
        day_of_year=doy, tbptt_steps=0)

A = {k: np.array(v) for k, v in rec.items()}
th = np.array(theta_hist)   # (NT, 3) theta FIN de pas

# z des couches (du static réellement alimenté)
pso = m.vertical_column._static["soil"]
z1 = float(pso["z1"].mean()); z2 = float(pso["z2"].mean()); z3 = float(pso["z3"].mean())
thetas = (float(pso["thetas1"].mean()), float(pso["thetas2"].mean()), float(pso["thetas3"].mean()))
# Δstockage : theta init (0.9·thetas) → theta final, en mm
frac = float(kw.get("column_theta_init_frac", 0.9))
th0 = np.array([frac * thetas[0], frac * thetas[1], frac * thetas[2]])
dstock = ((th[-1, 0] - th0[0]) * z1 + (th[-1, 1] - th0[1]) * z2 + (th[-1, 2] - th0[2]) * z3) * 1000.0

ap = A["apport"].sum()
surf = A["prod_surf"].sum(); hypo = A["prod_hypo"].sum(); base = A["prod_base"].sum()
etr = A["etr"].sum()
prod = surf + hypo + base
closure = ap - prod - etr - dstock

print(f"\n=== géométrie sol (alimentée) : z1={z1:.2f} z2={z2:.2f} z3={z3:.2f} m "
      f"(total {z1+z2+z3:.2f}) | thetas {thetas[0]:.2f}/{thetas[1]:.2f}/{thetas[2]:.2f} ===")
print(f"\n=== BILAN HYDRIQUE CUMULÉ (mm, moyenne bassin, {NT} pas) ===")
print(f"  apport (entrée)     : {ap:8.1f}   (100%)")
print(f"  ├─ prod_surf        : {surf:8.1f}   ({100*surf/ap:5.1f}%)")
print(f"  ├─ prod_hypo        : {hypo:8.1f}   ({100*hypo/ap:5.1f}%)")
print(f"  ├─ prod_base        : {base:8.1f}   ({100*base/ap:5.1f}%)")
print(f"  ├─ ETR (perte)      : {etr:8.1f}   ({100*etr/ap:5.1f}%)")
print(f"  ├─ Δstockage sol    : {dstock:8.1f}   ({100*dstock/ap:5.1f}%)")
print(f"  └─ fermeture/fuite  : {closure:8.1f}   ({100*closure/ap:5.1f}%)")
print(f"\n  production totale    : {prod:8.1f}   ({100*prod/ap:5.1f}% de l'apport)")
print(f"  ratio surf:hypo:base : {surf/prod:.2f} : {hypo/prod:.2f} : {base/prod:.2f}")

# ── analyse jours d'orage : top 20 apports, fraction ruisselée en surface ──
idx = np.argsort(A["apport"])[::-1][:20]
ap_storm = A["apport"][idx].sum()
surf_storm = A["prod_surf"][idx].sum(); hypo_storm = A["prod_hypo"][idx].sum()
prod_storm = A["prod_surf"][idx].sum() + A["prod_hypo"][idx].sum() + A["prod_base"][idx].sum()
print(f"\n=== JOURS D'ORAGE (top 20 apports) ===")
print(f"  apport orages        : {ap_storm:8.1f} mm")
print(f"  → surf {100*surf_storm/ap_storm:.1f}%  hypo {100*hypo_storm/ap_storm:.1f}%  "
      f"prod totale {100*prod_storm/ap_storm:.1f}%  infiltré/stocké {100*(1-prod_storm/ap_storm):.1f}%")
print(f"  theta1 médian ces jours: {np.median(th[idx,0]):.3f}  (thetas1={thetas[0]:.3f}, "
      f"saturation {100*np.median(th[idx,0])/thetas[0]:.0f}%)")
print("DONE")
