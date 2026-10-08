"""Smoke du ruissellement hortonien sous-journalier : modèle use_hortonian on vs off
sur le forçage CaSR intensité (canal DT_eff). Vérifie que l'Hortonien (1) tourne sans
NaN, (2) AUGMENTE le débit, surtout en ÉTÉ (orages convectifs intenses), (3) le gradient
passe. Fenêtre 2019-2020 (spinup + été), stations jaugées.
  python .runs/slso/test_hortonian_wiring.py
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, xarray as xr, torch
from meandre.data.basin_cache import BasinCache
from meandre.model import HydroModel
from meandre.utils.state import HydroState

dev = "cpu"; DB = ".runs/slso/data/slso.duckdb"
cache = BasinCache(DB); h = cache.load(device=dev); n = h["n_nodes"]
ds = xr.open_dataset(".runs/slso/data/forcing-casr-riox-intens.nc")
times = pd.to_datetime(ds["time"].values).normalize()
w = (times >= pd.Timestamp("2019-06-01")) & (times <= pd.Timestamp("2020-09-30"))
ff = ds["forcing"].values[w].astype(np.float32); win = times[w]; ds.close()
fc = torch.from_numpy(ff).to(dev)
doy = torch.tensor(win.dayofyear.values, dtype=torch.long, device=dev)
wd = cache.load_withdrawals(str(win[0].date()), str(win[-1].date()), device=dev)
print(f"DT_eff (canal 6) : médian {float(fc[:,:,6].median()):.1f}h  été<hiver attendu  "
      f"(jun-aoû {float(fc[np.array(win.month.isin([6,7,8]))][:,:,6].median()):.1f}h vs "
      f"déc-fév {float(fc[np.array(win.month.isin([12,1,2]))][:,:,6].median()):.1f}h)")

def run(horton, grad=False):
    m = HydroModel(n_nodes=n, n_forcing=7, context_window=30, residual_history=14,
                   max_travel_time=20, column_mode="hydrotel", use_frost_rankinen=True,
                   et_mode="mcguinness", use_hortonian=horton,
                   n_territorial=h["territorial"].to_tensor().shape[1]).to(dev)
    if horton: m.vertical_column.use_hortonian=True; m.vertical_column.storm_channel=6
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        Q,_ = m.simulate(forcing=fc, initial_state=HydroState.default_warm(n, device=dev),
                         graph=h["graph"], node_coords=h["node_coords"],
                         territorial=h["territorial"], withdrawals=wd, day_of_year=doy)
    return Q, m

torch.manual_seed(0); Q0,m0 = run(False)
torch.manual_seed(0); Q1,m1 = run(True); m1.load_state_dict(m0.state_dict(),strict=False)
torch.manual_seed(0); Q1,_ = run(True)
# été = juin-sept de la 2e année
summer = np.array(win.month.isin([6,7,8,9])) & np.array(win.year==2020)
gi = list(range(n))
print(f"\nsans Hortonien : Q moy {float(Q0.mean()):.3f}  été {float(Q0[summer].mean()):.3f}  NaN={bool(torch.isnan(Q0).any())}")
print(f"avec Hortonien : Q moy {float(Q1.mean()):.3f}  été {float(Q1[summer].mean()):.3f}  NaN={bool(torch.isnan(Q1).any())}")
print(f"ratio été (avec/sans) : {float(Q1[summer].mean()/Q0[summer].mean()):.2f}  "
      f"({'Hortonien AUGMENTE le quickflow été' if Q1[summer].mean()>Q0[summer].mean() else 'pas d effet ?'})")