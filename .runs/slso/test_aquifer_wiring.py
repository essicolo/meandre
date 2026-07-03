"""Smoke de l'aquifère restituant : modèle avec use_aquifer=True, vérifie que
S_gw accumule (réservoir se remplit), que le débit n'a pas de NaN, et que le
baseflow est RETARDÉ vs use_aquifer=False (lissage hiver->été). Forçage synthétique.
  python .runs/slso/test_aquifer_wiring.py
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, torch
from meandre.data.basin_cache import BasinCache
from meandre.model import HydroModel
from meandre.utils.state import HydroState

dev = "cpu"
h = BasinCache(".runs/slso/data/slso.duckdb").load(device=dev); n = h["n_nodes"]
T = 120
g = torch.linspace(0, 1, T).reshape(T, 1)
# pluie forte au début (recharge), sec ensuite (étiage) : teste le soutien
P = torch.where(g < 0.35, torch.full_like(g, 12.0), torch.zeros_like(g))
tmean = (5.0 + 12.0 * g)
def chan(v): return v.expand(T, n)
forcing = torch.stack([chan(P), chan(tmean - 3), chan(tmean + 3),
                       chan(8.0 + 0*g), chan(3.0 + 0*g), chan(0.7 + 0*g)], dim=-1).to(dev)
doy = torch.arange(120, 120 + T, dtype=torch.long, device=dev)
_d0 = pd.Timestamp("2020-05-01")
wd = BasinCache(".runs/slso/data/slso.duckdb").load_withdrawals(str(_d0.date()), str((_d0+pd.Timedelta(days=T-1)).date()), device=dev)

def run(use_aq):
    m = HydroModel(n_nodes=n, n_forcing=6, context_window=30, residual_history=14,
                   max_travel_time=20, column_mode="hydrotel", use_frost_rankinen=True,
                   et_mode="mcguinness", use_aquifer=use_aq,
                   n_territorial=h["territorial"].to_tensor().shape[1]).to(dev).eval()
    s0 = HydroState.default_warm(n, device=dev)
    with torch.no_grad():
        Q, st = m.simulate(forcing=forcing, initial_state=s0, graph=h["graph"],
                           node_coords=h["node_coords"], territorial=h["territorial"],
                           withdrawals=wd, day_of_year=doy)
    return Q, st

Q0, st0 = run(False)
Q1, st1 = run(True)
print(f"sans aquifère : Q moy {float(Q0.mean()):.3f}  NaN={bool(torch.isnan(Q0).any())}  S_gw final {float(st0.S_gw.mean()):.2f}")
print(f"avec aquifère : Q moy {float(Q1.mean()):.3f}  NaN={bool(torch.isnan(Q1).any())}  S_gw final {float(st1.S_gw.mean()):.2f}")
# baseflow d'étiage : moyenne du débit sur la 2e moitié (période sèche), aux nœuds jaugés
gi = [int(x) for x in h["graph"].edge_index[1].unique().tolist()[:50]] if False else list(range(n))
late0 = float(Q0[T//2:].mean()); late1 = float(Q1[T//2:].mean())
print(f"débit étiage (2e moitié sèche) : sans {late0:.3f}  avec {late1:.3f}  "
      f"({'aquifère SOUTIENT' if late1 > late0 else 'pas de soutien ?'})")
print(f"S_gw accumule (>0) : {'OUI' if float(st1.S_gw.mean())>0 else 'NON'}")
print(f"\n--- conservation masse ---  Q0.sum {float(Q0.sum()):.0f}  Q1.sum {float(Q1.sum()):.0f}  "
      f"ratio {float(Q1.sum()/Q0.sum()):.3f}")
# k_gw prédit par le NeRF + magnitude pb : reconstruire via la colonne
m = HydroModel(n_nodes=n, n_forcing=6, context_window=30, residual_history=14, max_travel_time=20,
               column_mode="hydrotel", use_frost_rankinen=True, et_mode="mcguinness", use_aquifer=True,
               n_territorial=h["territorial"].to_tensor().shape[1]).to(dev).eval()
sp = m.spatial_encoder(h["node_coords"], h["territorial"].to_tensor())
print(f"k_gw NeRF : min {float(sp.k_gw.min()):.4f}  méd {float(sp.k_gw.median()):.4f}  max {float(sp.k_gw.max()):.4f} (1/j)")
