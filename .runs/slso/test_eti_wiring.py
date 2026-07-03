"""Smoke END-TO-END du branchage ETI : construit le modèle en melt_mode='degree_day'
puis 'eti', simule la même fenêtre de forçage 8 canaux, et vérifie que (1) les deux
tournent sans NaN, (2) le débit DIFFÈRE (le mode circule bien jusqu'à la neige),
(3) tf/srf reçoivent du gradient. Forçage synthétique (saison de fonte) sur le vrai
réseau SLSO. CPU, rapide.
  python .runs/slso/test_eti_wiring.py
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np, torch
from meandre.data.basin_cache import BasinCache
from meandre.model import HydroModel
from meandre.utils.state import HydroState

dev = "cpu"
h = BasinCache(".runs/slso/data/slso.duckdb").load(device=dev); n = h["n_nodes"]
T = 80
# forçage synthétique 8 canaux : [P,Tmin,Tmax,R_n,u2,e_a,FB,FI]
g = torch.linspace(0, 1, T).reshape(T, 1)
tmean = (-8.0 + 18.0 * g)                      # -8 -> +10 °C
P = torch.where(g < 0.4, torch.full_like(g, 8.0), torch.zeros_like(g))   # neige début
FB = (80.0 + 240.0 * g)                        # 80 -> 320 W/m²
def chan(v): return v.expand(T, n)
forcing = torch.stack([chan(P), chan(tmean - 4), chan(tmean + 4),
                       chan(7.0 + 0*g), chan(4.0 + 0*g), chan(0.6 + 0*g),
                       chan(FB), chan(280.0 + 0*g)], dim=-1).to(dev)   # (T,n,8)
doy = torch.arange(60, 60 + T, dtype=torch.long, device=dev)
import pandas as pd
_d0 = pd.Timestamp("2020-01-01"); _d1 = _d0 + pd.Timedelta(days=T - 1)
wd = BasinCache(".runs/slso/data/slso.duckdb").load_withdrawals(str(_d0.date()), str(_d1.date()), device=dev)

def build(mode):
    m = HydroModel(n_nodes=n, n_forcing=8, context_window=30, residual_history=14,
                   max_travel_time=20, column_mode="hydrotel", use_frost_rankinen=True,
                   et_mode="mcguinness", melt_mode=mode,
                   n_territorial=h["territorial"].to_tensor().shape[1]).to(dev)
    m.eval(); return m

def run(m):
    Q, _ = m.simulate(forcing=forcing, initial_state=HydroState.default_warm(n, device=dev),
                      graph=h["graph"], node_coords=h["node_coords"],
                      territorial=h["territorial"], withdrawals=wd, day_of_year=doy)
    return Q

torch.manual_seed(0)
m_dd = build("degree_day")
# copie des poids pour comparer À PARAMÈTRES ÉGAUX (seul le mode diffère)
m_eti = build("eti"); m_eti.load_state_dict(m_dd.state_dict(), strict=False)
m_eti.vertical_column.melt_mode = "eti"; m_eti.vertical_column.sw_channel = 6

with torch.no_grad():
    Qdd = run(m_dd); Qeti = run(m_eti)
print(f"degree_day : Q moy {float(Qdd.mean()):.3f}  NaN={bool(torch.isnan(Qdd).any())}")
print(f"eti        : Q moy {float(Qeti.mean()):.3f}  NaN={bool(torch.isnan(Qeti).any())}")
d = (Qdd - Qeti).abs()
print(f"|Qdd - Qeti| max {float(d.max()):.4f}  moy {float(d.mean()):.4f}  "
      f"DIFFÈRENT={'OUI' if float(d.max())>1e-4 else 'NON (mode ne circule pas !)'}")

# gradient vers tf/srf en mode eti
m_eti.zero_grad()
Q = run(m_eti)
loss = (torch.arange(T, dtype=Q.dtype).reshape(T, 1) * Q).sum()   # moment temporel
loss.backward()
gtf = m_eti.vertical_column.sp_tf.grad; gsrf = m_eti.vertical_column.sp_srf.grad
print(f"grad sp_tf {None if gtf is None else float(gtf):.3e}  sp_srf {None if gsrf is None else float(gsrf):.3e}  "
      f"{'OK' if (gtf is not None and gsrf is not None and gtf.abs()>0) else 'ECHEC'}")
