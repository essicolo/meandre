"""Smoke test du squelette HydrotelColumn (Phase A) : chaîne neige → gel → ETP
→ ETR → sol (→ wetland) sur quelques nœuds/pas, vérifie l'exécution complète et
la différentiabilité. Occupation DELISLE UHRH1 (sandy_loam).

  python tests/smoke_hydrotel_column.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from meandre.vertical.hydrotel_column import HydrotelColumn, build_static_params
from hydrotel_clone.frost import n_intervalles

torch.set_default_dtype(torch.float64)
N = 3
occ = dict(feuillus=102 / 1754, ouverts=118 / 1754, humides=24 / 1754,
           urbain=1111 / 1754, routes=280 / 1754, eau=119 / 1754)
psnow, psoil, petr = build_static_params(
    N, lat=45.3, slope=0.026, orientation=7, texture="sandy_loam",
    z=(0.1, 0.4, 1.0), occupation=occ)

# rends krec apprenable pour tester le gradient bout-en-bout
krec = torch.tensor(1e-6, requires_grad=True)
psoil["krec"] = krec

col = HydrotelColumn(et_mode="mcguinness", use_frost=True)
col.set_static(psnow, psoil, petr, wetland=None, n_depth=n_intervalles(1.5, 0.05))
st = col.init_state(N, theta_init=(0.36, 0.36, 0.36))

T = lambda x: torch.full((N,), float(x))
print("forward 12 pas (froid j0-2, fonte j3-4, orage j7)")
for i in range(12):
    P = T(20.0 if i == 7 else (5.0 if i in (3, 4) else 0.5))
    tn = T(-8.0 if i < 3 else 6.0); tx = T(-1.0 if i < 3 else 16.0)
    prod, st, diag = col(P, tn, tx, T(15.0), T(2.0), T(1.0), float(20 + i), st)
    if i in (1, 4, 7):
        print(f"  j{i}: apport={float(diag['apport'].mean()):.2f} gel={float(diag['prof_gel_cm'].mean()):.1f}cm "
              f"couvert={float(diag['couvert_nival_mm'].mean()):.1f} prod={float(prod.mean()):.2f} theta1={float(st.theta1.mean()):.3f}")

loss = prod.mean()
loss.backward()
g = float(krec.grad)
assert g != 0.0 and g == g, "gradient nul ou NaN"
print(f"grad d(prod)/d(krec) = {g:.2f}  -> differentiable OK")
print("SMOKE OK")
