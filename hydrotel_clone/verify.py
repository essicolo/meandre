"""Vérifie le clone BV3C2 : conservation de masse, et surtout qu'il SATURE et
GÉNÈRE des pics au pas journalier, là où le sol de méandre échoue (Se reste à
0.63 en crue, jamais saturé). Petit test, secondes, aucun training.

  python hydrotel_clone/verify.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from hydrotel_clone.bv3c2 import BV3C2Clone, make_params, EPAISSEUR

dev = "cpu"
mod = BV3C2Clone(n_substep=48)
p = make_params(slope=0.04, fsa=0.90, fse=0.05, fsi=0.05, device=dev)
z1, z2, z3 = EPAISSEUR

# ── Forçage annuel boréal (apport = pluie+fonte au sol, mm/j ; gel hiver) ──
N = 365
rng = np.random.default_rng(0)
apport = np.zeros(N); etp = np.zeros(N); frozen_cm = np.zeros(N); swe = np.zeros(N)
# hiver j0-90 : neige s'accumule (apport sol ~0), sol gelé, couvert nival
frozen_cm[:110] = 30.0; swe[:100] = 200.0
# freshet j95-120 : grosse fonte sur sol encore gelé début, puis dégèle
apport[95:120] = rng.uniform(15, 40, 25); swe[95:105] = 50.0; swe[105:115] = 5.0
frozen_cm[108:] = 0.0
# été j150-270 : orages + ETP
storm = rng.choice(np.arange(150, 280), 18, replace=False); apport[storm] = rng.uniform(8, 55, 18)
etp[150:280] = 3.0
apport[290:330] += rng.uniform(0, 6, 40)

T = lambda x: torch.tensor(float(x))
t1 = T(0.30); t2 = T(0.27); t3 = T(0.27)
ro = np.zeros(N); sat = np.zeros(N); pinf_a = np.zeros(N)
in_tot = out_tot = 0.0; dstock0 = (t1.item()*z1 + t2.item()*z2 + t3.item()*z3)
prods = np.zeros(N)
for i in range(N):
    ps, ph, pb, rech, (t1, t2, t3), diag = mod(
        t1, t2, t3, T(apport[i]), T(etp[i]), T(frozen_cm[i]), T(swe[i]), p)
    ro[i] = ps.item()        # prod_surf (mm) = ruissellement de surface
    prods[i] = ps.item() + ph.item() + pb.item()
    sat[i] = diag["sat_t1"].item()
    in_tot += apport[i]; out_tot += ps.item() + ph.item() + pb.item() + rech.item()

dstock = (t1.item()*z1 + t2.item()*z2 + t3.item()*z3) - dstock0
resid = in_tot - out_tot - dstock*1000 - etp.sum()  # approx (ETP pas tout consommé)
print("=== Conservation (approx, ETP non bornée ici) ===")
print(f"  apport {in_tot:.0f}  sorties {out_tot:.0f}  dStock {dstock*1000:+.0f} mm")

fresh = slice(95, 125); summer = slice(150, 285)
print("=== SATURATION couche 1 (theta1/thetas, 1.0 = saturé) ===")
print(f"  moyenne année        : {np.median(sat):.2f}")
print(f"  pendant la FRESHET   : {np.median(sat[fresh]):.2f}  (méandre restait à ~0.73)")
print(f"  max sur l'année      : {sat.max():.2f}")
print("=== RUISSELLEMENT de surface (prod_surf, mm/j) ===")
print(f"  freshet : apport {apport[fresh].sum():.0f}  runoff {ro[fresh].sum():.0f}  coeff {ro[fresh].sum()/max(apport[fresh].sum(),1):.2f}")
print(f"  été     : apport {apport[summer].sum():.0f}  runoff {ro[summer].sum():.0f}  coeff {ro[summer].sum()/max(apport[summer].sum(),1):.2f}")
print(f"  pic journalier max   : {ro.max():.1f} mm/j")

# gradient
t1g = torch.tensor(0.40, requires_grad=True)
out = mod(t1g, T(0.30), T(0.27), T(30.0), T(1.0), T(0.0), T(0.0), p)
out[0].backward()
print(f"=== Différentiable : d(prod_surf)/d(theta1) = {t1g.grad.item():+.3f} (fini {torch.isfinite(t1g.grad).item()}) ===")
print("DONE")
