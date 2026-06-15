"""Vérifie la fidélité du BV3C2 Hydrotel : conservation de masse, loi de Campbell,
et les deux mécanismes de génération de pic qu'Hydrotel a et que méandre dissout
(porte gel = fonte sur sol gelé -> tout ruisselle ; plafond hortonien au Ks).

  python .runs/slso-od/verify_bv3c_hydrotel.py
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
import torch
from meandre.vertical.bv3c_hydrotel import BV3CHydrotel, make_params, campbell_K, campbell_psi

dev = "cpu"
mod = BV3CHydrotel()
p = make_params(device=dev)

def run(theta, apport, et, frozen, swe):
    t1, t2, t3 = (torch.tensor(float(theta[i]), device=dev) for i in range(3))
    out = mod(t1, t2, t3, torch.tensor(float(apport)), torch.tensor(float(et)),
              torch.tensor(frozen), torch.tensor(float(swe)), p)
    runoff, inter, base, rech, (n1, n2, n3), diag = out
    return runoff.item(), inter.item(), base.item(), rech.item(), (n1.item(), n2.item(), n3.item()), diag

print("=== 1. Campbell K/psi monotones et physiques (silt_loam) ===")
om = torch.linspace(0.1, 1.0, 5)
b = torch.tensor(1.0/0.234); ks = torch.tensor(0.0068); psis = torch.tensor(0.5087)
print("  omega :", [f"{x:.2f}" for x in om.tolist()])
print("  K(m/h):", [f"{x:.2e}" for x in campbell_K(om, ks, b).tolist()])
print("  psi(m):", [f"{x:+.3f}" for x in campbell_psi(om, psis, b).tolist()])

print("\n=== 2. Conservation de masse (jour calme, sol moyen) ===")
th0 = (0.30, 0.27, 0.27)
ro, it, ba, re, thn, dg = run(th0, apport=5.0, et=2.0, frozen=False, swe=0.0)
z = (p["z1"].item(), p["z2"].item(), p["z3"].item())
dstock = sum((thn[i]-th0[i])*z[i] for i in range(3))*1000.0  # mm
# bilan : apport = runoff + interflow + (baseflow+recharge) + ET + dStock
et_act = 2.0
bal = 5.0 - (ro + it + (ba+re) + et_act + dstock)
print(f"  apport=5.0  runoff={ro:.3f} inter={it:.3f} base+rech={ba+re:.3f} ET~={et_act} dStock={dstock:+.3f}")
print(f"  résidu bilan = {bal:+.4f} mm  (doit être ~0)")

print("\n=== 3. PORTE GEL : fonte sur sol gelé -> tout ruisselle ===")
for frozen, swe, lbl in [(False, 0.0, "dégelé"), (True, 5.0, "GELÉ, neige<10mm"), (True, 50.0, "gelé, neige>10mm")]:
    ro, it, ba, re, thn, dg = run(th0, apport=30.0, et=0.5, frozen=frozen, swe=swe)
    rc = ro/30.0
    print(f"  {lbl:22} apport=30  runoff={ro:6.2f}  coeff_ruiss={rc:.2f}  (pinf={dg['pinf_mm'].item():.2f})")

print("\n=== 4. PLAFOND HORTONIEN : ruissellement croît avec l'intensité ===")
print("  (silt_loam ks1=0.16 m/j ; au-delà l'excès ruisselle)")
for ap in [5.0, 50.0, 150.0, 300.0]:
    ro, it, ba, re, thn, dg = run((0.40, 0.30, 0.30), apport=ap, et=0.5, frozen=False, swe=0.0)
    print(f"  apport={ap:6.1f} mm  runoff={ro:7.2f} mm  coeff={ro/ap:.2f}")

print("\n=== 5. Différentiable : gradient passe (interflow vs theta2) ===")
# Le ruissellement hortonien ne dépend pas de theta1 (prec-min(prec,ks)),
# mais l'interflow q2=k2·sin(pente)·z2 dépend de theta2 via Campbell K.
t2 = torch.tensor(0.30, requires_grad=True)
out = mod(torch.tensor(0.30), t2, torch.tensor(0.27), torch.tensor(10.0),
          torch.tensor(1.0), torch.tensor(False), torch.tensor(0.0), p)
out[1].backward()   # interflow
g = t2.grad.item()
print(f"  d(interflow)/d(theta2) = {g:+.4f}  (fini, non-NaN: {torch.isfinite(t2.grad).item()})")
print("\nDONE")
