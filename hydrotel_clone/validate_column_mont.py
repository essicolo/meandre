"""Validation de la colonne BV3C2 sur les 1916 UHRH de MONT vs C++ Hydrotel 4.3.6
(protocole DELISLE porté à MONT ; pas de milieu humide isolé dans cette plateforme).
Référence générée le 2026-07-16 : run WSL complet 2020-2024 sur la plateforme copiée
(D:/meandre-data/quebec/mont-hy), sorties production/theta/etr par UHRH.
Le clone consomme apport/etp/etr C++ (sol isolé) et doit reproduire les productions.
  python hydrotel_clone/validate_column_mont.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from hydrotel_clone.bv3c2 import BV3C2Clone
import hydrotel_clone.validate_soil_all_uhrh as VS
from hydrotel_clone.hydrotel_params import load_project

torch.set_default_dtype(torch.float64)
MONT = "D:/meandre-data/quebec/mont-hy"
VS.DEL = MONT
VS.RES = MONT + "/simulation/simulation/resultat"

def main():
    proj = load_project(MONT)
    ids = proj["uhrh_ids"]; U = len(ids)
    print(f"{U} UHRH MONT (pas de milieu humide isolé)")
    g = lambda n: VS.read_cpp(n, U)
    apC, etpC = g("apport"), g("etp")
    e1C, e2C, e3C = g("etr1"), g("etr2"), g("etr3")
    th1C, th2C, th3C = g("theta1"), g("theta2"), g("theta3")
    psC, phC, pbC = g("production_surf"), g("production_hypo"), g("production_base")
    couv = g("couvert_nival")
    NT = min(map(len, [apC, th1C, couv]))
    print(f"NT = {NT} jours")

    P = VS.build_psoil(proj, ids)
    soil = BV3C2Clone(n_substep=2000)
    t1 = torch.tensor(th1C[0]); t2 = torch.tensor(th2C[0]); t3 = torch.tensor(th3C[0])
    z = torch.zeros(U)
    T = lambda a, i: torch.tensor(a[i])
    ps_s = np.zeros((NT, U)); ph_s = np.zeros((NT, U)); pb_s = np.zeros((NT, U))
    th1_s = np.zeros((NT, U))
    with torch.no_grad():
        for i in range(1, NT):
            surf, hyp, base, rech, (t1, t2, t3), _ = soil(
                t1, t2, t3, T(apC, i), T(etpC, i), z, T(couv, i), P,
                etr1_mm=T(e1C, i), etr2_mm=T(e2C, i), etr3_mm=T(e3C, i))
            ps_s[i] = surf.numpy(); ph_s[i] = hyp.numpy(); pb_s[i] = base.numpy()
            th1_s[i] = t1.numpy()

    sl = slice(1, NT)
    def rmse_per(a, b): return np.sqrt(np.nanmean((a[sl] - b[sl]) ** 2, axis=0))
    rps, rph, rpb = rmse_per(ps_s, psC), rmse_per(ph_s, phC), rmse_per(pb_s, pbC)
    rth = rmse_per(th1_s, th1C)
    def cum(a): return a[sl].sum(axis=0).mean()
    print(f"\n=== BILAN CUMULÉ moyen {U} UHRH (mm) clone | C++ 4.3.6 ===")
    print(f"  prod_surf : {cum(ps_s):8.1f} | {cum(psC):8.1f}")
    print(f"  prod_hypo : {cum(ph_s):8.1f} | {cum(phC):8.1f}")
    print(f"  prod_base : {cum(pb_s):8.1f} | {cum(pbC):8.1f}")
    print(f"  apport    : {'':8s} | {cum(apC):8.1f}   etr_total : {cum(e1C)+cum(e2C)+cum(e3C):8.1f}")
    print(f"\n=== RMSE par UHRH (médiane / p95 / max) ===")
    for nm, r in [("prod_surf", rps), ("prod_hypo", rph), ("prod_base", rpb), ("theta1", rth)]:
        print(f"  {nm:10s} méd {np.median(r):.4f}  p95 {np.percentile(r, 95):.4f}  max {np.max(r):.4f}")
    # pires UHRH pour investigation
    worst = np.argsort(-rps)[:5]
    print(f"\npires prod_surf : " + ", ".join(f"UHRH{ids[k]} (rmse {rps[k]:.3f})" for k in worst))
    print("DONE")

if __name__ == "__main__":
    main()
