"""Validation MULTI-UHRH du clone BV3C2 vs Hydrotel C++ (DELISLE). Pour chaque
UHRH (pente lue dans uhrh.csv), balaye la texture (theta1), ajuste la fraction
perméable fsa depuis l'interflow d'Hydrotel (fsa = prod_hypo_H / lhyp), puis
vérifie que theta ET prod_surf collent. Si ça tient sur des UHRH variés
(surface-dominés et interflow-dominés), le clone est robustement fidèle.

  python hydrotel_clone/validate_multi.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from hydrotel_clone.bv3c2 import BV3C2Clone, SOIL_TEXTURES, make_params

ROOT = r"C:/Users/parse01/documents-locaux/GitHub/hydrotel/DemoProject/DELISLE"
WDEL = f"{ROOT}/simulation/simulation/resultat"
Z = (0.1, 0.4, 1.0); KREC = 1e-6
mod = BV3C2Clone(n_substep=1500)
T = lambda x: torch.tensor(float(x))

def rd(name, col):
    rows = []
    for ln in open(f"{WDEL}/{name}.csv", encoding="latin-1").read().splitlines()[2:]:
        p = ln.split(";")
        if len(p) > col:
            try: rows.append(float(p[col]))
            except ValueError: pass
    return np.array(rows)

# pentes par UHRH (uhrh.csv : col 3 = PENTE MOYENNE ratio)
slopes = {}
for ln in open(f"{ROOT}/physitel/uhrh.csv", encoding="latin-1").read().splitlines():
    p = ln.split(";")
    if len(p) >= 4 and p[0].strip().isdigit():
        slopes[int(p[0])] = float(p[3])

def run(uhrh, tex, fsa, fse):
    ap = rd("apport", uhrh); th1h = rd("theta1", uhrh); th2h = rd("theta2", uhrh); th3h = rd("theta3", uhrh)
    e1 = rd("etr1", uhrh); e2 = rd("etr2", uhrh); e3 = rd("etr3", uhrh)
    N = min(len(ap), len(th1h), len(e1))
    p = make_params(tex, tex, tex, slope=slopes[uhrh], fsa=fsa, fse=fse, fsi=0.0, krec=KREC)
    for i in (1, 2, 3): p[f"z{i}"] = T(Z[i-1])
    t1, t2, t3 = T(th1h[0]), T(th2h[0]), T(th3h[0])
    th1c = np.zeros(N); psc = np.zeros(N); phc = np.zeros(N)
    th1c[0] = th1h[0]
    for i in range(N - 1):
        ps, ph, pb, rech, (t1, t2, t3), _ = mod(t1, t2, t3, T(ap[i]), T(0.0), T(0.0), T(0.0), p,
                                                etr1_mm=T(e1[i]), etr2_mm=T(e2[i]), etr3_mm=T(e3[i]))
        th1c[i+1] = float(t1); psc[i] = float(ps); phc[i] = float(ph)
    rmse1 = np.sqrt(np.nanmean((th1c - th1h[:N])**2))
    return rmse1, psc.sum(), phc.sum(), N

print(f"{'UHRH':>4} {'pente':>6} {'tex':>10} {'RMSE_th1':>8} | {'surf_C':>7} {'surf_H':>7} | {'hyp_C':>6} {'hyp_H':>6} | {'fsa':>5}")
for uhrh in [1, 2, 5, 10, 20, 40, 60]:
    surfH = rd("production_surf", uhrh).sum(); hypH = rd("production_hypo", uhrh).sum()
    # texture : meilleure par RMSE theta1 (fsa=1 ici, theta indep. de fsa)
    best = None
    for tex in SOIL_TEXTURES:
        r = run(uhrh, tex, 1.0, 0.0)
        if best is None or r[0] < best[1]: best = (tex, r[0], r[2])   # tex, rmse, lhyp(=hyp a fsa=1)
    tex, rmse1, lhyp = best
    fsa = float(np.clip(hypH / max(lhyp, 1e-6), 0.0, 1.0)); fse = 1.0 - fsa  # ajuste fsa depuis interflow H
    _, surfC, hypC, _ = run(uhrh, tex, fsa, fse)
    print(f"{uhrh:>4} {slopes[uhrh]:6.3f} {tex:>10} {rmse1:8.4f} | {surfC:7.1f} {surfH:7.1f} | {hypC:6.1f} {hypH:6.1f} | {fsa:5.2f}")
print("DONE")
