"""Validation NUMÉRIQUE du clone BV3C2 Python contre le Hydrotel C++ sur DELISLE.
Drive le clone avec l'apport + ETR d'Hydrotel (UHRH 1), part de son theta initial,
et compare la trajectoire theta1/2/3 et la production de surface, jour par jour.
Balaye la texture pour trouver celle d'Hydrotel (params bv3c uniformes connus).

  python hydrotel_clone/validate_vs_cpp.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from hydrotel_clone.bv3c2 import BV3C2Clone, SOIL_TEXTURES, make_params

DEL = "/mnt/c/Users/parse01/documents-locaux/GitHub/hydrotel/DemoProject/DELISLE/simulation/simulation/resultat"
# accès via le chemin Windows
WDEL = r"C:/Users/parse01/documents-locaux/GitHub/hydrotel/DemoProject/DELISLE/simulation/simulation/resultat"
UHRH = 1   # colonne (1-indexé) ; pente 0.026, z=(0.1,0.4,1.0), krec=1e-6

def read_hydrotel(name, col):
    """Lit une sortie Hydrotel (;-séparé, 2 lignes d'entête, UHRH en colonnes)."""
    path = f"{WDEL}/{name}.csv"
    rows = []
    with open(path, encoding="latin-1") as f:
        lines = f.read().splitlines()
    # ligne 1 = titre, ligne 2 = entête colonnes, reste = données
    for ln in lines[2:]:
        parts = ln.split(";")
        if len(parts) > col:
            try: rows.append(float(parts[col]))
            except ValueError: pass
    return np.array(rows)

ap = read_hydrotel("apport", UHRH)          # mm (pluie+fonte)
th1_h = read_hydrotel("theta1", UHRH)        # m3/m3
th2_h = read_hydrotel("theta2", UHRH)
th3_h = read_hydrotel("theta3", UHRH)
e1 = read_hydrotel("etr1", UHRH); e2 = read_hydrotel("etr2", UHRH); e3 = read_hydrotel("etr3", UHRH)
psurf_h = read_hydrotel("production_surf", UHRH)
N = min(len(ap), len(th1_h), len(e1))
print(f"UHRH {UHRH} : {N} jours. theta1 Hydrotel: min {th1_h.min():.3f} max {th1_h.max():.3f}")

# DELISLE : z=(0.1,0.4,1.0), krec=1e-6, slope=0.026, frozen=0 (pas de gel simulé)
Z = (0.1, 0.4, 1.0); KREC = 1e-6; SLOPE = 0.026
mod = BV3C2Clone(n_substep=1500)
T = lambda x: torch.tensor(float(x))

def run_texture(tex):
    p = make_params(tex, tex, tex, slope=SLOPE, fsa=1.0, fse=0.0, fsi=0.0,
                    krec=KREC, coef_recharge=0.0)
    for i in (1, 2, 3):
        p[f"z{i}"] = T(Z[i-1])
    # theta initial = celui d'Hydrotel jour 0
    t1, t2, t3 = T(th1_h[0]), T(th2_h[0]), T(th3_h[0])
    th1_c = np.zeros(N); th2_c = np.zeros(N); th3_c = np.zeros(N); ps_c = np.zeros(N)
    th1_c[0], th2_c[0], th3_c[0] = th1_h[0], th2_h[0], th3_h[0]
    for i in range(N - 1):
        ps, ph, pb, rech, (t1, t2, t3), _ = mod(
            t1, t2, t3, T(ap[i]), T(0.0), T(0.0), T(0.0), p,
            etr1_mm=T(e1[i]), etr2_mm=T(e2[i]), etr3_mm=T(e3[i]))
        th1_c[i+1], th2_c[i+1], th3_c[i+1] = float(t1), float(t2), float(t3)
        ps_c[i] = float(ps)
    rmse1 = np.sqrt(np.nanmean((th1_c - th1_h[:N])**2))
    return rmse1, th1_c, ps_c

print("\n=== balayage texture : RMSE theta1 clone vs Hydrotel ===")
best = None
for tex in SOIL_TEXTURES:
    rmse1, _, _ = run_texture(tex)
    print(f"  {tex:12} : RMSE theta1 = {rmse1:.4f}")
    if best is None or rmse1 < best[1]: best = (tex, rmse1)

tex = best[0]
print(f"\n=== meilleure texture : {tex} (RMSE {best[1]:.4f}) ===")
rmse1, th1_c, ps_c = run_texture(tex)
rmse_ps = np.sqrt(np.nanmean((ps_c[:N-1] - psurf_h[:N-1])**2))
print(f"  theta1 : RMSE {rmse1:.4f}  | clone vs Hydrotel corr {np.corrcoef(th1_c, th1_h[:N])[0,1]:.4f}")
print(f"  theta1 clone   jours 100-110 : {[round(x,3) for x in th1_c[100:110]]}")
print(f"  theta1 Hydrotel jours 100-110 : {[round(x,3) for x in th1_h[100:110]]}")
print(f"  production_surf : RMSE {rmse_ps:.4f}  (clone somme {ps_c.sum():.1f} vs Hydrotel {psurf_h[:N].sum():.1f} mm)")
print("DONE")
