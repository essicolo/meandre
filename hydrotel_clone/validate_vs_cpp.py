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
    p = make_params(tex, tex, tex, slope=SLOPE, fsa=0.14, fse=0.86, fsi=0.0,
                    krec=KREC, coef_recharge=0.0)
    for i in (1, 2, 3):
        p[f"z{i}"] = T(Z[i-1])
    # theta initial = celui d'Hydrotel jour 0
    t1, t2, t3 = T(th1_h[0]), T(th2_h[0]), T(th3_h[0])
    th1_c = np.zeros(N); th2_c = np.zeros(N); th3_c = np.zeros(N)
    ps_c = np.zeros(N); ph_c = np.zeros(N); pb_c = np.zeros(N)
    th1_c[0], th2_c[0], th3_c[0] = th1_h[0], th2_h[0], th3_h[0]
    for i in range(N - 1):
        ps, ph, pb, rech, (t1, t2, t3), _ = mod(
            t1, t2, t3, T(ap[i]), T(0.0), T(0.0), T(0.0), p,
            etr1_mm=T(e1[i]), etr2_mm=T(e2[i]), etr3_mm=T(e3[i]))
        th1_c[i+1], th2_c[i+1], th3_c[i+1] = float(t1), float(t2), float(t3)
        ps_c[i] = float(ps); ph_c[i] = float(ph); pb_c[i] = float(pb)
    rmse1 = np.sqrt(np.nanmean((th1_c - th1_h[:N])**2))
    return rmse1, th1_c, ps_c, th2_c, th3_c, ph_c, pb_c

print("\n=== balayage texture : RMSE theta1 clone vs Hydrotel ===")
best = None
for tex in SOIL_TEXTURES:
    rmse1 = run_texture(tex)[0]
    print(f"  {tex:12} : RMSE theta1 = {rmse1:.4f}")
    if best is None or rmse1 < best[1]: best = (tex, rmse1)

tex = best[0]
thsat = {1: SOIL_TEXTURES[tex]["thetas"], 2: SOIL_TEXTURES[tex]["thetas"], 3: SOIL_TEXTURES[tex]["thetas"]}
print(f"\n=== meilleure texture : {tex} (thetas={thsat[1]:.3f}) ===")
rmse1, th1_c, ps_c, th2_c, th3_c, ph_c, pb_c = run_texture(tex)
phyp_h = read_hydrotel("production_hypo", UHRH); pbase_h = read_hydrotel("production_base", UHRH)
print(f"== PARTITION (mm/an) clone vs Hydrotel ==")
print(f"  apport total            : {ap[:N].sum():.0f}")
print(f"  prod_surf (ruissel)     : clone {ps_c.sum():6.1f} | Hydrotel {psurf_h[:N].sum():6.1f}")
print(f"  prod_hypo (interflow)   : clone {ph_c.sum():6.1f} | Hydrotel {phyp_h[:N].sum():6.1f}")
print(f"  prod_base (baseflow)    : clone {pb_c.sum():6.1f} | Hydrotel {pbase_h[:N].sum():6.1f}")
print("\n== JOUR PAR JOUR pendant la fonte (jours 95-112) ==")
print(f"{'j':>3} {'apport':>7} {'ruis_C':>7} {'ruis_H':>7} {'hyp_C':>7} {'hyp_H':>7} {'th1_C':>6} {'th1_H':>6} {'th2_C':>6} {'th2_H':>6}")
for i in range(95, 112):
    print(f"{i:>3} {ap[i]:7.2f} {ps_c[i]:7.2f} {psurf_h[i]:7.2f} {ph_c[i]:7.2f} {phyp_h[i]:7.2f} "
          f"{th1_c[i]:6.3f} {th1_h[i]:6.3f} {th2_c[i]:6.3f} {th2_h[i]:6.3f}")
rmse_ps = np.sqrt(np.nanmean((ps_c[:N-1] - psurf_h[:N-1])**2))
print("== SATURATION : max atteint / thetas (1.0 = saturé) ==")
print(f"  couche 1 : clone {th1_c.max():.3f} ({th1_c.max()/thsat[1]:.2f})  | Hydrotel {th1_h[:N].max():.3f} ({th1_h[:N].max()/thsat[1]:.2f})")
print(f"  couche 2 : clone {th2_c.max():.3f} ({th2_c.max()/thsat[2]:.2f})  | Hydrotel {th2_h[:N].max():.3f} ({th2_h[:N].max()/thsat[2]:.2f})")
print(f"  couche 3 : clone {th3_c.max():.3f} ({th3_c.max()/thsat[3]:.2f})  | Hydrotel {th3_h[:N].max():.3f} ({th3_h[:N].max()/thsat[3]:.2f})")
print("== MOYENNES annuelles (clone vs Hydrotel) ==")
print(f"  theta1 {th1_c.mean():.3f}/{th1_h[:N].mean():.3f}  theta2 {th2_c.mean():.3f}/{th2_h[:N].mean():.3f}  theta3 {th3_c.mean():.3f}/{th3_h[:N].mean():.3f}")
print(f"== production_surf : clone {ps_c.sum():.1f} vs Hydrotel {psurf_h[:N].sum():.1f} mm (RMSE {rmse_ps:.3f}) ==")
print("DONE")
