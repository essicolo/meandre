"""Validation du clone milieu humide isolé contre Hydrotel C++ sur DELISLE
(UHRH 2, le premier milieu humide avec sauvegarde=1).

Le réservoir a une mémoire d'état (volume initial lu d'un fichier d'état non
disponible), donc on valide les ÉQUATIONS DU PAS en one-step-ahead : pour chaque
jour i, on part du Wetvol enregistré par Hydrotel au jour i−1, on applique un
pas, et on compare Wetvol/Wetsep/Wetflwo/Wetprod au jour i. La production `prod`
entrant (issue du sol, validé par ailleurs) est reconstruite depuis le Wetflwi
enregistré ; les autres flux (seepage depuis la surface, bilan de volume,
débordement, wetprod) sont des vérifications indépendantes.

  python hydrotel_clone/validate_wetland.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from hydrotel_clone.milieu_humide import init_wetland_geom, calcul_milieu_humide_isole

WDEL = r"C:/Users/parse01/documents-locaux/GitHub/hydrotel/DemoProject/DELISLE/simulation/simulation/resultat"
UHRH = 2
T = lambda x: torch.tensor(float(x), dtype=torch.float64)

# ── Lecture wetland_isole.csv, filtré sur IdUhrh==UHRH ──
rows = []
with open(f"{WDEL}/wetland_isole.csv", encoding="latin-1") as f:
    for ln in f.read().splitlines()[1:]:
        p = ln.split(";")
        if len(p) >= 12 and p[0].strip() == str(UHRH):
            rows.append([float(x) for x in p[:12]])
arr = np.array(rows)
# cols: 0 id,1 an,2 mois,3 jour,4 heure,5 Apport,6 Evp,7 Wetsep,8 Wetvol,9 Wetflwi,10 Wetflwo,11 Wetprod
Apport, Evp, Wetsep, Wetvol, Wetflwi, Wetflwo, Wetprod = (arr[:, i] for i in range(5, 12))
N = len(arr)
print(f"UHRH {UHRH} : {N} pas. Wetvol Hydrotel: {Wetvol.min():.0f}..{Wetvol.max():.0f} m3")

# ── Params UHRH2 (milieux_humides_isoles.csv) ──
UHRH_A, WET_A, WET_DRA_FR = 0.7972, 0.0448, 0.437531
FRAC, WETDNOR, WETDMAX = 0.8, 0.2, 0.3
KSAT_BS, C_EV, C_PROD = 0.5, 0.6, 10.0
HRU_HA = UHRH_A * 100.0
A, B, WETNVOL, WETMXVOL = init_wetland_geom(WET_A, WETDMAX, FRAC, WETDNOR)
print(f"A={A:.5f} B={B:.2f} wetnvol={WETNVOL:.1f} wetmxvol={WETMXVOL:.1f} m3")

vol_c = np.zeros(N); sep_c = np.zeros(N); flwo_c = np.zeros(N); prod_c = np.zeros(N)
for i in range(1, N):
    vol0 = T(Wetvol[i - 1])
    # surface au début du pas, pour reconstruire prod depuis Wetflwi enregistré
    wetsa = B * float(vol0) ** A / 10000.0
    denom = 10.0 * (HRU_HA * WET_DRA_FR - wetsa)
    prod_mm = Wetflwi[i] / denom if denom != 0 else 0.0
    v, sep, flwi, flwo, wprod = calcul_milieu_humide_isole(
        vol0, T(Apport[i]), T(Evp[i]), T(prod_mm), HRU_HA, WET_DRA_FR,
        A, B, WETNVOL, WETMXVOL, KSAT_BS, C_EV, C_PROD, pdt=24)
    vol_c[i] = float(v); sep_c[i] = float(sep); flwo_c[i] = float(flwo); prod_c[i] = float(wprod)

sl = slice(1, N)
def rmse(a, b): return np.sqrt(np.nanmean((a[sl] - b[sl]) ** 2))
print("\n=== one-step-ahead clone vs Hydrotel (RMSE) ===")
print(f"  Wetvol  (m3) : RMSE {rmse(vol_c, Wetvol):.4f}   (échelle ~{Wetvol.mean():.0f})")
print(f"  Wetsep  (m3) : RMSE {rmse(sep_c, Wetsep):.4f}")
print(f"  Wetflwo (m3) : RMSE {rmse(flwo_c, Wetflwo):.4f}")
print(f"  Wetprod (mm) : RMSE {rmse(prod_c, Wetprod):.6f}")

print("\n=== JOUR PAR JOUR (pas 1-15) ===")
print(f"{'i':>3} {'vol_C':>10} {'vol_H':>10} {'sep_C':>8} {'sep_H':>8} {'flwo_C':>8} {'flwo_H':>8} {'prod_C':>7} {'prod_H':>7}")
for i in range(1, min(16, N)):
    print(f"{i:>3} {vol_c[i]:10.2f} {Wetvol[i]:10.2f} {sep_c[i]:8.2f} {Wetsep[i]:8.2f} "
          f"{flwo_c[i]:8.2f} {Wetflwo[i]:8.2f} {prod_c[i]:7.3f} {Wetprod[i]:7.3f}")
print("DONE")
