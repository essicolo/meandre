"""Validation du clone ET (Hydro-Québec ETP + CalculeEtr par couche) contre
Hydrotel C++ sur DELISLE UHRH1. Pilote le clone avec les theta1/2/3 et les
Tmin/Tmax JOURNALIÈRES d'Hydrotel, et compare jour par jour etp, etr1, etr2,
etr3 aux sorties d'Hydrotel.

DELISLE : EVAPOTRANSPIRATION = HYDRO-QUEBEC, BILAN = BV3C. UHRH1 sol type 2 =
loamy_sand (thetacc 0.125, thetapf 0.055, alpha 6.0). _index_autres (classes
perméables, hors eau=8 / imperméable=5,6) contributrices : feuillus, milieux
ouverts, milieux humides (les autres ont 0 % d'occupation). z=(0.1,0.4,1.0),
des=0.6, coef_assech=1.0, BETA=1.1.

  python hydrotel_clone/validate_et.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from hydrotel_clone.et import hydro_quebec_etp, calcule_etr

WDEL = r"C:/Users/parse01/documents-locaux/GitHub/hydrotel/DemoProject/DELISLE/simulation/simulation/resultat"
UHRH = 1
T = lambda x: torch.tensor(float(x), dtype=torch.float64)


def read_col(name, col=UHRH, nhead=2):
    with open(f"{WDEL}/{name}.csv", encoding="latin-1") as f:
        L = f.read().splitlines()
    out = []
    for ln in L[nhead:]:
        p = ln.split(";")
        if len(p) > col:
            try: out.append(float(p[col]))
            except ValueError: pass
    return np.array(out)


th1 = read_col("theta1"); th2 = read_col("theta2"); th3 = read_col("theta3")
tnj = read_col("tmin_jour"); txj = read_col("tmax_jour")
etp_h = read_col("etp")
e1h = read_col("etr1"); e2h = read_col("etr2"); e3h = read_col("etr3")
N = min(map(len, [th1, th2, th3, tnj, txj, etp_h, e1h, e2h, e3h]))
print(f"UHRH {UHRH} : {N} jours.")

# ── Occupation UHRH1 (occupation_sol.cla, pixels) ; total 1754 ──
TOT = 1754.0
PCT = {  # classe -> (pourcentage, jours_bp, leaf_bp, root_bp)
    "feuillus": (102 / TOT,),
    "ouverts": (118 / TOT,),
    "humides": (24 / TOT,),
}
JBP = [1, 100, 135, 166, 180, 210, 244, 270, 274, 280, 365]
LEAF = {"feuillus": [3, 4, 5, 5, 5, 5, 5, 5, 5, 5, 3],
        "ouverts": [1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 1],
        "humides": [2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 2]}
ROOT = {"feuillus": [1.5] * 11, "ouverts": [0.5] * 11, "humides": [0.75] * 11}

# Props sol couche 1 (type 2 = loamy_sand) + géométrie
THETACC, THETAPF, ALPHA = T(0.125), T(0.055), T(6.0)
Z11, Z22, Z33 = 0.1, 0.4, 1.0
DES, COEF_ASSECH = T(0.6), T(1.0)

e1c = np.zeros(N); e2c = np.zeros(N); e3c = np.zeros(N); etpc = np.zeros(N)
for i in range(N):
    jour = (np.datetime64("2020-01-01") + np.timedelta64(i, "D")).astype(object).timetuple().tm_yday
    etp_tot = hydro_quebec_etp(T(tnj[i]), T(txj[i]))   # mm
    etpc[i] = float(etp_tot)
    etp_classes, roots, leaves = [], [], []
    for c, (pct, *_ ) in PCT.items():
        etp_classes.append((etp_tot * pct) / 1000.0)    # m, par classe
        roots.append(T(float(np.interp(jour, JBP, ROOT[c]))))
        leaves.append(T(float(np.interp(jour, JBP, LEAF[c]))))
    etr1, etr2, etr3 = calcule_etr(
        T(th1[i]), T(th2[i]), T(th3[i]), etp_classes, roots, leaves,
        THETACC, THETAPF, ALPHA, Z11, Z22, Z33, DES, COEF_ASSECH)
    e1c[i] = float(etr1) * 1000.0; e2c[i] = float(etr2) * 1000.0; e3c[i] = float(etr3) * 1000.0

print("\n=== BILAN (mm) clone vs Hydrotel ===")
print(f"  etp total : clone {etpc[:N].sum():7.1f} | Hydrotel {etp_h[:N].sum():7.1f}")
for nm, c, h in (("etr1", e1c, e1h), ("etr2", e2c, e2h), ("etr3", e3c, e3h)):
    rmse = np.sqrt(np.nanmean((c[:N] - h[:N]) ** 2))
    print(f"  {nm}      : clone {c[:N].sum():7.2f} | Hydrotel {h[:N].sum():7.2f}   RMSE {rmse:.4f}")
etr_tot_c = (e1c + e2c + e3c)[:N].sum(); etr_tot_h = (e1h + e2h + e3h)[:N].sum()
print(f"  etr TOTAL : clone {etr_tot_c:7.2f} | Hydrotel {etr_tot_h:7.2f}")

print("\n=== JOUR PAR JOUR (jours 120-145, saison active) ===")
print(f"{'j':>3} {'etp':>6} {'e1_C':>6} {'e1_H':>6} {'e2_C':>6} {'e2_H':>6} {'e3_C':>6} {'e3_H':>6}")
for i in range(120, min(145, N)):
    print(f"{i:>3} {etpc[i]:6.2f} {e1c[i]:6.3f} {e1h[i]:6.3f} {e2c[i]:6.3f} {e2h[i]:6.3f} {e3c[i]:6.3f} {e3h[i]:6.3f}")
print("DONE")
