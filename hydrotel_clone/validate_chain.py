"""Validation BOUT-EN-BOUT de la colonne verticale clonée contre Hydrotel C++ sur
DELISLE UHRH 1 (pas de milieu humide sur cette UHRH). Chaîne, dans l'ordre exact
d'Hydrotel (BV3C2::Calcule) :

  forçage (pluie, neige, Tmin/Tmax) → FONTE NEIGE (snow.py) → apport
  → ETP Hydro-Québec (et.py) → ETR par couche (et.py, sur theta début de pas)
  → BILAN SOL BV3C2 (bv3c2.py) → production_surf/hypo/base + theta

Teste que les modules, individuellement validés, SE COMPOSENT correctement.
On part de l'état Hydrotel jour 0 (theta = bv3c.csv 0.9·thetas ; neige = couvert
jour 0) et on compare la trajectoire theta1/2/3 et la production cumulée.

  python hydrotel_clone/validate_chain.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from hydrotel_clone.snow import DegreJourModifie, init_ce, init_state
from hydrotel_clone.et import hydro_quebec_etp, calcule_etr
from hydrotel_clone.bv3c2 import BV3C2Clone, make_params

WDEL = r"C:/Users/parse01/documents-locaux/GitHub/hydrotel/DemoProject/DELISLE/simulation/simulation/resultat"
UHRH = 1
T = lambda x: torch.tensor(float(x), dtype=torch.float64)


def rc(name, c=UHRH, h=2):
    L = open(f"{WDEL}/{name}.csv", encoding="latin-1").read().splitlines()
    return np.array([float(l.split(';')[c]) for l in L[h:] if len(l.split(';')) > c])


pl, ne, tn, tx = rc("pluie"), rc("neige"), rc("tmin"), rc("tmax")
tnj, txj = rc("tmin_jour"), rc("tmax_jour")
th1h, th2h, th3h = rc("theta1"), rc("theta2"), rc("theta3")
psurf, phypo, pbase = rc("production_surf"), rc("production_hypo"), rc("production_base")
aph = rc("apport")
N = min(map(len, [pl, ne, tn, tx, tnj, txj, th1h, th2h, th3h, psurf, phypo, pbase]))
print(f"UHRH {UHRH} : {N} jours")

# ── Occupation UHRH1 (pixels /1754) ──
TOT = 1754.0
PCT_FEU, PCT_OUV, PCT_HUM = 102 / TOT, 118 / TOT, 24 / TOT
FSE = 119 / TOT                       # eau (classe 8)
FSI = (1111 + 280) / TOT              # imperméable (classes 5,6)
FSA = 1.0 - FSE - FSI                 # autre (perméable)
print(f"fractions : fsa={FSA:.3f} fse={FSE:.3f} fsi={FSI:.3f}")

# ── Cycles foliaire / racinaire ──
JBP = [1, 100, 135, 166, 180, 210, 244, 270, 274, 280, 365]
LEAF = {"feuillus": [3, 4, 5, 5, 5, 5, 5, 5, 5, 5, 3], "ouverts": [1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 1],
        "humides": [2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 2]}
ROOT = {"feuillus": [1.5] * 11, "ouverts": [0.5] * 11, "humides": [0.75] * 11}
PCTC = {"feuillus": PCT_FEU, "ouverts": PCT_OUV, "humides": PCT_HUM}

# ── Params sol UHRH1 : sandy_loam (empreinte theta, cf validate_vs_cpp : type 2
# du .cla = sandy_loam, 3e texture du .sol → décalage d'indice). thetacc/thetapf/
# alpha de sandy_loam (l'ETR ne distingue pas la texture en période humide, stress
# clampé à 1). z=(0.1,0.4,1.0), krec 1e-6, cin 0.3. ──
THETACC, THETAPF, ALPHA = T(0.207), T(0.095), T(4.5)
DES, COEF_ASSECH = T(0.6), T(1.0)
psoil = make_params("sandy_loam", "sandy_loam", "sandy_loam", slope=0.026023,
                    fsa=FSA, fse=FSE, fsi=FSI, krec=1e-6, cin=0.3, coef_recharge=0.0)
for i in (1, 2, 3):
    psoil[f"z{i}"] = T([0.1, 0.4, 1.0][i - 1])

snow = DegreJourModifie(24)
soil = BV3C2Clone(n_substep=1500)
ce1, ce0 = init_ce(T(45.29459), T(0.026023), T(7))
psnow = dict(lat=T(45.29459), ce1=ce1, ce0=ce0, pct_conifers=T(0.0),
             pct_feuillus=T(PCT_FEU), pct_autres=T(1.0 - 0.0 - PCT_FEU),
             coeff_fonte_conifers=T(.012), coeff_fonte_feuillus=T(.014), coeff_fonte_decouver=T(.016),
             seuil_fonte_conifers=T(0.0), seuil_fonte_feuillus=T(0.0), seuil_fonte_decouver=T(0.0),
             taux_fonte_geo=T(0.5), densite_max=T(466.0), constante_tassement=T(0.1))

# états initiaux
sst = init_state(1, dtype=torch.float64)
swe0 = rc("couvert_nival")[0] / 1000.0
for c, pct in (("conifers", 0.0), ("feuillus", PCT_FEU), ("decouver", 1.0 - PCT_FEU)):
    s = T(swe0 * pct); sst[c] = (s, (s / 0.3 if pct > 0 else T(0.0)).reshape(()), T(0.0), T(0.0)); sst["albedo_" + c] = T(0.0)
t1, t2, t3 = T(th1h[0]), T(th2h[0]), T(th3h[0])

ps_c = np.zeros(N); ph_c = np.zeros(N); pb_c = np.zeros(N); ap_c = np.zeros(N)
t1c = np.zeros(N); t2c = np.zeros(N); t3c = np.zeros(N)
t1c[0], t2c[0], t3c[0] = th1h[0], th2h[0], th3h[0]
for i in range(1, N):
    jour = (np.datetime64("2020-01-01") + np.timedelta64(i, "D")).astype(object).timetuple().tm_yday
    # 1. neige → apport
    apport, sst = snow(T(tn[i]), T(tx[i]), T(pl[i]), T(ne[i]), T(jour), sst, psnow)
    ap_c[i] = float(apport)
    # 2. ETP
    etp_tot = hydro_quebec_etp(T(tnj[i]), T(txj[i]))
    # 3. ETR (sur theta début de pas)
    etpc = [etp_tot * PCTC[c] / 1000.0 for c in ("feuillus", "ouverts", "humides")]
    roots = [T(float(np.interp(jour, JBP, ROOT[c]))) for c in ("feuillus", "ouverts", "humides")]
    leaves = [T(float(np.interp(jour, JBP, LEAF[c]))) for c in ("feuillus", "ouverts", "humides")]
    e1, e2, e3 = calcule_etr(t1, t2, t3, etpc, roots, leaves, THETACC, THETAPF, ALPHA,
                             0.1, 0.4, 1.0, DES, COEF_ASSECH)
    # 4. sol
    ps, ph, pb, rech, (t1, t2, t3), _ = soil(
        t1, t2, t3, apport, etp_tot, T(0.0), T(swe0 * 0 + float(sst["couvert_nival_mm"])), psoil,
        etr1_mm=e1 * 1000.0, etr2_mm=e2 * 1000.0, etr3_mm=e3 * 1000.0)
    ps_c[i] = float(ps); ph_c[i] = float(ph); pb_c[i] = float(pb)
    t1c[i], t2c[i], t3c[i] = float(t1), float(t2), float(t3)

sl = slice(1, N)
def rmse(a, b): return float(np.sqrt(np.nanmean((a[sl] - b[sl]) ** 2)))
print("\n=== BILAN cumulé (mm) clone vs Hydrotel ===")
print(f"  apport    : {ap_c[sl].sum():7.1f} | {aph[sl].sum():7.1f}")
print(f"  prod_surf : {ps_c[sl].sum():7.1f} | {psurf[sl].sum():7.1f}")
print(f"  prod_hypo : {ph_c[sl].sum():7.1f} | {phypo[sl].sum():7.1f}")
print(f"  prod_base : {pb_c[sl].sum():7.1f} | {pbase[sl].sum():7.1f}")
print(f"\n=== theta trajectoire (RMSE) ===")
print(f"  theta1 {rmse(t1c, th1h):.4f}  theta2 {rmse(t2c, th2h):.4f}  theta3 {rmse(t3c, th3h):.4f}")
print(f"  prod_surf RMSE {rmse(ps_c, psurf):.3f}  prod_hypo {rmse(ph_c, phypo):.4f}  prod_base {rmse(pb_c, pbase):.5f}")

print("\n=== JOUR PAR JOUR (freshet 60-80) ===")
print(f"{'j':>3} {'ap_C':>6} {'ap_H':>6} {'pS_C':>6} {'pS_H':>6} {'t1_C':>6} {'t1_H':>6} {'t2_C':>6} {'t2_H':>6}")
for i in range(60, min(81, N)):
    print(f"{i:>3} {ap_c[i]:6.2f} {aph[i]:6.2f} {ps_c[i]:6.2f} {psurf[i]:6.2f} {t1c[i]:6.3f} {th1h[i]:6.3f} {t2c[i]:6.3f} {th2h[i]:6.3f}")
print("DONE")
