"""Validation de la CLASSE meandre `HydrotelColumn` (meandre/vertical/hydrotel_column.py)
contre Hydrotel C++ sur DELISLE UHRH1 — le maillon manquant de l'échelle.

`validate_chain.py` valide la chaîne `hydrotel_clone/` PURE (appels directs snow/et/bv3c2)
vs C++. Mais le modèle tourne `HydrotelColumn`, une RÉ-implémentation intégrée PyTorch
qui ré-orchestre ces mêmes modules. On n'avait jamais vérifié qu'elle reproduit la chaîne
validée sur des inputs identiques. Ce harnais comble ça : MÊMES forçage, MÊMES params,
routés à travers `HydrotelColumn.forward`, comparés à C++.

  - et_mode="hydro_quebec" (= validate_chain), use_frost=False (gel forcé 0 comme C++ ici),
    soil_n_substep=1500 (= validate_chain).
  - Split pluie/neige court-circuité : on injecte les pluie/neige PRÉ-SPLITÉS de C++
    (comme validate_chain) pour isoler l'orchestration du sol de la règle de split (TODO).

Verdict :
  prod_surf ≈ 375 (= validate_chain / C++ 350) → la classe est FIDÈLE, le déficit KGE
  vient du feeding NeRF (Z/theta/occupation), pas de l'assemblage.
  prod_surf très en-dessous → bug d'orchestration DANS la classe, localisé ici.

  python hydrotel_clone/validate_hydrotel_column.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from hydrotel_clone.snow import init_ce
from hydrotel_clone.bv3c2 import make_params
from meandre.vertical.hydrotel_column import HydrotelColumn

torch.set_default_dtype(torch.float64)
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
print(f"UHRH {UHRH} : {N} jours  (HydrotelColumn classe vs C++)")

# ── Occupation UHRH1 (idem validate_chain) ──
TOT = 1754.0
PCT_FEU, PCT_OUV, PCT_HUM = 102 / TOT, 118 / TOT, 24 / TOT
FSE = 119 / TOT
FSI = (1111 + 280) / TOT
FSA = 1.0 - FSE - FSI
print(f"fractions : fsa={FSA:.3f} fse={FSE:.3f} fsi={FSI:.3f}")

JBP = [1, 100, 135, 166, 180, 210, 244, 270, 274, 280, 365]
LEAF = {"feuillus": [3, 4, 5, 5, 5, 5, 5, 5, 5, 5, 3], "ouverts": [1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 1],
        "humides": [2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 2]}
ROOT = {"feuillus": [1.5] * 11, "ouverts": [0.5] * 11, "humides": [0.75] * 11}

# ── Params sol/etr IDENTIQUES à validate_chain ──
THETACC, THETAPF, ALPHA = T(0.207), T(0.095), T(4.5)
DES, COEF_ASSECH = T(0.6), T(1.0)
psoil = make_params("sandy_loam", "sandy_loam", "sandy_loam", slope=0.026023,
                    fsa=FSA, fse=FSE, fsi=FSI, krec=1e-6, cin=0.3, coef_recharge=0.0)
for i in (1, 2, 3):
    psoil[f"z{i}"] = T([0.1, 0.4, 1.0][i - 1])

ce1, ce0 = init_ce(T(45.29459), T(0.026023), T(7))
psnow = dict(lat=T(45.29459), ce1=ce1, ce0=ce0, pct_conifers=T(0.0),
             pct_feuillus=T(PCT_FEU), pct_autres=T(1.0 - 0.0 - PCT_FEU),
             coeff_fonte_conifers=T(.012), coeff_fonte_feuillus=T(.014), coeff_fonte_decouver=T(.016),
             seuil_fonte_conifers=T(0.0), seuil_fonte_feuillus=T(0.0), seuil_fonte_decouver=T(0.0),
             taux_fonte_geo=T(0.5), densite_max=T(466.0), constante_tassement=T(0.1))

# p_etr au format attendu par HydrotelColumn.forward (pe["classes"], z11/22/33, ...)
petr = dict(thetacc=THETACC, thetapf=THETAPF, alpha=ALPHA, des=DES, coef_assech=COEF_ASSECH,
            z11=T(0.1), z22=T(0.4), z33=T(1.0),
            classes=[(PCT_FEU, JBP, LEAF["feuillus"], ROOT["feuillus"]),
                     (PCT_OUV, JBP, LEAF["ouverts"], ROOT["ouverts"]),
                     (PCT_HUM, JBP, LEAF["humides"], ROOT["humides"])])

# ── La classe sous test ──
col = HydrotelColumn(et_mode="hydro_quebec", use_frost=False, soil_n_substep=1500)
col.set_static(psnow, psoil, petr, wetland=None, n_depth=1)
st = col.init_state(1, theta_init=(th1h[0], th2h[0], th3h[0]))

# état neige initial (idem validate_chain)
swe0 = rc("couvert_nival")[0] / 1000.0
for c, pct in (("conifers", 0.0), ("feuillus", PCT_FEU), ("decouver", 1.0 - PCT_FEU)):
    s = T(swe0 * pct)
    st.snow[c] = (s, (s / 0.3 if pct > 0 else T(0.0)).reshape(()), T(0.0), T(0.0))
    st.snow["albedo_" + c] = T(0.0)

# ── Court-circuit du split : on injecte les pluie/neige PRÉ-SPLITÉS de C++ ──
_cur = {"i": 0}
col._split_precip = lambda P, tmin, tmax: (
    torch.full_like(P, float(pl[_cur["i"]])), torch.full_like(P, float(ne[_cur["i"]])))

ps_c = np.zeros(N); ph_c = np.zeros(N); pb_c = np.zeros(N); ap_c = np.zeros(N)
t1c = np.zeros(N); t2c = np.zeros(N); t3c = np.zeros(N)
t1c[0], t2c[0], t3c[0] = th1h[0], th2h[0], th3h[0]
v = lambda x: torch.full((1,), float(x))
for i in range(1, N):
    jour = (np.datetime64("2020-01-01") + np.timedelta64(i, "D")).astype(object).timetuple().tm_yday
    _cur["i"] = i
    prod, st, diag = col.forward(v(pl[i] + ne[i]), v(tn[i]), v(tx[i]),
                                 v(0.0), v(0.0), v(0.0), jour, st,
                                 tmin_j=v(tnj[i]), tmax_j=v(txj[i]))
    ap_c[i] = float(diag["apport"])
    ps_c[i] = float(diag["prod_surf"]); ph_c[i] = float(diag["prod_hypo"]); pb_c[i] = float(diag["prod_base"])
    t1c[i], t2c[i], t3c[i] = float(st.theta1), float(st.theta2), float(st.theta3)

sl = slice(1, N)
def rmse(a, b): return float(np.sqrt(np.nanmean((a[sl] - b[sl]) ** 2)))
print("\n=== BILAN cumulé (mm) HydrotelColumn classe vs Hydrotel C++ ===")
print(f"  apport    : {ap_c[sl].sum():7.1f} | {aph[sl].sum():7.1f}")
print(f"  prod_surf : {ps_c[sl].sum():7.1f} | {psurf[sl].sum():7.1f}   (validate_chain clone: 375.3)")
print(f"  prod_hypo : {ph_c[sl].sum():7.1f} | {phypo[sl].sum():7.1f}   (validate_chain clone:  45.7)")
print(f"  prod_base : {pb_c[sl].sum():7.1f} | {pbase[sl].sum():7.1f}")
print(f"\n=== theta trajectoire (RMSE) ===")
print(f"  theta1 {rmse(t1c, th1h):.4f}  theta2 {rmse(t2c, th2h):.4f}  theta3 {rmse(t3c, th3h):.4f}")
print(f"  prod_surf RMSE {rmse(ps_c, psurf):.3f}  prod_hypo {rmse(ph_c, phypo):.4f}")

print("\n=== JOUR PAR JOUR (freshet 60-80) ===")
print(f"{'j':>3} {'ap_C':>6} {'ap_H':>6} {'pS_C':>6} {'pS_H':>6} {'t1_C':>6} {'t1_H':>6}")
for i in range(60, min(81, N)):
    print(f"{i:>3} {ap_c[i]:6.2f} {aph[i]:6.2f} {ps_c[i]:6.2f} {psurf[i]:6.2f} {t1c[i]:6.3f} {th1h[i]:6.3f}")
print("DONE")
