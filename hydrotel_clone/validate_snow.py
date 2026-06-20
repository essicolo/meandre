"""Validation NUMÉRIQUE du clone de fonte degré-jour modifié contre Hydrotel C++
sur DELISLE (UHRH 1). Pilote le clone avec pluie+neige+tmin+tmax d'Hydrotel,
part du couvert_nival jour 0, et compare jour par jour le couvert nival (EEN) et
l'apport (pluie+fonte au sol) — les deux sorties d'Hydrotel.

DELISLE : FONTE DE NEIGE = DEGRE JOUR MODIFIE. UHRH1 lat 45.295, pente 0.026,
orientation 7 ; occupation (conifères=classe 1, feuillus=classes 2+3) :
conif 0 %, feuillus ~5.8 %, autres ~94.2 %. Params degre_jour_modifie.csv :
taux conif/feuillus/decouv 12/14/16, seuils 0, densité_max 466, tassement 0.1,
taux géothermique 0.5.

  python hydrotel_clone/validate_snow.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from hydrotel_clone.snow import DegreJourModifie, init_ce, init_state

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


pluie = read_col("pluie"); neige = read_col("neige")
tmin = read_col("tmin"); tmax = read_col("tmax")
cn_h = read_col("couvert_nival"); ap_h = read_col("apport")
N = min(len(pluie), len(neige), len(tmin), len(tmax), len(cn_h), len(ap_h))
print(f"UHRH {UHRH} : {N} jours. couvert_nival Hydrotel: max {cn_h[:N].max():.1f} mm, "
      f"apport total {ap_h[:N].sum():.0f} mm")

# ── Params statiques UHRH1 ──
LAT, PENTE, ORIENT = 45.29459, 0.026023, 7
PCT_CONIF, PCT_FEUIL = 0.0, 0.0581
PCT_AUTRES = 1.0 - PCT_CONIF - PCT_FEUIL
ce1, ce0 = init_ce(T(LAT), T(PENTE), T(ORIENT))
p = dict(
    lat=T(LAT), ce1=ce1, ce0=ce0,
    pct_conifers=T(PCT_CONIF), pct_feuillus=T(PCT_FEUIL), pct_autres=T(PCT_AUTRES),
    coeff_fonte_conifers=T(12.0 / 1000.0), coeff_fonte_feuillus=T(14.0 / 1000.0),
    coeff_fonte_decouver=T(16.0 / 1000.0),
    seuil_fonte_conifers=T(0.0), seuil_fonte_feuillus=T(0.0), seuil_fonte_decouver=T(0.0),
    taux_fonte_geo=T(0.5), densite_max=T(466.0), constante_tassement=T(0.1),
)

mod = DegreJourModifie(pas_de_temps=24)

# ── État initial : couvert_nival[0] réparti par classe, densité rel 0.3 ──
st = init_state(1, dtype=torch.float64)
swe0_m = cn_h[0] / 1000.0
for c, pct in (("conifers", PCT_CONIF), ("feuillus", PCT_FEUIL), ("decouver", PCT_AUTRES)):
    stock = T(swe0_m * pct)
    haut = stock / 0.3 if pct > 0 else T(0.0)   # densité rel ~0.3
    st[c] = (stock, haut.reshape(()), T(0.0), T(0.0))
    st["albedo_" + c] = T(0.0)

cn_c = np.zeros(N); ap_c = np.zeros(N)
cn_c[0] = cn_h[0]
for i in range(1, N):
    jour = T((np.datetime64("2020-01-01") + np.timedelta64(i, "D")).astype("datetime64[D]").astype(object).timetuple().tm_yday)
    apport, st = mod(T(tmin[i]), T(tmax[i]), T(pluie[i]), T(neige[i]), jour, st, p)
    ap_c[i] = float(apport)
    cn_c[i] = float(st["couvert_nival_mm"])

print("\n=== BILAN (mm) clone vs Hydrotel ===")
print(f"  apport total       : clone {ap_c[1:N].sum():7.1f} | Hydrotel {ap_h[1:N].sum():7.1f}")
print(f"  couvert max        : clone {cn_c.max():7.1f} | Hydrotel {cn_h[:N].max():7.1f}")
rmse_cn = np.sqrt(np.nanmean((cn_c[1:N] - cn_h[1:N]) ** 2))
rmse_ap = np.sqrt(np.nanmean((ap_c[1:N] - ap_h[1:N]) ** 2))
print(f"  RMSE couvert_nival : {rmse_cn:.2f} mm   RMSE apport : {rmse_ap:.2f} mm")

print("\n=== JOUR PAR JOUR (freshet, jours 60-100) ===")
print(f"{'j':>3} {'tmoy':>5} {'neige':>6} {'pluie':>6} {'cn_C':>7} {'cn_H':>7} {'ap_C':>6} {'ap_H':>6}")
for i in range(60, min(100, N)):
    tmoy = (tmin[i] + tmax[i]) / 2
    print(f"{i:>3} {tmoy:5.1f} {neige[i]:6.2f} {pluie[i]:6.2f} {cn_c[i]:7.1f} {cn_h[i]:7.1f} {ap_c[i]:6.2f} {ap_h[i]:6.2f}")
print("DONE")
