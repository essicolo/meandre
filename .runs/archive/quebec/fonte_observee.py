"""La fonte, apprise DIRECTEMENT des observations plutot que devinee par formulation.

Idee d'Essi (2026-08-28) : « la fonte devrait neanmoins etre captee par la meteo et le
terrain. Penses-tu qu'elle pourrait etre predite avec un MLP ou meme un ridge ? ».

Renversement de methode. Depuis trois jours on devine une formulation (degre-jour,
modulation saisonniere, ETI, terme de vent), on la fait tourner dans le modele entier, et
on la juge sur un STOCK residuel -- qui ne montre que la difference cumulee entre
accumulation et fonte, donc ne peut refuter aucune hypothese sur l'une ou l'autre (R53).

Or la fonte est OBSERVABLE. CanSWE donne des series de masse au sol a intervalle median
d'UN JOUR (96 % des intervalles a 3 jours ou moins sur SAGU, 41 sites, 25 ans). Une
baisse de masse un jour SANS PRECIPITATION est une ablation mesuree. On a donc une cible
supervisee, et on peut ajuster la fonction fonte sur la meteo et le terrain sans passer
par le modele.

TROIS FAMILLES, trois questions distinctes :
  - RIDGE : ses coefficients SONT les parametres de Pellicciotti. Celui sur (T - seuil)
    est tf, celui sur le rayonnement absorbe est srf. On peut donc comparer ce que disent
    les donnees aux valeurs de litterature, et savoir si l'ETI est la bonne FORME.
  - GAM (splines) : les non-linearites restent lisibles courbe par courbe. Repond a
    « la reponse a la temperature est-elle vraiment lineaire ? ».
  - FORET : capte les interactions, mais n'extrapole pas. Sert de PLAFOND de performance
    atteignable, pas de modele a integrer -- le projet vit de l'extrapolation hors
    echantillon (scenarios climatiques), qu'une foret ne peut pas fournir.

    python .runs/quebec/fonte_observee.py sagu
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import xarray as xr

from meandre.data.basin_cache import BasinCache
from meandre.data.hydrotel_calib import load_occupation_sol
from meandre.utils import paths as _paths

REG = (sys.argv[1] if len(sys.argv) > 1 else "sagu").lower()
bc = BasinCache(_paths.data_path("quebec", f"{REG}.duckdb"))
h = bc.load(device="cpu")
mes, sites = bc.load_canswe("2000-01-01", "2024-12-31")

d = xr.open_dataset(_paths.data_path("quebec", f"forcing-{REG}-hyb.nc"))
F = d["forcing"].values
times = pd.DatetimeIndex(d["time"].values); d.close()
fsw = _paths.data_path("quebec", f"forcing-{REG}-swin.nc")
SW = None
if os.path.exists(fsw):
    d2 = xr.open_dataset(fsw); SW = d2["forcing"].values[:, :, 0]; d2.close()

# ── cible : ablation quotidienne mesuree, jours SANS precipitation ───────────
m = mes[mes.swe_mm.notna()].copy()
m["date"] = pd.to_datetime(m.date)
m = m.sort_values(["swe_station_id", "date"])
m["dswe"] = m.groupby("swe_station_id").swe_mm.diff()
m["dj"] = m.groupby("swe_station_id").date.diff().dt.days
m = m[(m.dj == 1) & m.dswe.notna()]
pos = pd.Series(np.arange(len(times)), index=times.normalize())
m["t"] = pos.reindex(pd.DatetimeIndex(m.date).normalize()).to_numpy()
m = m[np.isfinite(m.t)]
m["t"] = m.t.astype(int)
m["node"] = m.node_idx.astype(int)

P = F[m.t, m.node, 0]
# SANS PRECIPITATION : sinon la variation de masse melange accumulation et ablation, et
# la cible n'est plus une fonte. C'est le filtre qui rend l'exercice honnete.
garde = P < 0.5
m = m[garde]
tmin, tmax = F[m.t, m.node, 1], F[m.t, m.node, 2]
u2, ea = F[m.t, m.node, 4], F[m.t, m.node, 5]
sw = SW[m.t, m.node] if SW is not None else np.zeros(len(m))
tmoy = (tmin + tmax) / 2.0

# ablation = baisse de masse (mm/j). Les hausses sans precipitation sont du bruit de
# mesure ou de la redistribution par le vent : on les garde a zero plutot que de les
# jeter, sinon on biaiserait la cible vers le haut.
y = np.clip(-m.dswe.to_numpy(), 0.0, None)

occ = load_occupation_sol(f"{_paths.PLATFORMS_ROOT}/LN24HA/{REG.upper()}_LN24HA_2020",
                          h["node_ids"], device="cpu")
def att(k):
    v = occ.get(k)
    return v.numpy()[m.node] if v is not None else np.zeros(len(m))

lat = h["node_coords"].numpy()[m.node, 1]
doy = pd.DatetimeIndex(m.date).dayofyear.to_numpy()
X = pd.DataFrame({
    "t_positif": np.clip(tmoy, 0.0, None),      # terme degre-jour (tf)
    "sw_absorbe": sw * 0.4,                      # rayonnement absorbe (srf), albedo 0.6
    "tmoy": tmoy, "tmax": tmax, "amplitude": tmax - tmin,
    "vent": u2, "vapeur": ea, "swe": m.swe_mm.to_numpy(),
    "foret": att("f_forest_raw"), "conifere": att("f_forest_conifer_raw"),
    "agricole": att("f_agriculture_raw"), "humide": att("f_wetland_raw"),
    "lat": lat, "jour": np.sin(2 * np.pi * doy / 365.25),
})
print(f"[{REG}] {len(X):,} jours-site sans precipitation | ablation moyenne {y.mean():.2f} mm/j "
      f"| part de jours a fonte nulle {100 * (y == 0).mean():.0f} %")

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

# VALIDATION PAR SITE, pas aleatoire : deux jours consecutifs au meme site sont
# quasi identiques, un decoupage aleatoire donnerait un R2 fantome.
grp = m.swe_station_id.astype(str).to_numpy()
cv = GroupKFold(n_splits=5)

def evalue(nom, modele, cols, standardise=True):
    Xc = X[cols].to_numpy()
    r2 = []
    for tr, te in cv.split(Xc, y, grp):
        xa, xb = Xc[tr], Xc[te]
        if standardise:
            sc = StandardScaler().fit(xa); xa, xb = sc.transform(xa), sc.transform(xb)
        mo = modele()
        mo.fit(xa, y[tr])
        r2.append(r2_score(y[te], mo.predict(xb)))
    print(f"  {nom:38s} R2 = {np.mean(r2):.3f} (+/- {np.std(r2):.3f})")
    return np.mean(r2)

print("\n  validation croisee PAR SITE (5 groupes)")
evalue("degre-jour seul (T positif)", lambda: Ridge(1.0), ["t_positif"])
evalue("ETI (T positif + rayonnement)", lambda: Ridge(1.0), ["t_positif", "sw_absorbe"])
evalue("ETI + vent + vapeur", lambda: Ridge(1.0), ["t_positif", "sw_absorbe", "vent", "vapeur"])
evalue("ridge, toutes covariables", lambda: Ridge(1.0), list(X.columns))
evalue("foret (plafond, n'extrapole pas)",
       lambda: RandomForestRegressor(n_estimators=120, min_samples_leaf=20, n_jobs=-1,
                                     random_state=0), list(X.columns), standardise=False)

# ── coefficients ETI en unites physiques, comparables a Pellicciotti ─────────
print("\n  coefficients de l'ajustement ETI, en unites physiques")
xa = X[["t_positif", "sw_absorbe"]].to_numpy()
r = Ridge(1.0).fit(xa, y)
print(f"    tf  = {r.coef_[0]:.3f} mm/j/degC   (litterature Pellicciotti ~1.2 e-3 m/j "
      f"= {1.2:.1f} mm/j/degC)")
print(f"    srf = {r.coef_[1] * 1000:.4f} mm/j par kW/m2   (litterature 9.4e-6 m/j par "
      f"W/m2 = {9.4e-3 * 1000:.1f} mm/j par kW/m2)")
print(f"    ordonnee = {r.intercept_:.3f} mm/j")

# ── DEUX QUESTIONS SEPAREES ──────────────────────────────────────────────────
# 76 % des jours ont une fonte NULLE. Un R2 unique sur cette cible tres asymetrique
# melange deux problemes de nature differente et sous-estime un modele qui predirait
# bien les jours de fonte : « fond-il aujourd'hui ? » est une classification, « combien
# fond-il ? » une regression. On les separe.
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

fond = (y > 0.5).astype(int)     # 0.5 mm/j : au-dela du bruit de mesure
print("")
print(f"  DECLENCHEMENT : fond-il ? ({100 * fond.mean():.0f} % de jours de fonte)")
for nom, cols in (("degre-jour seul", ["t_positif"]),
                  ("ETI", ["t_positif", "sw_absorbe"]),
                  ("toutes covariables", list(X.columns))):
    auc = []
    for tr, te in cv.split(X, fond, grp):
        mo = RandomForestClassifier(n_estimators=120, min_samples_leaf=20, n_jobs=-1,
                                    random_state=0).fit(X[cols].to_numpy()[tr], fond[tr])
        auc.append(roc_auc_score(fond[te], mo.predict_proba(X[cols].to_numpy()[te])[:, 1]))
    print(f"    {nom:24s} AUC = {np.mean(auc):.3f}")

print("")
print(f"  INTENSITE : combien, LES JOURS OU IL FOND ({fond.sum():,} jours)")
Xf, yf, gf = X[fond == 1], y[fond == 1], grp[fond == 1]
for nom, cols, mk in (("degre-jour seul", ["t_positif"], lambda: Ridge(1.0)),
                      ("ETI", ["t_positif", "sw_absorbe"], lambda: Ridge(1.0)),
                      ("ridge, toutes covariables", list(X.columns), lambda: Ridge(1.0)),
                      ("foret", list(X.columns),
                       lambda: RandomForestRegressor(n_estimators=120, min_samples_leaf=20,
                                                     n_jobs=-1, random_state=0))):
    r2 = []
    for tr, te in GroupKFold(n_splits=5).split(Xf, yf, gf):
        mo = mk().fit(Xf[cols].to_numpy()[tr], yf[tr])
        r2.append(r2_score(yf[te], mo.predict(Xf[cols].to_numpy()[te])))
    print(f"    {nom:24s} R2 = {np.mean(r2):.3f}")

fo = RandomForestRegressor(n_estimators=120, min_samples_leaf=20, n_jobs=-1,
                           random_state=0).fit(X.to_numpy(), y)
imp = pd.Series(fo.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n  importance des covariables (foret)")
for k, v in imp.head(8).items():
    print(f"    {k:14s} {v:.3f}")
