"""Candidate 5.3, épreuve 3.3 : une loi de part ruisselée qui lit l'humidité simulée des couches 1
ou 2 prédit-elle le coefficient d'écoulement OBSERVÉ des orages d'été, aux stations jamais vues ?

Cible : coefficient observé de chaque orage (etat-predit-orages.parquet). Lois comparées, toutes
ajustées en logarithme, coefficient + 0,005 :
  pluie seule ; pluie et humidité relative de la couche 1 (theta1 / porosité 1) ; pluie et humidité
  relative de la couche 2 ; pluie et indice de pluies antérieures (témoin sans modèle) ;
  pluie et les deux couches ; arbres monotones sur les mêmes variables (plafond).
Validation : stations retirées (cinq plis) ; erreur absolue médiane du logarithme et corrélation de
rang avec l'observé ; et, pour comparaison, le coefficient simulé par le modèle actuel.

    .venv/Scripts/python.exe experiments/plan_ml/loi-aire-contributive.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

DOSSIER = "D:/meandre-data/experiments/plan_ml"


def main():
    E = pd.read_parquet(f"{DOSSIER}/etat-predit-orages.parquet")
    z = np.load("D:/meandre-data/quebec/modele-provincial-2026-10/reach-quebec-gele.npz", allow_pickle=True)
    q = np.load("D:/meandre-data/quebec/modele-provincial-2026-10/q-quebec-proba.npz", allow_pickle=True)
    ids = [str(s) for s in q["station_ids"]]
    por1 = dict(zip(ids, z["param_porosity_1"][z["station_idx"]]))
    por2 = dict(zip(ids, z["param_porosity_2"][z["station_idx"]]))
    E["s1"] = E.theta1 / E.station.map(por1)
    E["s2"] = E.theta2 / E.station.map(por2)
    E = E.dropna(subset=["s1", "s2"])
    y = np.log(E.c_obs + 0.005).to_numpy()
    X = pd.DataFrame({"log_pluie": np.log(E.pluie), "log_s1": np.log(E.s1.clip(lower=1e-3)), "log_s2": np.log(E.s2.clip(lower=1e-3)), "log_api": np.log(E.api + 1)})
    lois = {"pluie seule": ["log_pluie"], "pluie et couche 1": ["log_pluie", "log_s1"], "pluie et couche 2": ["log_pluie", "log_s2"], "pluie et API (sans modèle)": ["log_pluie", "log_api"], "pluie et couches 1 et 2": ["log_pluie", "log_s1", "log_s2"]}
    g = E.station.to_numpy()
    print(f"{len(E)} orages, {E.station.nunique()} stations ; humidité relative médiane couche 1 {E.s1.median():.2f}, couche 2 {E.s2.median():.2f}")
    res = []
    for nom, cols in list(lois.items()) + [("arbres monotones, pluie et couches 1 et 2", ["log_pluie", "log_s1", "log_s2"])]:
        pred = np.full(len(E), np.nan)
        coefs = []
        for tr, te in GroupKFold(5).split(X, y, g):
            if nom.startswith("arbres"):
                m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, min_samples_leaf=50, monotonic_cst=[1, 1, 1], random_state=0)
            else:
                m = LinearRegression()
            m.fit(X.iloc[tr][cols], y[tr])
            pred[te] = m.predict(X.iloc[te][cols])
            if hasattr(m, "coef_"):
                coefs.append(m.coef_)
        err = np.median(np.abs(pred - y))
        rho = pd.Series(pred).corr(pd.Series(y), method="spearman")
        res.append(dict(loi=nom, erreur=round(err, 3), correlation=round(rho, 3), coefficients=np.round(np.mean(coefs, axis=0), 2).tolist() if coefs else ""))
    sim = np.log(E.c_sim.clip(lower=0) + 0.005).to_numpy()
    res.append(dict(loi="modèle actuel (coefficient simulé)", erreur=round(np.median(np.abs(sim - y)), 3), correlation=round(pd.Series(sim).corr(pd.Series(y), method="spearman"), 3), coefficients=""))
    R = pd.DataFrame(res)
    R.to_csv(f"{DOSSIER}/loi-aire-contributive.csv", index=False)
    print("stations retirées (cinq plis) : écart absolu médian du log du coefficient, corrélation de rang avec l'observé, exposants moyens (pluie, couches)")
    print(R.to_string(index=False))
    for reg in ("mont", "slso", "outv"):
        s = E[E.territoire == reg]
        print(f"  {reg} : coefficient observé médian {s.c_obs.median():.3f}, simulé {s.c_sim.median():.3f}")


if __name__ == "__main__":
    main()
