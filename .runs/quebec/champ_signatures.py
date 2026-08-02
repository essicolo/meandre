"""CHAMP SPATIAL DES SIGNATURES (pivot 2026-08-02) : krigeage gaussien des
signatures hydrologiques observables, validé par BLOCS SPATIAUX INTERNES
(et non par région entière : un champ spatial interpole, il n'extrapole pas).
Cible : prédire les signatures d'un bassin non jaugé, avec incertitude.

  python .runs/quebec/champ_signatures.py
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.cluster import KMeans

d = pd.read_csv("reports/signatures_stations.csv")
SIGS = ["k_recession", "coeff_ecoul", "ratio_et", "cv_debit"]
# blocs spatiaux internes : k-means sur les coordonnées (indépendant du découpage
# administratif, et garantit que chaque bloc a des voisins dans l'entraînement)
d["bloc"] = KMeans(n_clusters=8, n_init=10, random_state=0).fit_predict(d[["lon", "lat"]])
print(f"{len(d)} stations | 8 blocs spatiaux ({d.groupby('bloc').size().min()}-{d.groupby('bloc').size().max()} stations/bloc)\n")
print(f"{'signature':>13} | {'R2 blocs':>9} | {'R2 région':>10} | {'couverture 90%':>14}")
res = {}
for sig in SIGS:
    sub = d.dropna(subset=[sig]).reset_index(drop=True)
    y = np.log(sub[sig].values) if sig in ("k_recession", "cv_debit") else sub[sig].values
    X = np.c_[sub.lon, sub.lat, np.log(sub.aire_km2)]
    ker = ConstantKernel(1.0) * Matern(length_scale=[2.0, 2.0, 3.0], nu=1.5) + WhiteKernel(0.05)
    def cv(groups):
        pr, sd, tr = [], [], []
        for g in np.unique(groups):
            m = groups != g
            if m.sum() < 10: continue
            gp = GaussianProcessRegressor(kernel=ker, normalize_y=True, alpha=1e-6).fit(X[m], y[m])
            p, s = gp.predict(X[~m], return_std=True)
            pr.append(p); sd.append(s); tr.append(y[~m])
        pr, sd, tr = np.concatenate(pr), np.concatenate(sd), np.concatenate(tr)
        r2 = 1 - ((pr - tr) ** 2).sum() / ((tr - tr.mean()) ** 2).sum()
        cov = float(np.mean(np.abs(pr - tr) < 1.645 * sd))
        return r2, cov
    r2b, covb = cv(sub.bloc.values)
    r2r, _ = cv(sub.region.values)
    res[sig] = (r2b, r2r, covb)
    print(f"{sig:>13} | {r2b:9.3f} | {r2r:10.3f} | {covb:14.2f}")
print("\nR2 blocs = interpolation spatiale (régime légitime du champ)")
print("R2 région = extrapolation à une région isolée (régime hostile, pour comparaison)")
print("couverture 90% : proportion d'observations dans l'intervalle prédit (cible 0.90)")
