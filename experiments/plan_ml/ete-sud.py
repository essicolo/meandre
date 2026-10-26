"""Volume d'été du sud agricole : où se perd-il, sur le modèle provincial gelé ? Sans simulation.

1. Décomposition de l'été (juin-septembre, 2019-2024, jours mesurés) en ligne de base (20e centile
   des débits de l'été, par station et par été) et en événements (volume au-dessus de la base), simulé
   sur observé, par territoire ; même chose pour la médiane de l'ensemble Hydrotel (2020-2024).
2. Conductivité de surface apprise (K_sat_1 du champ) par territoire et selon la part agricole des
   tronçons, comme au registre du 7 octobre, pour savoir si les bornes de texture ont fermé l'écart.

    .venv/Scripts/python.exe experiments/plan_ml/ete-sud.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from meandre.utils import paths as _p

DOSSIER = f"{_p.DATA_ROOT}/experiments/plan_ml"
MODELE = f"{_p.DATA_ROOT}/quebec/modele-provincial-2026-10"
ANALYSE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "meandre-analyse")


def decomposition(q, o):
    """Rapports simulé sur observé de la ligne de base et des événements d'été, médiane des étés."""
    d = pd.DataFrame({"q": q, "o": o}).dropna()
    d = d[d.index.month.isin([6, 7, 8, 9])]
    lignes = []
    for an, g in d.groupby(d.index.year):
        if len(g) < 90 or g.o.mean() <= 0:
            continue
        bo, bs = g.o.quantile(0.2), g.q.quantile(0.2)
        eo, es = (g.o - bo).clip(lower=0).sum(), (g.q - bs).clip(lower=0).sum()
        lignes.append((bs / bo if bo > 0 else np.nan, es / eo if eo > 0 else np.nan, g.q.mean() / g.o.mean()))
    if not lignes:
        return (np.nan, np.nan, np.nan)
    return tuple(np.nanmedian(np.array(lignes), axis=0))


def main():
    z = np.load(f"{MODELE}/q-quebec-proba.npz", allow_pickle=True)
    t = pd.DatetimeIndex(z["dates"])
    ids = [str(s) for s in z["station_ids"]]
    lignes = []
    for k, s in enumerate(ids):
        b, e, m = decomposition(pd.Series(z["q_sim"][:, k], index=t), pd.Series(z["q_obs"][:, k], index=t))
        lignes.append(dict(station=s, territoire=s.split(":")[0], modele="meandre", base=b, evenements=e, moyenne=m))
    # Hydrotel : médiane journalière des six membres, tirée du cache de meandre-analyse.
    import glob
    cache = sorted(glob.glob(f"{ANALYSE}/results/_cache/hydrotel_*.parquet"))
    if cache:
        h = pd.read_parquet(cache[-1])
        h = h.groupby(["date", "station"]).q.median().unstack()
        h.index = pd.DatetimeIndex(h.index)
        obs = pd.DataFrame(z["q_obs"], index=t, columns=ids)
        for s in h.columns:
            if s in obs.columns:
                b, e, m = decomposition(h[s], obs[s].reindex(h.index))
                lignes.append(dict(station=s, territoire=s.split(":")[0], modele="Hydrotel", base=b, evenements=e, moyenne=m))
    L = pd.DataFrame(lignes)
    L.to_csv(f"{DOSSIER}/ete-sud-decomposition.csv", index=False)
    print("1. Été (juin-septembre), simulé sur observé, médiane des stations : ligne de base (20e centile) ; événements au-dessus de la base ; moyenne")
    T = L.groupby(["territoire", "modele"])[["base", "evenements", "moyenne"]].median().round(2).unstack("modele")
    T["stations"] = L[L.modele == "meandre"].groupby("territoire").size()
    print(T.sort_values("stations", ascending=False).to_string())

    r = np.load(f"{MODELE}/reach-quebec-gele.npz", allow_pickle=True)
    raw = pd.read_parquet(f"{_p.DATA_ROOT}/quebec/territorial-raw-QC.parquet")
    regions = raw["region"].to_numpy()
    if len(regions) != len(r["param_K_sat_1"]):
        # La table brute porte vaud (201 tronçons) que le modèle provincial n'a pas.
        raw = raw[raw.region != "vaud"].reset_index(drop=True)
        regions = raw["region"].to_numpy()
    K = pd.DataFrame({"territoire": regions, "ksat1": r["param_K_sat_1"], "ksat2": r["param_K_sat_2"], "agri": raw["f_agriculture"].to_numpy(), "argile": raw["f_clay"].to_numpy()})
    print(f"\n2. Conductivité de surface apprise (m/j), tronçons : {len(K)} ; table brute alignée : {len(raw) == len(K)}")
    q = K.groupby("territoire").ksat1.quantile([0.1, 0.5, 0.9]).unstack().round(3)
    q["corr_rang_agri"] = K.groupby("territoire").apply(lambda g: g[["ksat1", "agri"]].corr(method="spearman").iloc[0, 1]).round(2)
    q["part_agri_moy"] = K.groupby("territoire").agri.mean().round(2)
    print(q.to_string())
    K["classe_agri"] = pd.cut(K.agri, [-0.01, 0.1, 0.3, 0.6, 1.0], labels=["< 10 %", "10-30 %", "30-60 %", "> 60 %"])
    print("\n   Montérégie et Saint-Laurent sud-ouest, K_sat_1 médian par part agricole du tronçon :")
    print(K[K.territoire.isin(["mont", "slso"])].groupby(["territoire", "classe_agri"], observed=True).ksat1.median().round(3).unstack().to_string())


if __name__ == "__main__":
    main()
