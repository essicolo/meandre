"""Quel état simulé prédit le coefficient d'écoulement OBSERVÉ des orages d'été ? Sans simulation.

Orages : pluie de bassin d'au moins 10 mm, de juin à septembre, après deux jours à moins de 2 mm,
2001-2024. Coefficient : hausse cumulée du débit spécifique sur cinq jours au-dessus du débit de la
veille, rapportée à la pluie des cinq jours, observé et simulé (modèle provincial gelé). États
simulés la veille, au tronçon de la station : humidité des couches 1 et 2, stock de la nappe ; et
l'indice de précipitation antérieure (API) comme témoin sans modèle. Corrélation de rang à
l'intérieur de chaque station, puis médiane des stations.

    .venv/Scripts/python.exe experiments/plan_ml/etat-predit-orages.py
"""
import numpy as np
import pandas as pd

DOSSIER = "D:/meandre-data/experiments/plan_ml"
REF = "D:/meandre-data/experiments/gru_residual"
SOURCES = {"mont": ("diag-gel-mont-a100.npz", "q-gel-mont-a100.npz"), "slso": ("diag-gel-slso.npz", "q-gel-slso.npz"), "outv": ("diag-gel-outv.npz", "q-gel-outv.npz")}


def main():
    fx = pd.read_parquet(f"{REF}/forcage_bassin.parquet")
    at = pd.read_parquet(f"{REF}/attributs.parquet")
    lignes = []
    for reg, (fd, fq) in SOURCES.items():
        D = np.load(f"{DOSSIER}/{fd}", allow_pickle=True)
        Z = np.load(f"{DOSSIER}/{fq}", allow_pickle=True)
        td, tq = pd.DatetimeIndex(D["dates"]), pd.DatetimeIndex(Z["dates"])
        idq = [str(s) for s in Z["station_ids"]]
        for j, s in enumerate([str(x) for x in D["station_ids"]]):
            st = f"{reg}:{s}"
            if st not in fx.columns.get_level_values(0) or s not in idq:
                continue
            k = idq.index(s)
            conv = 86.4 / float(at.loc[st, "aire_km2"])
            o = pd.Series(Z["q_obs"][:, k] * conv, index=tq)
            q = pd.Series(Z["q_sim"][:, k] * conv, index=tq)
            p = fx[st]["P"].reindex(tq)
            api = pd.Series(0.0, index=tq)
            pv = p.fillna(0).to_numpy()
            a = np.zeros(len(pv))
            for i in range(1, len(pv)):
                a[i] = 0.9 * a[i - 1] + pv[i - 1]
            api[:] = a
            etat = {v: pd.Series(D[v][:, j], index=td).reindex(tq) for v in ("theta1", "theta2", "s_gw")}
            sec = (p.shift(1) < 2) & (p.shift(2) < 2)
            ev = (p >= 10) & sec & tq.month.isin([6, 7, 8, 9])
            for t in tq[ev]:
                i = tq.get_loc(t)
                if i < 1 or i + 5 >= len(tq):
                    continue
                w = slice(i, i + 5)
                if o.iloc[w].isna().any() or np.isnan(o.iloc[i - 1]):
                    continue
                pl = p.iloc[w].sum()
                lignes.append(dict(station=st, territoire=reg, c_obs=(o.iloc[w] - o.iloc[i - 1]).clip(lower=0).sum() / pl, c_sim=(q.iloc[w] - q.iloc[i - 1]).clip(lower=0).sum() / pl, api=api.iloc[i], theta1=etat["theta1"].iloc[i - 1], theta2=etat["theta2"].iloc[i - 1], s_gw=etat["s_gw"].iloc[i - 1], q_veille_obs=o.iloc[i - 1], pluie=pl))
    E = pd.DataFrame(lignes).dropna()
    E.to_parquet(f"{DOSSIER}/etat-predit-orages.parquet")
    preds = ["api", "theta1", "theta2", "s_gw", "q_veille_obs", "pluie"]
    def par_station(cible):
        r = E.groupby("station").apply(lambda g: pd.Series({x: g[[cible, x]].corr("spearman").iloc[0, 1] for x in preds}) if len(g) >= 15 else None).dropna()
        return r
    for reg in SOURCES:
        sub = E[E.territoire == reg]
        ro = par_station("c_obs").loc[lambda d: d.index.str.startswith(reg)]
        rs = par_station("c_sim").loc[lambda d: d.index.str.startswith(reg)]
        print(f"{reg} : {len(sub)} orages, {ro.shape[0]} stations d'au moins 15 orages ; corrélation de rang dans la station, médiane des stations")
        print(pd.DataFrame({"coefficient observé": ro.median(), "coefficient simulé": rs.median()}).T.round(2).to_string())


if __name__ == "__main__":
    main()
