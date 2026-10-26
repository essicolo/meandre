"""Candidate 5.4, suite de l'épreuve 3.3 : la porte de gel est-elle en défaut, ou l'état qu'elle lit ?

Même statistique que sol_gele.py (réponse du débit aux pluies de novembre-décembre d'au moins
5 mm par température positive, hausse sur trois jours rapportée à la pluie, puis à la réponse
médiane de la station sur sol non gelé), mais classée selon l'état SIMULÉ au tronçon de la
station : profondeur de gel de Rankinen et équivalent en eau de la neige, la veille de la pluie.
Les diagnostics viennent du pilote (ETL_DUMP_STATIONS), modèle provincial gelé, zéro époque,
2001-2024, sur Ubuntu.

Trois questions :
  1. l'état simulé suit-il l'indice de gel atmosphérique (degrés-jours négatifs depuis le 15 octobre) ?
  2. la réponse OBSERVÉE s'ordonne-t-elle mieux selon l'état simulé que selon l'indice ? Si oui,
     l'état est bon et c'est la porte qui sature ; si non, l'état est en cause ;
  3. la réponse SIMULÉE suit-elle l'état simulé comme l'observée le suit ?

Limite : l'état est celui du tronçon exutoire, pas une moyenne du bassin.

    .venv/Scripts/python.exe experiments/plan_ml/gel_etat_simule.py mont outv slso
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from meandre.utils import paths as _p

DOSSIER = f"{_p.DATA_ROOT}/experiments/plan_ml"
REF = f"{_p.DATA_ROOT}/experiments/gru_residual"
CLASSES_GEL_CM = [-1, 0.5, 5, 15, 30, 1e9]
NOMS_GEL_CM = ["0", "0,5-5", "5-15", "15-30", "30+"]


def evenements(p, tm, q):
    saison = pd.Series(np.where(p.index.month >= 10, p.index.year, p.index.year - 1), index=p.index)
    froid = (-tm).clip(lower=0.0)
    froid[(p.index.month < 10) | ((p.index.month == 10) & (p.index.day < 15))] = 0.0
    indice = froid.groupby(saison).cumsum().shift(1)
    dq = (q.shift(-1).rolling(3).max().shift(-2) - q.shift(1)).clip(lower=0.0)
    ev = (p >= 5.0) & (tm > 0.5) & p.index.month.isin((11, 12)) & q.notna() & dq.notna() & q.shift(1).notna()
    return ev, (dq / p), indice


def main(regs):
    fx = pd.read_parquet(f"{REF}/forcage_bassin.parquet")
    at = pd.read_parquet(f"{REF}/attributs.parquet")
    lignes = []
    for reg in regs:
        d = np.load(f"{DOSSIER}/diag-gel-{reg}.npz", allow_pickle=True)
        z = np.load(f"{DOSSIER}/q-gel-{reg}.npz", allow_pickle=True)
        td = pd.DatetimeIndex(d["dates"])
        tq = pd.DatetimeIndex(z["dates"])
        for j, s in enumerate([str(x) for x in d["station_ids"]]):
            st = f"{reg}:{s}"
            if st not in fx.columns.get_level_values(0) or st not in at.index:
                continue
            conv = 86.4 / float(at.loc[st, "aire_km2"])
            k = list(z["station_ids"]).index(s) if s in list(z["station_ids"]) else None
            if k is None:
                continue
            qo = pd.Series(z["q_obs"][:, k] * conv, index=tq)
            qs = pd.Series(z["q_sim"][:, k] * conv, index=tq)
            f = fx[st].reindex(tq)
            p, tm = f["P"], (f["Tmin"] + f["Tmax"]) / 2
            gel = pd.Series(d["prof_gel_cm"][:, j], index=td).reindex(tq).shift(1)
            swe = pd.Series(d["swe"][:, j], index=td).reindex(tq).shift(1)
            ev, ro, indice = evenements(p, tm, qo)
            _, rs, _ = evenements(p, tm, qs.where(qo.notna()))
            for t in tq[ev]:
                lignes.append(dict(station=st, region=reg, date=t, rep_obs=ro[t], rep_sim=rs[t], indice=indice[t], gel_cm=gel[t], swe=swe[t]))
    E = pd.DataFrame(lignes).dropna()
    ref_o = E[E.indice < 5].groupby("station").rep_obs.median().rename("ro")
    ref_s = E[E.indice < 5].groupby("station").rep_sim.median().rename("rs")
    E = E.join(ref_o, on="station").join(ref_s, on="station")
    E = E[(E.ro > 0.01) & (E.rs > 0.01)]
    E["rel_obs"] = E.rep_obs / E.ro
    E["rel_sim"] = E.rep_sim / E.rs
    E.to_parquet(f"{DOSSIER}/gel_etat_simule.parquet")
    print(f"{len(E)} pluies de novembre-décembre, {E.station.nunique()} stations, {', '.join(regs)}, 2001-2024")
    c = lambda a, b: E[[a, b]].corr(method="spearman").iloc[0, 1]
    print(f"\n1. l'état simulé suit-il l'indice atmosphérique ? corrélation de rang indice~gel simulé {c('indice', 'gel_cm'):.2f}, indice~neige simulée {c('indice', 'swe'):.2f} ; part des pluies à gel simulé nul quand l'indice dépasse 50 °C·jour : {(E[E.indice > 50].gel_cm < 0.5).mean():.2f}")
    print(f"\n2. ce qui ordonne la réponse OBSERVÉE (corrélation de rang) : indice atmosphérique {c('indice', 'rel_obs'):+.2f}, gel simulé {c('gel_cm', 'rel_obs'):+.2f}, neige simulée {c('swe', 'rel_obs'):+.2f}")
    print(f"   ce qui ordonne la réponse SIMULÉE : indice {c('indice', 'rel_sim'):+.2f}, gel simulé {c('gel_cm', 'rel_sim'):+.2f}, neige simulée {c('swe', 'rel_sim'):+.2f}")
    E["classe_gel"] = pd.cut(E.gel_cm, CLASSES_GEL_CM, labels=NOMS_GEL_CM)
    E["classe_neige"] = pd.cut(E.swe, [-1, 5, 30, 1e9], labels=["< 5 mm", "5-30 mm", "> 30 mm"])
    print("\n3. par classe de gel SIMULÉ au tronçon (cm) : réponse relative médiane observée et simulée")
    t = E.groupby("classe_gel", observed=True).agg(n=("rel_obs", "size"), stations=("station", "nunique"), obs=("rel_obs", "median"), sim=("rel_sim", "median"), indice=("indice", "median"), neige=("swe", "median"))
    print(t.round(2).to_string())
    print("\n   par classe de gel simulé et de neige simulée (médiane observée / simulée, n) :")
    g = E.groupby(["classe_gel", "classe_neige"], observed=True).agg(obs=("rel_obs", "median"), sim=("rel_sim", "median"), n=("rel_obs", "size")).round(2)
    print(g.to_string())
    print("\n   par territoire, pluies à gel simulé de 5 cm et plus : observé, simulé")
    for r in regs:
        s = E[(E.region == r) & (E.gel_cm >= 5)]
        if len(s) >= 20:
            print(f"   {r}: obs {s.rel_obs.median():.2f}, sim {s.rel_sim.median():.2f} (n {len(s)})")


if __name__ == "__main__":
    main(sys.argv[1:] or ["mont", "outv", "slso"])
