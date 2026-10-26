"""Candidate 5.4, épreuve 3.4 : amplitude de la porte de gel continue, en passe avant à poids gelés.

Variantes (Ubuntu, Montérégie, 2001-2024, modèle provincial gelé) : amplitude 1 (porte d'origine),
0,5, 0,25, et 0,5 sans la coupure sous 10 mm de neige. Pour chacune : la réponse relative du
débit aux pluies de novembre-décembre (hausse sur trois jours rapportée à la pluie, puis à la
réponse de la station sur sol non gelé), observée et simulée, par classe de gel et de neige
SIMULÉS au tronçon de la station ; et la perte du pilote sur la période.

    .venv/Scripts/python.exe experiments/plan_ml/porte-gel-amplitude.py
"""
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from meandre.utils import paths as _p
from gel_etat_simule import evenements

DOSSIER = f"{_p.DATA_ROOT}/experiments/plan_ml"
REF = f"{_p.DATA_ROOT}/experiments/gru_residual"
REG = sys.argv[1] if len(sys.argv) > 1 else "mont"
VARIANTES = {"a100": "amplitude 1 (porte actuelle)", "a050": "amplitude 0,5", "a025": "amplitude 0,25", "a050neige": "amplitude 0,5, sans coupure sous la neige", "t050": "amplitude 0,5 sur infiltration et drainage", "t025": "amplitude 0,25 sur infiltration et drainage"}


def reponses(tag, fx, at):
    d = np.load(f"{DOSSIER}/diag-gel-{REG}-{tag}.npz", allow_pickle=True)
    z = np.load(f"{DOSSIER}/q-gel-{REG}-{tag}.npz", allow_pickle=True)
    td, tq = pd.DatetimeIndex(d["dates"]), pd.DatetimeIndex(z["dates"])
    ids_q = [str(x) for x in z["station_ids"]]
    lignes = []
    for j, s in enumerate([str(x) for x in d["station_ids"]]):
        st = f"{REG}:{s}"
        if st not in fx.columns.get_level_values(0) or st not in at.index or s not in ids_q:
            continue
        k = ids_q.index(s)
        conv = 86.4 / float(at.loc[st, "aire_km2"])
        qo = pd.Series(z["q_obs"][:, k] * conv, index=tq)
        qs = pd.Series(z["q_sim"][:, k] * conv, index=tq)
        f = fx[st].reindex(tq)
        p, tm = f["P"], (f["Tmin"] + f["Tmax"]) / 2
        gel = pd.Series(d["prof_gel_cm"][:, j], index=td).reindex(tq).shift(1)
        swe = pd.Series(d["swe"][:, j], index=td).reindex(tq).shift(1)
        ev, ro, indice = evenements(p, tm, qo)
        _, rs, _ = evenements(p, tm, qs.where(qo.notna()))
        for t in tq[ev]:
            lignes.append(dict(station=st, rep_obs=ro[t], rep_sim=rs[t], indice=indice[t], gel_cm=gel[t], swe=swe[t]))
    E = pd.DataFrame(lignes).dropna()
    ref_o = E[E.indice < 5].groupby("station").rep_obs.median().rename("ro")
    ref_s = E[E.indice < 5].groupby("station").rep_sim.median().rename("rs")
    E = E.join(ref_o, on="station").join(ref_s, on="station")
    E = E[(E.ro > 0.01) & (E.rs > 0.01)]
    E["rel_obs"], E["rel_sim"] = E.rep_obs / E.ro, E.rep_sim / E.rs
    return E


def perte(tag):
    txt = open(f"{DOSSIER}/gel-{REG}-{tag}.log", encoding="utf-8", errors="replace").read()
    kge = re.search(rf"HELD-OUT [0-9-]+ {REG}: n=\d+ \| médian ([0-9.]+)", txt)
    tot = re.search(rf"PERTE [0-9-]+ {REG} .*?: total ([0-9.]+)", txt)
    eti = re.search(rf"PERTE [0-9-]+ {REG} .*?etiage ([0-9.]+)", txt)
    return dict(kge=float(kge.group(1)) if kge else np.nan, perte=float(tot.group(1)) if tot else np.nan, etiage=float(eti.group(1)) if eti else np.nan)


def main():
    fx = pd.read_parquet(f"{REF}/forcage_bassin.parquet")
    at = pd.read_parquet(f"{REF}/attributs.parquet")
    classes = [("sol non gelé, sans neige", lambda e: (e.gel_cm < 0.5) & (e.swe < 5)), ("gel 5 à 15 cm, sans neige", lambda e: e.gel_cm.between(5, 15) & (e.swe < 5)), ("gel 15 cm et plus, sans neige", lambda e: (e.gel_cm >= 15) & (e.swe < 5)), ("gel 15 cm et plus, plus de 30 mm de neige", lambda e: (e.gel_cm >= 15) & (e.swe > 30)), ("sol non gelé, 5 mm de neige et plus", lambda e: (e.gel_cm < 0.5) & (e.swe >= 5))]
    lignes = []
    for tag, nom in VARIANTES.items():
        if not os.path.exists(f"{DOSSIER}/diag-gel-{REG}-{tag}.npz"):
            print(f"{nom} : sorties absentes")
            continue
        E = reponses(tag, fx, at)
        r = dict(variante=nom, pluies=len(E), **perte(tag))
        for cl, f in classes:
            s = E[f(E)]
            r[f"{cl} : obs"] = round(s.rel_obs.median(), 2)
            r[f"{cl} : sim"] = round(s.rel_sim.median(), 2)
            r[f"{cl} : n"] = len(s)
        lignes.append(r)
    T = pd.DataFrame(lignes).set_index("variante").T
    T.to_csv(f"{DOSSIER}/porte-gel-amplitude-{REG}.csv")
    print(f"{REG}, 2001-2024, pluies de novembre-décembre, réponse relative médiane (observée, simulée, nombre) ; perte, KGE et étiage du pilote sur la même période")
    print(T.to_string())


if __name__ == "__main__":
    main()
