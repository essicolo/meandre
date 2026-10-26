"""Candidate 5.6, épreuve 3.1 : quelle forme a la réponse rapide observée, et le modèle la
rend-il ?

Les mesures du jour disent que le modèle propage la crue au bon jour (retard amont-aval juste)
et n'atténue pas assez dans le réseau, alors que ses hydrogrammes aux stations sont trop
amortis. Les deux ne tiennent ensemble que si la réponse rapide est étalée dès le versant.
Reste à la caractériser : montée trop lente, ressuyage trop long, ou les deux. Sans simulation
nouvelle, sur les débits observés et simulés du modèle provincial gelé et le forçage de bassin.

Trois grandeurs par orage d'été, en débit spécifique (mm/j) :
  - le jour du maximum de hausse par rapport à la veille de l'orage, observé et simulé ;
  - la part de la hausse des cinq jours déjà acquise le jour même, puis à un et deux jours ;
  - la hausse restante quatre jours après le début, rapportée au maximum : un ressuyage long
    laisse une part élevée.

    .venv/Scripts/python.exe experiments/plan_ml/forme_reponse.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from meandre.utils import paths as _p

REF = f"{_p.DATA_ROOT}/experiments/gru_residual"
DOSSIER = f"{_p.DATA_ROOT}/experiments/plan_ml"
FENETRE = 7


def orages(p, q):
    """Un enregistrement par orage : jour du pic, parts cumulées, reste au septième jour."""
    sec = (p.shift(1) < 2.0) & (p.shift(2) < 2.0)
    ev = (p >= 10.0) & sec & p.index.month.isin((6, 7, 8, 9))
    lignes = []
    for t in p.index[ev]:
        i = p.index.get_loc(t)
        if i + FENETRE >= len(p) or i < 1:
            continue
        base = q.iloc[i - 1]
        if not np.isfinite(base):
            continue
        hausse = (q.iloc[i:i + FENETRE] - base).clip(lower=0).to_numpy()
        if not np.all(np.isfinite(hausse)):
            continue
        total = float(hausse.sum())
        pic = int(np.argmax(hausse))
        if total <= 1e-4 or hausse[pic] <= 1e-3:
            continue
        cumul = np.cumsum(hausse)
        lignes.append(dict(date=t, pluie=float(p.iloc[i:i + 5].sum()), pic=pic,
                           part_j0=cumul[0] / total, part_j1=cumul[1] / total,
                           part_j2=cumul[2] / total, reste_j6=hausse[6] / total,
                           part_pic=float(hausse[pic]) / total,
                           hausse_pic=float(hausse[pic])))
    return pd.DataFrame(lignes)


def main():
    q = pd.read_parquet(f"{REF}/debits.parquet")
    fx = pd.read_parquet(f"{REF}/forcage_bassin.parquet")
    at = pd.read_parquet(f"{REF}/attributs.parquet")
    blocs = []
    for st in q["q_obs"].columns:
        o, s = q["q_obs"][st], q["q_sim"][st]
        if o.notna().sum() < 365 or st not in at.index or at.loc[st, "aire_km2"] <= 0:
            continue
        conv = 86.4 / float(at.loc[st, "aire_km2"])
        p = fx[st]["P"].reindex(q.index)
        eo = orages(p, o * conv)
        es = orages(p, s * conv)
        if len(eo) < 20 or len(es) < 20:
            continue
        eo["region"] = st.split(":")[0]
        es["region"] = st.split(":")[0]
        eo["station"] = st
        es["station"] = st
        blocs.append((eo, es))
    if not blocs:
        print("aucun orage exploitable")
        return
    O = pd.concat([b[0] for b in blocs], ignore_index=True)
    S = pd.concat([b[1] for b in blocs], ignore_index=True)
    O.to_parquet(f"{DOSSIER}/forme_reponse-obs.parquet")
    S.to_parquet(f"{DOSSIER}/forme_reponse-sim.parquet")
    print(f"{len(O)} orages observés et {len(S)} simulés, {O.station.nunique()} stations, 2019-2024")
    lignes = []
    for nom, d in (("observé", O), ("simulé", S)):
        par_st = d.groupby("station").median(numeric_only=True)
        lignes.append(dict(source=nom, pic=par_st.pic.median(), part_j0=par_st.part_j0.median(),
                           part_j1=par_st.part_j1.median(), part_j2=par_st.part_j2.median(),
                           part_pic=par_st.part_pic.median(), reste_j6=par_st.reste_j6.median()))
    t = pd.DataFrame(lignes).set_index("source").round(3)
    print("\nmédiane par station, orages de juin à septembre, 144 stations :")
    print(t.to_string())
    print("\npar territoire (médiane par station du jour du pic, parts cumulées) :")
    for reg in sorted(O.region.unique()):
        o = O[O.region == reg].groupby("station").median(numeric_only=True)
        s = S[S.region == reg].groupby("station").median(numeric_only=True)
        if len(o) < 3:
            continue
        print(f"  {reg}: n {len(o)} | pic obs {o.pic.median():.2f} sim {s.pic.median():.2f} | "
              f"part j0 obs {o.part_j0.median():.2f} sim {s.part_j0.median():.2f} | "
              f"part j2 obs {o.part_j2.median():.2f} sim {s.part_j2.median():.2f} | "
              f"reste j6 obs {o.reste_j6.median():.2f} sim {s.reste_j6.median():.2f}")


if __name__ == "__main__":
    main()
