"""Candidate 5.3, suite : quel chemin porte la réponse rapide, et répond-il à l'état du sol ?

L'exposant des aires saturées ne rend pas la dépendance observée (épreuve 3.4 du 23 octobre :
le coefficient simulé décroît avec l'humidité antécédente à b = 2, 8 et 16). Deux chemins
restent possibles : la production de surface (ruissellement de saturation et hortonien) qui
lit theta1, et l'écoulement latéral de la couche 2, qui ne le lit pas. Ce script les sépare
et les rapporte à l'humidité de surface, à l'échelle de l'orage et à celle de la station.

Entrées : les diagnostics journaliers aux stations écrits par le pilote
(`ETL_DUMP_STATIONS`, une série par grandeur et par station) et le forçage de bassin, qui
donne la pluie de l'événement. Aucune simulation nouvelle.

    python chemin_rapide.py --dumps '/home/essi/planml-out/diag-{reg}.npz' \
        --qdumps '/home/essi/planml-out/vsa-b8-{reg}.npz'

Sortie, par territoire puis réunis : le coefficient d'écoulement (hausse du débit
spécifique sur cinq jours, divisée par la pluie), la production de surface et l'écoulement
latéral de la même fenêtre, rapportés à la pluie, et theta1 de la veille, classé par
quartiles. Le rapport du débit à la production de surface dit ce qui est retenu entre le
versant et la rivière.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from meandre.utils import paths as _p

REF = f"{_p.DATA_ROOT}/experiments/gru_residual"
GRANDEURS = ("prod_surf", "prod_hypo", "prod_base", "lateral_mm", "theta1", "theta2")


def un_territoire(f, qf, reg, fx, aire):
    """Orages d'un territoire, avec les productions, l'état du sol de la veille et le débit."""
    z = np.load(f, allow_pickle=True)
    idx = pd.DatetimeIndex(z["dates"])
    # Le débit simulé vient de la même passe, sauvegardé à part (ETL_DUMP_Q) : c'est lui qui
    # donne le coefficient d'écoulement, à comparer aux productions internes.
    zq = np.load(qf, allow_pickle=True) if qf and os.path.exists(qf) else None
    blocs = []
    for j, sid in enumerate(z["station_ids"]):
        st = f"{reg}:{sid}"
        if st not in fx.columns.get_level_values(0) or st not in aire.index or aire.loc[st] <= 0:
            continue
        p = fx[st]["P"].reindex(idx)
        if p.notna().sum() < 300:
            continue
        conv = 86.4 / float(aire.loc[st])
        qs = None
        if zq is not None and sid in set(zq["station_ids"].tolist()):
            jq = int(np.flatnonzero(zq["station_ids"] == sid)[0])
            qs = pd.Series(zq["q_sim"][:, jq], index=pd.DatetimeIndex(zq["dates"])) * conv
        sec = (p.shift(1) < 2.0) & (p.shift(2) < 2.0)
        ev = (p >= 10.0) & sec & p.index.month.isin((6, 7, 8, 9))
        for t in p.index[ev]:
            i = p.index.get_loc(t)
            if i + 5 >= len(p) or i < 1:
                continue
            fen = slice(i, i + 5)
            pl = float(p.iloc[fen].sum())
            if pl <= 0:
                continue
            l = dict(date=t, pluie=pl, station=st, region=reg,
                     theta1_avant=float(z["theta1"][i - 1, j]),
                     surf=float(np.clip(z["prod_surf"][fen, j], 0, None).sum()) / pl,
                     hypo=float(np.clip(z["prod_hypo"][fen, j], 0, None).sum()) / pl,
                     base=float(np.clip(z["prod_base"][fen, j], 0, None).sum()) / pl,
                     coef=np.nan)
            l["lat"] = float(np.clip(z["lateral_mm"][fen, j], 0, None).sum()) / pl
            if qs is not None and t in qs.index:
                w = qs.iloc[i:i + 5]
                if np.isfinite(qs.iloc[i - 1]) and w.notna().all():
                    l["coef"] = float((w - qs.iloc[i - 1]).clip(lower=0).sum()) / pl
            blocs.append(l)
    return pd.DataFrame(blocs)


def rapport(E, titre):
    print("")
    print(f"===== {titre} : {len(E)} orages, {E.station.nunique()} stations =====")
    if not len(E):
        return
    E = E.copy()
    E["classe_theta"] = pd.qcut(E.theta1_avant, 4, duplicates="drop")
    t = E.groupby("classe_theta", observed=True).agg(
        n=("surf", "size"), theta1=("theta1_avant", "median"),
        surf=("surf", "median"), hypo=("hypo", "median"), lat=("lat", "median"),
        coef=("coef", "median"))
    print(t.round(3).to_string())
    print("corrélations de rang avec theta1 de la veille : " + ", ".join(
        f"{c} {E[['theta1_avant', c]].corr(method='spearman').iloc[0, 1]:+.2f}"
        for c in ("surf", "hypo", "lat", "coef")))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dumps", required=True, help="motif des diagnostics, avec {reg}")
    p.add_argument("--qdumps", default=None, help="motif des séries de débit de la même passe, avec {reg}")
    p.add_argument("--regions", nargs="+", default=["mont", "slso", "outv"])
    a = p.parse_args()
    fx = pd.read_parquet(f"{REF}/forcage_bassin.parquet")
    at = pd.read_parquet(f"{REF}/attributs.parquet")
    aire = at["aire_km2"]
    tous = []
    for reg in a.regions:
        f = a.dumps.format(reg=reg)
        if not os.path.exists(f):
            print(f"  (absent : {f})")
            continue
        qf = a.qdumps.format(reg=reg) if a.qdumps else None
        E = un_territoire(f, qf, reg, fx, aire)
        rapport(E, reg)
        tous.append(E)
    if tous:
        rapport(pd.concat(tous, ignore_index=True), "réunis")


if __name__ == "__main__":
    main()
