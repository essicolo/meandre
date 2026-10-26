"""D'où vient le retard d'un jour du débit simulé ? Deux mesures, sans simulation.

1. Délai de réponse à la pluie : décalage k qui maximise la corrélation entre la pluie de
   bassin du jour t et la hausse du débit du jour t + k, sur l'observé et sur le simulé.
   Un délai simulé plus long que l'observé désigne un retard interne au modèle ; des délais
   égaux désignent une erreur de forçage ou de calendrier.
2. Lacs en série : en routage « operator-lagged », chaque lac relâche selon son stockage de
   la veille. On compte le plus grand nombre de lacs traversés sur un chemin vers la station
   et on le croise avec le retard mesuré.

    .venv/Scripts/python.exe experiments/gru_residual/decalage.py
"""
import os
import sys
from collections import defaultdict

import duckdb
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from meandre.data.basin_cache import BasinCache
from meandre.utils import paths as _p
from donnees import amont

DOSSIER = f"{_p.DATA_ROOT}/experiments/gru_residual"


def delai(p, q, kmax=6):
    hausse = q.diff().clip(lower=0)
    c = {k: p.corr(hausse.shift(-k)) for k in range(0, kmax + 1)}
    return max(c, key=c.get)


def lacs_en_serie(graph, idx, exutoire):
    """Plus grand nombre de lacs sur un chemin d'une source vers l'exutoire, dans le
    sous-graphe idx."""
    lac = graph.is_lake.numpy().astype(int)
    dans = set(int(v) for v in idx)
    pere = defaultdict(list)
    for s, t in zip(graph.edge_index[0].numpy(), graph.edge_index[1].numpy()):
        if int(s) in dans and int(t) in dans:
            pere[int(t)].append(int(s))
    prof = {}
    for v in graph.topo_order.numpy():
        v = int(v)
        if v in dans:
            prof[v] = lac[v] + max((prof[u] for u in pere[v] if u in prof), default=0)
    return prof[int(exutoire)], int(lac[list(dans)].sum())


def main():
    q = pd.read_parquet(f"{DOSSIER}/debits.parquet")
    fx = pd.read_parquet(f"{DOSSIER}/forcage_bassin.parquet")
    ev = pd.read_csv(f"{DOSSIER}/evenements.csv").set_index("station")
    lignes = []
    for reg in sorted({s.split(":")[0] for s in ev.index}):
        base = f"{_p.DATA_ROOT}/quebec/{reg}.duckdb"
        d = BasinCache(base).load(device=torch.device("cpu"))
        con = duckdb.connect(base, read_only=True)
        noeuds = dict(con.execute("select station_id, node_idx from stations").fetchall())
        aires = dict(con.execute("select station_id, drainage_area_km2 from stations").fetchall())
        con.close()
        for st in [s for s in ev.index if s.startswith(reg + ":")]:
            code = st.split(":")[1]
            idx = amont(d["graph"], noeuds[code])
            serie, total = lacs_en_serie(d["graph"], idx, noeuds[code])
            p = fx[st]["P"].reindex(q.index)
            lignes.append(dict(station=st, region=reg, aire=aires.get(code), n_troncons=len(idx),
                               lacs_serie=serie, lacs_total=total,
                               delai_obs=delai(p, q["q_obs"][st]), delai_sim=delai(p, q["q_sim"][st]),
                               decalage=ev.loc[st, "decalage"], kge=ev.loc[st, "kge"], kge_decale=ev.loc[st, "kge_decale"]))
    L = pd.DataFrame(lignes)
    L["retard_interne"] = L.delai_sim - L.delai_obs
    L.to_csv(f"{DOSSIER}/decalage.csv", index=False)
    print(f"{len(L)} stations")
    print("délai de réponse à la pluie, observé (jours) :", L.delai_obs.value_counts().sort_index().to_dict())
    print("délai de réponse à la pluie, simulé (jours)  :", L.delai_sim.value_counts().sort_index().to_dict())
    print("délai simulé moins observé :", L.retard_interne.value_counts().sort_index().to_dict())
    print("\nretard du simulé (corrélation croisée) selon le nombre de lacs en série :")
    L["classe_lacs"] = pd.cut(L.lacs_serie, [-1, 0, 1, 2, 4, 1000], labels=["0", "1", "2", "3-4", "5+"])
    print(L.groupby("classe_lacs", observed=True).agg(n=("station", "size"), decalage=("decalage", "mean"), retard_interne=("retard_interne", "mean"), gain_kge=("kge_decale", "mean")).round(2).to_string())
    L["gain"] = L.kge_decale - L.kge
    print("\nselon la taille du bassin (tronçons) :")
    L["classe_taille"] = pd.qcut(L.n_troncons, 4)
    print(L.groupby("classe_taille", observed=True).agg(n=("station", "size"), decalage=("decalage", "mean"), retard_interne=("retard_interne", "mean"), lacs_serie=("lacs_serie", "median"), gain=("gain", "mean")).round(2).to_string())
    print(f"\ncorrélation de rang décalage / lacs en série : {L.decalage.corr(L.lacs_serie, method='spearman'):.2f} ; décalage / tronçons : {L.decalage.corr(L.n_troncons, method='spearman'):.2f}")


if __name__ == "__main__":
    main()
