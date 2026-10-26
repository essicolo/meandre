"""Candidate 5.6, épreuve 3.1 : entre deux stations emboîtées, le modèle propage-t-il la crue
avec le bon retard et la bonne atténuation ? Sans simulation nouvelle.

Pour chaque paire (station amont, station aval) où l'amont est dans le bassin de l'aval,
sur 2019-2024 : décalage qui maximise la corrélation croisée des hausses journalières de
débit entre amont et aval, pour l'observé et pour le simulé ; et rapport des amplitudes de
crue (hausse maximale sur trois jours à l'aval rapportée à l'amont, par événement, en débit
spécifique). Un retard simulé plus long et une atténuation plus forte que l'observé désignent
le routage ; des valeurs justes renvoient le défaut d'amortissement au versant.

    .venv/Scripts/python.exe experiments/plan_ml/propagation.py
"""
import os
import sys
from collections import defaultdict, deque

import duckdb
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from meandre.data.basin_cache import BasinCache
from meandre.utils import paths as _p

REF = f"{_p.DATA_ROOT}/experiments/gru_residual"
DOSSIER = f"{_p.DATA_ROOT}/experiments/plan_ml"


def amont(graph, depart):
    src, dst = graph.edge_index[0].numpy(), graph.edge_index[1].numpy()
    pere = defaultdict(list)
    for s, t in zip(src, dst):
        pere[int(t)].append(int(s))
    vus, q = {int(depart)}, deque([int(depart)])
    while q:
        u = q.popleft()
        for v in pere[u]:
            if v not in vus:
                vus.add(v)
                q.append(v)
    return vus


def decalage(a, b, kmax=5):
    """k ≥ 0 maximisant la corrélation entre la hausse amont au jour t et la hausse aval au jour t + k."""
    ha, hb = a.diff().clip(lower=0), b.diff().clip(lower=0)
    c = {k: ha.corr(hb.shift(-k)) for k in range(0, kmax + 1)}
    k = max(c, key=c.get)
    return k, c[k]


def main():
    q = pd.read_parquet(f"{REF}/debits.parquet")
    at = pd.read_parquet(f"{REF}/attributs.parquet")
    paires = []
    for reg in sorted({s.split(":")[0] for s in q["q_obs"].columns}):
        base = f"{_p.DATA_ROOT}/quebec/{reg}.duckdb"
        d = BasinCache(base).load(device=torch.device("cpu"))
        con = duckdb.connect(base, read_only=True)
        st = con.execute("select station_id, node_idx, drainage_area_km2 from stations").df()
        con.close()
        st = st[("%s:" % reg + st.station_id).isin(q["q_obs"].columns)]
        bassins = {r.station_id: amont(d["graph"], int(r.node_idx)) for r in st.itertuples()}
        for a in st.itertuples():
            for b in st.itertuples():
                if a.station_id != b.station_id and int(a.node_idx) in bassins[b.station_id] and a.drainage_area_km2 < 0.8 * b.drainage_area_km2:
                    paires.append((f"{reg}:{a.station_id}", f"{reg}:{b.station_id}", float(a.drainage_area_km2), float(b.drainage_area_km2), len(bassins[b.station_id]) - len(bassins[a.station_id])))
    lignes = []
    for sa, sb, aa, ab, n_entre in paires:
        oa, ob = q["q_obs"][sa], q["q_obs"][sb]
        sma, smb = q["q_sim"][sa], q["q_sim"][sb]
        ok = oa.notna() & ob.notna()
        if ok.sum() < 730:
            continue
        ko, co = decalage(oa[ok], ob[ok])
        ks, cs = decalage(sma[ok], smb[ok])
        # atténuation : hausse maximale sur 3 jours à l'aval / à l'amont, en débit spécifique, sur les 20 plus fortes hausses amont
        ha = (oa.rolling(3).max() - oa.shift(1)).clip(lower=0) / aa
        hb = (ob.shift(-ko).rolling(3).max() - ob.shift(1 - ko)).clip(lower=0) / ab
        top = ha[ok].nlargest(20).index
        att_o = float(np.median(hb.reindex(top) / ha.reindex(top)))
        hsa = (sma.rolling(3).max() - sma.shift(1)).clip(lower=0) / aa
        hsb = (smb.shift(-ks).rolling(3).max() - smb.shift(1 - ks)).clip(lower=0) / ab
        tops = hsa[ok].nlargest(20).index
        att_s = float(np.median(hsb.reindex(tops) / hsa.reindex(tops)))
        lignes.append(dict(amont=sa, aval=sb, aire_amont=aa, aire_aval=ab, troncons_entre=n_entre, retard_obs=ko, retard_sim=ks, corr_obs=co, corr_sim=cs, attenuation_obs=att_o, attenuation_sim=att_s))
    L = pd.DataFrame(lignes)
    L.to_csv(f"{DOSSIER}/propagation.csv", index=False)
    print(f"{len(L)} paires de stations emboîtées (amont < 80 % de l'aval), 2019-2024, jours communs")
    print(f"retard amont-aval (jours) : observé moyen {L.retard_obs.mean():.2f}, simulé {L.retard_sim.mean():.2f} ; simulé moins observé : " + str(dict((L.retard_sim - L.retard_obs).value_counts().sort_index())))
    print(f"corrélation croisée au meilleur décalage : observée {L.corr_obs.median():.2f}, simulée {L.corr_sim.median():.2f}")
    print(f"atténuation des 20 plus fortes hausses (aval/amont, débit spécifique) : observée médiane {L.attenuation_obs.median():.2f}, simulée {L.attenuation_sim.median():.2f}")
    L["classe"] = pd.cut(L.troncons_entre, [0, 20, 60, 150, 10000], labels=["1-20", "21-60", "61-150", "150+"])
    print("\npar nombre de tronçons entre les deux stations :")
    print(L.groupby("classe", observed=True).agg(n=("amont", "size"), retard_obs=("retard_obs", "mean"), retard_sim=("retard_sim", "mean"), att_obs=("attenuation_obs", "median"), att_sim=("attenuation_sim", "median")).round(2).to_string())
    print("\npar territoire :")
    L["region"] = L.amont.str.split(":").str[0]
    print(L.groupby("region").agg(n=("amont", "size"), retard_obs=("retard_obs", "mean"), retard_sim=("retard_sim", "mean"), att_obs=("attenuation_obs", "median"), att_sim=("attenuation_sim", "median")).round(2).to_string())


if __name__ == "__main__":
    main()
