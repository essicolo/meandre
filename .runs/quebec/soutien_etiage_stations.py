"""Qu'est-ce qui porte l'étiage d'août ? Test croisé sur toutes les stations, sans simulation.

La décrue d'été des deux sous-bassins du banc s'arrête sur un plancher (exposant de Brutsaert
et Nieber au-dessus de 2), qu'aucun réservoir en loi de puissance ne produit. Avant d'ajouter
un second soutien lent au modèle, on demande aux stations ce qui le porte. Pour chaque station
mesurée, deux grandeurs d'étiage : le soutien, débit minimal sur 7 jours d'août-septembre
rapporté au débit moyen annuel, médiane des années ; et l'exposant de récession d'été. En face,
les attributs du bassin amont, pondérés par l'aire locale des tronçons : lacs, milieux humides,
forêt, texture, profondeur au roc. Un attribut qui explique le soutien en validation croisée
par territoire est un candidat pour paramétrer le second soutien ; le contraire dit que le
soutien n'est pas prédictible par le terrain, et qu'il faut l'ancrer sur la récession observée.

    python .runs/quebec/soutien_etiage_stations.py outv mont slso slno sagu gasp
"""
import os
import sys
from collections import defaultdict, deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import duckdb
import numpy as np
import pandas as pd

from meandre.data import recession_anchor as ra
from meandre.utils import paths as _p

ATTRIBUTS = ["lake_fraction", "f_wetland", "f_water", "f_forest", "f_agriculture", "f_sand", "f_clay", "depth_to_bedrock_m", "mean_slope_pct"]


def amont(edges, n):
    """Ensemble des tronçons amont de chaque nœud, par parcours du graphe orienté aval."""
    parents = defaultdict(list)
    for src, dst in edges:
        parents[int(dst)].append(int(src))
    cache = {}

    def _amont(i):
        vus, pile = {i}, deque([i])
        while pile:
            j = pile.popleft()
            for k in parents.get(j, ()):
                if k not in vus:
                    vus.add(k); pile.append(k)
        return np.fromiter(vus, dtype=int)
    return _amont


def exposant_ete(q, dates):
    q = np.asarray(q, dtype=float)
    bons = ra.segments_de_decrue(np.nan_to_num(q, nan=0.0))
    dqdt = np.concatenate([[np.nan], -np.diff(q)])
    qm = np.concatenate([[np.nan], 0.5 * (q[1:] + q[:-1])])
    garde = bons & np.isfinite(dqdt) & np.isfinite(qm) & (dqdt > 0) & (qm > 0) & np.isin(dates.month, (7, 8, 9))
    if garde.sum() < 60:
        return np.nan
    ex, ey = ra._enveloppe_inferieure(np.log(qm[garde]), np.log(dqdt[garde]))
    if ex is None or len(ex) < 10:
        return np.nan
    return float(np.polyfit(ex, ey, 1)[0])


def stations_du_territoire(reg):
    con = duckdb.connect(f"{_p.DATA_ROOT}/quebec/{reg}.duckdb", read_only=True)
    st = con.execute("select station_id, node_idx, drainage_area_km2 from stations").df()
    terr = con.execute("select * from territorial").df().set_index("node_idx")
    edges = con.execute("select * from edges").fetchall()
    obs = con.execute("select station_id, date, discharge, reconstructed from observations where discharge is not null").df()
    con.close()
    edges = [(e[0], e[1]) for e in edges]
    _amont = amont(edges, len(terr))
    aire = terr["area_km2_local"].to_numpy()
    lignes = []
    for _, s in st.iterrows():
        o = obs[obs.station_id == s.station_id]
        o = o[~o.reconstructed.fillna(True).astype(bool)]
        if len(o) < 2000:
            continue
        q = pd.Series(o.discharge.values, index=pd.DatetimeIndex(o.date)).asfreq("D")
        qmoy = q.mean()
        # Soutien : minimum glissant sur 7 jours d'aout-septembre, par annee, rapporte au debit moyen.
        soutiens = []
        for an, qa in q.groupby(q.index.year):
            ete = qa[qa.index.month.isin((8, 9))]
            if ete.notna().sum() < 40:
                continue
            soutiens.append(ete.rolling(7, min_periods=7).mean().min() / qmoy)
        if len(soutiens) < 5:
            continue
        up = _amont(int(s.node_idx))
        w = aire[up]; w = w / w.sum() if w.sum() > 0 else np.ones(len(up)) / len(up)
        ligne = {"territoire": reg, "station": s.station_id, "aire_km2": float(s.drainage_area_km2), "annees": len(soutiens), "soutien": float(np.median(soutiens)), "b_ete": exposant_ete(q.values, q.index)}
        for a in ATTRIBUTS:
            if a in terr.columns:
                ligne[a] = float((terr[a].to_numpy()[up] * w).sum())
        lignes.append(ligne)
    return pd.DataFrame(lignes)


def main(regs):
    t = pd.concat([stations_du_territoire(r) for r in regs], ignore_index=True)
    t = t.replace([np.inf, -np.inf], np.nan)
    print(f"{len(t)} stations mesurees sur {len(regs)} territoires")
    print(f"  soutien d'etiage (min 7 j aout-sept / debit moyen) : mediane {t.soutien.median():.3f}, quartiles {t.soutien.quantile(.25):.3f} a {t.soutien.quantile(.75):.3f}")
    b = t.b_ete.dropna()
    print(f"  exposant de recession d'ete : mediane {b.median():.2f}, part des stations au-dessus de 2 : {100 * (b > 2).mean():.0f} % sur {len(b)}")
    attrs = [a for a in ATTRIBUTS if a in t.columns]
    print("  correlation (Spearman) du soutien avec chaque attribut amont :")
    for a in attrs:
        r = t[[a, "soutien"]].dropna().corr(method="spearman").iloc[0, 1]
        rb = t[[a, "b_ete"]].dropna().corr(method="spearman").iloc[0, 1]
        print(f"    {a:20s} soutien {r:+.2f} | exposant d'ete {rb:+.2f}")
    # Validation croisee par territoire : chaque territoire predit par un modele lineaire ajuste sur les autres.
    y = np.log(t.soutien.clip(lower=1e-4))
    X = t[attrs].fillna(t[attrs].median())
    X = (X - X.mean()) / X.std().replace(0, 1)
    def _cv(cols):
        err, temoin = [], []
        for reg in t.territoire.unique():
            tr, te = t.territoire != reg, t.territoire == reg
            if te.sum() < 3 or tr.sum() < 10:
                continue
            A = np.column_stack([np.ones(tr.sum()), X.loc[tr, cols]])
            beta, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
            pred = np.column_stack([np.ones(te.sum()), X.loc[te, cols]]) @ beta
            err.append(np.mean(np.abs(pred - y[te])))
            temoin.append(np.mean(np.abs(y[tr].mean() - y[te])))
        return float(np.mean(err)), float(np.mean(temoin))
    print("  validation croisee par territoire, erreur absolue moyenne sur log(soutien), contre la moyenne des autres territoires :")
    for nom, cols in (("lacs + milieux humides", ["lake_fraction", "f_wetland", "f_water"]), ("texture + roc", ["f_sand", "f_clay", "depth_to_bedrock_m"]), ("couvert", ["f_forest", "f_agriculture"]), ("tout", attrs)):
        cols = [c for c in cols if c in attrs]
        e, tm = _cv(cols)
        print(f"    {nom:24s} : {e:.3f} contre temoin {tm:.3f} ({100 * (tm - e) / tm:+.0f} %)")
    f = f"{_p.DERIVED_ROOT}/auxiliaires/soutien-etiage-stations.csv"
    t.to_csv(f, index=False)
    print(f"  ecrit : {f}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["outv", "mont", "slso", "slno", "sagu", "gasp"])
