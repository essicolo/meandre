"""Les attributs de l'IRDA expliquent-ils l'erreur du modele sur les petits bassins ?

Test SANS entrainement. Pour chaque station jaugee, on cumule les fractions pedologiques
de tous les troncons de son bassin, puis on demande si ces fractions expliquent une part
du KGE, du biais de volume ou du rapport des variabilites, une fois l'aire drainee prise
en compte. Si elles n'expliquent rien, la piste s'arrete la et n'a coute que du temps de
processeur.

    .venv/Scripts/python.exe .runs/quebec/test_irda_petits_bassins.py
"""
import os
import sys

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from meandre.utils import paths as _paths


def kge(o, s):
    m = np.isfinite(o) & np.isfinite(s)
    if m.sum() < 365:
        return None
    o, s = o[m].astype(float), s[m].astype(float)
    if o.std() < 1e-9 or s.std() < 1e-9 or o.mean() <= 0:
        return None
    r = float(np.corrcoef(o, s)[0, 1])
    b = float(s.mean() / o.mean())
    g = float((s.std() / s.mean()) / (o.std() / o.mean()))
    return 1 - float(np.sqrt((r - 1) ** 2 + (b - 1) ** 2 + (g - 1) ** 2)), r, b, g


def main():
    racine = _paths.DATA_ROOT
    f = f"{racine}/irda/irda-troncons.parquet"
    if not os.path.exists(f):
        print(f"croisement absent : {f}")
        return 1
    ir = pd.read_parquet(f)
    cols = [c for c in ir.columns if c.startswith(("drainage_", "materiau_", "roc_"))]
    lignes = []
    for reg, sous in ir.groupby("region"):
        base = f"{racine}/quebec/{reg}.duckdb"
        q = f"{racine}/quebec/flotte/q-{reg}-B.npz"
        if not (os.path.exists(base) and os.path.exists(q)):
            continue
        z = np.load(q, allow_pickle=True)
        cx = duckdb.connect(base, read_only=True)
        nd = cx.sql("select node_idx, node_id, topo_order from nodes order by node_idx").fetchdf()
        ed = cx.sql("select src, dst from edges").fetchdf()
        st = cx.sql("""select station_id, node_idx, drainage_area_km2 a from stations
            where drainage_area_km2 is not null""").fetchdf()
        cx.close()
        n = len(nd)
        # Fractions par noeud, via l'identifiant de troncon.
        m = sous.set_index("troncon")
        par_noeud = np.zeros((n, len(cols)))
        aire_n = np.zeros(n)
        for i, tid in zip(nd.node_idx.values, nd.node_id.values):
            if tid in m.index:
                par_noeud[i] = m.loc[tid, cols].values.astype(float)
                aire_n[i] = float(m.loc[tid, "aire_m2"])
        # Cumul amont, pondere par l'aire locale.
        enf = [[] for _ in range(n)]
        for a, b in zip(ed.src.values.astype(int), ed.dst.values.astype(int)):
            enf[b].append(a)
        ordre = nd.sort_values("topo_order").node_idx.values.astype(int)
        num = par_noeud * aire_n[:, None]
        den = aire_n.copy()
        for j in ordre:
            for k in enf[j]:
                num[j] += num[k]
                den[j] += den[k]
        frac = np.divide(num, np.maximum(den, 1e-9)[:, None])
        ids = [str(x) for x in z["station_ids"]]
        for r in st.itertuples():
            if str(r.station_id) not in ids:
                continue
            jj = ids.index(str(r.station_id))
            k = kge(z["q_obs"][:, jj], z["q_sim"][:, jj])
            if k is None:
                continue
            d = {"region": reg, "station": str(r.station_id), "aire_km2": float(r.a),
                 "kge": k[0], "r": k[1], "beta": k[2], "gamma": k[3],
                 "couvert": float(frac[int(r.node_idx)][[cols.index(c) for c in cols
                                                         if c.startswith("drainage_")]].sum())}
            for c, v in zip(cols, frac[int(r.node_idx)]):
                d[c] = float(v)
            lignes.append(d)
    d = pd.DataFrame(lignes)
    if d.empty:
        print("aucune station appariée")
        return 1
    print(f"{len(d)} stations, {d.region.nunique()} régions")
    print(f"couverture pédologique du bassin : médiane {100 * d.couvert.median():.0f} %, "
          f"{int((d.couvert > 0.5).sum())} stations couvertes à plus de 50 %")
    q = d[d.couvert > 0.5].copy()
    if len(q) < 10:
        print("trop peu de stations bien couvertes pour conclure")
        return 0
    from scipy.stats import spearmanr
    print(f"\nsur les {len(q)} stations couvertes à plus de 50 pour cent")
    print(f"  aire médiane {q.aire_km2.median():.0f} km², KGE médian {q.kge.median():.3f}")
    print(f"\n{'attribut':>26} {'rho(KGE)':>9} {'rho(beta)':>10} {'rho(gamma)':>11} {'p(KGE)':>8}")
    for c in cols + ["couvert"]:
        if q[c].std() < 1e-9:
            continue
        rk, pk = spearmanr(q[c], q.kge)
        rb, _ = spearmanr(q[c], q.beta)
        rg, _ = spearmanr(q[c], q.gamma)
        print(f"{c:>26} {rk:+9.3f} {rb:+10.3f} {rg:+11.3f} {pk:8.4f}")
    sortie = os.environ.get("MEANDRE_CACHES", ".reports/quebec/caches")
    if os.path.isdir(sortie):
        d.to_csv(f"{sortie}/irda-stations.csv", index=False)
        print(f"\ncache : {sortie}/irda-stations.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
