"""L'IRDA apporte-t-elle de l'information que les attributs actuels n'ont pas ? Sans simulation.

Pour chaque station dont le bassin est couvert à plus de 50 % par l'IRDA, on moyenne sur le
bassin amont, pondérés par l'aire locale, les attributs territoriaux actuels et la composition
de l'IRDA. On prédit ensuite le biais de volume et le KGE des modèles du 15 septembre par une
régression ridge, jugée en validation croisée par exclusion d'une station. Le gain se juge
contre un témoin où les lignes de l'IRDA sont permutées entre stations.

    .venv/Scripts/python.exe .runs/quebec/test_irda_information.py
"""
import os
import sys

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from meandre.utils import paths as _paths

RACINE = f"{_paths.DATA_ROOT}/quebec"
MODELES = f"{RACINE}/checkpoints-ref-2026-09-15"
ACTUELS = ["drainage_area_km2", "mean_slope_pct", "mean_elevation_m", "f_forest", "f_agriculture", "f_urban", "f_wetland", "f_water", "f_sand", "f_silt", "f_clay"]
IRDA = ["drainage_Excessif", "drainage_Bon", "drainage_Imparfait", "drainage_Mauvais", "drainage_Très mauvais", "materiau_ARGILE", "materiau_TILL", "materiau_SABLE", "materiau_LIMON", "materiau_GRAVIER", "materiau_ORGANIQUE", "roc_AFFLEUREMENT"]


def kge(o, s):
    ok = np.isfinite(o) & np.isfinite(s)
    if ok.sum() < 365:
        return None
    o, s = o[ok], s[ok]
    r = np.corrcoef(o, s)[0, 1]
    b = s.mean() / o.mean()
    g = (s.std() / s.mean()) / (o.std() / o.mean())
    return 1 - np.sqrt((r - 1) ** 2 + (b - 1) ** 2 + (g - 1) ** 2), b


def stations_region(reg, terr, ir):
    cx = duckdb.connect(f"{RACINE}/{reg}.duckdb", read_only=True)
    nd = cx.sql("select node_idx, node_id, topo_order from nodes order by node_idx").fetchdf()
    ed = cx.sql("select src, dst from edges").fetchdf()
    st = cx.sql("select station_id, node_idx from stations").fetchdf()
    aire = cx.sql("select area_km2_local from territorial order by node_idx").fetchdf().area_km2_local.to_numpy()
    cx.close()
    n = len(nd)
    t = terr[terr.region == reg]
    if len(t) != n:
        return []
    x = t[ACTUELS].to_numpy(dtype=float)
    y = np.zeros((n, len(IRDA)))
    m = ir[ir.region == reg].set_index("troncon")
    for i, tid in zip(nd.node_idx.values, nd.node_id.values):
        if tid in m.index:
            y[i] = [float(m.loc[tid].get(c, 0.0)) for c in IRDA]
    num = np.hstack([x, y]) * aire[:, None]
    den = aire.copy()
    enf = [[] for _ in range(n)]
    for a, b in zip(ed.src.values.astype(int), ed.dst.values.astype(int)):
        enf[b].append(a)
    for j in nd.sort_values("topo_order").node_idx.values.astype(int):
        for k in enf[j]:
            num[j] += num[k]
            den[j] += den[k]
    cumul = num / np.maximum(den, 1e-9)[:, None]
    f = f"{MODELES}/q-{reg}-variations.npz"
    if not os.path.exists(f):
        return []
    z = np.load(f, allow_pickle=True)
    ids = [str(s) for s in z["station_ids"]]
    lignes = []
    for r in st.itertuples():
        if str(r.station_id) not in ids:
            continue
        j = ids.index(str(r.station_id))
        k = kge(z["q_obs"][:, j], z["q_sim"][:, j])
        if k is None:
            continue
        v = cumul[int(r.node_idx)]
        lignes.append({"region": reg, "station": str(r.station_id), "kge": k[0], "log_beta": np.log(k[1]), **dict(zip(ACTUELS + IRDA, v))})
    return lignes


def ridge_loo(X, y, alpha=1.0):
    X = (X - X.mean(0)) / np.maximum(X.std(0), 1e-9)
    pred = np.zeros_like(y)
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        xm, ym = X[m], y[m]
        mu = ym.mean()
        w = np.linalg.solve(xm.T @ xm + alpha * np.eye(X.shape[1]), xm.T @ (ym - mu))
        pred[i] = mu + X[i] @ w
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main():
    terr = pd.read_parquet(f"{RACINE}/territorial-raw-QC.parquet")
    ir = pd.read_parquet(f"{_paths.DATA_ROOT}/irda/irda-troncons.parquet")
    lignes = []
    for reg in sorted(ir.region.unique()):
        lignes += stations_region(reg, terr, ir)
    d = pd.DataFrame(lignes)
    d["couvert"] = d[IRDA[:5]].sum(axis=1)
    q = d[d.couvert > 0.5].reset_index(drop=True)
    print(f"{len(d)} stations, dont {len(q)} couvertes à plus de 50 % par l'IRDA ({', '.join(f'{r} {n}' for r, n in q.region.value_counts().items())})")
    rng = np.random.default_rng(0)
    for cible in ("log_beta", "kge"):
        y = q[cible].to_numpy()
        base = ridge_loo(q[ACTUELS].to_numpy(), y, alpha=5.0)
        avec = ridge_loo(q[ACTUELS + IRDA].to_numpy(), y, alpha=5.0)
        nul = []
        for _ in range(200):
            perm = q[IRDA].to_numpy()[rng.permutation(len(q))]
            nul.append(ridge_loo(np.hstack([q[ACTUELS].to_numpy(), perm]), y, alpha=5.0))
        nul = np.array(nul)
        print(f"{cible} : R² hors échantillon {base:.3f} avec les attributs actuels, {avec:.3f} avec l'IRDA ; IRDA permutée {np.median(nul):.3f} en médiane, p = {(nul >= avec).mean():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
