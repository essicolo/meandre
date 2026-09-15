"""Le sur-lissage suit-il une variable CONTINUE, ou faut-il decouper par region ?

Essi, 2026-09-15 : « on peut plus jouer sur le loss et la validation croisee qu'un ancrage
discontinu par region arbitraire ». La region est un artefact de production, pas une entite
hydrologique, et un reglage par region serait une bidouille.

On teste donc l'hypothese continue. Pour chaque station, on mesure la nervosite du debit
simule rapportee a l'observee, puis on cherche quelle combinaison d'attributs TERRITORIAUX,
cumules sur le bassin amont, la predit. Si une variable continue la porte, le champ peut
s'en servir sans qu'aucun decoupage soit necessaire ; si seul l'identifiant de region la
porte, il n'y a pas de generalisation possible et il faut le dire.

Le juge est la prediction sur des stations RETIREES, pas sur les stations vues.

    .venv/Scripts/python.exe .runs/quebec/gradient_nervosite.py
"""
import os
import sys

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from meandre.utils import paths as _paths

REGIONS = ["outv", "gasp", "sagu", "mont", "slno", "slso", "abit", "cnda", "cndb",
           "cndc", "cndd", "cnde", "labi", "outm"]


def main():
    racine = _paths.DATA_ROOT
    rep = f"{racine}/quebec/rapport"
    rw = pd.read_parquet(f"{racine}/quebec/territorial-raw-QC.parquet")
    lignes = []
    for reg in REGIONS:
        f = f"{rep}/rap-{reg}-q.npz"
        base = f"{racine}/quebec/{reg}.duckdb"
        if not (os.path.exists(f) and os.path.exists(base)):
            continue
        z = np.load(f, allow_pickle=True)
        cx = duckdb.connect(base, read_only=True)
        nd = cx.sql("select node_idx, topo_order from nodes order by node_idx").fetchdf()
        ed = cx.sql("select src, dst from edges").fetchdf()
        st = cx.sql("""select station_id, node_idx, drainage_area_km2 a from stations
            where drainage_area_km2 is not null""").fetchdf()
        cx.close()
        r = rw[rw.region == reg].reset_index(drop=True)
        if len(r) != len(nd):
            continue
        attrs = [c for c in r.columns if c not in ("region",)]
        n = len(nd)
        aire = r["area_km2_local"].values if "area_km2_local" in r.columns else np.ones(n)
        aire = np.clip(np.nan_to_num(aire, nan=1.0), 1e-6, None)
        A = r[attrs].values.astype(float)
        enf = [[] for _ in range(n)]
        for a_, b_ in zip(ed.src.values.astype(int), ed.dst.values.astype(int)):
            enf[b_].append(a_)
        ordre = nd.sort_values("topo_order").node_idx.values.astype(int)
        num = A * aire[:, None]
        den = aire.copy()
        for j in ordre:
            for k in enf[j]:
                num[j] += num[k]
                den[j] += den[k]
        moy = num / np.maximum(den, 1e-9)[:, None]
        ids = [str(x) for x in z["station_ids"]]
        for row in st.itertuples():
            if str(row.station_id) not in ids:
                continue
            j = ids.index(str(row.station_id))
            o, s = z["q_obs"][:, j].astype(float), z["q_sim"][:, j].astype(float)
            m = np.isfinite(o) & np.isfinite(s)
            if m.sum() < 365 or o[m].mean() <= 0 or s[m].mean() <= 0:
                continue
            vo = np.std(np.diff(o[m])) / o[m].mean()
            vs = np.std(np.diff(s[m])) / s[m].mean()
            if vo <= 0:
                continue
            d = {"region": reg, "station": str(row.station_id), "aire_km2": float(row.a),
                 "nervosite": float(vs / vo)}
            for c, v in zip(attrs, moy[int(row.node_idx)]):
                d[c] = float(v)
            lignes.append(d)
    d = pd.DataFrame(lignes).dropna(axis=1, how="all")
    print(f"{len(d)} stations, {d.region.nunique()} régions")
    print(f"nervosité : médiane {d.nervosite.median():.2f}, "
          f"q10-q90 {d.nervosite.quantile(.1):.2f}-{d.nervosite.quantile(.9):.2f}")

    from scipy.stats import spearmanr
    expl = [c for c in d.columns if c not in ("region", "station", "nervosite")
            and d[c].std() > 1e-9]
    res = []
    for c in expl:
        rho, p = spearmanr(d[c], d.nervosite, nan_policy="omit")
        res.append({"attribut": c, "rho": rho, "p": p})
    t = pd.DataFrame(res).reindex(pd.DataFrame(res).rho.abs().sort_values(ascending=False).index)
    print("\ncorrélation de rang de la nervosité avec chaque attribut du bassin amont")
    print(t.head(10).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # Prediction sur stations RETIREES, avec et sans l'identifiant de region.
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold, KFold
    X = d[expl].values
    y = d.nervosite.values
    def score(X_, groupes=None):
        cv = GroupKFold(n_splits=5) if groupes is not None else KFold(5, shuffle=True, random_state=0)
        pred = np.zeros(len(y))
        for tr, te in (cv.split(X_, y, groupes) if groupes is not None else cv.split(X_)):
            m = HistGradientBoostingRegressor(max_iter=250, random_state=0).fit(X_[tr], y[tr])
            pred[te] = m.predict(X_[te])
        ss = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
        return ss, float(spearmanr(pred, y)[0])
    s1 = score(X)
    s2 = score(X, d.region.values)
    print(f"\nprédiction de la nervosité par les seuls attributs continus")
    print(f"  stations retirées au hasard      : R² {s1[0]:+.3f} | rang {s1[1]:+.3f}")
    print(f"  RÉGIONS entières retirées        : R² {s2[0]:+.3f} | rang {s2[1]:+.3f}")
    moy_reg = d.groupby("region").nervosite.median()
    pred_reg = d.region.map(moy_reg).values
    ss = 1 - np.sum((y - pred_reg) ** 2) / np.sum((y - y.mean()) ** 2)
    print(f"  référence : la seule médiane de sa région : R² {ss:+.3f} (ajustée, non retirée)")
    sortie = os.environ.get("MEANDRE_CACHES", ".reports/quebec/caches")
    if os.path.isdir(sortie):
        d.to_csv(f"{sortie}/gradient-nervosite.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
