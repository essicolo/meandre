"""Candidate 5.4, épreuve 3.1 : la réponse du débit à une pluie d'automne varie-t-elle de façon
CONTINUE avec l'état de gel du sol, et dans quel sens ? Sans simulation.

Reprend la statistique de septembre (reponse_sol_gele_archive.py) avec le forçage CaSR brut du
modèle et des classes continues d'indice de gel au lieu de deux états. Pour chaque pluie de
novembre à décembre d'au moins 5 mm tombée par température moyenne positive (> 0,5 °C), la
réponse est la hausse du débit spécifique sur les trois jours suivants, rapportée à la pluie.
L'indice de gel est la somme des degrés-jours négatifs depuis le 15 octobre, veille de
l'événement ; le manteau est un stock de degrés-jours (accumulation sous 0 °C, fonte 3 mm/°C).

Ce qu'une porte continue f(gel) dans [0, 1], décroissante, prédit : une réponse qui CROÎT avec
l'indice de gel, sans palier net ; ce qu'une porte à seuil prédit : deux niveaux. Ce que le
manteau change : sous une neige épaisse le sol gèle moins et la pluie est retenue par la neige.

    .venv/Scripts/python.exe experiments/plan_ml/sol_gele.py outv mont slno sagu gasp slso
"""
import os
import sys
from collections import defaultdict, deque

import duckdb
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from meandre.utils import paths as _p

DOSSIER = f"{_p.DATA_ROOT}/experiments/plan_ml"
CLASSES_GEL = [-1, 5, 20, 50, 100, 200, 1e9]
NOMS_GEL = ["0-5", "5-20", "20-50", "50-100", "100-200", "200+"]


def amont(edges, n):
    parents = defaultdict(list)
    for src, dst in edges:
        parents[int(dst)].append(int(src))

    def _amont(i):
        vus, pile = {i}, deque([i])
        while pile:
            j = pile.popleft()
            for k in parents.get(j, ()):
                if k not in vus:
                    vus.add(k)
                    pile.append(k)
        return np.fromiter(vus, dtype=int)
    return _amont


def evenements(p, tm, q):
    saison = pd.Series(np.where(p.index.month >= 10, p.index.year, p.index.year - 1), index=p.index)
    froid = (-tm).clip(lower=0.0)
    froid[(p.index.month < 10) | ((p.index.month == 10) & (p.index.day < 15))] = 0.0
    indice = froid.groupby(saison).cumsum().shift(1)
    swe = np.zeros(len(p))
    s = 0.0
    pv, tv = p.to_numpy(), tm.to_numpy()
    for k in range(len(p)):
        s = s + pv[k] if tv[k] < 0 else max(0.0, s - 3.0 * tv[k])
        swe[k] = s
    swe = pd.Series(swe, index=p.index).shift(1)
    dq = (q.shift(-1).rolling(3).max().shift(-2) - q.shift(1)).clip(lower=0.0)
    ev = (p >= 5.0) & (tm > 0.5) & p.index.month.isin((11, 12)) & q.notna() & dq.notna() & q.shift(1).notna()
    e = pd.DataFrame({"reponse": (dq / p)[ev], "gel": indice[ev], "neige": swe[ev], "pluie": p[ev], "q_avant": q.shift(1)[ev]})
    e["date"] = e.index
    return e


def territoire(reg):
    con = duckdb.connect(f"{_p.DATA_ROOT}/quebec/{reg}.duckdb", read_only=True)
    st = con.execute("select station_id, node_idx, drainage_area_km2 from stations").df()
    terr = con.execute("select * from territorial").df().set_index("node_idx")
    edges = [(e[0], e[1]) for e in con.execute("select * from edges").fetchall()]
    cols = {c[0] for c in con.execute("describe observations").fetchall()}
    sel = "station_id, date, discharge" + (", reconstructed" if "reconstructed" in cols else "")
    obs = con.execute(f"select {sel} from observations where discharge is not null").df()
    con.close()
    if "reconstructed" not in obs:
        obs["reconstructed"] = False
    _amont = amont(edges, len(terr))
    aire = terr["area_km2_local"].to_numpy()
    ds = xr.open_dataset(f"{_p.DATA_ROOT}/quebec/forcing-{reg}-casr-brut.nc")
    F = ds["forcing"]
    temps = pd.DatetimeIndex(ds.time.values)
    blocs = []
    for _, s in st.iterrows():
        o = obs[obs.station_id == s.station_id]
        o = o[~o.reconstructed.fillna(True).astype(bool)]
        if len(o) < 2000:
            continue
        up = _amont(int(s.node_idx))
        w = aire[up]
        w = w / w.sum() if w.sum() > 0 else np.ones(len(up)) / len(up)
        f = F.isel(node=up).values
        p = pd.Series((f[:, :, 0] * w[None, :]).sum(axis=1), index=temps)
        tm = pd.Series((0.5 * (f[:, :, 1] + f[:, :, 2]) * w[None, :]).sum(axis=1), index=temps)
        conv = 86.4 / float(s.drainage_area_km2)
        q = pd.Series(o.discharge.values * conv, index=pd.DatetimeIndex(o.date)).reindex(temps)
        e = evenements(p, tm, q)
        if len(e) < 10:
            continue
        e["station"] = f"{reg}:{s.station_id}"
        e["region"] = reg
        for a in ("f_agriculture", "f_forest", "f_wetland", "lake_fraction", "mean_slope_pct"):
            if a in terr.columns:
                e[a] = float((terr[a].to_numpy()[up] * w).sum())
        blocs.append(e)
    ds.close()
    return pd.concat(blocs) if blocs else pd.DataFrame()


def main(regs):
    E = pd.concat([territoire(r) for r in regs], ignore_index=True)
    E = E[np.isfinite(E.gel) & np.isfinite(E.neige)]
    # La réponse est normalisée par la réponse médiane de la station aux pluies sur sol non gelé
    # (indice < 5) pour comparer les stations entre elles.
    ref = E[E.gel < 5].groupby("station").reponse.median().rename("ref")
    E = E.join(ref, on="station")
    E = E[E.ref > 0.01]
    E["rel"] = E.reponse / E.ref
    E["classe_gel"] = pd.cut(E.gel, CLASSES_GEL, labels=NOMS_GEL)
    E["classe_neige"] = pd.cut(E.neige, [-1, 5, 30, 1e9], labels=["< 5 mm", "5-30 mm", "> 30 mm"])
    print(f"{len(E)} pluies de novembre-décembre (≥ 5 mm, T > 0,5 °C) sur {E.station.nunique()} stations, {len(regs)} territoires, jours mesurés")
    print("\nréponse relative (hausse du débit sur 3 jours / pluie, rapportée à la réponse de la station sur sol non gelé) par classe d'indice de gel (°C·jour) :")
    t = E.groupby("classe_gel", observed=True).agg(n=("rel", "size"), stations=("station", "nunique"), mediane=("rel", "median"), q25=("rel", lambda v: v.quantile(0.25)), q75=("rel", lambda v: v.quantile(0.75)), neige_moy=("neige", "median"))
    print(t.round(2).to_string())
    print("\npar classe de gel et de manteau (médiane de la réponse relative, n) :")
    t2 = E.groupby(["classe_gel", "classe_neige"], observed=True).rel.agg(["median", "size"]).round(2)
    print(t2.unstack("classe_neige").to_string())
    print("\npar territoire : réponse relative médiane sous gel 20-100 contre 0-5, et part des stations où le gel répond plus :")
    for r in regs:
        s = E[E.region == r]
        g = s[(s.gel >= 20) & (s.gel < 100)]
        par_st = g.groupby("station").rel.median()
        if len(g):
            print(f"  {r} : {g.rel.median():.2f} (n {len(g)}, {len(par_st)} stations, gel répond plus dans {100 * (par_st > 1).mean():.0f} %)")
    g = E[(E.gel >= 20) & (E.gel < 100)].groupby("station").agg(rel=("rel", "median"), agri=("f_agriculture", "first"), pente=("mean_slope_pct", "first"), lacs=("lake_fraction", "first"), foret=("f_forest", "first"))
    print("\ncorrélation de rang de la réponse relative sous gel (20-100) avec les attributs amont : " + ", ".join(f"{a} {g[['rel', a]].dropna().corr(method='spearman').iloc[0, 1]:+.2f}" for a in ("agri", "pente", "lacs", "foret")))
    E.to_parquet(f"{DOSSIER}/sol_gele_evenements.parquet")


if __name__ == "__main__":
    main(sys.argv[1:] or ["outv", "mont", "slno", "sagu", "gasp", "slso"])
