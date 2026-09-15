"""Le materiau parental est-il deja deductible des attributs que le modele possede ?

La couverture pedologique de l'IRDA ne couvre que 11 pour cent de la superficie modelisee.
Si le materiau parental se predit a partir des seize attributs deja consommes par le champ,
alors l'information circule deja et les zones non cartographiees ne perdent rien. S'il ne
se predit pas, c'est de l'information neuve, et il faut decider quoi faire hors couverture.

Le juge est la prediction sur des TRONCONS RETIRES, et non sur des troncons vus, sinon on
mesure la memorisation. On retire par BLOCS SPATIAUX et non au hasard : deux troncons
voisins se ressemblent, et un tirage au hasard laisserait le voisin d'un troncon retire
dans le jeu d'apprentissage, ce qui gonfle le score sans rien prouver.

    .venv/Scripts/python.exe .runs/quebec/predire_materiau.py
"""
import os
import sys

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from meandre.utils import paths as _paths

MATER = ["ARGILE", "TILL", "SABLE", "LIMON", "GRAVIER", "ORGANIQUE"]


def main():
    racine = _paths.DATA_ROOT
    ir = pd.read_parquet(f"{racine}/irda/irda-troncons.parquet")
    cm = [f"materiau_{m}" for m in MATER if f"materiau_{m}" in ir.columns]
    ir["couvert"] = ir[[c for c in ir.columns if c.startswith("drainage_")]].sum(axis=1)
    ir = ir[ir.couvert > 0.7].copy()
    rw = pd.read_parquet(f"{racine}/quebec/territorial-raw-QC.parquet")
    lignes = []
    for reg, sous in ir.groupby("region"):
        cx = duckdb.connect(f"{racine}/quebec/{reg}.duckdb", read_only=True)
        nd = cx.sql("select node_idx, node_id, lon, lat from nodes order by node_idx").fetchdf()
        cx.close()
        r = rw[rw.region == reg].reset_index(drop=True)
        if len(r) != len(nd):
            continue
        par_tid = dict(zip(nd.node_id.astype(int), nd.node_idx.astype(int)))
        for t in sous.itertuples():
            i = par_tid.get(int(t.troncon))
            if i is None:
                continue
            d = {"region": reg, "lon": float(nd.lon.values[i]), "lat": float(nd.lat.values[i])}
            for c in r.columns:
                if c != "region":
                    d[c] = float(r[c].values[i])
            for c in cm:
                d[c] = float(getattr(t, c))
            lignes.append(d)
    d = pd.DataFrame(lignes).dropna()
    if len(d) < 200:
        print(f"trop peu de tronçons ({len(d)})")
        return 1
    d["dominant"] = d[cm].idxmax(axis=1).str.replace("materiau_", "", regex=False)
    print(f"{len(d)} tronçons couverts à plus de 70 pour cent, {d.region.nunique()} régions")
    print(d.dominant.value_counts().to_string())

    attrs = [c for c in d.columns if c not in ("region", "lon", "lat", "dominant")
             and not c.startswith("materiau_") and d[c].std() > 1e-9]
    print(f"\n{len(attrs)} attributs explicatifs : {', '.join(attrs)}")

    # Blocs spatiaux : une grille d'un demi-degre, retiree en entier.
    d["bloc"] = (np.floor(d.lon * 2).astype(int).astype(str) + "_"
                 + np.floor(d.lat * 2).astype(int).astype(str))
    blocs = d.bloc.unique()
    rng = np.random.default_rng(1234)
    rng.shuffle(blocs)
    parts = np.array_split(blocs, 5)
    print(f"{len(blocs)} blocs d'un demi-degré, validation croisée en 5 plis")

    from sklearn.ensemble import HistGradientBoostingClassifier
    X = d[attrs].values
    y = d.dominant.values
    bon, n = 0, 0
    for k, p in enumerate(parts):
        te = d.bloc.isin(p).values
        tr = ~te
        if te.sum() < 20 or tr.sum() < 100:
            continue
        m = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(X[tr], y[tr])
        pred = m.predict(X[te])
        bon += int((pred == y[te]).sum())
        n += int(te.sum())
    base = d.dominant.value_counts(normalize=True).max()
    print(f"\nprédiction du matériau dominant sur des blocs RETIRÉS")
    print(f"  exactitude {100 * bon / n:.1f} % sur {n} tronçons")
    print(f"  référence, toujours prédire la classe la plus fréquente : {100 * base:.1f} %")
    print(f"  gain sur la référence : {100 * (bon / n - base):+.1f} points")
    return 0


if __name__ == "__main__":
    sys.exit(main())
