"""Les propriétés de sol de l'IRDA sont-elles redondantes avec les attributs du champ ? Sans simulation.

La question n'est pas ce que la donnée prédit du débit, mais ce qu'elle ajoute à l'entrée du
champ spatial. Deux mesures, sur les tronçons couverts à plus de moitié.

1. Redondance. Chaque propriété de l'IRDA est prédite à partir de l'ensemble des attributs
   actuels, compositions en ilr, par gradient boosté en validation croisée par blocs spatiaux
   d'un degré. Un R² hors bloc élevé signifie que l'attribut n'apporte rien de nouveau.
2. Séparation. Parmi les paires de tronçons voisins que les attributs actuels rendent presque
   identiques, part de celles que l'IRDA distingue. Et rang effectif de l'entrée, avec et sans
   l'IRDA.

    .venv/Scripts/python.exe .runs/quebec/test_irda_redondance.py
"""
import os
import sys

import duckdb
import nuee
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from meandre.utils import paths as _paths

RACINE = f"{_paths.DATA_ROOT}/quebec"
IRDA = ["ilr_1", "ilr_2", "drainage", "log_permeabilite", "structure", "groupe_hydro", "lithique", "organique"]


def ilr_parts(parts):
    return nuee.ilr(nuee.multiplicative_replacement(nuee.closure(parts)))


def attributs_actuels(t):
    """Attributs du champ, compositions passées en ilr."""
    tex = ilr_parts(t[["f_sand", "f_silt", "f_clay"]].to_numpy(dtype=float))
    occ = t[["f_forest", "f_agriculture", "f_urban", "f_wetland", "f_water"]].to_numpy(dtype=float)
    occ = np.hstack([occ, np.clip(1.0 - occ.sum(axis=1, keepdims=True), 0.0, 1.0)])
    occ = ilr_parts(occ)
    autres = t[["drainage_area_km2", "strahler_order", "mean_slope_pct", "mean_elevation_m", "sin_aspect", "cos_aspect", "dist_to_outlet_km", "lake_fraction"]].to_numpy(dtype=float, copy=True)
    autres[:, 2] = np.log(autres[:, 2])
    noms = ["texture_ilr_1", "texture_ilr_2"] + [f"occupation_ilr_{i + 1}" for i in range(occ.shape[1])] + ["log_aire", "strahler", "log_pente", "altitude", "sin_aspect", "cos_aspect", "dist_exutoire", "part_lac"]
    return pd.DataFrame(np.hstack([tex, occ, autres]), columns=noms)


def charger():
    terr = pd.read_parquet(f"{RACINE}/territorial-raw-QC.parquet")
    ir = pd.read_parquet(f"{_paths.DATA_ROOT}/irda/irda-proprietes-troncons.parquet")
    lignes, voisins = [], []
    decalage = 0
    for reg, t in terr.groupby("region", sort=False):
        base = f"{RACINE}/{reg}.duckdb"
        if not os.path.exists(base):
            continue
        cx = duckdb.connect(base, read_only=True)
        nd = cx.sql("select node_idx, node_id, lon, lat from nodes order by node_idx").fetchdf()
        ed = cx.sql("select src, dst from edges").fetchdf()
        cx.close()
        if len(nd) != len(t):
            continue
        a = attributs_actuels(t.reset_index(drop=True))
        a["region"], a["lon"], a["lat"] = reg, nd.lon.values, nd.lat.values
        m = ir[ir.region == reg].set_index("troncon")
        props = m.reindex(nd.node_id.values)
        for v in IRDA:
            a[v] = props[v].values
            a[f"couv_{v}"] = props[f"couv_{v}"].fillna(0.0).values
        lignes.append(a)
        voisins.append(np.column_stack([ed.src.values + decalage, ed.dst.values + decalage]))
        decalage += len(nd)
    return pd.concat(lignes, ignore_index=True), np.vstack(voisins)


def rang_effectif(x):
    z = (x - x.mean(0)) / np.maximum(x.std(0), 1e-9)
    ev = np.clip(np.linalg.eigvalsh(np.cov(z, rowvar=False)), 0, None)
    return ev.sum() ** 2 / (ev ** 2).sum()


def main():
    d, voisins = charger()
    actuels = [c for c in d.columns if c.startswith(("texture_", "occupation_")) or c in ("log_aire", "strahler", "log_pente", "altitude", "sin_aspect", "cos_aspect", "dist_exutoire", "part_lac")]
    blocs = (np.floor(d.lon) * 1000 + np.floor(d.lat)).astype(int).values
    print(f"{len(d)} tronçons, {len(actuels)} attributs actuels\n")
    print("1. redondance : R² hors bloc spatial de chaque propriété IRDA prédite par les attributs actuels")
    for v in IRDA:
        m = (d[f"couv_{v}"] >= 0.5).values & d[v].notna().values
        x, y, g = d.loc[m, actuels].to_numpy(), d.loc[m, v].to_numpy(), blocs[m]
        pred = np.zeros_like(y)
        for tr, te in GroupKFold(n_splits=5).split(x, y, g):
            pred[te] = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(x[tr], y[tr]).predict(x[te])
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        print(f"   {v:18s} {m.sum():6d} tronçons, {len(np.unique(g)):3d} blocs | R² {r2:+.3f}")

    couverts = np.all([(d[f"couv_{v}"] >= 0.5).values & d[v].notna().values for v in IRDA], axis=0)
    xa = d[actuels].to_numpy()
    xi = d[IRDA].to_numpy()
    za = (xa - xa[couverts].mean(0)) / np.maximum(xa[couverts].std(0), 1e-9)
    zi = (xi - np.nanmean(xi[couverts], 0)) / np.maximum(np.nanstd(xi[couverts], 0), 1e-9)
    p = voisins[couverts[voisins[:, 0]] & couverts[voisins[:, 1]]]
    da = np.linalg.norm(za[p[:, 0]] - za[p[:, 1]], axis=1) / np.sqrt(za.shape[1])
    di = np.linalg.norm(zi[p[:, 0]] - zi[p[:, 1]], axis=1) / np.sqrt(zi.shape[1])
    proches = da <= np.quantile(da, 0.25)
    print(f"\n2. séparation, sur {len(p)} paires de tronçons voisins couverts pour toutes les propriétés")
    print(f"   quart des paires les plus semblables en attributs actuels : écart moyen {da[proches].mean():.3f} écart-type par attribut")
    print(f"   parmi elles, écart IRDA médian {np.median(di[proches]):.3f} ; part au-dessus de 0,5 écart-type : {(di[proches] > 0.5).mean():.3f}")
    print(f"   rang effectif de l'entrée : {rang_effectif(xa[couverts]):.2f} sur {len(actuels)} sans l'IRDA, {rang_effectif(np.hstack([xa[couverts], xi[couverts]])):.2f} sur {len(actuels) + len(IRDA)} avec")
    return 0


if __name__ == "__main__":
    sys.exit(main())
