"""La distance de Gower predit-elle le bon donneur ? (test propose par Essi, 2026-09-01)

Le rapport transferait un donneur unique sur les neuf regions peu jaugees. La mesure du
7 aout montre que le choix du donneur vaut 0.20 de KGE sur OUTV. Mesurer les 54 couples
region-donneur coute cinq heures ; si une distance calculee sur les attributs du
territoire designe le meme gagnant, la regle devient gratuite et s'applique a toute
region future, jaugee ou non.

La distance de Gower convient ici parce que les attributs sont heterogenes : des
fractions bornees a [0, 1], une aire drainee qui court sur quatre ordres de grandeur,
un ordre de Strahler entier. Chaque variable est ramenee a son etendue avant d'etre
moyennee, ce qu'une distance euclidienne ne fait pas.

RESERVE ASSUMEE : les fractions granulometriques et d'occupation sont compositionnelles,
et les traiter comme des variables independantes est incorrect (dette 24 du registre).
La version 1.1 passera par les log-ratios isometriques du paquet nuee. Ce script mesure
d'abord si l'idee tient avec le traitement naif.

  python .runs/quebec/gower_donneur.py
"""
import os
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd

from meandre.utils import paths as _paths

MESURES = f"{_paths.DATA_ROOT}/quebec/donneurs.txt"
LOCALES = ["outv", "slno", "gasp", "sagu", "mont", "slso"]
TRANSFEREES = ["abit", "cnda", "cndb", "cndc", "cndd", "cnde", "labi", "outm", "vaud"]
# Attributs retenus : ceux qui gouvernent la physique de la colonne et du versant.
ATTRIBUTS = ["mean_slope_pct", "mean_elevation_m", "f_forest", "f_agriculture", "f_urban",
             "f_wetland", "f_water", "f_sand", "f_silt", "f_clay",
             "depth_to_bedrock_m", "lake_fraction"]


def profil(reg):
    """Profil d'une region : mediane de chaque attribut, ponderee par l'aire du troncon."""
    f = f"{_paths.DATA_ROOT}/quebec/territorial-raw-{reg}.parquet"
    if not os.path.exists(f):
        return None
    d = pd.read_parquet(f)
    cols = [c for c in ATTRIBUTS if c in d.columns]
    return d[cols].median()


def gower(profils):
    """Matrice de distances de Gower : moyenne des ecarts absolus normalises par l'etendue."""
    M = pd.DataFrame(profils).T
    etendue = (M.max() - M.min()).replace(0, np.nan)
    n = len(M)
    D = pd.DataFrame(np.zeros((n, n)), index=M.index, columns=M.index)
    for a in M.index:
        for b in M.index:
            D.loc[a, b] = float(((M.loc[a] - M.loc[b]).abs() / etendue).mean(skipna=True))
    return D


profils = {r: p for r in LOCALES + TRANSFEREES if (p := profil(r)) is not None}
D = gower(profils)

if not os.path.exists(MESURES):
    print(f"Mesures absentes ({MESURES}) : distances seules.")
    print(D.round(3).to_string())
    raise SystemExit

m = pd.read_csv(MESURES, sep=" ", names=["region", "donneur", "kge"])
m = m[m.kge.astype(str).str.replace(".", "", 1).str.lstrip("-").str.isdigit()]
m["kge"] = m.kge.astype(float)
m["source"] = m.donneur.str.split("-").str[0]

lignes = []
for reg, g in m.groupby("region"):
    if reg not in D.index:
        continue
    g = g[g.source.isin(D.index)].copy()
    g["gower"] = [float(D.loc[reg, s]) for s in g.source]
    mesure = g.loc[g.kge.idxmax()]
    proche = g.loc[g.gower.idxmin()]
    lignes.append({
        "région": reg,
        "meilleur mesuré": mesure.source, "KGE": round(mesure.kge, 3),
        "plus proche (Gower)": proche.source, "KGE de ce choix": round(proche.kge, 3),
        "coût de la règle": round(mesure.kge - proche.kge, 3),
        "corrélation rang": round(g.kge.corr(g.gower, method="spearman"), 2)})

t = pd.DataFrame(lignes).set_index("région")
print(t.to_string())
print(f"\nCoût médian de choisir par Gower plutôt que par mesure : "
      f"{t['coût de la règle'].median():.3f} de KGE")
print(f"Corrélation de rang médiane entre distance et KGE (négative = la règle marche) : "
      f"{t['corrélation rang'].median():.2f}")
t.to_csv(f"{_paths.DATA_ROOT}/quebec/gower_donneur.csv")
