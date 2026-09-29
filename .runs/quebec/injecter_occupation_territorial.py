"""Écrit les fractions BRUTES d'occupation du sol par tronçon dans la table `territorial`.

Les bases construites avant le 2026-08-10 n'ont que les fractions centrées-réduites, invisibles
pour la physique : la colonne y met toute la forêt en feuillus et ne peut pas distinguer, pour
la phénologie, ce qui perd ses feuilles à l'équinoxe de ce qui n'en perd pas. On recalcule ici
les mêmes fractions que le chargeur PHYSITEL (`meandre/data/physitel_loader.py`), à partir de
`occupation_sol.cla` et `troncon.trl` du projet, et on les ajoute en colonnes `_raw`, que le
cache range dans le dictionnaire physique sans normalisation. Le nombre d'attributs du champ
spatial ne change pas : les points de reprise se rechargent.

    python .runs/quebec/injecter_occupation_territorial.py outv mont
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import duckdb
import numpy as np
import pandas as pd

from meandre.data.physitel_loader import _parse_occupation_sol_cla, _parse_troncon, _parse_uhrh
from meandre.utils import paths as _p

# Classes de la grille d'occupation d'Hydrotel, dans l'ordre du fichier .cla.
# 0 no_data, 1 eau, 2 sol_nu, 3 foret_feuillus, 4 agricole_paturage, 5 foret_coniferes,
# 6 impermeable, 7 tourbiere, 8 milieu_humide
COLONNES = {
    "f_forest_raw": (3, 5),
    "f_forest_deciduous_raw": (3,),
    "f_forest_conifer_raw": (5,),
    "f_agriculture_raw": (2, 4),
    "f_bare_raw": (2,),
    "f_urban_raw": (6,),
    "f_wetland_raw": (7, 8),
    "f_water_raw": (1,),
}


def projet(reg):
    racine = Path(os.environ.get("MEANDRE_PLATEFORMES", f"{_p.DATA_ROOT}/plateformes-hydrotel")) / "LN24HA"
    cand = sorted(racine.glob(f"{reg.upper()}_*"))
    if not cand:
        raise FileNotFoundError(f"aucun projet PHYSITEL pour {reg} sous {racine}")
    return cand[0] / "physitel"


def fractions(reg):
    ph = projet(reg)
    uhrh = _parse_uhrh(ph / "uhrh.csv")
    occ = _parse_occupation_sol_cla(ph / "occupation_sol.cla", uhrh)
    troncons = _parse_troncon(ph / "troncon.trl")
    lignes = []
    for t in troncons:
        total = np.zeros(9)
        for u in t["uhrh_ids"]:
            total += occ.get(u, np.zeros(9))
        n = total[1:].sum()
        ligne = {"node_id": int(t["id"])}
        for col, cls in COLONNES.items():
            ligne[col] = float(sum(total[c] for c in cls) / n) if n > 0 else float("nan")
        lignes.append(ligne)
    return pd.DataFrame(lignes)


def main(regs):
    for reg in regs:
        f = fractions(reg)
        con = duckdb.connect(f"{_p.DATA_ROOT}/quebec/{reg}.duckdb")
        nodes = con.execute("SELECT node_idx, node_id FROM nodes").df()
        f = nodes.merge(f, on="node_id", how="left").drop(columns="node_id")
        presentes = {c[0] for c in con.execute("DESCRIBE territorial").fetchall()}
        con.register("occ", f)
        for col in COLONNES:
            if col not in presentes:
                con.execute(f"ALTER TABLE territorial ADD COLUMN {col} FLOAT")
            con.execute(f"UPDATE territorial SET {col} = occ.{col} FROM occ WHERE territorial.node_idx = occ.node_idx")
        con.unregister("occ")
        moy = con.execute("SELECT avg(f_forest_deciduous_raw), avg(f_forest_conifer_raw), avg(f_agriculture_raw), count(*) FILTER (WHERE f_forest_raw IS NULL) FROM territorial").fetchone()
        con.close()
        print(f"{reg} : {len(f)} troncons | feuillus {moy[0]:.3f}, coniferes {moy[1]:.3f}, agricole {moy[2]:.3f} en moyenne | {moy[3]} troncons sans occupation")


if __name__ == "__main__":
    main(sys.argv[1:])
