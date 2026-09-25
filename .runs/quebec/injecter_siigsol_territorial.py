"""Ecrit la texture SIIGSOL par troncon dans une table `territorial_siigsol` de chaque base,
que le chargeur joint aux attributs du champ quand MEANDRE_TERRITORIAL_EXTRA la nomme.

Motif (registre, 2026-09-24) : les attributs que le champ recoit, texture PHYSITEL en trois
classes, profondeur au roc et pente, predisent la part souterraine observee a -9 % contre le
temoin ; la texture SIIGSOL a 100 m, ingeree par troncon mais jamais donnee au champ, la
predit a +42 %. La table est separee de `territorial` pour ne pas changer la dimension
d'entree des points de reprise existants.

    .venv/Scripts/python.exe .runs/quebec/injecter_siigsol_territorial.py mont slso slno outv gasp sagu
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import duckdb
import pandas as pd

from meandre.utils import paths as _paths

SOURCE = f"{_paths.DERIVED_ROOT}/auxiliaires/siigsol-troncons.parquet"
TABLE = "territorial_siigsol"


def injecter(reg, reduit=False):
    """`reduit` : trois attributs au lieu de dix-huit, moyennes sur les profondeurs des deux
    coordonnees ilr de texture et du logarithme de la matiere organique, table
    `territorial_siigsol3`. Les dix-huit attributs, dont 38 % de noeuds a la moyenne en
    Monteregie, ont fait diverger le champ a une graine sur deux (banc du 2026-09-25)."""
    s = pd.read_parquet(SOURCE)
    s = s[s.region.str.lower() == reg]
    cols = [c for c in s.columns if c not in ("region", "troncon", "area_m2", "aire_m2") and not c.startswith("couv_")]
    if reduit:
        for base in ("texture_ilr_1", "texture_ilr_2", "log_matiere_organique"):
            s[base] = s[[c for c in cols if c.startswith(base)]].mean(axis=1)
        cols = ["texture_ilr_1", "texture_ilr_2", "log_matiere_organique"]
    table = TABLE + ("3" if reduit else "")
    t = s[["troncon"] + cols].copy()
    t["node_idx"] = t.troncon.astype(int) - 1
    t = t.drop(columns="troncon")
    con = duckdb.connect(f"{_paths.DATA_ROOT}/quebec/{reg}.duckdb")
    n = con.execute("SELECT count(*) FROM territorial").fetchone()[0]
    con.execute(f"DROP TABLE IF EXISTS {table}")
    con.execute(f"CREATE TABLE {table} AS SELECT * FROM t")
    couverts = con.execute(f"SELECT count(*) FROM {table} WHERE {cols[0]} IS NOT NULL").fetchone()[0]
    con.close()
    print(f"{reg}: {len(cols)} attributs SIIGSOL dans {table}, {couverts} tronçons renseignes sur {n}")


if __name__ == "__main__":
    reduit = "--reduit" in sys.argv
    for r in [a.lower() for a in sys.argv[1:] if not a.startswith("--")] or ["mont"]:
        injecter(r, reduit=reduit)
