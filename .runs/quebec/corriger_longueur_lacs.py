"""Remet la longueur d'un troncon de lac a la racine de sa surface, dans les bases et dans
les attributs provinciaux, sans toucher a la numerotation des noeuds.

Mesure du 2026-09-15 : la colonne `edge_attr_0` de la table `edges` contient l'AIRE du lac
en metres carres pour les troncons de type lac, rapport a `lake_area_m2` de 1,0000 avec un
ecart-type nul sur les 348 aretes lacustres du Saguenay. Le code ne fait plus cette erreur
depuis le 3 septembre, mais aucune base n'a ete reconstruite depuis. La distance a
l'exutoire, cumulee sur cette colonne et consommee par le champ spatial comme l'un de ses
seize canaux, vaut donc 4 124 km en moyenne provinciale au lieu de 252 a 394 selon la
region, et melange l'ordre des troncons la ou les lacs abondent.

Ce script ne reconstruit PAS les bases : il corrige les deux colonnes fautives. Une
reconstruction complete deplacerait la numerotation des noeuds et invaliderait les points
de reprise et tous les caches.

    .venv/Scripts/python.exe .runs/quebec/corriger_longueur_lacs.py --verifier
    .venv/Scripts/python.exe .runs/quebec/corriger_longueur_lacs.py --appliquer
"""
import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from meandre.data.physitel_loader import _parse_troncon
from meandre.utils import paths as _paths

REGIONS = ["outv", "gasp", "sagu", "mont", "slno", "slso", "abit", "cnda", "cndb",
           "cndc", "cndd", "cnde", "labi", "outm", "vaud"]
PLATE = os.environ.get("IRDA_PLATEFORME", "LN24HA")


def longueurs_physitel(reg):
    p = Path(f"{_paths.PLATFORMS_ROOT}/{PLATE}/{reg.upper()}_{PLATE}_2020/physitel/troncon.trl")
    if not p.exists():
        return None
    return {t["id"]: t["length_m"] for t in _parse_troncon(p)}


def distance_exutoire(nd, ed, lg_par_noeud):
    """Distance le long du reseau jusqu'a l'exutoire, en kilometres."""
    n = len(nd)
    aval = -np.ones(n, dtype=int)
    aval[ed.src.values.astype(int)] = ed.dst.values.astype(int)
    d = np.zeros(n)
    for j in nd.sort_values("topo_order", ascending=False).node_idx.values.astype(int):
        k = aval[j]
        d[j] = lg_par_noeud[j] + (d[k] if k >= 0 else 0.0)
    return d / 1000.0


def main(appliquer):
    racine = _paths.DATA_ROOT
    rapport = []
    for reg in REGIONS:
        base = f"{racine}/quebec/{reg}.duckdb"
        if not os.path.exists(base):
            continue
        lg = longueurs_physitel(reg)
        if lg is None:
            print(f"{reg}: projet PHYSITEL absent, ignoré")
            continue
        cx = duckdb.connect(base, read_only=not appliquer)
        nd = cx.sql("select node_idx, node_id, topo_order from nodes order by node_idx").fetchdf()
        ed = cx.sql("select rowid as rid, src, dst, edge_attr_0 from edges").fetchdf()
        nid = dict(zip(nd.node_idx.astype(int), nd.node_id.astype(int)))
        neuf = np.array([lg.get(nid.get(int(s)), np.nan) for s in ed.src.values])
        ok = np.isfinite(neuf)
        change = ok & (np.abs(neuf - ed.edge_attr_0.values) > 1.0)
        n = len(nd)
        lgn_avant = np.zeros(n)
        lgn_avant[ed.src.values.astype(int)] = ed.edge_attr_0.values
        lgn_apres = lgn_avant.copy()
        lgn_apres[ed.src.values.astype(int)[ok]] = neuf[ok]
        d_avant = distance_exutoire(nd, ed, lgn_avant)
        d_apres = distance_exutoire(nd, ed, lgn_apres)
        rapport.append({"region": reg, "aretes": len(ed), "corrigees": int(change.sum()),
                        "dist_avant_med": float(np.median(d_avant)),
                        "dist_apres_med": float(np.median(d_apres)),
                        "dist_avant_max": float(d_avant.max()),
                        "dist_apres_max": float(d_apres.max())})
        if appliquer:
            # La distance a l'exutoire vit a DEUX endroits que le champ lit : la table
            # `territorial` de la base, employee quand la normalisation est regionale, et
            # `territorial-raw-QC.parquet`, employee quand elle est provinciale. Corriger
            # la seule table des aretes ne changerait rien pour le modele.
            _cx2 = duckdb.connect(base)
            try:
                # La table territoriale stocke les canaux continus CENTRES ET REDUITS,
                # moyenne nulle et ecart-type un ; y ecrire des kilometres bruts placerait
                # ce canal sur une echelle cent fois celle des quinze autres. Bevue commise
                # puis corrigee le 2026-09-15.
                _z = (d_apres - d_apres.mean()) / (d_apres.std() + 1e-9)
                _cx2.register("_d", pd.DataFrame({"node_idx": nd.node_idx.values.astype(int),
                                                  "dd": _z.astype("float32")}))
                _cx2.execute("UPDATE territorial SET dist_to_outlet_km = "
                             "(SELECT dd FROM _d WHERE _d.node_idx = territorial.node_idx)")
                print(f"{reg}: distance à l'exutoire corrigée dans la table territoriale")
            except Exception as _e:
                print(f"{reg}: table territoriale non corrigée ({type(_e).__name__}: {_e})")
            finally:
                _cx2.close()
            _pq = f"{racine}/quebec/territorial-raw-QC.parquet"
            if os.path.exists(_pq):
                _rw = pd.read_parquet(_pq)
                _m = _rw.region == reg
                if int(_m.sum()) == len(nd):
                    _rw.loc[_m, "dist_to_outlet_km"] = d_apres.astype("float32")
                    _rw.to_parquet(_pq, index=False)
                    print(f"{reg}: distance à l'exutoire corrigée dans les attributs provinciaux")
                else:
                    print(f"{reg}: attributs provinciaux ignorés ({int(_m.sum())} vs {len(nd)})")
        if appliquer and change.any():
            cx.execute("CREATE OR REPLACE TEMP TABLE _corr (rid BIGINT, L FLOAT)")
            cx.register("_c", pd.DataFrame({"rid": ed.rid.values[change],
                                            "L": neuf[change].astype("float32")}))
            cx.execute("INSERT INTO _corr SELECT rid, L FROM _c")
            cx.execute("UPDATE edges SET edge_attr_0 = (SELECT L FROM _corr WHERE _corr.rid = edges.rowid) "
                       "WHERE rowid IN (SELECT rid FROM _corr)")
            print(f"{reg}: {int(change.sum())} arêtes corrigées dans la base")
        cx.close()
    d = pd.DataFrame(rapport)
    print(d.to_string(index=False))
    print("\ndistance à l'exutoire en kilomètres ; « appliquer » écrit dans les bases")
    return 0


if __name__ == "__main__":
    sys.exit(main("--appliquer" in sys.argv))
