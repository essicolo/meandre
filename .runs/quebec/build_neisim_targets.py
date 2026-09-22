"""Cible de masse du manteau tirée de NEISIM, pour un territoire.

Le réseau au sol ne couvre pas le domaine : 76 sites en Outaouais, aucun au Saint-Laurent
sud-ouest. Là où il se tait, rien ne contraint la masse du manteau. NEISIM couvre tout le
Québec méridional et s'accorde aux relevés au sol à 0,98 de rapport médian et 0,87 de
corrélation (`neisim_contre_canswe.py`), ce qui en fait une extension défendable de la
contrainte, à la condition de dire que c'est un modèle et non une mesure.

Le fichier écrit porte la même forme que les cibles CanSWE, valeurs (T, n_sites) et nœud de
chaque site, pour entrer dans la perte sans terme nouveau.

    .venv/Scripts/python.exe .runs/quebec/build_neisim_targets.py OUTV --pas 4
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from meandre.data import neisim_loader as _nl
from meandre.utils import paths as _paths

DATE_START, DATE_END = "2000-01-01", "2024-12-31"


def noeuds_du_territoire(reg):
    """Coordonnées des nœuds, dans l'ordre de node_idx."""
    import duckdb

    chemin = f"{_paths.DATA_ROOT}/quebec/{reg.lower()}.duckdb"
    if not os.path.exists(chemin):
        raise SystemExit(f"base absente : {chemin}")
    con = duckdb.connect(chemin, read_only=True)
    df = con.execute("SELECT node_id, lon, lat FROM nodes ORDER BY node_idx").df()
    con.close()
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("regions", nargs="+")
    p.add_argument("--pas", type=int, default=1,
                   help="ne garder qu'un nœud sur N ; les nœuds voisins partagent la cellule")
    a = p.parse_args()

    if not _nl.fichier_present():
        raise SystemExit(f"fichier NEISIM absent : {_nl.DEFAULT_FILE}")
    times = pd.date_range(DATE_START, DATE_END, freq="D")

    # UNE SEULE LECTURE POUR TOUS LES TERRITOIRES. Le fichier se lit par tranches de temps
    # et la traversee complete prend quelques minutes : on reunit d'abord les cellules de
    # tous les territoires demandes, on lit une fois, puis on decoupe.
    plans = {}
    toutes = []
    for reg in [r.lower() for r in a.regions]:
        df = noeuds_du_territoire(reg)
        garde = np.arange(0, len(df), a.pas)
        cellules, rang, _lo, _la, temps_n = _nl.cellules_des_noeuds(df["lon"].to_numpy()[garde],
                                                                   df["lat"].to_numpy()[garde])
        plans[reg] = (garde, cellules, rang, len(df))
        toutes.append(cellules)
        print(f"{reg} : {len(df)} nœuds, {len(garde)} retenus, {len(cellules)} cellules de grille")
    union, retour = np.unique(np.concatenate(toutes), axis=0, return_inverse=True)
    print(f"{len(union)} cellules distinctes en tout", flush=True)

    series = _nl.lire_cellules(union)
    _lo, _la, temps_n = _nl._grille(_nl.DEFAULT_FILE)
    axe = pd.DatetimeIndex(times).normalize()
    pos = pd.Series(np.arange(len(temps_n)), index=temps_n)
    pos = pos[~pos.index.duplicated()]
    ou = pos.reindex(axe).to_numpy()
    vu = np.isfinite(ou)

    debut = 0
    for reg, (garde, cellules, rang, n_total) in plans.items():
        dans_union = np.ravel(retour)[debut:debut + len(cellules)]
        debut += len(cellules)
        valeurs = np.full((len(axe), len(garde)), np.nan, dtype="float32")
        if vu.any():
            valeurs[vu] = series[:, ou[vu].astype(int)].T[:, dans_union[rang]]
        hiver = np.isin(times.month, (11, 12, 1, 2, 3, 4, 5))
        print("")
        print(f"{reg} : {100 * np.isfinite(valeurs).mean():.1f} % de valeurs renseignées")
        print(f"  équivalent en eau moyen de novembre à mai : {np.nanmean(valeurs[hiver]):.0f} mm")
        print(f"  maximum médian par nœud : {np.nanmedian(np.nanmax(valeurs, axis=0)):.0f} mm")
        sortie = f"{_paths.DERIVED_ROOT}/auxiliaires/neisim-{reg}.npz"
        os.makedirs(os.path.dirname(sortie), exist_ok=True)
        np.savez_compressed(sortie, valeurs=valeurs, node_idx=garde.astype("int64"),
                            n_nodes=n_total, times=times.to_numpy().astype("datetime64[D]"))
        print(f"  écrit : {sortie} ({os.path.getsize(sortie) / 1e6:.1f} Mo)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
