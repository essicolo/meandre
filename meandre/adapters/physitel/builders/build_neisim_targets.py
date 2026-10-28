"""Cible de masse du manteau tirée de NEISIM, pour un territoire.

Le réseau au sol ne couvre pas le domaine : 76 sites en Outaouais, aucun au Saint-Laurent
sud-ouest. Là où il se tait, rien ne contraint la masse du manteau. NEISIM couvre tout le
Québec méridional et s'accorde aux relevés au sol à 0,98 de rapport médian et 0,87 de
corrélation (`neisim_contre_canswe.py`), ce qui en fait une extension défendable de la
contrainte, à la condition de dire que c'est un modèle et non une mesure.

Le fichier écrit porte la même forme que les cibles CanSWE, valeurs (T, n_sites) et nœud de
chaque site, pour entrer dans la perte sans terme nouveau.

    python -m meandre.adapters.physitel.builders.build_neisim_targets OUTV --pas 4
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd


from meandre.adapters.physitel import neisim_loader as _nl
from meandre.utils import paths as _paths

DATE_START, DATE_END = "2000-01-01", "2024-12-31"


def territory_nodes(reg):
    """Coordonnées des nœuds, dans l'ordre de node_idx."""
    import duckdb

    file_path = f"{_paths.DATA_ROOT}/quebec/{reg.lower()}.duckdb"
    if not os.path.exists(file_path):
        raise SystemExit(f"base absente : {file_path}")
    con = duckdb.connect(file_path, read_only=True)
    df = con.execute("SELECT node_id, lon, lat FROM nodes ORDER BY node_idx").df()
    con.close()
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("regions", nargs="+")
    p.add_argument("--pas", type=int, default=1,
                   help="ne garder qu'un nœud sur N ; les nœuds voisins partagent la cellule")
    a = p.parse_args()

    if not _nl.file_present():
        raise SystemExit(f"fichier NEISIM absent : {_nl.DEFAULT_FILE}")
    times = pd.date_range(DATE_START, DATE_END, freq="D")

    # UNE SEULE LECTURE POUR TOUS LES TERRITOIRES. Le fichier se lit par tranches de temps
    # et la traversee complete prend quelques minutes : on reunit d'abord les cellules de
    # tous les territoires demandes, on lit une fois, puis on decoupe.
    plans = {}
    toutes = []
    for reg in [r.lower() for r in a.regions]:
        df = territory_nodes(reg)
        kept = np.arange(0, len(df), a.pas)
        cells, rank, _lo, _la, time_n = _nl.node_cells(df["lon"].to_numpy()[kept],
                                                                   df["lat"].to_numpy()[kept])
        plans[reg] = (kept, cells, rank, len(df))
        toutes.append(cells)
        print(f"{reg} : {len(df)} nœuds, {len(kept)} retenus, {len(cells)} cellules de grille")
    union, retour = np.unique(np.concatenate(toutes), axis=0, return_inverse=True)
    print(f"{len(union)} cellules distinctes en tout", flush=True)

    series = _nl.read_cells(union)
    _lo, _la, time_n = _nl._grid(_nl.DEFAULT_FILE)
    axe = pd.DatetimeIndex(times).normalize()
    pos = pd.Series(np.arange(len(time_n)), index=time_n)
    pos = pos[~pos.index.duplicated()]
    ou = pos.reindex(axe).to_numpy()
    vu = np.isfinite(ou)

    start = 0
    for reg, (kept, cells, rank, n_total) in plans.items():
        in_union = np.ravel(retour)[start:start + len(cells)]
        start += len(cells)
        value_list = np.full((len(axe), len(kept)), np.nan, dtype="float32")
        if vu.any():
            value_list[vu] = series[:, ou[vu].astype(int)].T[:, in_union[rank]]
        winter = np.isin(times.month, (11, 12, 1, 2, 3, 4, 5))
        print("")
        print(f"{reg} : {100 * np.isfinite(value_list).mean():.1f} % de valeurs renseignées")
        print(f"  équivalent en eau moyen de novembre à mai : {np.nanmean(value_list[winter]):.0f} mm")
        print(f"  maximum médian par nœud : {np.nanmedian(np.nanmax(value_list, axis=0)):.0f} mm")
        step_output = f"{_paths.DERIVED_ROOT}/auxiliaires/neisim-{reg}.npz"
        os.makedirs(os.path.dirname(step_output), exist_ok=True)
        np.savez_compressed(step_output, value_list=value_list, node_idx=kept.astype("int64"),
                            n_nodes=n_total, times=times.to_numpy().astype("datetime64[D]"))
        print(f"  écrit : {step_output} ({os.path.getsize(step_output) / 1e6:.1f} Mo)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
