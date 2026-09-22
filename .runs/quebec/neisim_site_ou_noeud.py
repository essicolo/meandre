"""Combien coûte l'échantillonnage de NEISIM au NŒUD plutôt qu'au SITE.

La comparaison des trois sources prend NEISIM au centroïde du tronçon qui porte le site, et
non au site lui-même. Le filtre de représentativité accepte jusqu'à quinze kilomètres de
distance : une part de l'écart au réseau au sol peut donc venir de ce déplacement plutôt
que du modèle. Ce banc la mesure, en lisant la grille aux deux endroits.

    .venv/Scripts/python.exe .runs/quebec/neisim_site_ou_noeud.py OUTV
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from meandre.data import neisim_loader as _nl
from meandre.data.basin_cache import BasinCache
from meandre.data.canswe_loader import build_swe_targets
from meandre.utils import paths as _paths

DATE_START, DATE_END = "2000-01-01", "2024-12-31"
MOIS = ("jan", "fev", "mar", "avr", "mai", "jui", "jul", "aou", "sep", "oct", "nov", "dec")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("region")
    a = p.parse_args()
    reg = a.region.lower()

    import duckdb

    cache = BasinCache(f"{_paths.DATA_ROOT}/quebec/{reg}.duckdb")
    times = pd.date_range(DATE_START, DATE_END, freq="D")
    mes, sit = cache.load_canswe(DATE_START, DATE_END)
    obs, node_idx, gardes = build_swe_targets(mes, sit, times)
    if obs is None:
        raise SystemExit(f"{reg} : aucun site ne passe les filtres de representativite")
    obs = obs.cpu().numpy()
    node_idx = node_idx.cpu().numpy()
    print(f"{reg} : {len(gardes)} sites, distance au nœud mediane {gardes['dist_km'].median():.1f} km, "
          f"maximum {gardes['dist_km'].max():.1f} km")

    con = duckdb.connect(f"{_paths.DATA_ROOT}/quebec/{reg}.duckdb", read_only=True)
    noeuds = con.execute("SELECT lon, lat FROM nodes ORDER BY node_idx").df()
    con.close()

    lon_site, lat_site = gardes["lon"].to_numpy(), gardes["lat"].to_numpy()
    lon_noeud = noeuds["lon"].to_numpy()[node_idx]
    lat_noeud = noeuds["lat"].to_numpy()[node_idx]
    # Une seule traversee du fichier pour les deux echantillonnages.
    cel_s, rang_s, _lo, _la, temps_n = _nl.cellules_des_noeuds(lon_site, lat_site)
    cel_n, rang_n, _lo, _la, _t = _nl.cellules_des_noeuds(lon_noeud, lat_noeud)
    union, retour = np.unique(np.concatenate([cel_s, cel_n]), axis=0, return_inverse=True)
    retour = np.ravel(retour)
    series = _nl.lire_cellules(union)

    axe = pd.DatetimeIndex(times).normalize()
    pos = pd.Series(np.arange(len(temps_n)), index=temps_n)
    pos = pos[~pos.index.duplicated()]
    ou = pos.reindex(axe).to_numpy()
    vu = np.isfinite(ou)
    col_s = retour[:len(cel_s)][rang_s]
    col_n = retour[len(cel_s):][rang_n]
    v_site = np.full((len(axe), len(lon_site)), np.nan, dtype="float32")
    v_noeud = np.full_like(v_site, np.nan)
    v_site[vu] = series[:, ou[vu].astype(int)].T[:, col_s]
    v_noeud[vu] = series[:, ou[vu].astype(int)].T[:, col_n]

    mois = times.month.to_numpy()
    print("")
    print(f"{'mois':<6s} {'releves':>8s} {'au sol':>9s} {'au site':>9s} {'au nœud':>9s} "
          f"{'site/sol':>9s} {'nœud/sol':>9s}")
    for m in range(1, 13):
        o = obs[mois == m]
        if int(np.isfinite(o).sum()) < 30:
            continue
        vus = np.isfinite(o).any(axis=0)
        sol = float(np.nanmean(o[:, vus]))
        s = float(np.nanmean(v_site[mois == m][:, vus]))
        n = float(np.nanmean(v_noeud[mois == m][:, vus]))
        print(f"{MOIS[m - 1]:<6s} {int(np.isfinite(o).sum()):>8d} {sol:>7.0f} mm {s:>7.0f} mm "
              f"{n:>7.0f} mm {s / max(sol, 1e-6):>9.2f} {n / max(sol, 1e-6):>9.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
