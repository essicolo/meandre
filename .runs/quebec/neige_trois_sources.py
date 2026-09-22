"""Le manteau simulé, les relevés au sol et NEISIM, aux MÊMES nœuds et au même mois.

Le chantier de la neige porte un chiffre répété partout : 121 mm de manteau simulé contre
238 mesurés en Outaouais, soit la moitié. La comparaison à NEISIM sur les 3412 nœuds du
territoire donne pourtant un rapport de 1,04. Les deux ne peuvent pas être vrais du même
objet : ou les populations diffèrent, les sites du réseau n'étant pas le territoire, ou la
grandeur diffère.

Ce banc pose les trois sources sur la même population et la même grandeur : la moyenne
mensuelle, aux seuls nœuds qui portent un site du réseau.

    .venv/bin/python .runs/quebec/neige_trois_sources.py outv --variante finale-n2
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from meandre.data.basin_cache import BasinCache
from meandre.data.canswe_loader import build_swe_targets
from meandre.utils import paths as _paths

DERIVES = f"{os.environ.get('MEANDRE_DERIVES', _paths.DERIVED_ROOT)}/auxiliaires"
DATE_START, DATE_END = "2000-01-01", "2024-12-31"
MOIS = ("jan", "fev", "mar", "avr", "mai", "jui", "jul", "aou", "sep", "oct", "nov", "dec")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("region")
    p.add_argument("--variante", required=True)
    a = p.parse_args()
    reg = a.region.lower()

    cache = BasinCache(f"{_paths.DATA_ROOT}/quebec/{reg}.duckdb")
    if not cache.has_canswe():
        raise SystemExit(f"{reg} : pas de releves du reseau au sol dans la base")
    times = pd.date_range(DATE_START, DATE_END, freq="D")
    mes, sit = cache.load_canswe(DATE_START, DATE_END)
    obs, node_idx, gardes = build_swe_targets(mes, sit, times)
    if obs is None:
        raise SystemExit(f"{reg} : aucun site ne passe les filtres de representativite")
    obs = obs.cpu().numpy()
    node_idx = node_idx.cpu().numpy()
    print(f"{reg} : {len(gardes)} sites du reseau retenus, "
          f"{int(np.isfinite(obs).sum())} releves, sur {len(np.unique(node_idx))} nœuds distincts")

    z = np.load(f"{DERIVES}/reach-{reg}-{a.variante}.npz")
    sim_mens = z["swe_mensuel"][:, node_idx]
    zn = np.load(f"{DERIVES}/neisim-{reg}.npz")
    rang = {int(n): i for i, n in enumerate(zn["node_idx"])}
    col = np.array([rang.get(int(n), -1) for n in node_idx])
    tn = pd.DatetimeIndex(zn["times"])
    nei_mens = np.stack([np.nanmean(zn["valeurs"][tn.month == m], axis=0) for m in range(1, 13)])
    nei_mens = np.where(col[None, :] >= 0, nei_mens[:, np.clip(col, 0, None)], np.nan)

    mois_obs = times.month.to_numpy()
    print("")
    print(f"{'mois':<6s} {'releves':>8s} {'au sol':>9s} {'simule':>9s} {'NEISIM':>9s} "
          f"{'sim/sol':>9s} {'nei/sol':>9s}")
    for m in range(1, 13):
        o = obs[mois_obs == m]
        n_rel = int(np.isfinite(o).sum())
        if n_rel < 30:
            continue
        # Un site n'est compare que par les sites REELLEMENT releves ce mois-la.
        vus = np.isfinite(o).any(axis=0)
        sol = float(np.nanmean(o[:, vus]))
        sim = float(np.nanmean(sim_mens[m - 1, vus]))
        nei = float(np.nanmean(nei_mens[m - 1, vus]))
        print(f"{MOIS[m - 1]:<6s} {n_rel:>8d} {sol:>7.0f} mm {sim:>7.0f} mm {nei:>7.0f} mm "
              f"{sim / max(sol, 1e-6):>9.2f} {nei / max(sol, 1e-6):>9.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
