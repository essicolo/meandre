"""L'indice d'écoulement de base du modèle, mesuré comme celui des observations.

Comparer la part SOUTERRAINE de la production simulée à l'indice d'écoulement de base observé
est fautif, et c'est l'erreur qui guettait après l'ajout de la sortie latérale de la couche
profonde. Le filtre de Lyne et Hollick sépare le lent du rapide, pas les chemins de l'eau : un
écoulement hypodermique amorti par le versant et le tronçon lui apparaît comme de la base. La
seule comparaison licite passe le même filtre sur les deux séries, au même endroit.

    .venv/bin/python .runs/quebec/indice_base_simule.py outv slno --variantes s-temoin-g1234 lr3e-5-g1234
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

_racine = os.environ.get("MEANDRE_DERIVES")
if not _racine:
    from meandre.utils import paths as _paths

    _racine = _paths.DERIVED_ROOT
DERIVES = f"{_racine}/auxiliaires"

from importlib.machinery import SourceFileLoader

_ib = SourceFileLoader("ib", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "indice_base_observe.py")).load_module()


def indices(region, variante):
    """Indice simulé par station, et indice observé aux mêmes stations."""
    import glob

    f = f"{DERIVES}/reach-{region}-{variante}-journalier.npz"
    r = f"{DERIVES}/reach-{region}-{variante}.npz"
    if not (os.path.exists(f) and os.path.exists(r)):
        return None
    z, meta = np.load(f, allow_pickle=True), np.load(r, allow_pickle=True)
    if "station_idx" not in meta.files:
        return None
    flotte = os.environ.get("MEANDRE_FLOTTE", f"{os.environ.get('MEANDRE_DATA', '.')}/quebec/flotte")
    fichiers = sorted(glob.glob(f"{flotte}/q-{region}-*.npz"))
    if not fichiers:
        return None
    obs = np.load(fichiers[0], allow_pickle=True)["q_obs"]
    sim, noeuds = z["q"], meta["station_idx"]
    n = min(len(z["dates"]), obs.shape[0])
    sims, obss = [], []
    for j, nd in enumerate(noeuds):
        if j >= obs.shape[1] or nd < 0 or nd >= sim.shape[1]:
            continue
        o = obs[:n, j].astype(float)
        fini = np.isfinite(o)
        if fini.sum() < 700:
            continue
        s = sim[:n, int(nd)].astype(float)
        if not np.isfinite(s).all() or s.sum() <= 0:
            continue
        o = np.where(fini, o, np.nanmedian(o[fini]))
        sims.append(_ib.lyne_hollick(s).sum() / s.sum())
        obss.append(_ib.lyne_hollick(o).sum() / o.sum())
    return (np.array(sims), np.array(obss)) if sims else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("regions", nargs="+")
    p.add_argument("--variantes", nargs="+", required=True)
    a = p.parse_args()
    print(f"{'territoire':11s} {'variante':18s} | {'simule':>8s} | {'observe':>8s} | ecart")
    for reg in [r.lower() for r in a.regions]:
        for v in a.variantes:
            res = indices(reg, v)
            if res is None:
                continue
            s, o = res
            print(f"{reg:11s} {v:18s} | {np.median(s):8.3f} | {np.median(o):8.3f} | "
                  f"{np.median(s) - np.median(o):+.3f}  sur {len(s)} stations")
    print("\nLe meme filtre sur les deux series : c'est la seule comparaison licite, la part")
    print("souterraine de la PRODUCTION ne se compare pas a un indice mesure sur le debit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
