"""L'indice d'écoulement de base répond-il MÉCANIQUEMENT à la variabilité.

Sur neuf modèles, l'indice d'écoulement de base et le facteur gamma du KGE, qui est le
rapport des variabilités simulée et observée, vont en sens contraires avec une corrélation de
rang de -0,93. Deux lectures s'opposent : ou les modèles les plus variables ont réellement
trop d'écoulement rapide, ou le filtre de séparation baisse tout seul quand la variabilité
monte, auquel cas la relation ne dit rien des chemins de l'eau.

Le contrôle se fait sur l'OBSERVATION, qui n'a pas de paramètres : on déforme chaque
hydrogramme observé pour lui donner la variabilité voulue, et on regarde ce que le filtre en
dit. La déformation se fait en logarithme, qui préserve la positivité et la forme relative :
q_k = exp(moy(log q) + k (log q - moy(log q))).

    .venv/bin/python .runs/quebec/indice_base_contre_gamma.py outv
"""
import argparse
import os
import sys
from importlib.machinery import SourceFileLoader

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from meandre.utils import paths as _paths

_ici = os.path.dirname(os.path.abspath(__file__))
_ib = SourceFileLoader("ib", os.path.join(_ici, "indice_base_observe.py")).load_module()

DATE_START, DATE_END = "2000-01-01", "2024-12-31"


def _deforme(q, k):
    """Hydrogramme de même moyenne géométrique, de variabilité multipliée par k en log."""
    l = np.log(np.clip(q, 1e-6, None))
    return np.exp(l.mean() + k * (l - l.mean()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("region")
    p.add_argument("--variantes", nargs="*", default=[],
                   help="triplets nom:gamma:indice, lus dans les journaux d'evaluation")
    a = p.parse_args()
    import duckdb

    con = duckdb.connect(f"{_paths.DATA_ROOT}/quebec/{a.region.lower()}.duckdb", read_only=True)
    d = con.execute("SELECT station_id, date, discharge FROM observations").df()
    con.close()
    d = d.dropna(subset=["discharge"])
    axe = pd.date_range(DATE_START, DATE_END, freq="D")
    large = d.pivot_table(index="date", columns="station_id", values="discharge", aggfunc="mean")
    large = large.reindex(axe)

    print(f"{a.region.lower()} : {large.shape[1]} stations\n")
    print(f"{'facteur en log':>15s} {'gamma obtenu':>13s} {'indice de base':>15s}")
    base = None
    for k in (0.80, 0.90, 1.00, 1.10, 1.20, 1.30):
        gammas, indices = [], []
        for c in large.columns:
            q = large[c].to_numpy()
            fini = np.isfinite(q)
            if fini.sum() < 2000:
                continue
            qo = q[fini]
            qk = _deforme(qo, k)
            # gamma du KGE : rapport des coefficients de variation.
            g = (qk.std() / qk.mean()) / (qo.std() / qo.mean())
            gammas.append(g)
            indices.append(float(np.nansum(_ib.lyne_hollick(qk)) / np.nansum(qk)))
        if not indices:
            continue
        if k == 1.0:
            base = np.median(indices)
        print(f"{k:>15.2f} {np.median(gammas):>13.3f} {np.median(indices):>15.3f}")
    print("")
    print(f"Indice de l'observation intacte : {base:.3f}.")

    # DEFICIT A VARIABILITE EGALE. Comparer l'indice d'un modele a celui de l'observation
    # INTACTE melange deux choses : ce que le filtre fait mecaniquement quand la variabilite
    # change, et ce que le modele met reellement dans le chemin rapide. On compare donc
    # chaque modele a l'observation DEFORMEE jusqu'a sa propre variabilite.
    if not a.variantes:
        return 0
    courbe_g, courbe_i = [], []
    for k in np.arange(0.6, 1.8, 0.05):
        gs, ins = [], []
        for c in large.columns:
            q = large[c].to_numpy()
            fini = np.isfinite(q)
            if fini.sum() < 2000:
                continue
            qo = q[fini]
            qk = _deforme(qo, k)
            gs.append((qk.std() / qk.mean()) / (qo.std() / qo.mean()))
            ins.append(float(np.nansum(_ib.lyne_hollick(qk)) / np.nansum(qk)))
        courbe_g.append(np.median(gs))
        courbe_i.append(np.median(ins))
    courbe_g, courbe_i = np.array(courbe_g), np.array(courbe_i)
    print("")
    print(f"{'variante':<24s} {'gamma':>7s} {'indice':>8s} {'attendu':>9s} {'deficit':>9s}")
    for spec in a.variantes:
        nom, g, i = spec.split(":")
        g, i = float(g), float(i)
        attendu = float(np.interp(g, courbe_g, courbe_i))
        print(f"{nom:<24s} {g:>7.3f} {i:>8.3f} {attendu:>9.3f} {i - attendu:>9.3f}")
    print("")
    print("Le deficit est ce que le filtre ne peut PAS expliquer par la variabilite.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
