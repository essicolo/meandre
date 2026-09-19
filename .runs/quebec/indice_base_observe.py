"""Indice d'écoulement de base MESURÉ sur les hydrogrammes observés.

Pourquoi. Le modèle retenu produit 0,10 de son débit par la nappe en Outaouais et 0,13 au
Saint-Laurent nord-ouest, et il fallait savoir si c'est peu. Les ordres de grandeur de la
littérature sur bassins forestiers, de 0,3 à 0,5, ne valent pas une mesure sur les stations
qui servent à calculer le KGE. Le filtre récursif de Lyne et Hollick, trois passes, est le
séparateur standard depuis Nathan et McMahon (1990) ; il n'a aucun paramètre à caler au-delà
de son coefficient, dont 0,925 est la valeur usuelle au pas journalier.

Mesure du 2026-09-19 : 0,58 en médiane sur les 16 stations de l'Outaouais, quartiles 0,51 à
0,64, et 0,54 sur les 25 stations du Saint-Laurent nord-ouest. Le modèle en produit quatre à
six fois moins.

    .venv/bin/python .runs/quebec/indice_base_observe.py outv slno
"""
import glob
import os
import sys

import numpy as np

FLOTTE = os.environ.get("MEANDRE_FLOTTE", f"{os.environ.get('MEANDRE_DATA', '.')}/quebec/flotte")


def lyne_hollick(q, alpha: float = 0.925, passes: int = 3):
    """Sépare l'écoulement de base par le filtre récursif standard, trois passes."""
    y = q.copy()
    for p in range(passes):
        if p % 2 == 1:
            y = y[::-1]
        f = np.zeros_like(y)
        f[0] = y[0]
        for i in range(1, len(y)):
            f[i] = alpha * f[i - 1] + (1 + alpha) / 2 * (y[i] - y[i - 1])
        y = np.minimum(y - np.maximum(f, 0.0), y)
        if p % 2 == 1:
            y = y[::-1]
    return y


def indices(region: str, jours_minimum: int = 700):
    fichiers = sorted(glob.glob(f"{FLOTTE}/q-{region}-*.npz"))
    if not fichiers:
        return np.array([])
    z = np.load(fichiers[0], allow_pickle=True)
    obs = z["q_obs"]
    sortie = []
    for j in range(obs.shape[1]):
        s = obs[:, j]
        fini = np.isfinite(s)
        if fini.sum() < jours_minimum:
            continue
        s = np.where(fini, s, np.nanmedian(s))
        sortie.append(lyne_hollick(s).sum() / s.sum())
    return np.array(sortie)


def main(regions):
    for reg in regions:
        b = indices(reg)
        if len(b) == 0:
            print(f"{reg} : aucune série de station trouvée sous {FLOTTE}")
            continue
        print(f"{reg} {len(b):2d} stations | indice mesuré : médiane {np.median(b):.2f}, "
              f"quartiles {np.percentile(b, 25):.2f} à {np.percentile(b, 75):.2f}, "
              f"étendue {b.min():.2f} à {b.max():.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main([a.lower() for a in sys.argv[1:]] or ["outv", "slno"]))
