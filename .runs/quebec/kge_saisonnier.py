"""KGE et biais de volume PAR SAISON, la forme sous laquelle l'équipe d'Hydrotel publie.

Comparer notre KGE ANNUEL à leur KGE SAISONNIER n'est pas licite : calculer un KGE à
l'intérieur d'une saison retire le cycle saisonnier, c'est-à-dire la part du signal la plus
facile à reproduire. Le même modèle rend donc systématiquement moins en saisonnier qu'en
annuel, et l'écart n'a rien à voir avec sa qualité. C'est la quatrième fois que ce projet
bute sur une unité d'agrégation ; la règle est d'écrire celle des DEUX côtés avant de
comparer.

Ce banc calcule, sur les séries journalières aux stations, le KGE et le biais de volume par
saison, dans les mêmes bornes que la figure de l'École de technologie supérieure : hiver
décembre à février, printemps mars à mai, été juin à août, automne septembre à novembre.

    .venv/bin/python .runs/quebec/kge_saisonnier.py outv gasp --suffixe terr-affine
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

_racine = os.environ.get("MEANDRE_DERIVES")
if not _racine:
    from meandre.utils import paths as _paths

    _racine = _paths.DERIVED_ROOT
DERIVES = f"{_racine}/auxiliaires"
SAISONS = {"hiver": (12, 1, 2), "printemps": (3, 4, 5), "ete": (6, 7, 8), "automne": (9, 10, 11)}


def _kge(o, s):
    """KGE de Gupta 2009, et ses trois facteurs."""
    if o.size < 30 or o.std() == 0 or s.std() == 0:
        return np.nan, np.nan, np.nan, np.nan
    r = float(np.corrcoef(o, s)[0, 1])
    beta = float(s.mean() / o.mean())
    gamma = float((s.std() / s.mean()) / (o.std() / o.mean()))
    return 1.0 - float(np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)), r, beta, gamma


def main():
    p = argparse.ArgumentParser()
    p.add_argument("regions", nargs="+")
    p.add_argument("--suffixe", required=True, help="etiquette du dump, ex. terr-affine")
    a = p.parse_args()

    print(f"{'territoire':<12s} {'saison':<11s} {'n':>4s} {'KGE med':>8s} {'r':>7s} "
          f"{'beta':>7s} {'gamma':>7s} {'PBIAS %':>9s}")
    for reg in [x.lower() for x in a.regions]:
        f = f"{DERIVES}/q-{reg}-{a.suffixe}.npz"
        if not os.path.exists(f):
            print(f"{reg:<12s} sortie absente : {os.path.basename(f)}")
            continue
        z = np.load(f, allow_pickle=True)
        qs, qo = z["q_sim"], z["q_obs"]
        mois = pd.DatetimeIndex(z["dates"].astype(str)).month.to_numpy()
        for sais, m in SAISONS.items():
            masque = np.isin(mois, m)
            kges, rs, betas, gammas, pbias = [], [], [], [], []
            for j in range(qo.shape[1]):
                o, s = qo[masque, j], qs[masque, j]
                fini = np.isfinite(o) & np.isfinite(s)
                if fini.sum() < 30:
                    continue
                k, r, b, g = _kge(o[fini], s[fini])
                if not np.isfinite(k):
                    continue
                kges.append(k); rs.append(r); betas.append(b); gammas.append(g)
                pbias.append(100.0 * (s[fini].sum() - o[fini].sum()) / o[fini].sum())
            if not kges:
                continue
            print(f"{reg:<12s} {sais:<11s} {len(kges):>4d} {np.median(kges):>8.2f} "
                  f"{np.median(rs):>7.2f} {np.median(betas):>7.2f} {np.median(gammas):>7.2f} "
                  f"{np.median(pbias):>9.1f}")
    print("")
    print("Un KGE calcule DANS une saison retire le cycle saisonnier : il est")
    print("systematiquement plus bas qu'un KGE annuel du meme modele.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
