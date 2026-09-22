"""Les champs du NeRF sont-ils plats, et qu'est-ce qui les tient.

La crainte est fondée et ancienne : sur le bassin ouvert le champ s'était effondré vers des
paramètres quasi uniformes, coefficients de variation de 0,006 à 0,09 contre 0,2 à 0,47 sur
le réseau PHYSITEL, et l'ancrage sur la littérature en était la cause mathématique, sa forme
d'alors pénalisant la variance spatiale au même poids que le biais de moyenne.

Mais GRACE intègre sur des centaines de kilomètres : il contraint l'amplitude TEMPORELLE du
stockage moyen d'un bassin, pas un motif spatial. La masse du manteau et
l'évapotranspiration, elles, sont résolues dans l'espace. Retirer l'un ou l'autre ne retire
donc pas la même chose, et la question se tranche en mesurant.

La sortie par tronçon porte les 43 champs nœud par nœud : aucune simulation n'est nécessaire.

    .venv/bin/python .runs/quebec/platitude_des_champs.py outv --variantes depart essai3-vigueur
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
# Seuil sous lequel un champ est dit plat, repris du diagnostic d'effondrement de 2026-06-12.
PLAT = 0.10


def _cv(a):
    """Coefficient de variation spatial. En log pour les champs qui courent sur des
    ordres de grandeur, sinon l'écart-type suit la moyenne et ne dit rien."""
    a = np.asarray(a, dtype="float64")
    a = a[np.isfinite(a)]
    if a.size < 2:
        return np.nan
    m = a.mean()
    if m > 0 and a.min() > 0 and a.max() / a.min() > 20:
        l = np.log(a)
        return float(l.std() / max(abs(l.mean()), 1e-9))
    return float(a.std() / max(abs(m), 1e-9))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("region")
    p.add_argument("--variantes", nargs="+", required=True)
    p.add_argument("--detail", action="store_true", help="une ligne par champ")
    a = p.parse_args()
    reg = a.region.lower()

    tables = {}
    for var in a.variantes:
        f = f"{DERIVES}/reach-{reg}-{var}.npz"
        if not os.path.exists(f):
            print(f"{var} : sortie absente")
            continue
        z = np.load(f)
        tables[var] = {k[6:]: _cv(z[k]) for k in z.files if k.startswith("param_")}
    if not tables:
        return 1

    champs = sorted(set().union(*[set(t) for t in tables.values()]))
    noms = list(tables)
    print(f"{reg} : {len(champs)} champs du NeRF, coefficient de variation spatial\n")
    if a.detail:
        print(f"{'champ':<26s}" + "".join(f"{n[:16]:>18s}" for n in noms))
        for c in champs:
            print(f"{c:<26s}" + "".join(f"{tables[n].get(c, np.nan):>18.4f}" for n in noms))
        print("")
    print(f"{'variante':<26s} {'médiane':>9s} {'champs plats':>13s} {'les plus plats':>40s}")
    for n in noms:
        v = np.array([tables[n].get(c, np.nan) for c in champs])
        ok = np.isfinite(v)
        plats = [champs[i] for i in np.argsort(np.where(ok, v, np.inf))[:3]]
        print(f"{n:<26s} {np.nanmedian(v):>9.4f} {int((v[ok] < PLAT).sum()):>6d} / {int(ok.sum()):<4d} "
              f"{', '.join(plats):>40s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
