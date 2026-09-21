"""La recharge est-elle identifiable, et laquelle de ses quatre facettes.

« Identifiable » ne veut rien dire tant qu'on n'a pas dit DE QUOI on parle. La recharge a
quatre facettes, et elles n'ont pas le même sort :

  volume    la lame annuelle, en millimètres par an ;
  phase     le mois du maximum et la forme du cycle saisonnier ;
  espace    sa variation d'un tronçon à l'autre ;
  paramètres les grandeurs de la colonne qui la produisent, percolation, plafond, temps de
            séjour.

La mesure repose sur l'hypothèse qu'une collègue formule ainsi : à l'échelle d'une station
hydrométrique, toute la recharge finit par faire résurgence en débit de base. La lame de
recharge se lit donc dans l'hydrogramme observé, comme la part souterraine du débit total. Le
prix de cette hypothèse est qu'elle hérite de l'ambiguïté du filtre de séparation, qui est
mesurée ici et non supposée petite.

    .venv/bin/python .runs/quebec/recharge_identifiable.py --variante lr3e-5-g1234
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from importlib.machinery import SourceFileLoader

_ici = os.path.dirname(os.path.abspath(__file__))
_fdb = SourceFileLoader("fdb", os.path.join(_ici, "filtres_debit_base.py")).load_module()

_racine = os.environ.get("MEANDRE_DERIVES")
if not _racine:
    from meandre.utils import paths as _paths

    _racine = _paths.DERIVED_ROOT
DERIVES = f"{_racine}/auxiliaires"

FILTRES = {"Lyne-Hollick": None, "Eckhardt 0,50": 0.50, "Eckhardt 0,65": 0.65,
           "Eckhardt 0,80": 0.80}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--region", default="outv")
    p.add_argument("--variante", required=True)
    a = p.parse_args()

    j = np.load(f"{DERIVES}/reach-{a.region}-{a.variante}-journalier.npz", allow_pickle=True)
    meta = np.load(f"{DERIVES}/reach-{a.region}-{a.variante}.npz", allow_pickle=True)
    mois = np.array([int(str(d)[5:7]) for d in j["dates"]])

    print("VOLUME. Lame de recharge annuelle du modele, et ce que l'observation permet d'en dire.\n")
    rech = float(j["recharge"].mean() * 365.25)
    print(f"  modele : {rech:.0f} mm/an")

    # La lame observee se deduit du debit de base rapporte a la surface. Faute d'une aire par
    # station dans le cache, on passe par la LAME totale du modele, qui est la meme quantite
    # geometrique, et on applique l'indice observe : c'est la part souterraine appliquee a la
    # lame ecoulee, et l'ambiguite mesuree est celle du filtre, pas celle de la surface.
    lame_totale = float((j["prod_surf"] + j["prod_hypo"] + j["prod_base"]).mean() * 365.25) \
        if "prod_hypo" in j.files else np.nan
    flotte = os.environ.get("MEANDRE_FLOTTE", f"{os.environ.get('MEANDRE_DATA', '.')}/quebec/flotte")
    g = sorted(glob.glob(f"{flotte}/q-{a.region}-*.npz"))
    obs = np.load(g[0], allow_pickle=True)["q_obs"]
    indices = {}
    for nom, bfi in FILTRES.items():
        vals = []
        for k in range(obs.shape[1]):
            q = obs[:, k].astype(float)
            fini = np.isfinite(q) & (q > 0)
            if fini.sum() < 700:
                continue
            qq = q[fini]
            b = (_fdb._ib.lyne_hollick(qq) if bfi is None
                 else _fdb.eckhardt(qq, _fdb.constante_recession(qq), bfi))
            vals.append(float(b.sum() / qq.sum()))
        indices[nom] = float(np.median(vals))
    print(f"  lame ecoulee du modele : {lame_totale:.0f} mm/an")
    for nom, ind in indices.items():
        print(f"  observation avec {nom:14s} : indice {ind:.3f} -> {ind * lame_totale:5.0f} mm/an")
    bas, haut = min(indices.values()), max(indices.values())
    print(f"\n  => le seul choix du filtre etale l'estimation d'un facteur {haut / bas:.2f}, "
          f"de {bas * lame_totale:.0f} a {haut * lame_totale:.0f} mm/an")

    print("\nPHASE. Cycle saisonnier de la recharge simulee, part de chaque mois.\n")
    cyc = np.array([j["recharge"][mois == m].mean() for m in range(1, 13)])
    cyc = cyc / max(cyc.sum(), 1e-12)
    noms = ["jan", "fev", "mar", "avr", "mai", "jun", "jul", "aou", "sep", "oct", "nov", "dec"]
    print("  " + "  ".join(f"{n} {100 * v:4.1f}%" for n, v in zip(noms, cyc)))
    print(f"  maximum en {noms[int(cyc.argmax())]}, amplitude saisonniere "
          f"{cyc.max() / max(cyc.min(), 1e-12):.0f} entre le mois fort et le mois faible")

    print("\nESPACE. Variation de la recharge d'un troncon a l'autre.\n")
    par_troncon = j["recharge"].mean(axis=0) * 365.25
    print(f"  mediane {np.median(par_troncon):.0f} mm/an, "
          f"etendue {np.percentile(par_troncon, 5):.0f} a {np.percentile(par_troncon, 95):.0f}, "
          f"variation {par_troncon.std() / par_troncon.mean():.2f}")
    print(f"  {len(par_troncon)} troncons pour {obs.shape[1]} stations : "
          f"{len(par_troncon) // max(obs.shape[1], 1)} troncons par observation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
