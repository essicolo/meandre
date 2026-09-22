"""Toutes les répétitions d'UNE passe SWOT, pour obtenir une série temporelle propre.

Chaque passe voit son propre jeu de tronçons. Mélanger les passes donne une matrice trouée
où chaque colonne existe à des dates différentes, ce qui brouille toute mesure de rang. Une
passe unique, répétée tous les vingt-et-un jours, donne le même ensemble de tronçons à chaque
visite : c'est la seule forme sur laquelle le nombre de directions indépendantes se mesure.

La passe 326 est celle qui couvre le mieux l'Outaouais, avec 177 tronçons de plus de cent
mètres de large.

    .venv/Scripts/python.exe .runs/quebec/swot_passe_unique.py --passe 326
"""
import argparse
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from meandre.utils import paths as _paths

warnings.filterwarnings("ignore")
DOSSIER = f"{_paths.DATA_ROOT}/swot"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--passe", default="326")
    p.add_argument("--emprise", default="-79.5,45.0,-73.5,48.5")
    p.add_argument("--debut", default="2023-08-01")
    p.add_argument("--fin", default="2026-09-22")
    p.add_argument("--max", type=int, default=80)
    a = p.parse_args()
    import earthaccess

    earthaccess.login(strategy="netrc")
    bb = tuple(float(x) for x in a.emprise.split(","))
    # La recherche est gratuite : on prend TOUTE la metadonnee, puis on ne telecharge que
    # les granules de la passe voulue.
    tous = earthaccess.search_data(short_name="SWOT_L2_HR_RiverSP_reach_D",
                                   bounding_box=bb, temporal=(a.debut, a.fin), count=3000)
    motif = f"_{a.passe}_NA_"
    garde = [g for g in tous if motif in str(g)]
    print(f"{len(tous)} granules sur l'emprise, {len(garde)} pour la passe {a.passe}")
    garde = garde[:a.max]
    os.makedirs(DOSSIER, exist_ok=True)
    deja = set(os.listdir(DOSSIER))
    manquants = [g for g in garde if not any(str(g).find(n.replace(".zip", "")) >= 0 for n in deja)]
    print(f"{len(garde) - len(manquants)} deja presents, {len(manquants)} a telecharger",
          flush=True)
    if manquants:
        earthaccess.download(manquants, DOSSIER)
    print("fait")
    return 0


if __name__ == "__main__":
    sys.exit(main())
