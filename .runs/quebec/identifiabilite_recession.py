"""Les covariables prédisent-elles la NON-LINÉARITÉ de la vidange, et pas seulement sa part.

L'analyse des récessions observées a montré que l'exposant de Brutsaert-Nieber vaut 1,65 en
médiane mais s'étend de 1,31 en Montérégie à 1,83 au Saguenay. Cet exposant décide de
l'étalement des temps de vidange, donc de la forme de l'étiage, qui est la grandeur visée par
le projet. S'il est prédictible par des attributs de terrain, le champ spatial doit prédire la
NON-LINÉARITÉ du réservoir souterrain et non seulement son plafond de percolation.

Même protocole que pour l'indice d'écoulement de base : covariables moyennées sur le bassin
amont, chaque territoire prédit par un modèle ajusté sur les autres, mesure sur l'ERREUR et non
sur la part de variance, territoires sous un effectif minimal écartés.

    .venv/bin/python .runs/quebec/identifiabilite_recession.py outv slno gasp mont sagu --sources siigsol
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
_idb = SourceFileLoader("idb", os.path.join(_ici, "identifiabilite_indice_base.py")).load_module()
_rec = SourceFileLoader("rec", os.path.join(_ici, "recessions_observees.py")).load_module()


def exposants(region):
    """Exposant de récession par station, avec sa colonne dans le cache de débits."""
    flotte = os.environ.get("MEANDRE_FLOTTE", f"{os.environ.get('MEANDRE_DATA', '.')}/quebec/flotte")
    fichiers = sorted(glob.glob(f"{flotte}/q-{region}-*.npz"))
    if not fichiers:
        return pd.DataFrame()
    obs = np.load(fichiers[0], allow_pickle=True)["q_obs"]
    lignes = []
    for j in range(obs.shape[1]):
        q = obs[:, j].astype(float)
        fini = np.isfinite(q) & (q > 0)
        if fini.sum() < 700:
            continue
        r = _rec.exposant_station(q[fini])
        if r is None:
            continue
        lignes.append({"region": region, "colonne": j, "exposant": r["b"],
                       "tau_bas": r["tau_basses_eaux_j"], "tau_haut": r["tau_hautes_eaux_j"]})
    return pd.DataFrame(lignes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("regions", nargs="+")
    p.add_argument("--cache", default="appariement")
    p.add_argument("--sources", nargs="+", default=["siigsol"])
    p.add_argument("--minimum", type=int, default=10)
    p.add_argument("--cible", default="exposant", choices=("exposant", "tau_bas", "tau_haut"))
    a = p.parse_args()
    morceaux = []
    for reg in [r.lower() for r in a.regions]:
        ind = exposants(reg)
        if ind.empty:
            continue
        att = _idb.attributs_de_station(reg, a.cache, ind.colonne.to_numpy(), sources=a.sources)
        if att is None:
            print(f"{reg} : cache {a.cache} sans appariement station-troncon")
            continue
        morceaux.append(pd.concat([ind.reset_index(drop=True), att], axis=1))
    if not morceaux:
        return 1
    t = pd.concat(morceaux, ignore_index=True).dropna()
    gros = t.region.value_counts()
    t = t[t.region.isin(gros[gros >= a.minimum].index)].reset_index(drop=True)
    cols = [c for c in t.columns if "__" in c]
    y = t[a.cible].to_numpy()
    print(f"{len(t)} stations, {t.region.nunique()} territoires, {len(cols)} attributs")
    print(f"{a.cible} : mediane {np.median(y):.2f}, etendue {np.min(y):.2f} a {np.max(y):.2f}, "
          f"ecart-type {np.std(y):.3f}")
    blocs = t.region.astype("category").cat.codes.to_numpy()
    err, temoin = _idb.erreur_hors_bloc(t[cols].to_numpy(), y, blocs)
    print(f"erreur du temoin, moyenne des autres territoires : {temoin:.3f}")
    print(f"erreur avec les covariables : {err:.3f}, soit {100 * (1 - err / temoin):+.0f} %")
    for reg in sorted(t.region.unique()):
        te = (t.region == reg).to_numpy()
        e, w = _idb.erreur_hors_bloc(t[cols].to_numpy(), y, np.where(te, 0, -1), evalue=0)
        print(f"  {reg:6s} {te.sum():3d} stations | mediane {np.median(y[te]):.2f}"
              f" | erreur {e:.3f} contre {w:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
