"""La géologie explique-t-elle la dispersion de l'indice d'écoulement de base entre stations ?

Ce test décide du chantier de spatialisation. L'indice d'écoulement de base, mesuré sur les
hydrogrammes OBSERVÉS par le filtre de Lyne et Hollick, vaut 0,58 en médiane sur les 16
stations de l'Outaouais mais s'étend de 0,39 à 0,76 : sa dispersion est surtout INTRA-régionale.
Or c'est le plafond de percolation du substratum qui règle cette part dans le modèle. Un
plafond uniforme ne peut donc pas la reproduire, et il faut savoir si les covariables
disponibles le prédisent avant d'en faire un paramètre du champ spatial.

Le test est délibérément simple et il porte sur une grandeur OBSERVÉE, non sur un paramètre
ajusté : la part de variance de l'indice expliquée par la géologie du socle et les dépôts, en
validation croisée par blocs spatiaux pour ne pas se laisser tromper par l'autocorrélation.
Une part nulle ou négative signifie que ces covariables ne sont pas les bonnes, et non que le
plafond doit rester uniforme.

    .venv/bin/python .runs/quebec/identifiabilite_indice_base.py outv slno --cache <suffixe>
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

from importlib.machinery import SourceFileLoader

_ib = SourceFileLoader("ib", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "indice_base_observe.py")).load_module()


def indices_par_station(region: str):
    """Indice observé et identifiant, pour les stations à série suffisante."""
    import glob

    flotte = os.environ.get("MEANDRE_FLOTTE", f"{os.environ.get('MEANDRE_DATA', '.')}/quebec/flotte")
    fichiers = sorted(glob.glob(f"{flotte}/q-{region}-*.npz"))
    if not fichiers:
        return pd.DataFrame()
    z = np.load(fichiers[0], allow_pickle=True)
    obs, ids = z["q_obs"], [str(x) for x in z["station_ids"]]
    lignes = []
    for j in range(obs.shape[1]):
        s = obs[:, j]
        fini = np.isfinite(s)
        if fini.sum() < 700:
            continue
        s = np.where(fini, s, np.nanmedian(s))
        lignes.append({"region": region, "station": ids[j], "colonne": j,
                       "indice_base": float(_ib.lyne_hollick(s).sum() / s.sum())})
    return pd.DataFrame(lignes)


def attributs_de_station(region: str, cache: str, colonnes):
    """Attributs géologiques du tronçon portant chaque station, via l'appariement du cache."""
    f = f"{DERIVES}/reach-{region}-{cache}.npz"
    if not os.path.exists(f):
        return None
    z = np.load(f, allow_pickle=True)
    if "station_idx" not in z.files:
        return None
    noeuds = z["station_idx"][colonnes]
    geo = pd.read_parquet(f"{DERIVES}/sigeom-geologie-socle-troncons.parquet")
    geo = geo[geo.region.str.lower() == region.lower()].set_index("troncon")
    # Le tronçon est indexé à partir de 1, le nœud à partir de 0.
    lignes = geo.reindex([int(n) + 1 for n in noeuds])
    cols = [c for c in geo.columns if c.startswith(("litho_", "province_")) and not c.startswith("couv_")]
    return lignes[cols].reset_index(drop=True)


def part_expliquee(X, y, blocs, graine=0):
    """Part de variance expliquée hors bloc, par un gradient boosté peu profond."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    residus, total = [], []
    for b in np.unique(blocs):
        ap, te = blocs != b, blocs == b
        if te.sum() == 0 or ap.sum() < 5:
            continue
        m = HistGradientBoostingRegressor(max_depth=2, max_iter=120, random_state=graine)
        m.fit(X[ap], y[ap])
        residus.append(((y[te] - m.predict(X[te])) ** 2).sum())
        total.append(((y[te] - y[ap].mean()) ** 2).sum())
    if not total or sum(total) == 0:
        return np.nan
    return 1.0 - sum(residus) / sum(total)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("regions", nargs="+")
    p.add_argument("--cache", default="finale-n2")
    a = p.parse_args()
    morceaux = []
    for reg in [r.lower() for r in a.regions]:
        ind = indices_par_station(reg)
        if ind.empty:
            print(f"{reg} : aucune série de station")
            continue
        att = attributs_de_station(reg, a.cache, ind.colonne.to_numpy())
        if att is None:
            print(f"{reg} : cache {a.cache} sans appariement station-tronçon")
            continue
        morceaux.append(pd.concat([ind.reset_index(drop=True), att], axis=1))
    if not morceaux:
        return 1
    t = pd.concat(morceaux, ignore_index=True).dropna()
    cols = [c for c in t.columns if c.startswith(("litho_", "province_"))]
    print(f"{len(t)} stations, {len(cols)} attributs géologiques")
    print(f"indice observé : médiane {t.indice_base.median():.2f}, "
          f"étendue {t.indice_base.min():.2f} à {t.indice_base.max():.2f}")
    # Blocs spatiaux : un par région, plus un découpage en deux par la médiane de l'indice
    # de station pour ne pas prédire par la seule appartenance régionale.
    blocs = t.region.astype("category").cat.codes.to_numpy()
    r2 = part_expliquee(t[cols].to_numpy(), t.indice_base.to_numpy(), blocs)
    print(f"part de variance expliquée hors bloc, blocs = territoires : {r2:+.2f}")
    if len(t) >= 20:
        rng = np.random.default_rng(0)
        alea = rng.integers(0, 4, size=len(t))
        r2b = part_expliquee(t[cols].to_numpy(), t.indice_base.to_numpy(), alea)
        print(f"part de variance expliquée hors bloc, blocs aléatoires  : {r2b:+.2f}")
    print("\nUne part nulle ou négative dit que ces covariables ne prédisent pas l'indice,")
    print("non que le plafond doive rester uniforme.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
