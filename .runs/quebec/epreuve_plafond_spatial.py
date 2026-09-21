"""Le champ apprend-il un plafond de percolation qui vaut quelque chose, ou un plafond déguisé.

Épreuve écrite AVANT la simulation, comme le demande la règle du 2026-09-13 : si aucune issue
ne change une décision, le calcul est inutile. Trois questions, et ce que chacune déciderait.

1. Le plafond appris varie-t-il dans l'espace. Un champ qui rend la même valeur partout a
   simplement appris à ignorer une sortie, et la spatialisation n'a rien apporté. Seuil :
   coefficient de variation supérieur à 0,15, faute de quoi le chantier se referme.

2. Sa structure est-elle celle de la texture. L'argument de la spatialisation est que la
   texture explique la part souterraine du débit à 27 % contre le témoin. Si le plafond appris
   n'est pas prédictible par la texture en TRANSFERT entre territoires, le champ a trouvé autre
   chose, peut-être un effet de position, et il faut savoir quoi avant d'en tirer une recette
   provinciale.

3. Le plafond appris prédit-il l'indice d'écoulement de base OBSERVÉ aux stations. C'est la
   question physique, et la seule qui décide vraiment. Le plafond gouverne la part souterraine ;
   un plafond plus haut doit aller avec un bassin plus soutenu. Une corrélation de rang positive
   et significative valide le mécanisme ; une corrélation nulle dit que le champ a ajusté un
   paramètre sans signification ; une corrélation négative dit que le modèle compense autre
   chose par ce paramètre, ce qui serait un avertissement sérieux.

    .venv/bin/python .runs/quebec/epreuve_plafond_spatial.py outv slno --variante pile-spatiale
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from importlib.machinery import SourceFileLoader

_ici = os.path.dirname(os.path.abspath(__file__))
_idb = SourceFileLoader("idb", os.path.join(_ici, "identifiabilite_indice_base.py")).load_module()
DERIVES = _idb.DERIVES

# Reference du champ, en metres par heure : un millimetre par jour. Un plafond appris qui
# reste colle dessus signale une sortie que l'optimiseur n'a pas touchee.
KSUB_REF = 1.0e-3 / 24.0
MM_PAR_JOUR = 1000.0 * 24.0


def champ(region, variante):
    """Plafond appris par tronçon, en mm/jour, et le cache qui le porte."""
    f = f"{DERIVES}/reach-{region}-{variante}.npz"
    if not os.path.exists(f):
        return None, None
    z = np.load(f, allow_pickle=True)
    if "param_k_sub" not in z.files:
        return None, z
    return z["param_k_sub"] * MM_PAR_JOUR, z


def attributs_par_troncon(region, sources, n_noeuds):
    """Attributs de terrain de chaque tronçon, alignés sur l'indice de nœud.

    Le tronçon est indexé à partir de 1 dans les tables d'ingestion, le nœud à partir de 0.
    """
    morceaux = []
    for src in sources:
        f = f"{DERIVES}/{src}-troncons.parquet"
        if not os.path.exists(f):
            continue
        t = pd.read_parquet(f)
        t = t[t.region.str.lower() == region.lower()].set_index("troncon")
        garde = [c for c in t.columns
                 if c not in ("region", "area_m2") and not c.startswith("couv_")]
        morceaux.append(t[garde].add_prefix(f"{src}__"))
    if not morceaux:
        return None
    table = pd.concat(morceaux, axis=1).fillna(0.0)
    return table.reindex(np.arange(1, n_noeuds + 1)).fillna(0.0).reset_index(drop=True)


def rang(x, y):
    """Corrélation de rang, sans dépendance à scipy."""
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else np.nan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("regions", nargs="+")
    p.add_argument("--variante", required=True)
    p.add_argument("--sources", nargs="+", default=["siigsol"])
    a = p.parse_args()
    regions = [r.lower() for r in a.regions]

    print("1. Le plafond appris varie-t-il dans l'espace")
    tout, blocs_reg = [], []
    for reg in regions:
        k, z = champ(reg, a.variante)
        if k is None:
            print(f"  {reg:6s} : pas de param_k_sub dans le cache")
            continue
        cv = float(np.std(k) / max(np.mean(k), 1e-12))
        print(f"  {reg:6s} {len(k):6d} troncons | mediane {np.median(k):6.2f} mm/j"
              f" | etendue {np.percentile(k, 5):5.2f} a {np.percentile(k, 95):6.2f}"
              f" | variation {cv:.2f} {'' if cv > 0.15 else '  UNIFORME, le chantier se referme'}")
        tout.append(k)
        blocs_reg.append(np.full(len(k), reg))
    if not tout:
        return 1

    print("\n2. Sa structure est-elle celle de la texture, en transfert entre territoires")
    lignes = []
    for reg in regions:
        k, z = champ(reg, a.variante)
        if k is None:
            continue
        # Les attributs se lisent PAR TRONCON et non par station : le plafond est une sortie
        # du champ sur chaque troncon, pas une signature integree sur un bassin amont.
        att = attributs_par_troncon(reg, a.sources, len(k))
        if att is None:
            print(f"  {reg:6s} : attributs de troncon indisponibles")
            continue
        lignes.append(pd.concat([pd.DataFrame({"region": reg, "k_sub": k}), att], axis=1))
    if len(lignes) >= 2:
        t = pd.concat(lignes, ignore_index=True).dropna()
        cols = [c for c in t.columns if "__" in c]
        blocs = t.region.astype("category").cat.codes.to_numpy()
        err, temoin = _idb.erreur_hors_bloc(t[cols].to_numpy(), t.k_sub.to_numpy(), blocs)
        print(f"  erreur du temoin {temoin:.3f} mm/j | avec la texture {err:.3f} mm/j"
              f" | {100 * (1 - err / temoin):+.0f} %")
    else:
        print("  moins de deux territoires exploitables : rien a transferer")

    print("\n3. Le plafond appris predit-il l'indice d'ecoulement de base OBSERVE")
    for reg in regions:
        k, z = champ(reg, a.variante)
        if k is None or z is None or "station_idx" not in z.files:
            continue
        ind = _idb.indices_par_station(reg)
        if ind.empty:
            continue
        noeuds = z["station_idx"][ind.colonne.to_numpy()]
        valides = (noeuds >= 0) & (noeuds < len(k))
        if valides.sum() < 5:
            continue
        # Le plafond du seul troncon de la station ne dit rien : la signature integre tout
        # l'amont. On moyenne donc sur le bassin, comme pour les attributs de terrain.
        if "edge_index" in z.files:
            moyennes = []
            for n in noeuds[valides]:
                bassin = _idb.bassin_amont(z["edge_index"], len(k), int(n))
                moyennes.append(float(np.mean(k[bassin])))
            x = np.array(moyennes)
        else:
            x = k[noeuds[valides]]
        y = ind.indice_base.to_numpy()[valides]
        print(f"  {reg:6s} {len(x):3d} stations | correlation de rang {rang(x, y):+.2f}"
              f" | plafond median {np.median(x):.2f} mm/j | indice median {np.median(y):.2f}")
    print("\nUne correlation positive valide le mecanisme ; nulle, le champ a ajuste un")
    print("parametre sans signification ; negative, il compense autre chose par lui.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
