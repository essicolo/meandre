"""Combien de constantes de temps les récessions OBSERVÉES exigent-elles.

La question est celle d'Essi, 2026-09-20 : découper le sol en trois tranches est une
convention, pas de la physique. Ce que les données peuvent trancher n'est pas le nombre de
tranches mais le nombre de constantes de temps que la vidange du bassin réclame. Si une seule
suffit, le troisième compartiment et sa sortie latérale sont un habillage. Si le taux de
récession varie fortement avec le débit, il faut soit plusieurs réservoirs, soit un réservoir
non linéaire, et la question devient laquelle des deux descriptions est la plus économe.

Le test ne demande aucune simulation et porte sur les hydrogrammes observés.

Méthode de Brutsaert et Nieber, 1977 : sur les segments de décrue, on ajuste
−dQ/dt = a · Q^b. L'exposant b est le diagnostic.
  b proche de 1   un réservoir LINÉAIRE unique, une seule constante de temps ;
  b entre 1 et 2  vidange non linéaire, ou mélange de réservoirs de constantes différentes ;
  b proche de 2   écoulement de nappe libre de Boussinesq, régime de courte durée.

Les segments sont pris après un délai suivant la pointe, pour écarter le ressuyage rapide et
la crue elle-même. L'ajustement se fait sur l'ENVELOPPE INFÉRIEURE du nuage, comme le
prescrit la méthode : le bas du nuage correspond aux décrues non perturbées par la pluie,
que l'on ne peut pas identifier autrement faute de forçage à la station.

    .venv/bin/python .runs/quebec/recessions_observees.py outv slno gasp mont sagu
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def segments(q, delai=2, duree_min=5):
    """Indices des jours de décrue franche, hors les `delai` jours suivant une pointe."""
    dq = np.diff(q)
    decroit = np.concatenate([[False], dq < 0])
    bons = np.zeros(len(q), dtype=bool)
    i = 0
    while i < len(q):
        if not decroit[i]:
            i += 1
            continue
        j = i
        while j < len(q) and decroit[j]:
            j += 1
        if j - i >= duree_min + delai:
            bons[i + delai:j] = True
        i = j
    return bons


def enveloppe_inferieure(x, y, n_classes=20, quantile=0.10):
    """Bas du nuage, pris par quantile dans des tranches d'abscisse d'effectif égal.

    Le découpage sert UNIQUEMENT à estimer l'enveloppe, jamais à représenter la relation :
    l'ajustement qui suit reste continu sur les points retenus.
    """
    ordre = np.argsort(x)
    x, y = x[ordre], y[ordre]
    bornes = np.linspace(0, len(x), n_classes + 1).astype(int)
    px, py = [], []
    for a, b in zip(bornes[:-1], bornes[1:]):
        if b - a < 5:
            continue
        seuil = np.quantile(y[a:b], quantile)
        garde = y[a:b] <= seuil
        if garde.sum() == 0:
            continue
        px.append(x[a:b][garde])
        py.append(y[a:b][garde])
    if not px:
        return None, None
    return np.concatenate(px), np.concatenate(py)


def exposant_station(q, delai=2, duree_min=5):
    """Exposant b de Brutsaert-Nieber, et taux de récession aux hautes et basses eaux."""
    bons = segments(q, delai, duree_min)
    if bons.sum() < 100:
        return None
    dqdt = np.concatenate([[np.nan], -np.diff(q)])
    qm = np.concatenate([[np.nan], 0.5 * (q[1:] + q[:-1])])
    garde = bons & np.isfinite(dqdt) & np.isfinite(qm) & (dqdt > 0) & (qm > 0)
    if garde.sum() < 100:
        return None
    lx, ly = np.log(qm[garde]), np.log(dqdt[garde])
    ex, ey = enveloppe_inferieure(lx, ly)
    if ex is None or len(ex) < 20:
        return None
    b, loga = np.polyfit(ex, ey, 1)
    # Taux de recession instantane k = -(dQ/dt)/Q, en 1/jour, aux deux extremites du nuage.
    k = dqdt[garde] / qm[garde]
    bas = qm[garde] <= np.quantile(qm[garde], 0.20)
    haut = qm[garde] >= np.quantile(qm[garde], 0.80)
    return {"b": float(b), "n_jours": int(garde.sum()),
            "tau_basses_eaux_j": float(1.0 / max(np.median(k[bas]), 1e-9)),
            "tau_hautes_eaux_j": float(1.0 / max(np.median(k[haut]), 1e-9)),
            "rapport_debits": float(np.median(qm[garde][haut]) / max(np.median(qm[garde][bas]), 1e-12))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("regions", nargs="+")
    p.add_argument("--minimum-stations", type=int, default=5)
    a = p.parse_args()
    flotte = os.environ.get("MEANDRE_FLOTTE", f"{os.environ.get('MEANDRE_DATA', '.')}/quebec/flotte")
    lignes = []
    for reg in [r.lower() for r in a.regions]:
        fichiers = sorted(glob.glob(f"{flotte}/q-{reg}-*.npz"))
        if not fichiers:
            continue
        z = np.load(fichiers[0], allow_pickle=True)
        obs = z["q_obs"]
        for j in range(obs.shape[1]):
            q = obs[:, j].astype(float)
            fini = np.isfinite(q) & (q > 0)
            if fini.sum() < 700:
                continue
            r = exposant_station(np.where(fini, q, np.nan)[fini])
            if r is not None:
                lignes.append({"region": reg, "station": j, **r})
    if not lignes:
        print("aucune station exploitable")
        return 1
    t = pd.DataFrame(lignes)
    print(f"{len(t)} stations sur {t.region.nunique()} territoires\n")
    print("Exposant de Brutsaert-Nieber, par territoire :")
    for reg, g in t.groupby("region"):
        if len(g) < a.minimum_stations:
            continue
        print(f"  {reg:6s} {len(g):3d} stations | b mediane {g.b.median():.2f}"
              f" | etendue {g.b.quantile(0.1):.2f} a {g.b.quantile(0.9):.2f}"
              f" | tau hautes eaux {g.tau_hautes_eaux_j.median():5.1f} j"
              f" | tau basses eaux {g.tau_basses_eaux_j.median():6.1f} j")
    print(f"\nToutes stations : b median {t.b.median():.2f}, "
          f"rapport des deux temps {t.tau_basses_eaux_j.median() / t.tau_hautes_eaux_j.median():.1f}")
    # UN SEUL RESERVOIR NON LINEAIRE SUFFIT-IL. Pour -dQ/dt = a·Q^b, le temps de vidange
    # apparent vaut tau(Q) = Q / (-dQ/dt) = Q^(1-b)/a. L'etalement des temps entre hautes et
    # basses eaux se DEDUIT alors de l'exposant et du rapport des debits, sans invoquer le
    # moindre second reservoir. Si l'etalement predit reproduit l'etalement mesure, empiler
    # des compartiments pour fabriquer un spectre est inutile : la non-linearite le fabrique.
    t["etalement_mesure"] = t.tau_basses_eaux_j / t.tau_hautes_eaux_j
    t["etalement_predit"] = t.rapport_debits ** (t.b - 1.0)
    ok = np.isfinite(t.etalement_predit) & (t.etalement_predit > 0)
    print(f"\nEtalement des temps de recession entre basses et hautes eaux :")
    print(f"  mesure  {t.etalement_mesure[ok].median():5.1f}")
    print(f"  predit par le seul exposant  {t.etalement_predit[ok].median():5.1f}")
    _r = np.log(t.etalement_predit[ok]) / np.log(t.etalement_mesure[ok].replace(1.0, np.nan))
    print(f"  la non-linearite explique {100 * _r.median():.0f} % de l'etalement, en log")
    print("\nb proche de 1 : un seul reservoir lineaire suffit, une seule constante de temps.")
    print("b nettement superieur a 1 : la vidange est non lineaire, ou plusieurs reservoirs")
    print("de constantes differentes se superposent. Le rapport des temps de recession entre")
    print("basses et hautes eaux dit l'etendue du spectre a representer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
