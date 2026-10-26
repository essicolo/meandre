"""Candidate 5.4, épreuve 3.3 : quelle forme physique de porte de gel rend la réponse
observée aux pluies d'automne, et transfère-t-elle aux stations, aux années et aux
territoires jamais vus ?

L'épreuve 3.1 a montré que la réponse relative croît de façon continue avec l'indice de
gel, de 1,00 à 2,74 entre 0 et 200 °C·jour, et que le manteau l'amplifie encore ; le
modèle actuel, une rampe linéaire en profondeur de gel, sature à 1,5 à 1,9 et se trompe
de territoire. Ici on ajuste des formes physiques communes, positives et monotones, aux
médianes observées par cellule de gel et de manteau, sans simulation nouvelle. La cellule
est le bon grain : les réponses individuelles sont bruitées, et 8 % valent zéro.

Formes comparées :
  - rampe saturante de la colonne, 1 + A min(gel/g_sat, 1) ;
  - linéaire, 1 + a gel ;
  - puissance, (1 + gel/g0)^p ;
  - exponentielle, exp(a gel), inverse d'une porte exp(-a gel) ;
  - chacune, au choix, multipliée par un facteur de manteau 1 + c neige/(neige + s0).

Trois retraits : stations, années, territoires. Le critère est l'écart aux médianes des
cellules retenues, en logarithme.

    .venv/Scripts/python.exe experiments/plan_ml/porte_gele_forme.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from meandre.utils import paths as _p

DOSSIER = f"{_p.DATA_ROOT}/experiments/plan_ml"
CLASSES_GEL = np.array([-1.0, 5.0, 20.0, 50.0, 100.0, 200.0, 1e9])
CLASSES_NEIGE = np.array([-1.0, 5.0, 30.0, 1e9])
NOMS_GEL = ["0-5", "5-20", "20-50", "50-100", "100-200", "200+"]
MINIMUM_CELLULE = 20
GRAINE = 0


def rampe(x, gel, neige):
    return 1.0 + x[0] * np.minimum(gel / x[1], 1.0)


def rampe_manteau(x, gel, neige):
    return rampe(x[:2], gel, neige) * (1.0 + x[2] * neige / (neige + x[3]))


def lineaire(x, gel, neige):
    return 1.0 + x[0] * gel


def puissance(x, gel, neige):
    return (1.0 + gel / x[0]) ** x[1]


def puissance_manteau(x, gel, neige):
    return puissance(x[:2], gel, neige) * (1.0 + x[2] * neige / (neige + x[3]))


def exponentielle(x, gel, neige):
    return np.exp(x[0] * gel)


def exponentielle_manteau(x, gel, neige):
    return np.exp(x[0] * gel) * (1.0 + x[1] * neige / (neige + x[2]))


FORMES = {
    "rampe saturante": (rampe, [(0.0, 10.0), (5.0, 500.0)], [[2.0, 100.0], [3.0, 50.0], [1.0, 200.0]]),
    "linéaire": (lineaire, [(0.0, 0.05)], [[0.01], [0.005]]),
    "puissance": (puissance, [(1.0, 500.0), (0.05, 2.0)], [[20.0, 0.5], [5.0, 1.0], [50.0, 0.3]]),
    "exponentielle": (exponentielle, [(0.0, 0.05)], [[0.01], [0.005]]),
    "rampe et manteau": (rampe_manteau, [(0.0, 10.0), (5.0, 500.0), (0.0, 5.0), (1.0, 200.0)],
                          [[2.0, 100.0, 1.0, 20.0], [3.0, 50.0, 2.0, 30.0]]),
    "puissance et manteau": (puissance_manteau, [(1.0, 500.0), (0.05, 2.0), (0.0, 5.0), (1.0, 200.0)],
                              [[20.0, 0.5, 1.0, 20.0], [50.0, 0.4, 2.0, 30.0], [10.0, 0.8, 0.5, 50.0]]),
    "exponentielle et manteau": (exponentielle_manteau, [(0.0, 0.05), (0.0, 5.0), (1.0, 200.0)],
                                  [[0.01, 1.0, 20.0], [0.005, 2.0, 30.0]]),
}


def cellules(gel, neige, rel):
    """Médiane observée par cellule de gel et de manteau, cellules assez peuplées."""
    cg = np.digitize(gel, CLASSES_GEL) - 1
    cn = np.digitize(neige, CLASSES_NEIGE) - 1
    d = pd.DataFrame({"cg": cg, "cn": cn, "gel": gel, "neige": neige, "rel": rel})
    g = d.groupby(["cg", "cn"]).agg(n=("rel", "size"), gel=("gel", "median"),
                                    neige=("neige", "median"), rel=("rel", "median")).reset_index()
    return g[g.n >= MINIMUM_CELLULE].reset_index(drop=True)


def ajuste(nom, tab):
    """Ajuste une forme sur le logarithme des médianes de cellule, poids de la cellule."""
    forme, bornes, departs = FORMES[nom]
    gel, neige = tab.gel.to_numpy(), tab.neige.to_numpy()
    cible = np.log(tab.rel.to_numpy())
    poids = tab.n.to_numpy(dtype=float)

    def objectif(x):
        pred = forme(x, gel, neige)
        if np.any(~np.isfinite(pred)) or np.any(pred <= 0):
            return 1e9
        return float(np.sum(poids * (np.log(pred) - cible) ** 2))

    meilleur = None
    for depart in departs:
        res = minimize(objectif, np.asarray(depart, dtype=float), method="L-BFGS-B", bounds=bornes)
        if meilleur is None or res.fun < meilleur.fun:
            meilleur = res
    return meilleur.x


def predit(nom, x, gel, neige):
    return np.clip(FORMES[nom][0](x, gel, neige), 0.02, 50.0)


def erreur(nom, x, tab):
    """Écart aux médianes des cellules retenues, en logarithme, et biais signé."""
    pred = predit(nom, x, tab.gel.to_numpy(), tab.neige.to_numpy())
    ecart = np.log(pred) - np.log(tab.rel.to_numpy())
    return float(np.mean(np.abs(ecart))), float(np.mean(ecart))


def principale():
    rng = np.random.default_rng(GRAINE)
    d = pd.read_parquet(f"{DOSSIER}/sol_gele_evenements.parquet")
    d = d[np.isfinite(d.rel) & np.isfinite(d.gel) & (d.gel >= 0) & np.isfinite(d.neige) & (d.neige >= 0)]
    d = d.reset_index(drop=True)
    gel, neige = d.gel.to_numpy(), d.neige.to_numpy()
    rel = d.rel.to_numpy()
    stations = d.station.to_numpy()
    regions = d.region.to_numpy()
    annees = pd.DatetimeIndex(d.date).year.to_numpy()
    tab = cellules(gel, neige, rel)
    print(f"{len(d)} pluies, {d.station.nunique()} stations, {d.region.nunique()} territoires, "
          f"{annees.min()}-{annees.max()} ; {len(tab)} cellules de gel et de manteau "
          f"d'au moins {MINIMUM_CELLULE} pluies")

    # Ajustement sur tout l'échantillon et courbe par classe de gel.
    cg = np.digitize(gel, CLASSES_GEL) - 1
    obs = [np.median(rel[cg == c]) for c in range(len(NOMS_GEL))]
    print("\nformes ajustées sur tout l'échantillon, médiane par classe de gel et paramètres :")
    print("  observé                  " + " ".join(f"{v:.2f}" for v in obs))
    ajustes = {}
    for nom in FORMES:
        x = ajuste(nom, tab)
        ajustes[nom] = x
        pred = predit(nom, x, gel, neige)
        courbe = [np.median(pred[cg == c]) if (cg == c).sum() >= MINIMUM_CELLULE else np.nan
                  for c in range(len(NOMS_GEL))]
        e, b = erreur(nom, x, tab)
        print(f"  {nom:24s} " + " ".join(f"{v:.2f}" for v in courbe) +
              f" | écart {e:.3f} biais {b:+.3f} | " + ", ".join(f"{v:.3f}" for v in x))

    print("\nretrait de stations, cinq plis, écart moyen aux cellules retenues (log) :")
    uniques = np.array(sorted(set(stations)))
    pli = {s: int(i % 5) for i, s in enumerate(rng.permutation(uniques))}
    app = np.array([pli[s] for s in stations])
    for nom in FORMES:
        vals = []
        for f in range(5):
            tr, te = app != f, app == f
            if te.sum() < 100:
                continue
            x = ajuste(nom, cellules(gel[tr], neige[tr], rel[tr]))
            vals.append(erreur(nom, x, cellules(gel[te], neige[te], rel[te]))[0])
        print(f"  {nom:24s} médiane {np.median(vals):.3f} | plis " + " ".join(f"{v:.3f}" for v in vals))

    print("\nretrait d'années, forme apprise jusqu'à 2014, éprouvée de 2015 à 2024 :")
    tr, te = annees <= 2014, annees >= 2015
    for nom in FORMES:
        x = ajuste(nom, cellules(gel[tr], neige[tr], rel[tr]))
        e, b = erreur(nom, x, cellules(gel[te], neige[te], rel[te]))
        print(f"  {nom:24s} écart {e:.3f} biais {b:+.3f}")

    print("\nretrait d'un territoire entier, écart moyen aux cellules du territoire (log) :")
    for reg in sorted(set(regions)):
        te = regions == reg
        if te.sum() < 100:
            continue
        tr = ~te
        valeurs = []
        for nom in FORMES:
            x = ajuste(nom, cellules(gel[tr], neige[tr], rel[tr]))
            valeurs.append(erreur(nom, x, cellules(gel[te], neige[te], rel[te]))[0])
        print(f"  {reg:5s} {int(te.sum()):4d} pluies | " +
              " ".join(f"{nom.split()[0][:9]} {v:.2f}" for nom, v in zip(FORMES, valeurs)))

    # Ce que la meilleure forme prédit par territoire, sous gel 20 à 100.
    nom = min(FORMES, key=lambda n: erreur(n, ajustes[n], tab)[0])
    x = ajustes[nom]
    pred = predit(nom, x, gel, neige)
    m = (gel >= 20) & (gel < 100)
    print(f"\nforme la plus proche des cellules : {nom}")
    print("  territoire | observé sous gel 20-100 | prédit | écart")
    for reg in sorted(set(regions)):
        r = m & (regions == reg)
        if r.sum() < 20:
            continue
        print(f"  {reg:5s} | {np.median(rel[r]):.2f} | {np.median(pred[r]):.2f} | "
              f"{np.median(pred[r]) - np.median(rel[r]):+.2f}")


if __name__ == "__main__":
    principale()
