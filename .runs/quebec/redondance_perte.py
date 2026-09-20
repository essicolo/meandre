"""Combien d'information DISTINCTE portent les termes de la perte.

Question d'Essi, 2026-09-20 : la perte porte le KGE, l'écart quadratique et le biais de
volume, soit plusieurs fois la même information. Le KGE se décompose exactement en
corrélation, rapport des moyennes et rapport des écarts-types ; le biais de volume EST son
deuxième facteur, et l'écart quadratique mélange les trois. La redondance est donc attendue
par construction, et la seule question utile est son ampleur.

La mesure ne demande aucune simulation. On reprend les déformations du banc de perte, qui
sont celles que le diagnostic a relevées sur les sorties réelles, on note chaque terme sur
chacune, et on regarde si les termes se distinguent. Un terme dont la note se déduit des
autres n'ajoute rien à l'optimiseur, quel que soit son poids.

Trois lectures, de la plus grossière à la plus utile :
  corrélation de rang entre termes, deux à deux ;
  part de la variation d'un terme expliquée par TOUS les autres, qui dit la redondance ;
  nombre de directions indépendantes dans le jeu de termes, par valeurs singulières.

    .venv/Scripts/python.exe .runs/quebec/redondance_perte.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from importlib.machinery import SourceFileLoader

_bp = SourceFileLoader("bp", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "banc_perte.py")).load_module()

# Poids de la recette du socle, pour dire ce que la redondance coûte en pratique.
POIDS = _bp.POIDS

# Jeux de termes à comparer à nombre égal. La recette en vigueur contre la proposition
# d'Essi du 2026-09-20 : décomposer le KGE en ses trois facteurs plutôt que le porter entier.
BASES = {
    "recette en vigueur": ["KGE", "biais de volume", "écart quadratique",
                           "écart quadratique log", "pics"],
    "KGE décomposé": ["calendrier (1-r)", "volume (beta-1)^2", "amplitude (gamma-1)^2"],
    "KGE décomposé + étiage": ["calendrier (1-r)", "volume (beta-1)^2",
                               "amplitude (gamma-1)^2", "soutien d'étiage"],
    "KGE décomposé + étiage + variations": ["calendrier (1-r)", "volume (beta-1)^2",
                                            "amplitude (gamma-1)^2", "soutien d'étiage",
                                            "variations d'un jour"],
}


def matrice():
    """Une ligne par couple station-déformation, une colonne par terme, notes centrées réduites.

    Le centrage est fait PAR STATION : les termes n'ont pas la même échelle d'un bassin à
    l'autre et une corrélation calculée sur les valeurs brutes ne mesurerait que cela.
    """
    lignes = []
    for reg in _bp.REGS:
        f = f"{_bp.DOS}/q-{reg}-A-v4.npz"
        if not os.path.exists(f):
            continue
        z = np.load(f, allow_pickle=True)
        mois = pd.to_datetime([str(v)[:10] for v in z["dates"]]).month.to_numpy()
        for j in range(z["q_obs"].shape[1]):
            q = z["q_obs"][:, j].astype(float)
            if not np.isfinite(q).all() or (q <= 0).any():
                continue
            var, q75 = float(q.var()), float(np.quantile(q, 0.75))
            for nom, s in _bp.deformations(q, mois).items():
                t = _bp.termes(q, s, var, q75)
                lignes.append({"region": reg, "station": j, "deformation": nom,
                               **t, **facteurs_kge(q, s)})
    return pd.DataFrame(lignes)


def facteurs_kge(q_obs, q_sim):
    """Les trois facteurs du KGE, pris separement, en forme de PERTE.

    Proposition d'Essi, 2026-09-20 : decomposer le KGE plutot que de le porter entier. Le
    KGE vaut 1 moins la distance a l'optimum du triplet (correlation, rapport des moyennes,
    rapport des ecarts-types). Porte entier, il melange une erreur de calendrier, une erreur
    de volume et une erreur d'amplitude en un seul nombre, et l'optimiseur ne peut plus
    choisir laquelle reduire. Les trois facteurs sont ici mis a l'echelle d'une perte, zero
    au mieux, pour etre comparables aux autres termes.
    """
    o, s = np.asarray(q_obs, dtype=float), np.asarray(q_sim, dtype=float)
    r = float(np.corrcoef(o, s)[0, 1])
    beta = float(s.mean() / o.mean()) if o.mean() != 0 else np.nan
    gamma = float((s.std() / s.mean()) / (o.std() / o.mean())) if s.mean() != 0 else np.nan
    return {"calendrier (1-r)": 1.0 - r,
            "volume (beta-1)^2": (beta - 1.0) ** 2,
            "amplitude (gamma-1)^2": (gamma - 1.0) ** 2}


def reduire_par_station(t, cols):
    """Centre et réduit chaque terme à l'intérieur de chaque station."""
    g = t.groupby(["region", "station"])[cols]
    return ((t[cols] - g.transform("mean")) / g.transform("std").replace(0.0, np.nan)).dropna()


def part_expliquee_par_les_autres(X, j):
    """Part de la variation de la colonne j expliquée par toutes les autres, aux moindres carrés."""
    autres = np.delete(X, j, axis=1)
    y = X[:, j]
    A = np.column_stack([autres, np.ones(len(autres))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ coef
    return 1.0 - float(res.var() / y.var()) if y.var() > 0 else np.nan


def main():
    t = matrice()
    if t.empty:
        print("aucune station lisible")
        return 1
    cols = [c for c in t.columns if c not in ("region", "station", "deformation")]
    print(f"{t.station.nunique()} stations x {t.deformation.nunique()} deformations "
          f"= {len(t)} notes, {len(cols)} termes\n")
    Z = reduire_par_station(t, cols)
    X = Z.to_numpy()

    print("Correlation de rang entre termes, sur les notes centrees par station :")
    r = Z.rank().corr(method="pearson")
    print(r.round(2).to_string())

    print("\nPart de la variation d'un terme expliquee par TOUS les autres :")
    for j, c in enumerate(cols):
        p = part_expliquee_par_les_autres(X, j)
        etat = "REDONDANT" if p > 0.95 else ("tres redondant" if p > 0.9 else "distinct")
        poids = POIDS.get(c)
        print(f"  {c:24s} {p:6.3f}  {etat:15s}"
              + (f"  poids {poids}" if poids is not None else "  hors recette"))

    print("\nNombre de directions independantes, par valeurs singulieres :")
    s = np.linalg.svd(X - X.mean(0), compute_uv=False)
    part = s ** 2 / (s ** 2).sum()
    cum = np.cumsum(part)
    for k, (p, c) in enumerate(zip(part, cum), start=1):
        print(f"  direction {k} : {100 * p:5.1f} % | cumul {100 * c:5.1f} %")
    n90 = int(np.searchsorted(cum, 0.90) + 1)
    print(f"\n{len(cols)} termes, mais {n90} directions suffisent a porter 90 % de leur variation.")
    print("\nQuel petit jeu de termes engendre tous les autres. A chaque etape on retient le")
    print("terme le plus mal reconstruit par ceux deja pris : c'est celui qui ajoute le plus.")
    retenus, restants = [], list(range(len(cols)))

    def manque(j, pris):
        """Part de la variation de j que les termes deja pris ne reconstruisent pas."""
        if not pris:
            return 1.0
        A = np.column_stack([X[:, pris], np.ones(len(X))])
        coef, *_ = np.linalg.lstsq(A, X[:, j], rcond=None)
        return float((X[:, j] - A @ coef).var() / X[:, j].var())

    while restants:
        j = max(restants, key=lambda k: manque(k, retenus))
        retenus.append(j)
        restants.remove(j)
        reste = max((manque(k, retenus) for k in restants), default=0.0)
        print(f"  {len(retenus)}. {cols[j]:24s} | le terme restant le plus mal reconstruit"
              f" garde {100 * reste:4.1f} % de variation propre")
        if reste < 0.10:
            print(f"\n  {len(retenus)} termes suffisent : tout le reste s'en deduit a 90 % au moins.")
            break

    print("\nLa recette en vigueur contre le KGE DECOMPOSE, a nombre de termes egal :")
    for nom, base in BASES.items():
        pris = [cols.index(c) for c in base if c in cols]
        autres = [j for j in range(len(cols)) if j not in pris]
        m = [(manque(j, pris), cols[j]) for j in autres]
        pire, quel = max(m)
        print(f"  {nom}")
        print(f"    {len(pris)} termes | manque moyen {100 * np.mean([x for x, _ in m]):4.1f} %"
              f" | angle mort le plus grand {100 * pire:4.1f} % sur {quel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
