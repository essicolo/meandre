"""Qu'est-ce qui CHANGE pour un meme troncon selon les regions chargees ?

FAIT A EXPLIQUER (2026-08-31). Le meme fichier de poids, repere FIXE, evalue sur les
memes stations, rend :

              6 regions   14 regions
    gasp        0.681       0.364
    sagu        0.723       0.302
    slno        0.750       0.620
    outv        0.705       0.606

Aucun poids ne change. La remarque d'Essi est decisive : la continuite du champ aux
coutures entre regions ne touche qu'une poignee de noeuds au pourtour et ne peut pas
produire ca. C'est donc une ENTREE qui change.

Deux pistes deja ecartees par mesure : le repere de projection (corrige, deplacement
nul) et l'intersection des champs physiques (identique sur 6 et sur 14).

Ce script ne devine plus. Il charge les deux domaines et compare, pour les noeuds
COMMUNS, chaque tableau que `load_domain` fabrique : attributs territoriaux, coordonnees,
occupation du sol, ancrages de fonte, ETP Linacre, champ k_gw, sol calibre, fraction
agricole, prelevements et forcage. Ce qui differe est la cause.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, ".runs/quebec")
sys.path.insert(0, ".")

S6 = ["outv", "gasp", "mont", "sagu", "slno", "slso"]
S14 = S6 + ["abit", "cnda", "cndb", "cndc", "cndd", "cnde", "labi", "vaud"]


def aplati(x):
    """Rend un tableau numpy 1D ou 2D, ou None si ce n'est pas comparable."""
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return None


def compare(nom, a, b, n):
    """Compare les n premieres lignes de deux tableaux et rapporte l'ecart."""
    a, b = aplati(a), aplati(b)
    if a is None or b is None:
        return
    if a.shape[0] < n or b.shape[0] < n:
        print(f"  {nom:26s} formes {a.shape} contre {b.shape} : NON COMPARABLE")
        return
    d = np.abs(np.asarray(a[:n], dtype=float) - np.asarray(b[:n], dtype=float))
    if not np.isfinite(d).any():
        return
    m = float(np.nanmax(d))
    if m > 1e-9:
        rel = m / (float(np.nanmax(np.abs(a[:n]))) + 1e-12)
        print(f"  {nom:26s} DIFFERE : ecart max {m:.6g} ({100 * rel:.2f} % de l'echelle)")


def main():
    from domain_data import load_domain

    print("chargement du domaine a 6 regions", flush=True)
    d6 = load_domain(S6, {}, device="cpu")
    print("chargement du domaine a 14 regions", flush=True)
    d14 = load_domain(S14, {}, device="cpu")

    n = int(d6["n_nodes"])
    print(f"\n{n:,} noeuds communs (les 6 premieres regions sont dans le meme ordre)\n")
    print("ce qui DIFFERE pour un meme troncon :")

    compare("node_coords", d6["node_coords"], d14["node_coords"], n)
    compare("territorial", d6["territorial"].data, d14["territorial"].data, n)
    for k in sorted(set(d6["territorial"].physical) & set(d14["territorial"].physical)):
        compare(f"phys/{k}", d6["territorial"].physical[k],
                d14["territorial"].physical[k], n)
    compare("kgw", d6.get("kgw"), d14.get("kgw"), n)
    for cle in ("land_cover", "melt_params", "soil"):
        a, b = d6.get(cle) or {}, d14.get(cle) or {}
        for k in sorted(set(a) & set(b)):
            compare(f"{cle}/{k}", a[k], b[k], n)
        seuls6, seuls14 = sorted(set(a) - set(b)), sorted(set(b) - set(a))
        if seuls6:
            print(f"  {cle}: champs presents SEULEMENT a 6 regions : {seuls6}")
        if seuls14:
            print(f"  {cle}: champs presents SEULEMENT a 14 regions : {seuls14}")
    if d6.get("linacre") and d14.get("linacre"):
        for i, (a, b) in enumerate(zip(d6["linacre"], d14["linacre"])):
            compare(f"linacre[{i}]", a, b, n)

    t6, t14 = d6["train_data"], d14["train_data"]
    compare("forcage (100 premiers pas)", t6.forcing[:100, :n], t14.forcing[:100, :n], 100)
    for k in ("_vals", "_vals_gw"):
        a = getattr(t6.withdrawals, k, None)
        b = getattr(t14.withdrawals, k, None)
        if a is not None and b is not None and a.shape == b.shape:
            compare(f"prelev/{k}", a, b, a.shape[0])
        elif a is not None and b is not None:
            print(f"  prelev/{k:20s} formes {tuple(a.shape)} contre {tuple(b.shape)}")

    print("\nLECTURE. Toute ligne ci-dessus est une entree qui depend des regions chargees,")
    print("donc une cause possible du 0.68 contre 0.36. S'il n'y en a aucune, la cause est")
    print("dans le graphe ou le routage, pas dans les tableaux d'entree.")


if __name__ == "__main__":
    main()
