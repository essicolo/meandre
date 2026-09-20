"""Lit le carré physique x contrainte à taux stable et rend une case par ligne.

Le carré croise deux facteurs sur l'Outaouais, chacun sur deux graines : la pile de corrections
du drainage profond, et la contrainte sur les niveaux de puits mesurés. Quatre cases, huit
passes. C'est le premier protocole de comparaison défendable du chantier, parce que le taux
d'apprentissage y est de 3e-5 : à 1e-4 l'affinage est bistable et donne deux valeurs séparées
de 0,108 sans rapport avec la configuration.

Trois grandeurs par case, et la dispersion entre graines à côté de chacune, car un écart plus
petit que la dispersion n'existe pas.

    .venv/bin/python .runs/quebec/lire_carre.py
"""
import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

_racine = os.environ.get("MEANDRE_DERIVES")
if not _racine:
    from meandre.utils import paths as _paths

    _racine = _paths.DERIVED_ROOT
DERIVES = f"{_racine}/auxiliaires"

# Nom de passe par case et par graine. La case pile-libre vient de la série sur le taux
# d'apprentissage, qui portait exactement la même configuration et les mêmes graines.
CASES = {
    ("temoin", "libre"): ["s-temoin-g1234", "s-temoin-g7"],
    ("temoin", "contraint"): ["s-temoinc-g1234", "s-temoinc-g7"],
    ("pile", "libre"): ["lr3e-5-g1234", "lr3e-5-g7"],
    ("pile", "contraint"): ["s-pilec-g1234", "s-pilec-g7"],
}


def lire(nom, region="outv"):
    """KGE tenu de côté, blocs jetés pour gradient non fini, et meilleure époque retenue."""
    f = f"{DERIVES}/eval-{region}-{nom}.log"
    if not os.path.exists(f):
        return None
    texte = open(f, encoding="utf-8", errors="replace").read()
    kge = re.findall(r"HELD-OUT.*?médian\s+([-0-9.]+)", texte)
    jetes = re.findall(r"\((\d+) bloc\(s\) jete", texte)
    epoque = re.findall(r"best checkpoint saved \(kge_median=([-0-9.]+)\)", texte)
    return {"nom": nom,
            "kge": float(kge[-1]) if kge else np.nan,
            "jetes": int(jetes[-1]) if jetes else 0,
            "val": float(epoque[-1]) if epoque else np.nan,
            "n_sauvegardes": len(epoque)}


def structure(nom, region="outv"):
    """Chemins de l'eau, troncature et recharge, lus dans la sortie journalière de la passe.

    Ce sont les grandeurs qui décident, le KGE ne départageant pas les variantes. L'indice
    d'écoulement de base vaut 0,54 en médiane sur les hydrogrammes observés, la nervosité de
    la composante rapide 2,045.
    """
    f = f"{DERIVES}/reach-{region}-{nom}-journalier.npz"
    if not os.path.exists(f):
        return None
    z = np.load(f, allow_pickle=True)
    if not all(k in z.files for k in ("prod_surf", "prod_hypo", "prod_base")):
        return None
    base, surf, hypo = z["prod_base"], z["prod_surf"], z["prod_hypo"]
    tot = base.mean() + surf.mean() + hypo.mean()
    rapide = (surf + hypo).mean(axis=1)
    b = base.mean(axis=1)
    out = {"indice_base": float(base.mean() / tot) if tot > 0 else np.nan,
           "part_surface": float(surf.mean() / tot) if tot > 0 else np.nan,
           "variation_base": float(b.std() / max(b.mean(), 1e-9)),
           "nervosite_rapide": float(rapide.std() / max(rapide.mean(), 1e-9))}
    if "temps_non_traite" in z.files:
        out["part_non_traitee"] = float(z["temps_non_traite"].mean())
    if "recharge" in z.files:
        out["recharge_mm_an"] = float(z["recharge"].mean() * 365.25)
    return out


OBSERVE = {"indice_base": 0.54, "nervosite_rapide": 2.045, "variation_base": 0.68}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--region", default="outv")
    a = p.parse_args()
    print(f"{'physique':9s} {'contrainte':11s} | {'KGE tenu de cote':22s} | {'blocs jetes':13s} | validation")
    resume = {}
    for (phys, contr), noms in CASES.items():
        passes = [r for n in noms if (r := lire(n, a.region)) is not None]
        if not passes:
            print(f"{phys:9s} {contr:11s} | pas encore de sortie")
            continue
        k = np.array([r["kge"] for r in passes])
        j = np.array([r["jetes"] for r in passes])
        v = np.array([r["val"] for r in passes])
        etendue = f"{k.min():.4f} a {k.max():.4f}" if len(k) > 1 else "une seule passe"
        print(f"{phys:9s} {contr:11s} | {np.mean(k):.4f}  ({etendue}) | "
              f"{int(j.mean()):5d} en moyenne | {np.mean(v):.4f}")
        resume[(phys, contr)] = k
    if len(resume) < 4:
        print("\nCarre incomplet : les effets ne se lisent pas encore.")
        return 0
    disp = max(k.max() - k.min() for k in resume.values())
    print(f"\ndispersion maximale entre graines d'une meme case : {disp:.4f}")
    eff_phys = (np.mean(list(resume[("pile", "libre")]) + list(resume[("pile", "contraint")]))
                - np.mean(list(resume[("temoin", "libre")]) + list(resume[("temoin", "contraint")])))
    eff_contr = (np.mean(list(resume[("temoin", "contraint")]) + list(resume[("pile", "contraint")]))
                 - np.mean(list(resume[("temoin", "libre")]) + list(resume[("pile", "libre")])))
    inter = (np.mean(resume[("pile", "contraint")]) - np.mean(resume[("pile", "libre")])
             - np.mean(resume[("temoin", "contraint")]) + np.mean(resume[("temoin", "libre")]))
    for nom, val in [("effet de la pile de corrections", eff_phys),
                     ("effet de la contrainte sur les puits", eff_contr),
                     ("interaction des deux", inter)]:
        verdict = "lisible" if abs(val) > disp else "SOUS la dispersion, non lisible"
        print(f"{nom:36s} : {val:+.4f}  {verdict}")
    print("\nStructure des chemins de l'eau, moyenne des deux graines :")
    cles = ["indice_base", "part_surface", "variation_base", "nervosite_rapide",
            "part_non_traitee", "recharge_mm_an"]
    print(f"{'case':22s} | " + " | ".join(f"{c[:15]:>15s}" for c in cles))
    for (phys, contr), noms in CASES.items():
        st = [s for n in noms if (s := structure(n, a.region)) is not None]
        if not st:
            continue
        moy = {c: np.mean([s[c] for s in st if c in s]) for c in cles if any(c in s for s in st)}
        print(f"{phys + ' ' + contr:22s} | "
              + " | ".join(f"{moy[c]:15.3f}" if c in moy else f"{'':>15s}" for c in cles))
    print(f"{'observe':22s} | "
          + " | ".join(f"{OBSERVE[c]:15.3f}" if c in OBSERVE else f"{'':>15s}" for c in cles))
    return 0


if __name__ == "__main__":
    sys.exit(main())
