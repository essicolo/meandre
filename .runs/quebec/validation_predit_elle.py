"""La validation prédit-elle le tenu de côté, et dans quel sens.

Le point de reprise est choisi sur le KGE médian de VALIDATION, période comprise dans
l'entraînement. Le seul chiffre qui compte est le KGE médian TENU DE CÔTÉ, 2022-2024, jamais
vu. Si les deux ne vont pas dans le même sens, la sélection d'époque travaille contre le
résultat, et le protocole de comparaison doit changer avant le prochain chantier.

Le test lit toutes les passes d'un territoire qui portent les deux chiffres et rend le
coefficient de rang de Spearman, insensible à la forme de la relation.

    .venv/bin/python .runs/quebec/validation_predit_elle.py --region outv
"""
import argparse
import glob
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


def passes(region):
    """Nom, validation retenue, tenu de côté et taux sommital, pour chaque passe entraînée."""
    for f in sorted(glob.glob(f"{DERIVES}/eval-{region}-*.log")):
        texte = open(f, encoding="utf-8", errors="replace").read()
        kge = re.findall(r"HELD-OUT.*?médian\s+([-0-9.]+)", texte)
        val = re.findall(r"best checkpoint saved \(kge_median=([-0-9.]+)\)", texte)
        lrs = [float(x) for x in re.findall(r"lr=([0-9.e+-]+)", texte)]
        if not kge or not val or not lrs:
            continue
        yield {"nom": os.path.basename(f)[len(f"eval-{region}-"):-4],
               "val": float(val[-1]), "tenu": float(kge[-1]), "lr_sommet": max(lrs)}


def spearman(x, y):
    """Coefficient de rang, sans dépendance à scipy."""
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else np.nan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--region", default="outv")
    p.add_argument("--seuil-stable", type=float, default=1e-4,
                   help="taux sommital au-dessous duquel l'affinage est reproductible")
    a = p.parse_args()
    tout = list(passes(a.region))
    if len(tout) < 4:
        print(f"{len(tout)} passes : trop peu pour conclure")
        return 1
    stables = [r for r in tout if r["lr_sommet"] < a.seuil_stable]
    for nom, jeu in [("toutes les passes", tout), ("passes a taux stable", stables)]:
        if len(jeu) < 4:
            print(f"{nom} : {len(jeu)} passes, trop peu")
            continue
        v = np.array([r["val"] for r in jeu])
        t = np.array([r["tenu"] for r in jeu])
        print(f"{nom:24s} | {len(jeu):3d} passes | rang de Spearman {spearman(v, t):+.2f} | "
              f"validation {v.mean():.3f}, tenu de cote {t.mean():.3f}")
    print("\nDetail des passes a taux stable, triees par validation decroissante :")
    for r in sorted(stables, key=lambda r: -r["val"]):
        print(f"  {r['nom']:20s} | validation {r['val']:.4f} | tenu de cote {r['tenu']:.4f}"
              f" | sommet du taux {r['lr_sommet']:.1e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
