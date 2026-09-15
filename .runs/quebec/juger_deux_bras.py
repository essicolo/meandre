"""Verdict de l'epreuve appariee du terme des variations d'un jour.

Lit les journaux des vingt-huit taches et rend, region par region, les quatre grandeurs qui
jugent : la nervosite, la corrélation, le rapport des pointes et la platitude. Le KGE sert
de garde-fou, jamais de juge.

TROIS ATTENDUS, ecrits AVANT l'epreuve, au registre du 2026-09-15.
  1. Le terme supprime la preference de la perte pour le lissage sur environ quatre-vingt-dix
     pour cent des stations, et sur un peu moins de la moitie pour le rabotage des pics.
  2. Les suites plates d'hiver raccourcissent dans les DEUX bras, puisque le depart a froid
     entraine la physique du gel pour la premiere fois. Reference : 74 a 116 jours le matin
     du 15 septembre.
  3. Le terme etant SYMETRIQUE, il ameliore les deux populations : les trois regions dont la
     nervosite vaut 0,57 a 0,65 et celles ou elle depasse 1, jusqu'a 2,12.

    .runs/quebec/juger_deux_bras.py ~/scratch/meandre/journaux
"""
import os
import re
import sys

import pandas as pd

MOT = {
    "kge": re.compile(r"HELD-OUT [\d-]+ (\w+): n=(\d+) \| médian ([-\d.]+)"),
    "forme": re.compile(r"FORME (\w+) .*?: plat ([\d.]+) % \(obs ([\d.]+) %\).*?"
                        r"ete ([\d.]+) % \(obs ([\d.]+) %\).*?suite plate (\d+) j.*?"
                        r"pointes sim/obs ([\d.]+) \| q99 sim/obs ([\d.]+)"),
    "wdq": re.compile(r"w_dq = ([\d.]+)"),
}


def lire(f):
    d = {}
    with open(f, encoding="utf-8", errors="replace") as h:
        for l in h:
            m = MOT["kge"].search(l)
            if m:
                d.update(region=m.group(1), stations=int(m.group(2)), kge=float(m.group(3)))
            m = MOT["forme"].search(l)
            if m:
                d.update(plat=float(m.group(2)), plat_obs=float(m.group(3)),
                         ete=float(m.group(4)), ete_obs=float(m.group(5)),
                         suite=int(m.group(6)), pointes=float(m.group(7)),
                         q99=float(m.group(8)))
            m = MOT["wdq"].search(l)
            if m:
                d["w_dq"] = float(m.group(1))
    return d


def main(dossier):
    dossier = os.path.expanduser(dossier)
    lignes = []
    for f in sorted(os.listdir(dossier)):
        if not f.startswith("ref-") or not f.endswith(".log"):
            continue
        p = os.path.join(dossier, f)
        d = lire(p)
        if "kge" not in d:
            print(f"  {f} : inachevé ou sans score")
            continue
        parts = f.split("-")
        d["bras"] = parts[2]
        lignes.append(d)
    if not lignes:
        print("aucun journal exploitable")
        return 1
    t = pd.DataFrame(lignes)
    print(f"{len(t)} tâches lues, {t.region.nunique()} régions\n")
    for col, nom in (("pointes", "rapport des pointes"), ("suite", "plus longue suite plate (j)"),
                     ("plat", "jours plats (%)"), ("kge", "KGE médian")):
        p = t.pivot_table(index="region", columns="bras", values=col)
        if {"temoin", "variations"} <= set(p.columns):
            p["écart"] = p["variations"] - p["temoin"]
        print(nom)
        print(p.to_string(float_format=lambda v: f"{v:.3f}"))
        print()
    if "suite" in t:
        print(f"suites plates : témoin médian {t[t.bras=='temoin'].suite.median():.0f} j, "
              f"variations médian {t[t.bras=='variations'].suite.median():.0f} j, "
              f"référence du matin 74 à 116 j")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "~/scratch/meandre/journaux"))
