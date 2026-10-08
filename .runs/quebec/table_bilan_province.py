"""Table provinciale de l'exces d'hiver-printemps et du deficit d'ete, a partir du journal
`bilan_province.log` (passes avant a zero epoque, bilan mensuel de la colonne, rapport simule
sur observe par mois). Une ligne par territoire : rapport simule/observe decembre-mars,
avril-mai, juin-aout, septembre-octobre ; evaporation de juin a aout en mm par mois ; neige
accumulee de decembre a mars en mm ; volume annuel.

    python .runs/quebec/table_bilan_province.py <bilan_province.log>
"""
import re
import sys

import numpy as np


def main(chemin):
    texte = open(chemin, encoding="utf-8", errors="replace").read()
    blocs = re.split(r"^=== (\w+) ", texte, flags=re.M)[1:]
    print(f"{'territoire':10s} {'KGE':>5s} {'vol':>6s} | {'dec-mar':>7s} {'avr-mai':>7s} {'jun-aou':>7s} {'sep-oct':>7s} | {'ETR jun-aou':>11s} {'prod jun-aou':>12s} {'neige dec-mar':>13s}")
    for reg, corps in zip(blocs[::2], blocs[1::2]):
        m = re.search(r"m[ée]dian ([0-9.]+)", corps)
        kge = float(m.group(1)) if m else np.nan
        v = re.search(r"ANNEE\s+\d+\s+\d+\s+[+-]?\d+\s+([+-]?[0-9.]+) %", corps)
        vol = 1 + float(v.group(1)) / 100 if v else np.nan
        r = re.search(r"simule/observe par mois : (.*)", corps)
        if not r:
            print(f"{reg:10s} (pas de bilan : {corps.strip().splitlines()[-1][:80] if corps.strip() else ''})")
            continue
        mois = {int(k): float(x) for k, x in re.findall(r"(\d\d)=([0-9.]+)", r.group(1))}
        lignes = re.findall(r"^\s+(\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*$", corps, flags=re.M)
        bil = {int(l[0]): tuple(int(x) for x in l[1:]) for l in lignes}
        def moy(ms):
            return np.mean([mois[k] for k in ms if k in mois])
        etr = sum(bil[k][1] for k in (6, 7, 8) if k in bil) / 3
        prod = sum(bil[k][2] for k in (6, 7, 8) if k in bil) / 3
        neige = sum(bil[k][3] for k in (12, 1, 2, 3) if k in bil)
        print(f"{reg:10s} {kge:5.2f} {vol:6.2f} | {moy((12, 1, 2, 3)):7.2f} {moy((4, 5)):7.2f} {moy((6, 7, 8)):7.2f} {moy((9, 10)):7.2f} | {etr:11.0f} {prod:12.0f} {neige:13.0f}")


if __name__ == "__main__":
    main(sys.argv[1])
