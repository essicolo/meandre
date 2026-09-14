"""Le sur-lissage nait-il dans la COLONNE ou dans le RESEAU ?

Le modele bouge deux fois moins vite que la riviere d'un jour a l'autre : la variabilite
des variations de debit vaut 0,54 de l'observee en mediane sur 129 stations, et lisser
l'observation ameliore la correlation sur 95 pour cent d'entre elles. Aucun facteur
structurel ne l'explique, ni les lacs, ni le nombre de troncons, ni la pluie d'entree, et
abaisser la borne du Muskingum sous quatre heures n'a rien donne en aout 2026.

Deux endroits restent possibles et ce script les separe, sans entrainement. On compare la
variabilite des variations d'un jour a l'autre de trois series, toutes rapportees a leur
propre moyenne pour etre sans dimension :

  la PRODUCTION de la colonne, sommee sur les trois chemins et moyennee sur les troncons
    du bassin, avant tout routage ;
  le DEBIT SIMULE a l'exutoire, apres routage ;
  le DEBIT OBSERVE.

Si la production est deja lissee au niveau du debit simule, le canal est hors de cause et
le defaut est dans la colonne ou la reponse du versant. Si elle est aussi vive que
l'observation, tout se perd entre la colonne et l'exutoire.

Le sous-bassin doit PORTER le defaut : en choisir un dont l'indice regional est bas.

    .venv/bin/python .runs/quebec/lissage_ou.py sagu 062803
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def variabilite(x):
    """Ecart-type des variations d'un jour a l'autre, rapporte a la moyenne de la serie."""
    x = np.asarray(x, dtype=float)
    m = np.isfinite(x)
    x = x[m]
    if len(x) < 100 or x.mean() <= 0:
        return np.nan
    return float(np.std(np.diff(x)) / x.mean())


def main(reg, station):
    import banc_sousbassin as banc

    os.environ.setdefault("JOINT_FX_SUFFIX", "-budyko")
    os.environ.setdefault("ETL_SEED", "1234")
    temps, q, o = banc.simuler(reg, station, melt_saison=0.5, sol="sauf_ks", annees=6)
    part = banc._DERNIERE_PARTITION
    if not part:
        print("partition des chemins indisponible")
        return 1
    prod = sum(part.values())
    an = np.asarray(temps.year)
    g = an > an.min()
    m = g & np.isfinite(o) & np.isfinite(q)

    v_prod = variabilite(prod[m])
    v_sim = variabilite(q[m])
    v_obs = variabilite(o[m])
    print(f"\n{reg.upper()} / {station} : {int(m.sum())} jours retenus")
    print(f"  variabilité des variations d'un jour à l'autre, rapportée à la moyenne")
    print(f"    production de la colonne, avant routage : {v_prod:.3f}")
    print(f"    débit simulé, après routage             : {v_sim:.3f}")
    print(f"    débit observé                           : {v_obs:.3f}")
    print(f"\n  production sur observé : {v_prod / v_obs:.2f}")
    print(f"  simulé sur production  : {v_sim / v_prod:.2f}   (ce que le réseau retire)")
    print(f"  simulé sur observé     : {v_sim / v_obs:.2f}   (le défaut à expliquer)")
    print("\n  lecture : si le premier rapport vaut déjà 0,5, la colonne produit trop lisse")
    print("  et le canal est hors de cause. S'il vaut 1 et le second 0,5, tout se perd au routage.")

    # Par saison, puisque la crue porte 59 pour cent de la variance annuelle
    mois = np.asarray(temps.month)
    print(f"\n  {'saison':18s} {'production/obs':>16s} {'simulé/production':>19s} {'simulé/obs':>13s}")
    for nom, ms in (("crue avril-mai", (4, 5)), ("été juin-sept", (6, 7, 8, 9)),
                    ("hiver déc-mars", (12, 1, 2, 3))):
        k = m & np.isin(mois, ms)
        if k.sum() < 120:
            continue
        a, b, c = variabilite(prod[k]), variabilite(q[k]), variabilite(o[k])
        print(f"  {nom:18s} {a / c:16.2f} {b / a:19.2f} {b / c:13.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "sagu",
                  sys.argv[2] if len(sys.argv) > 2 else "062803"))
