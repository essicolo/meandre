"""Domaine de convergence des formes du catalogue, en fonction du plafond de sous-pas.

Le catalogue ouvert le 2026-09-20 expose des EXPOSANTS LIBRES, alors que la convergence n'avait
été mesurée que pour l'exposant un, et seulement sur la loi écrite à la main. Une forme non
convergée au plafond employé ne mesure pas ce qu'on croit : sous le plafond, le temps non
traité est versé en ruissellement de surface par la fermeture du bilan, si bien que la masse
est conservée mais que le chemin de l'eau ne l'est pas.

La règle du dépôt est de valider une pièce sur la plus petite unité AVANT de l'intégrer. Ce
banc le fait pour chaque forme et chaque exposant, sur une colonne fictive, sans carte.

Le critère est la production cumulée sur trente jours, rapportée à celle obtenue au plafond de
référence de 512 sous-pas. Une forme est déclarée convergée à un plafond donné quand l'écart
relatif tombe sous deux pour cent.

    .venv/Scripts/python.exe .runs/quebec/convergence_formes.py
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from hydrotel_clone.bv3c2 import BV3C2Clone, make_params
from meandre.vertical import soil_processes as sp

PLAFONDS = (16, 32, 64, 128, 512)
N_JOURS = 30
PLUIE_MM = 4.0
N_NOEUDS = 8


def profil(forme, params, kind="lateral"):
    """Profil minimal : la couche 2 garde sa loi d'Hydrotel, la couche 3 porte la forme visée."""
    procs = [sp.SoilProcess(layer=2, kind="lateral", form="BASE_LINEAR")]
    if kind == "lateral":
        procs.append(sp.SoilProcess(layer=3, kind="lateral", form=forme, params=params))
        procs.append(sp.SoilProcess(layer=3, kind="percolation", form="PERC_LINEAR",
                                    params={"krec": 2e-5}))
    else:
        procs.append(sp.SoilProcess(layer=3, kind="percolation", form=forme, params=params))
    return sp.SoilProfile(layers=3, processes=tuple(procs))


def cumul(prof, n_substep, theta3_depart=0.45):
    """Production hypodermique et de base cumulée sur la période, en millimètres."""
    p = dict(make_params())
    p["soil_profile"] = prof
    p["thetacc2"] = torch.full((N_NOEUDS,), 0.30)
    p["thetacc3"] = torch.full((N_NOEUDS,), 0.30)
    col = BV3C2Clone(n_substep=n_substep)
    t1 = torch.full((N_NOEUDS,), 0.32)
    t2 = torch.full((N_NOEUDS,), 0.34)
    t3 = torch.full((N_NOEUDS,), theta3_depart)
    zero = torch.zeros(N_NOEUDS)
    hypo = base = 0.0
    for _ in range(N_JOURS):
        ps, ph, pb, _rech, (t1, t2, t3), _d = col.forward(
            t1, t2, t3, torch.full((N_NOEUDS,), PLUIE_MM), torch.full((N_NOEUDS,), 1.0),
            zero, zero, p)
        hypo += float(ph.mean())
        base += float(pb.mean())
    return hypo, base


def main():
    essais = [
        ("BASE_THRESH_POWER", {"tau": 2 * 24.0}, "lateral", "latéral, tau 2 j, exposant 1"),
        ("BASE_THRESH_POWER", {"tau": 5 * 24.0}, "lateral", "latéral, tau 5 j, exposant 1"),
        ("BASE_THRESH_POWER", {"tau": 20 * 24.0}, "lateral", "latéral, tau 20 j, exposant 1"),
        ("BASE_THRESH_POWER", {"tau": 5 * 24.0, "exponent": 1.5}, "lateral",
         "latéral, tau 5 j, exposant 1,5"),
        ("BASE_THRESH_POWER", {"tau": 5 * 24.0, "exponent": 2.0}, "lateral",
         "latéral, tau 5 j, exposant 2"),
        ("BASE_THRESH_POWER", {"tau": 5 * 24.0, "exponent": 3.0}, "lateral",
         "latéral, tau 5 j, exposant 3"),
        ("PERC_THRESH_POWER", {"tau": 2 * 24.0}, "percolation", "percolation, tau 2 j"),
        ("PERC_LINEAR", {"krec": 2e-5}, "percolation", "percolation d'Hydrotel"),
        ("PERC_POWER_LAW", {"krec": 2e-5, "exponent": 3.0}, "percolation",
         "percolation en puissance, exposant 3"),
    ]
    lignes = []
    for forme, params, kind, nom in essais:
        prof = profil(forme, params, kind)
        ref = sum(cumul(prof, max(PLAFONDS)))
        for n in PLAFONDS[:-1]:
            tot = sum(cumul(prof, n))
            lignes.append({"forme": nom, "plafond": n,
                           "ecart": abs(tot - ref) / max(abs(ref), 1e-9)})
    t = pd.DataFrame(lignes)
    piv = 100 * t.pivot_table(index="forme", columns="plafond", values="ecart")
    pd.set_option("display.width", 200)
    print(f"Ecart relatif a la solution a {max(PLAFONDS)} sous-pas, en pour cent.")
    print("Converge = sous deux pour cent.\n")
    print(piv.round(1).to_string())
    print("\nPlafond minimal convergeant, par forme :")
    for nom, g in t.groupby("forme"):
        bons = sorted(g[g.ecart < 0.02].plafond.tolist())
        print(f"  {nom:36s} {bons[0] if bons else 'AUCUN des plafonds essayes'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
