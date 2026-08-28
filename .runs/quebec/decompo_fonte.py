"""D'ou vient la difference entre degre-jour et ETI ? On MESURE au lieu de raconter.

Motif (Essi, 2026-08-28) : deux narrations que j'ai avancees sans les verifier sont
tombees. (1) J'ai decrit OUTV et GASP comme « plus venteux » alors que mon propre
tableau donnait OUTV a 3.98 m/s contre SAGU a 4.57 -- OUTV est le MOINS vente. (2) J'ai
explique le gain de l'ETI par une fonte de redoux que le degre-jour ne freine pas, ce
qui predit un deficit maximal la ou les redoux sont les plus frequents : c'est OUTV
(35.7 % de jours a Tmax>0), et c'est justement la que le degre-jour accumule le mieux.

Ce script ne suppose rien. Il decompose, mois par mois, la FONTE effective de chaque
formulation (somme des baisses journalieres de SWE) et l'accumulation (somme des
hausses), pour les trois regions. La difference de manteau est alors attribuable a
l'une ou l'autre, sans interpretation.
"""
import sys
sys.path.insert(0, ".")
sys.path.insert(0, ".runs/quebec")
import numpy as np
import pandas as pd
import torch
import importlib

sb = importlib.import_module("snow_bench")


def bilan(d, swe, etiquette):
    """Fonte et accumulation mensuelles, moyennes sur les noeuds de sites (mm/mois)."""
    x = swe.numpy()
    delta = np.diff(x, axis=0)
    mois = pd.DatetimeIndex(d["times"][1:]).month
    annees = pd.DatetimeIndex(d["times"][1:]).year.nunique()
    fonte = np.where(delta < 0, -delta, 0.0).mean(axis=1)
    accum = np.where(delta > 0, delta, 0.0).mean(axis=1)
    mm = [10, 11, 12, 1, 2, 3, 4, 5]
    f = [fonte[mois == m].sum() / annees for m in mm]
    a = [accum[mois == m].sum() / annees for m in mm]
    print(f"  {etiquette}")
    print("      mois " + "".join(f"{m:7d}" for m in mm))
    print("   accum   " + "".join(f"{v:7.0f}" for v in a))
    print("   fonte   " + "".join(f"{v:7.0f}" for v in f))
    return np.array(a), np.array(f)


for reg in ("sagu", "outv"):
    sb.REG = reg
    sb.PLAT = f"{sb._paths.PLATFORMS_ROOT}/LN24HA/{reg.upper()}_LN24HA_2020"
    d = sb.charger()
    print(f"\n########## {reg} ##########")
    a_dj, f_dj = bilan(d, sb.simuler(d, amp=0.5), "degre-jour (recette 1.0, amp 0.5)")
    a_eti, f_eti = bilan(d, sb.simuler(d, amp=None, melt_mode="eti",
                                       tf=1.2e-3, srf=9.4e-6), "ETI litterature")
    mm = [10, 11, 12, 1, 2, 3, 4, 5]
    print("  ecart ETI moins degre-jour")
    print("      mois " + "".join(f"{m:7d}" for m in mm))
    print("   accum   " + "".join(f"{v:+7.0f}" for v in (a_eti - a_dj)))
    print("   fonte   " + "".join(f"{v:+7.0f}" for v in (f_eti - f_dj)))
