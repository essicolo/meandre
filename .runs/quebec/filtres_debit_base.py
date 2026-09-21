"""Le choix du filtre de séparation change-t-il l'indice d'écoulement de base.

Tous les chiffres d'indice cités depuis le 2026-09-19, de 0,54 à 0,58 selon le territoire,
viennent du filtre récursif de Lyne et Hollick à trois passes. Une collègue qui modélise les
mêmes étiages emploie celui d'Eckhardt, et pose l'hypothèse qu'à l'échelle d'une station toute
la recharge finit par faire résurgence en débit de base. Si les deux filtres ne donnent pas la
même chose, la cible change, et avec elle le verdict sur la part souterraine du modèle.

Les deux filtres. Lyne et Hollick, 1979, sépare la composante rapide par un passe-haut du
premier ordre appliqué en aller-retour ; son unique paramètre est l'amortissement, pris à
0,925. Eckhardt, 2005, ajoute une borne physique, la part maximale que le débit de base peut
atteindre, et se règle par deux grandeurs interprétables :

    b_k = ((1 - BFImax) a b_(k-1) + (1 - a) BFImax Q_k) / (1 - a BFImax),   b_k <= Q_k

où `a` est la constante de récession journalière et `BFImax` la part maximale. Eckhardt
recommande 0,80 pour un cours d'eau pérenne sur aquifère poreux, 0,50 sur socle fracturé,
0,25 pour un cours d'eau intermittent. Le Québec méridional relève des deux premiers cas selon
qu'on est dans les basses-terres ou sur le Bouclier.

La constante `a` n'est pas devinée ici : elle se déduit du temps de vidange aux basses eaux,
mesuré station par station par la même analyse de récession qui sert à l'ancrage de l'exposant.

    .venv/Scripts/python.exe .runs/quebec/filtres_debit_base.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from importlib.machinery import SourceFileLoader

from meandre.data.recession_anchor import exposant_recession, segments_de_decrue

_ib = SourceFileLoader("ib", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "indice_base_observe.py")).load_module()

REGS = ["outv", "slno", "gasp", "mont", "sagu"]


def constante_recession(q):
    """Constante de récession journalière `a`, depuis les segments de décrue franche."""
    bons = segments_de_decrue(q)
    r = q[1:] / np.clip(q[:-1], 1e-9, None)
    garde = bons[1:] & np.isfinite(r) & (r > 0.2) & (r < 1.0)
    return float(np.median(r[garde])) if garde.sum() > 50 else 0.95


def eckhardt(q, a, bfi_max):
    """Filtre d'Eckhardt, une passe, borné par le débit lui-même."""
    b = np.empty_like(q)
    b[0] = q[0] * bfi_max
    c = 1.0 - a * bfi_max
    for k in range(1, len(q)):
        val = ((1.0 - bfi_max) * a * b[k - 1] + (1.0 - a) * bfi_max * q[k]) / c
        b[k] = min(val, q[k])
    return b


def main():
    flotte = os.environ.get("MEANDRE_FLOTTE", f"{os.environ.get('MEANDRE_DATA', '.')}/quebec/flotte")
    lignes = []
    for reg in REGS:
        f = sorted(glob.glob(f"{flotte}/q-{reg}-*.npz"))
        if not f:
            continue
        obs = np.load(f[0], allow_pickle=True)["q_obs"]
        for j in range(obs.shape[1]):
            q = obs[:, j].astype(float)
            fini = np.isfinite(q) & (q > 0)
            if fini.sum() < 700:
                continue
            q = q[fini]
            a = constante_recession(q)
            ligne = {"region": reg, "station": j, "a": a,
                     "lyne_hollick": float(_ib.lyne_hollick(q).sum() / q.sum())}
            for bfi_max in (0.50, 0.65, 0.80):
                ligne[f"eckhardt {bfi_max:.2f}"] = float(eckhardt(q, a, bfi_max).sum() / q.sum())
            b = exposant_recession(q)
            ligne["exposant"] = b if b is not None else np.nan
            lignes.append(ligne)
    if not lignes:
        print("aucune station lisible")
        return 1
    t = pd.DataFrame(lignes)
    cols = ["lyne_hollick"] + [c for c in t.columns if c.startswith("eckhardt")]
    pd.set_option("display.width", 200)
    print(f"{len(t)} stations, constante de recession mediane {t.a.median():.3f}\n")
    print("Indice d'ecoulement de base median, par territoire et par filtre :\n")
    print(t.groupby("region")[cols].median().round(3).to_string())
    print("\nToutes stations :")
    print(t[cols].median().round(3).to_string())
    print("\nEcart entre filtres, par station :")
    for c in cols[1:]:
        d = t[c] - t["lyne_hollick"]
        print(f"  {c:16s} contre Lyne-Hollick : mediane {d.median():+.3f}, "
              f"etendue {d.quantile(0.1):+.3f} a {d.quantile(0.9):+.3f}")
    print("\nCorrelation de rang entre filtres, sur les stations :")
    print(t[cols].rank().corr().round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
