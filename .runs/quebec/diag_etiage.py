"""Sens de l'erreur sur les jours d'étiage, par saison, à partir d'un export ETL_DUMP_Q.

Le terme d'étiage de la perte est l'écart absolu des logarithmes sur les jours où le débit
observé est sous son 30e centile. Il ne dit pas si le modèle est trop haut ou trop bas, ni
en quelle saison. Ce script le dit, station par station, puis en médiane.

Usage : python diag_etiage.py export.npz [export2.npz ...]
"""
import sys

import numpy as np
import pandas as pd

EPS = 1e-3
SAISONS = {"hiver (dec-mars)": (12, 1, 2, 3), "printemps (avr-mai)": (4, 5),
           "ete (juin-sept)": (6, 7, 8, 9), "automne (oct-nov)": (10, 11)}


def diagnostic(chemin: str) -> None:
    z = np.load(chemin, allow_pickle=True)
    qs, qo = z["q_sim"].astype(float), z["q_obs"].astype(float)
    mois = pd.DatetimeIndex(z["dates"]).month.to_numpy()
    lignes = []
    for s in range(qo.shape[1]):
        ok = np.isfinite(qo[:, s]) & np.isfinite(qs[:, s])
        if ok.sum() < 60:
            continue
        seuil = np.quantile(qo[ok, s], 0.3)
        bas = ok & (qo[:, s] <= seuil)
        e = np.log(np.clip(qs[:, s], 0, None) + EPS) - np.log(qo[:, s] + EPS)
        lig = {"station": str(z["station_ids"][s]), "terme": np.abs(e[bas]).mean(), "biais_log": e[bas].mean()}
        for nom, m in SAISONS.items():
            sel = bas & np.isin(mois, m)
            lig[nom] = (sel.sum() / bas.sum(), np.median(e[sel]) if sel.sum() >= 5 else np.nan)
        lignes.append(lig)
    print(f"== {chemin} : {len(lignes)} stations")
    print(f"   terme d'etiage moyen {np.mean([l['terme'] for l in lignes]):.3f} | biais log moyen {np.mean([l['biais_log'] for l in lignes]):+.3f}"
          f" (positif : simule trop haut) | stations trop hautes {sum(l['biais_log'] > 0 for l in lignes)}/{len(lignes)}")
    for nom in SAISONS:
        part = np.nanmean([l[nom][0] for l in lignes])
        med = np.nanmedian([l[nom][1] for l in lignes])
        print(f"   {nom:<20} part des jours d'etiage {part:5.2f} | rapport sim/obs median {np.exp(med):5.2f}")


if __name__ == "__main__":
    for c in sys.argv[1:]:
        diagnostic(c)
