"""Rapport simule sur observe par saison, MEDIANE DES STATIONS, pour des exports de stations
(ETL_DUMP_Q, un fichier par territoire). Remplace la ligne « simule/observe par mois » du
pilote, qui est un rapport de SOMMES et explose la ou l'observe d'hiver est nul ou absent
(Abitibi, Cote-Nord : 1e12 le 2026-10-08).

    python .runs/quebec/saisons_stations.py q-*.npz
"""
import os
import sys

import numpy as np
import pandas as pd

SAISONS = {"dec-mar": (12, 1, 2, 3), "avr-mai": (4, 5), "jun-aou": (6, 7, 8), "sep-oct": (9, 10)}


def rapports(f):
    d = np.load(f, allow_pickle=True)
    dates = pd.to_datetime(d["dates"])
    mois = dates.month
    out = {}
    for nom, ms in SAISONS.items():
        sel = np.isin(mois, ms)
        r = []
        for k in range(d["q_sim"].shape[1]):
            o, s = d["q_obs"][sel, k], d["q_sim"][sel, k]
            m = np.isfinite(o) & np.isfinite(s)
            if m.sum() >= 60 and o[m].mean() > 0:
                r.append(s[m].mean() / o[m].mean())
        out[nom] = (float(np.median(r)) if r else np.nan, len(r))
    o, s = d["q_obs"], d["q_sim"]
    m = np.isfinite(o) & np.isfinite(s)
    vols = [s[m[:, k], k].mean() / o[m[:, k], k].mean() for k in range(o.shape[1]) if m[:, k].sum() >= 365 and o[m[:, k], k].mean() > 0]
    out["annee"] = (float(np.median(vols)) if vols else np.nan, len(vols))
    return out


def main(fichiers):
    print(f"{'fichier':28s} {'n':>3s} {'annee':>6s} | {'dec-mar':>7s} {'avr-mai':>7s} {'jun-aou':>7s} {'sep-oct':>7s}   (mediane des stations, 2022-2024)")
    for f in fichiers:
        r = rapports(f)
        print(f"{os.path.basename(f)[:-4]:28s} {r['annee'][1]:3d} {r['annee'][0]:6.2f} | " + " ".join(f"{r[s][0]:7.2f}" for s in SAISONS))


if __name__ == "__main__":
    main(sys.argv[1:])
