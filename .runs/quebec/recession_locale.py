"""Exposant de vidange d'un sous-bassin, mesuré sur ses propres débits observés.

L'ancre de territoire (`meandre/data/recession_anchor.py`) est une médiane sur les grandes
stations ; un sous-bassin peut vidanger autrement. On mesure ici l'exposant de Brutsaert et
Nieber sur la station seule, toute l'année puis sur les seules décrues de juillet à septembre,
et on le convertit en exposant de stock n = 1/(2 - b).

    python .runs/quebec/recession_locale.py outv 040110 mont 030905
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import duckdb
import numpy as np
import pandas as pd

from meandre.data import recession_anchor as ra
from meandre.utils import paths as _p


def exposant(q, dates, mois=None):
    q = np.asarray(q, dtype=float)
    bons = ra.segments_de_decrue(np.nan_to_num(q, nan=0.0))
    dqdt = np.concatenate([[np.nan], -np.diff(q)])
    qm = np.concatenate([[np.nan], 0.5 * (q[1:] + q[:-1])])
    garde = bons & np.isfinite(dqdt) & np.isfinite(qm) & (dqdt > 0) & (qm > 0)
    if mois is not None:
        garde &= np.isin(dates.month, mois)
    if garde.sum() < 60:
        return None, int(garde.sum())
    ex, ey = ra._enveloppe_inferieure(np.log(qm[garde]), np.log(dqdt[garde]))
    if ex is None or len(ex) < 10:
        return None, int(garde.sum())
    return float(np.polyfit(ex, ey, 1)[0]), int(garde.sum())


def main(paires):
    for reg, st in paires:
        con = duckdb.connect(f"{_p.DATA_ROOT}/quebec/{reg}.duckdb", read_only=True)
        obs = con.execute("select date, discharge, reconstructed from observations where station_id = ? order by date", [st]).fetchdf()
        con.close()
        s = pd.Series(obs.discharge.values, index=pd.DatetimeIndex(obs.date))
        s = s[~obs.reconstructed.fillna(True).values]  # jours mesures seulement
        s = s.asfreq("D")
        print(f"{reg} {st} : {int(np.isfinite(s.values).sum())} jours mesures, {s.index.year.min()}-{s.index.year.max()}")
        for nom, mois in (("annee entiere", None), ("juillet-septembre", (7, 8, 9)), ("aout", (8,))):
            b, n = exposant(s.values, s.index, mois)
            if b is None:
                print(f"  {nom:18s} : insuffisant ({n} jours de decrue)")
                continue
            nn = 1.0 / (2.0 - b) if b < ra.B_MAX else float("inf")
            print(f"  {nom:18s} : b = {b:.2f} sur {n} jours de decrue, exposant de stock n = {nn:.2f}")
    print("references de territoire (registre) : MONT 1,36, OUTV 4,65 ; Boussinesq 2,0 ; lineaire 1,0")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(list(zip(a[::2], a[1::2])))
