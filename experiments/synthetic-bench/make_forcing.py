"""Météo fabriquée du banc fictif : trois années, du 1er janvier 2000 au 31 décembre 2002.

Climat de type québécois méridional, le même sur tous les tronçons : température moyenne de
−11 °C en janvier à +19 °C en juillet, 8 mm de précipitation tous les trois jours (970 mm
par an hors épisodes fabriqués, 905 mm en moyenne sur les trois ans ; neige l'hiver par la
température), rayonnement net saisonnier, vent de 3,5 m/s.

- 2000 : mise en régime.
- 2001 : sécheresse du 15 juillet au 15 septembre, pour la récession estivale et l'étiage.
- 2002 : du 25 mai au 31 octobre, la pluie de fond est retirée et remplacée par huit orages de
  10, 20, 40 et 60 mm, chacun après dix jours secs ou après trois jours de 10 mm, suivis de
  dix jours sans pluie.

Les dates des orages sont écrites dans un second fichier, lu par l'analyse.
Usage : python make_forcing.py <dossier de sortie>
"""
import os
import sys

import numpy as np
import pandas as pd

STORMS_MM = [10.0, 20.0, 40.0, 60.0]


def main(out):
    dates = pd.date_range("2000-01-01", "2002-12-31", freq="D")
    doy = dates.dayofyear.to_numpy()
    t_mean = 4.0 + 15.0 * np.sin(2.0 * np.pi * (doy - 110) / 365.0)
    t_min, t_max = t_mean - 5.0, t_mean + 5.0
    rn = np.clip(7.0 + 7.0 * np.sin(2.0 * np.pi * (doy - 80) / 365.0), 0.5, None)
    ea = 0.611 * np.exp(17.27 * t_min / (t_min + 237.3))
    p = np.where(np.arange(len(dates)) % 3 == 0, 8.0, 0.0)
    drought = (dates >= "2001-07-15") & (dates <= "2001-09-15")
    p[drought] = 0.0
    season = (dates >= "2002-05-25") & (dates <= "2002-10-31")
    p[season] = 0.0
    events = []
    day = int(np.flatnonzero(dates == pd.Timestamp("2002-06-05"))[0])
    for wet in (False, True):
        for mm in STORMS_MM:
            if wet:
                p[day - 3:day] = 10.0
            p[day] = mm
            events.append((str(dates[day].date()), mm, wet))
            day += 20
    os.makedirs(out, exist_ok=True)
    pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "P": p, "Tmin": t_min, "Tmax": t_max, "Rn": rn, "u2": 3.5, "ea": ea}).to_csv(f"{out}/fictif-v1.csv", index=False)
    pd.DataFrame(events, columns=["date", "pluie_mm", "apres_pluie"]).to_csv(f"{out}/fictif-v1-orages.csv", index=False)
    print(f"{out}/fictif-v1.csv : {len(dates)} jours, {p.sum() / 3:.0f} mm/an en moyenne, {len(events)} orages")


if __name__ == "__main__":
    main(sys.argv[1])
