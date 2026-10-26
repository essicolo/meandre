"""Bilan de la colonne de juin à septembre par année, médiane des bassins jaugés (moyennes de
bassin écrites par ETL_DUMP_STATIONS) : apport au sol, ETR, écoulement latéral et ses trois
composantes, teneur en eau de surface.

    .venv/Scripts/python.exe experiments/plan_ml/bilan-ete-annee.py diag-hiv-mont-por072
"""
import sys

import numpy as np
import pandas as pd

D = "D:/meandre-data/experiments/plan_ml"


def main(name):
    z = np.load(f"{D}/{name}.npz", allow_pickle=True)
    t = pd.DatetimeIndex(z["dates"])
    rows = []
    for y in sorted(set(t.year)):
        m = (t.year == y) & t.month.isin([6, 7, 8, 9])
        total = lambda a: np.median(z["basin_" + a][m].sum(0))
        rows.append(dict(an=y, apport=total("snowmelt"), etr=total("etr"), lateral=total("lateral_mm"), surf=total("prod_surf"), hypo=total("prod_hypo"), base=total("prod_base"), theta1=np.median(z["basin_theta1"][m].mean(0))))
    print(name)
    print(pd.DataFrame(rows).round(2).to_string(index=False))


if __name__ == "__main__":
    for n in sys.argv[1:]:
        main(n)
