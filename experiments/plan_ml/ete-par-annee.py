"""Volume de juin à septembre simulé sur observé, par année, médiane des stations ayant plus de
100 jours mesurés dans l'été, pour des passes du pilote (ETL_DUMP_Q) d'un même territoire.

    .venv/Scripts/python.exe experiments/plan_ml/ete-par-annee.py mont hiv casr-brut ks0.33 ks0.1
"""
import sys

import numpy as np
import pandas as pd

D = "D:/meandre-data/experiments/plan_ml"


def main(reg, prefix, tags):
    rows = {}
    for tag in tags:
        z = np.load(f"{D}/q-{prefix}-{reg}-{tag}.npz", allow_pickle=True)
        t = pd.DatetimeIndex(z["dates"])
        qs, qo = z["q_sim"], z["q_obs"]
        r = {}
        for y in sorted(set(t.year)):
            m = (t.year == y) & t.month.isin([6, 7, 8, 9])
            ok = np.isfinite(qo[m])
            n = ok.sum(0)
            s = np.where(ok, qs[m], 0).sum(0)
            o = np.where(ok, qo[m], 0).sum(0)
            k = n > 100
            r[y] = np.median(s[k] / o[k]) if k.any() else np.nan
        rows[tag] = r
    print(pd.DataFrame(rows).T.round(2).to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
