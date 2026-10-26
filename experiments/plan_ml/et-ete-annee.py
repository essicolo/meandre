"""ETR de juin à septembre par année : simulée au tronçon des stations (ETL_DUMP_STATIONS) contre
MOD16 au même tronçon, mm/j, médiane des stations ; avec le rapport simulé sur observé du débit d'été.

    .venv/Scripts/python.exe experiments/plan_ml/et-ete-annee.py mont diag-agri-mont-temoin q-agri-mont-temoin
"""
import sys

import duckdb
import numpy as np
import pandas as pd

D = "D:/meandre-data/experiments/plan_ml"


def main(reg, diag, qname):
    z = np.load(f"{D}/{diag}.npz", allow_pickle=True)
    t = pd.DatetimeIndex(z["dates"])
    ids = [str(s).split(":")[-1] for s in z["station_ids"]]
    con = duckdb.connect(f"D:/meandre-data/quebec/{reg}.duckdb", read_only=True)
    st = con.sql("select station_id, node_idx from stations").df().set_index("station_id")["node_idx"]
    nodes = [int(st[i]) for i in ids]
    m16 = con.sql(f"select date, node_idx, etr_mm_day from modis_et where quality_ok and node_idx in ({','.join(map(str, set(nodes)))})").df()
    con.close()
    m16 = m16.pivot_table(index="date", columns="node_idx", values="etr_mm_day")
    m16.index = pd.DatetimeIndex(m16.index)
    q = np.load(f"{D}/{qname}.npz", allow_pickle=True)
    tq = pd.DatetimeIndex(q["dates"])
    rows = []
    for y in sorted(set(t.year)):
        m = (t.year == y) & t.month.isin([6, 7, 8, 9])
        sim = z["etr"][m].mean(0)
        mm = (m16.index.year == y) & m16.index.month.isin([6, 7, 8, 9])
        obs = np.array([m16.loc[mm, n].mean() if n in m16 else np.nan for n in nodes])
        mq = (tq.year == y) & tq.month.isin([6, 7, 8, 9])
        ok = np.isfinite(q["q_obs"][mq])
        k = ok.sum(0) > 100
        rq = np.median((np.where(ok, q["q_sim"][mq], 0).sum(0) / np.where(ok, q["q_obs"][mq], 0).sum(0))[k]) if k.any() else np.nan
        rows.append(dict(an=y, etr_sim=np.nanmedian(sim), etr_mod16=np.nanmedian(obs), rapport=np.nanmedian(sim / obs), debit_sim_obs=rq))
    print(pd.DataFrame(rows).round(2).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:4])
