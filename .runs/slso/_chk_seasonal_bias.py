"""Caractérise le biais SYSTÉMATIQUE de CaSR : climatologie mensuelle du débit simulé
vs observé, par mois calendaire, moyenne sur les stations. Révèle QUELS mois CaSR
sur/sous-estime systématiquement (la structure à corriger). Compare à PHYSITEL.
Test 2022-2024.
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, duckdb

T0, T1 = "2022-01-01", "2024-12-31"
c = duckdb.connect(".runs/slso/data/slso.duckdb", read_only=True)
st = c.execute("SELECT station_id, node_idx FROM stations").fetchdf()
obs = c.execute(f"SELECT station_id,date,discharge FROM observations WHERE date>='{T0}' AND date<='{T1}'").fetchdf()
c.close()
obs["date"] = pd.to_datetime(obs["date"]).dt.normalize()

def monthly_clim(tag):
    d = np.load(f".runs/slso/results/eval_{tag}.npz", allow_pickle=True)
    Q = d["Q"]; dd = pd.to_datetime(d["dates"].astype(str)).normalize()
    ratios = {m: [] for m in range(1, 13)}
    for _, s in st.iterrows():
        o = obs[obs.station_id == s.station_id][["date", "discharge"]]
        if len(o) < 60: continue
        mo = o.merge(pd.DataFrame({"date": dd, "qm": Q[:, int(s.node_idx)]}), on="date")
        if len(mo) < 60: continue
        mo["mon"] = mo.date.dt.month
        g = mo.groupby("mon").mean(numeric_only=True)
        for m in g.index:
            if g.loc[m, "discharge"] > 1e-6:
                ratios[m].append(g.loc[m, "qm"] / g.loc[m, "discharge"])
    return {m: np.median(v) if v else np.nan for m, v in ratios.items()}

rc = monthly_clim("aquifer"); rq = monthly_clim("ksat05")
moisn = ["", "jan", "fév", "mar", "avr", "mai", "jun", "jul", "aoû", "sep", "oct", "nov", "déc"]
print("mois | sim/obs AQUIFERE | sim/obs ksat05   (1.0 = parfait, >1 sur-estime)")
for m in range(1, 13):
    fc = "  <<<" if (np.isfinite(rc[m]) and (rc[m] > 1.25 or rc[m] < 0.8)) else ""
    print(f" {moisn[m]} |    {rc[m]:.2f}      |    {rq[m]:.2f}{fc}")
print("\nLecture : les mois où CaSR s'écarte le plus de 1.0 (et plus que PHYSITEL)")
print("= la signature du biais systématique de forçage à corriger.")
