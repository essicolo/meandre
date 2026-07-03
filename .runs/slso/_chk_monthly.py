"""KGE de meandre au pas JOURNALIER vs MENSUEL (comparaison à armes égales avec
les modèles de bilan d'eau type PyHELP, évalués au mois). Le mensuel moyenne le
bruit de timing journalier. Test 2022-2024, par station.
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

def kge(s, o):
    m = np.isfinite(s) & np.isfinite(o) & (o >= 0)
    if m.sum() < 12: return None
    s, o = s[m], o[m]
    if s.std() == 0 or o.std() == 0: return None
    r = np.corrcoef(s, o)[0, 1]
    return 1 - np.sqrt((r-1)**2 + (s.mean()/o.mean()-1)**2 + ((s.std()/s.mean())/(o.std()/o.mean())-1)**2)

def analyse(tag):
    d = np.load(f".runs/slso/results/eval_{tag}.npz", allow_pickle=True)
    Q = d["Q"]; dd = pd.to_datetime(d["dates"].astype(str)).normalize()
    day, mon = [], []
    for _, s in st.iterrows():
        o = obs[obs.station_id == s.station_id][["date", "discharge"]]
        if len(o) < 60: continue
        mo = o.merge(pd.DataFrame({"date": dd, "qm": Q[:, int(s.node_idx)]}), on="date")
        if len(mo) < 60: continue
        kd = kge(mo.qm.to_numpy(), mo.discharge.to_numpy())
        # agrégation mensuelle (moyenne)
        mo2 = mo.set_index("date").resample("MS").mean()
        km = kge(mo2.qm.to_numpy(), mo2.discharge.to_numpy())
        if kd is not None: day.append(kd)
        if km is not None: mon.append(km)
    return np.array(day), np.array(mon)

print("modèle  | KGE journalier (méd / >0.8) | KGE mensuel (méd / >0.8)")
for tag, name in [("ksat05", "ksat05"), ("retention", "RETENTION")]:
    try:
        day, mon = analyse(tag)
        print(f"{name:8s}| {np.median(day):.3f} / {int((day>0.8).sum())}/{len(day)}            "
              f"| {np.median(mon):.3f} / {int((mon>0.8).sum())}/{len(mon)}")
    except Exception as e:
        print(f"{name}: {e}")
