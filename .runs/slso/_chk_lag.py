"""Le décalage de l'Hortonien est-il SYSTÉMATIQUE (réparable, routage/shift) ou
ALÉATOIRE (timing convectif irréductible) ? Corrélation décalée sim vs obs : pour
chaque station, r à décalage -3..+3 jours. Si r bondit à un décalage CONSTANT
(~2j), c'est systématique -> corrigeable. Si l'optimum est dispersé et le gain
faible, c'est aléatoire. Test 2022-2024.
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

def analyse(tag):
    d = np.load(f".runs/slso/results/eval_{tag}.npz", allow_pickle=True)
    Q = d["Q"]; dd = pd.to_datetime(d["dates"].astype(str)).normalize()
    best_lags, r0s, rbest = [], [], []
    for _, s in st.iterrows():
        o = obs[obs.station_id == s.station_id][["date", "discharge"]]
        if len(o) < 200: continue
        mo = o.merge(pd.DataFrame({"date": dd, "qm": Q[:, int(s.node_idx)]}), on="date").sort_values("date")
        if len(mo) < 200: continue
        qo = mo.discharge.to_numpy(); qs = mo.qm.to_numpy()
        rs = {}
        for k in range(-3, 4):
            if k >= 0: a, b = qs[k:], qo[:len(qo)-k] if k>0 else qo
            else: a, b = qs[:k], qo[-k:]
            m = np.isfinite(a) & np.isfinite(b)
            rs[k] = np.corrcoef(a[m], b[m])[0,1] if m.sum() > 100 else np.nan
        kbest = max(rs, key=lambda k: (rs[k] if np.isfinite(rs[k]) else -9))
        best_lags.append(kbest); r0s.append(rs[0]); rbest.append(rs[kbest])
    return np.array(best_lags), np.array(r0s), np.array(rbest)

for tag in ["hortonian", "ksat05"]:
    try:
        bl, r0, rb = analyse(tag)
        print(f"=== {tag} ===")
        print(f"  décalage optimal : médian {np.median(bl):+.0f}j  | distribution {np.bincount(bl+3, minlength=7)} (pour lags -3..+3)")
        print(f"  r à lag 0 : médian {np.median(r0):.3f}  | r au lag optimal : médian {np.median(rb):.3f}  | GAIN {np.median(rb)-np.median(r0):+.3f}")
        # systématique si l'optimum est concentré ET le gain notable
        conc = (bl == np.median(bl)).mean()
        print(f"  {conc*100:.0f}% des stations partagent le décalage médian  "
              f"-> {'SYSTÉMATIQUE (corrigeable)' if conc>0.5 and abs(np.median(bl))>=1 else 'dispersé/faible = plutôt aléatoire'}")
    except Exception as e:
        print(f"{tag}: {e}")
