"""Chiffre HONNÊTE du benchmark : KGE poolé (toutes stations concaténées, flatte)
vs médiane/moyenne par station (ce que les reviewers regardent) + distribution.
Pour PHYSITEL+quebec (qb) vs CaSR (ksat05). Test 2022-2024.
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
    if m.sum() < 60: return None
    s, o = s[m], o[m]
    if s.std() == 0 or o.std() == 0: return None
    r = np.corrcoef(s, o)[0, 1]
    return 1 - np.sqrt((r-1)**2 + (s.mean()/o.mean()-1)**2 + ((s.std()/s.mean())/(o.std()/o.mean())-1)**2)

def bench(tag):
    d = np.load(f".runs/slso/results/eval_{tag}.npz", allow_pickle=True)
    Q = d["Q"]; dd = pd.to_datetime(d["dates"].astype(str)).normalize()
    per, S, O = [], [], []
    for _, s in st.iterrows():
        o = obs[obs.station_id == s.station_id][["date", "discharge"]]
        if len(o) < 60: continue
        qm = pd.DataFrame({"date": dd, "qm": Q[:, int(s.node_idx)]})
        mo = o.merge(qm, on="date")
        if len(mo) < 60: continue
        k = kge(mo.qm.to_numpy(), mo.discharge.to_numpy())
        if k is not None:
            per.append(k); S.append(mo.qm.to_numpy()); O.append(mo.discharge.to_numpy())
    per = np.array(per)
    pooled = kge(np.concatenate(S), np.concatenate(O))   # toutes stations concaténées
    print(f"{tag:8s} | poolé {pooled:.3f} | médiane {np.median(per):.3f} | moyenne {per.mean():.3f} | "
          f">0.7 {int((per>0.7).sum())}/{len(per)} | >0.5 {int((per>0.5).sum())}/{len(per)} | min {per.min():.2f}")

print("benchmark test 2022-2024 (40 stations nettoyées)\n")
print(f"{'modele':8s} | {'poole':>7s} | {'mediane':>9s} | moyenne | distribution")
for t in ["qb", "ksat05", "casr3"]:
    try: bench(t)
    except Exception as e: print(f"{t}: {e}")
