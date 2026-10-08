"""Attribution du déficit KGE CaSR (casr3) vs quebec (qb) : où est l'écart ?
Par station sur le test 2022-2024 : décompose KGE en termes (r, beta, gamma),
r par saison, et localise gamma (CV sur hauts débits >p90 vs reste). Dit quel
levier bouge vraiment le KGE.
  python .runs/slso/diag_kge_attribution.py
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, duckdb

T0, T1 = "2022-01-01", "2024-12-31"
SEAS = {"DJF": [12, 1, 2], "MAM": [3, 4, 5], "JJA": [6, 7, 8], "SON": [9, 10, 11]}
c = duckdb.connect(".runs/slso/data/slso.duckdb", read_only=True)
st = c.execute("SELECT station_id, node_idx FROM stations").fetchdf()
obs = c.execute(f"SELECT station_id, date, discharge FROM observations WHERE date>='{T0}' AND date<='{T1}'").fetchdf()
c.close()
obs["date"] = pd.to_datetime(obs["date"]).dt.normalize()

def load(tag):
    d = np.load(f".runs/slso/results/eval_{tag}.npz", allow_pickle=True)
    return d["Q"], pd.to_datetime(d["dates"].astype(str)).normalize()

def kge_terms(s, o):
    m = np.isfinite(s) & np.isfinite(o) & (o >= 0)
    if m.sum() < 60: return None
    s, o = s[m], o[m]
    r = np.corrcoef(s, o)[0, 1]; beta = s.mean() / o.mean()
    gamma = (s.std() / s.mean()) / (o.std() / o.mean())
    kge = 1 - np.sqrt((r - 1)**2 + (beta - 1)**2 + (gamma - 1)**2)
    return r, beta, gamma, kge

def analyse(tag):
    Q, dd = load(tag)
    rows = []
    for _, s in st.iterrows():
        o = obs[obs.station_id == s.station_id][["date", "discharge"]]
        if len(o) < 60: continue
        qm = pd.DataFrame({"date": dd, "qm": Q[:, int(s.node_idx)]})
        mo = o.merge(qm, on="date")
        if len(mo) < 60: continue
        qo = mo.discharge.to_numpy(); qs = mo.qm.to_numpy(); mon = mo.date.dt.month.to_numpy()
        t = kge_terms(qs, qo)
        if t is None: continue
        rec = {"r": t[0], "beta": t[1], "gamma": t[2], "kge": t[3]}
        for sn, mm in SEAS.items():
            ss = np.isin(mon, mm)
            rec["r_" + sn] = np.corrcoef(qs[ss], qo[ss])[0, 1] if ss.sum() > 20 and np.std(qs[ss]) > 0 else np.nan
        # flashiness : CV sim/obs sur les hauts débits (>p90 obs)
        hi = qo > np.percentile(qo, 90)
        rec["gamma_hi"] = ((qs[hi].std() / qs[hi].mean()) / (qo[hi].std() / qo[hi].mean())) if hi.sum() > 10 and qs[hi].mean() > 0 else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)

A = analyse("qb"); B = analyse("casr3")
def med(df, k): return np.nanmedian(df[k])
print(f"=== médianes par station (test {T0[:4]}-{T1[:4]}) ===")
print(f"{'':10s} {'KGE':>6s} {'r':>6s} {'beta':>6s} {'gamma':>6s} | {'rDJF':>5s} {'rMAM':>5s} {'rJJA':>5s} {'rSON':>5s} | {'g_hi':>5s}")
for tag, df in [("quebec", A), ("CaSR", B)]:
    print(f"{tag:10s} {med(df,'kge'):6.3f} {med(df,'r'):6.3f} {med(df,'beta'):6.3f} {med(df,'gamma'):6.3f} | "
          f"{med(df,'r_DJF'):5.2f} {med(df,'r_MAM'):5.2f} {med(df,'r_JJA'):5.2f} {med(df,'r_SON'):5.2f} | {med(df,'gamma_hi'):5.2f}")

# contribution de chaque terme au déficit KGE (médianes)
rq, bq, gq = med(A,'r'), med(A,'beta'), med(A,'gamma')
rc, bc, gc = med(B,'r'), med(B,'beta'), med(B,'gamma')
print(f"\n=== termes (1-x)^2 du KGE, médianes ===")
print(f"          r-term   beta-term  gamma-term")
print(f"quebec  {(1-rq)**2:8.4f} {(1-bq)**2:9.4f} {(1-gq)**2:10.4f}")
print(f"CaSR    {(1-rc)**2:8.4f} {(1-bc)**2:9.4f} {(1-gc)**2:10.4f}")
print(f"écart   {(1-rc)**2-(1-rq)**2:+8.4f} {(1-bc)**2-(1-bq)**2:+9.4f} {(1-gc)**2-(1-gq)**2:+10.4f}")
print(f"\nLecture : le plus gros écart positif = le terme qui creuse le KGE de CaSR.")
print(f"gamma_hi >1 = pics trop flashy (sur-dispersés). r par saison localise le timing.")
