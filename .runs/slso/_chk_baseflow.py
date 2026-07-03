"""Décortique l'incohérence : bfi sim (0.39) > obs (0.27) MAIS été sous-produit.
Sépare baseflow/quickflow (filtre Lyne-Hollick) sur sim (ksat05) et obs, puis
climatologie mensuelle de CHAQUE composante, sim vs obs. Révèle OÙ est l'excès de
baseflow (hiver ?) et le déficit d'été. Test 2022-2024, stations jaugées.
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, duckdb

T0, T1 = "2022-01-01", "2024-12-31"

def lyne_hollick(q, alpha=0.925, passes=3):
    """Sépare le quickflow (filtre passe-haut récursif). Retourne baseflow."""
    q = np.asarray(q, float); b = q.copy()
    for p in range(passes):
        f = np.zeros_like(b); fwd = (p % 2 == 0)
        idx = range(1, len(b)) if fwd else range(len(b) - 2, -1, -1)
        prev = 0.0
        for i in idx:
            j = i - 1 if fwd else i + 1
            f[i] = alpha * prev + (1 + alpha) / 2 * (b[i] - b[j])
            prev = f[i]
        quick = np.clip(f, 0, None)
        b = np.clip(b - quick, 0, b)
    return b

c = duckdb.connect(".runs/slso/data/slso.duckdb", read_only=True)
st = c.execute("SELECT station_id, node_idx FROM stations").fetchdf()
obs = c.execute(f"SELECT station_id,date,discharge FROM observations WHERE date>='{T0}' AND date<='{T1}'").fetchdf()
c.close()
obs["date"] = pd.to_datetime(obs["date"]).dt.normalize()
d = np.load(".runs/slso/results/eval_hortonian.npz", allow_pickle=True)
Q = d["Q"]; dd = pd.to_datetime(d["dates"].astype(str)).normalize()

# accumulateurs : baseflow & quickflow par mois, sim & obs
bf_s = {m: [] for m in range(1, 13)}; bf_o = {m: [] for m in range(1, 13)}
qf_s = {m: [] for m in range(1, 13)}; qf_o = {m: [] for m in range(1, 13)}
bfi_s, bfi_o = [], []
for _, s in st.iterrows():
    o = obs[obs.station_id == s.station_id][["date", "discharge"]]
    if len(o) < 200: continue
    mo = o.merge(pd.DataFrame({"date": dd, "qm": Q[:, int(s.node_idx)]}), on="date").sort_values("date")
    if len(mo) < 200: continue
    qo = mo.discharge.to_numpy(); qs = mo.qm.to_numpy(); mon = mo.date.dt.month.to_numpy()
    bo = lyne_hollick(qo); bs = lyne_hollick(qs)
    bfi_o.append(bo.sum() / qo.sum()); bfi_s.append(bs.sum() / qs.sum())
    for m in range(1, 13):
        k = mon == m
        if k.sum() > 5:
            bf_o[m].append(bo[k].mean()); bf_s[m].append(bs[k].mean())
            qf_o[m].append((qo - bo)[k].mean()); qf_s[m].append((qs - bs)[k].mean())

print(f"BFI médian : sim {np.median(bfi_s):.3f}  obs {np.median(bfi_o):.3f}\n")
mn = ["", "jan", "fév", "mar", "avr", "mai", "jun", "jul", "aoû", "sep", "oct", "nov", "déc"]
print("mois | BASEFLOW sim/obs | QUICKFLOW sim/obs   (1=parfait)")
for m in range(1, 13):
    rb = np.median(bf_s[m]) / max(np.median(bf_o[m]), 1e-9)
    rq = np.median(qf_s[m]) / max(np.median(qf_o[m]), 1e-9)
    fb = "  <<" if (rb > 1.3 or rb < 0.77) else ""
    print(f" {mn[m]} |   bf {rb:.2f}        |  qf {rq:.2f}{fb}")
print("\nLecture : baseflow sim/obs >1 en hiver = excès de baseflow hivernal ;")
print("baseflow sim/obs <1 en été = déficit de soutien d'étiage. Si les deux,")
print("le baseflow est MAL TIMÉ (hiver au lieu d'été) -> incohérence résolue.")
