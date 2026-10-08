"""QA dates : la jointure pluie/débit est-elle correcte ? Test MODÈLE-LIBRE :
corrélation décalée entre pluie (forçage au nœud) et débit observé. Physiquement le
débit SUIT la pluie de +1 à +3 j. Si l'optimum est 0 ou négatif, les dates sont mal
alignées (bug jointure ou fuseau ou off-by-one dans les débits assemblés).
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, duckdb, xarray as xr

T0, T1 = "2015-01-01", "2021-12-31"
# 1) conventions brutes
c = duckdb.connect(".runs/slso/data/slso.duckdb", read_only=True)
st = c.execute("SELECT station_id, node_idx FROM stations").fetchdf()
obs = c.execute(f"SELECT station_id,date,discharge FROM observations WHERE date>='{T0}' AND date<='{T1}'").fetchdf()
smpl = c.execute("SELECT date FROM observations ORDER BY date LIMIT 3").fetchdf()
c.close()
print("obs.date brut (3 premiers) :", smpl.date.tolist(), "| dtype", obs.date.dtype)
obs["date"] = pd.to_datetime(obs["date"]).dt.normalize()

ds = xr.open_dataset(".runs/slso/data/forcing-casr-riox-intens.nc")
ft = pd.to_datetime(ds["time"].values)
print("forçage time brut (3 premiers) :", [str(x) for x in ds["time"].values[:3]])
sl = (ft.normalize() >= pd.Timestamp(T0)) & (ft.normalize() <= pd.Timestamp(T1))
P = ds["forcing"].values[sl][..., 0]; ftn = ft[sl].normalize(); ds.close()

# 2) lag pluie->débit par station, uniquement l'été (réponse rapide, pas la fonte)
best = []
for _, s in st.iterrows():
    o = obs[obs.station_id == s.station_id][["date", "discharge"]]
    if len(o) < 300: continue
    pf = pd.DataFrame({"date": ftn, "p": P[:, int(s.node_idx)]})
    m = o.merge(pf, on="date").sort_values("date")
    m = m[m.date.dt.month.isin([6, 7, 8, 9])]   # été : réponse d'orage rapide
    if len(m) < 200: continue
    q = m.discharge.to_numpy(); p = m.p.to_numpy()
    rs = {}
    for k in range(-3, 6):   # débit décalé de k vs pluie : k>0 = débit APRÈS pluie
        if k >= 0: a, b = q[k:], p[:len(p)-k] if k > 0 else p
        else: a, b = q[:k], p[-k:]
        mm = np.isfinite(a) & np.isfinite(b)
        rs[k] = np.corrcoef(a[mm], b[mm])[0, 1] if mm.sum() > 100 else np.nan
    kbest = max(rs, key=lambda k: (rs[k] if np.isfinite(rs[k]) else -9))
    best.append(kbest)
best = np.array(best)
print(f"\nlag optimal PLUIE->DÉBIT (été), {len(best)} stations :")
print(f"  médian {np.median(best):+.0f} j | distribution (lags -3..+5) {np.bincount(best+3, minlength=9)}")
print(f"  {'>>> PHYSIQUE (débit suit la pluie de +1..+3j) : dates OK' if 1<=np.median(best)<=3 else '>>> ANORMAL : lag 0 ou négatif = DÉCALAGE DE DATES probable'}")
