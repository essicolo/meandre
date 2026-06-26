"""BISSECTION génération vs routage. ZERO run.
Le déficit de r vient-il de la GÉNÉRATION (la colonne produit un ruissellement
déjà trop lissé) ou du ROUTAGE (Muskingum diffusif qui écrase un bon signal) ?

Reconstruit l'hydrogramme NON routé à chaque exutoire = somme du ruissellement
amont (lateral_mm × aire / 86.4), MÊME jour, sans Muskingum. Compare son r a l'obs
au r du Q routé (sortie du run).
  - r(non routé) >> r(routé)  -> le ROUTAGE dégrade (diffusion). Fix = routage.
  - r(non routé) ~ r(routé) bas -> la GÉNÉRATION est déjà lisse. Fix = colonne.

  python .runs/slso/diag_gen_vs_routing.py
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd
import duckdb
import xarray as xr
import torch
from collections import deque
from meandre.data.basin_cache import BasinCache

DB = ".runs/slso/data/slso.duckdb"
PARQUET = ".runs/slso/results/reach-physitel-hydrotel-overnight.parquet"
FIELDS = ".runs/slso/results/fields-physitel-hydrotel-overnight.nc"
T0, T1 = "2022-01-01", "2024-12-31"

h = BasinCache(DB).load(device="cpu")
g = h["graph"]
ei = g.edge_index.numpy()   # (2, E) : [src, dst]
n = h["n_nodes"]
area = h["territorial"].get_physical("area_km2_local").numpy()   # km2 par noeud

# Sens des aretes : on teste src->dst = aval. Ancetres(g) = noeuds atteignant g.
# adjacence inverse : pour remonter de g vers ses amonts.
pred = [[] for _ in range(n)]   # pred[d] = liste des src amont directs
for s, d in zip(ei[0], ei[1]):
    pred[int(d)].append(int(s))

def upstream(node):
    seen = {node}; q = deque([node])
    while q:
        u = q.popleft()
        for p in pred[u]:
            if p not in seen:
                seen.add(p); q.append(p)
    return np.array(sorted(seen))

# lateral_mm (mm/j) par noeud sur la fenetre
ds = xr.open_dataset(FIELDS)
lat = ds["lateral_mm"].sel(time=slice(T0, T1))
tf = pd.to_datetime(lat["time"].values).normalize()
latv = lat.values   # (time, node)
ds.close()
# Q non route par noeud (m3/s) = lateral_mm * area / 86.4
qspec = latv * area[None, :] / 86.4   # (time, node)

c = duckdb.connect(DB, read_only=True)
st = c.execute("SELECT station_id, node_idx, drainage_area_km2 FROM stations").fetchdf()
obs = c.execute(f"SELECT station_id, date, discharge FROM observations WHERE date>='{T0}' AND date<='{T1}'").fetchdf()
c.close()
obs["date"] = pd.to_datetime(obs["date"]).dt.normalize()
sim = duckdb.sql(f"SELECT date, reach_id, Q_sim_m3s FROM '{PARQUET}' WHERE date>='{T0}' AND date<='{T1}'").df()
sim["date"] = pd.to_datetime(sim["date"]).dt.normalize()

def r_(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 30 or a[m].std() < 1e-9 or b[m].std() < 1e-9:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])

# sanity : l'exutoire de plus grosse aire doit avoir bcp d'amont
amax = int(st.loc[st.drainage_area_km2.idxmax(), "node_idx"])
print(f"sanity amont : noeud max-aire a {len(upstream(amax))} amonts (sur {n})")

rows = []
df_lat = pd.DataFrame(qspec, index=tf)
for _, s in st.iterrows():
    sid, nidx, ar = s["station_id"], int(s["node_idx"]), s["drainage_area_km2"]
    o = obs[obs["station_id"] == sid][["date", "discharge"]]
    if len(o) < 30: continue
    up = upstream(nidx)
    q_unrouted = pd.DataFrame({"date": tf, "qu": qspec[:, up].sum(axis=1)})
    q_routed = sim[sim["reach_id"] == nidx+1][["date", "Q_sim_m3s"]].rename(columns={"Q_sim_m3s": "qr"})
    m = o.merge(q_unrouted, on="date").merge(q_routed, on="date")
    if len(m) < 60: continue
    qo = m["discharge"].to_numpy(); qu = m["qu"].to_numpy(); qr = m["qr"].to_numpy()
    rows.append((sid, ar, len(up), r_(qu, qo), r_(qr, qo),
                 qu.std()/qo.std(), qr.std()/qo.std(), qu.mean()/qo.mean()))

df = pd.DataFrame(rows, columns=["station","area","n_up","r_unrouted","r_routed","amp_unr","amp_rout","beta_unr"]).dropna()
df = df.sort_values("area")
pd.set_option("display.float_format", lambda x: f"{x:6.2f}"); pd.set_option("display.width", 200)
print(f"\n=== GÉNÉRATION (non routé) vs ROUTÉ, r a l'obs, test {T0}..{T1}, {len(df)} stations ===")
print(df.to_string(index=False))
print(f"\nr_unrouted median {df.r_unrouted.median():.3f}   r_routed median {df.r_routed.median():.3f}")
print(f"gain du routage (r_routed - r_unrouted) median : {(df.r_routed-df.r_unrouted).median():+.3f}")
print(f"amplitude non-routé {df.amp_unr.median():.2f}  vs routé {df.amp_rout.median():.2f}  vs obs 1.0")
print(f"\nLecture: r_unrouted >> r_routed -> ROUTAGE diffuse. r_unrouted ~ r_routed bas -> GÉNÉRATION lisse.")
