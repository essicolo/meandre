"""Vérifie chaque station contre la TOPOLOGIE réelle : aire de drainage RECONSTRUITE
en remontant le réseau (edges src->dst + area_km2_local accumulée sur tout l'amont)
vs aire officielle de la jauge. Plus le test physique Q/P (>1 = impossible).
territorial.drainage_area_km2 s'est révélé être l'aire LOCALE (~1-3 km²), pas
l'accumulée — d'où ce calcul direct depuis le graphe.
  python .runs/slso/diag_stations_topo.py
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, xarray as xr, duckdb
from collections import defaultdict, deque

Y0, Y1 = "2013-01-01", "2021-12-31"
c = duckdb.connect(".runs/slso/data/slso.duckdb", read_only=True)
st = c.execute("SELECT station_id, node_idx, drainage_area_km2 AS area_off FROM stations").fetchdf()
edges = c.execute("SELECT src, dst FROM edges").fetchdf()
area_loc = dict(c.execute("SELECT node_idx, area_km2_local FROM territorial").fetchall())
obs = c.execute(f"""SELECT station_id, AVG(discharge) q, COUNT(*) nobs FROM observations
                    WHERE date>='{Y0}' AND date<='{Y1}' GROUP BY station_id""").fetchdf()
et = c.execute(f"""SELECT node_idx, AVG(etr_mm_day)*365.25 et FROM modis_et
                   WHERE quality_ok AND date>='{Y0}' AND date<='{Y1}' GROUP BY node_idx""").fetchdf()
c.close()
qm = dict(zip(obs.station_id, obs.q)); nm = dict(zip(obs.station_id, obs.nobs))
etm = dict(zip(et.node_idx, et.et))
ds = xr.open_dataset(".runs/slso/data/forcing-casr-riox.nc")
t = pd.to_datetime(ds["time"].values); sl = (t >= pd.Timestamp(Y0)) & (t <= pd.Timestamp(Y1))
Pn = ds["forcing"].values[sl][..., 0].mean(axis=0) * 365.25; ds.close()

# parents[node] = noeuds qui s'écoulent DANS node (src tel que dst==node)
parents = defaultdict(list)
for _, e in edges.iterrows():
    parents[int(e.dst)].append(int(e.src))
def upstream_area(node):
    """Aire accumulée = area_km2_local de tout l'amont (BFS sur les parents)."""
    seen = {node}; dq = deque([node]); tot = 0.0
    while dq:
        u = dq.popleft(); tot += area_loc.get(u, 0.0)
        for p in parents[u]:
            if p not in seen:
                seen.add(p); dq.append(p)
    return tot, len(seen)

rows = []
for _, s in st.iterrows():
    sid, ni = s.station_id, int(s.node_idx); ao = s.area_off
    if sid not in qm: continue
    area_acc, n_up = upstream_area(ni)
    P = float(Pn[ni]); ET = etm.get(ni, np.nan)
    Q = qm[sid] * 31557.6 / ao if ao and ao > 0 else np.nan   # lame avec aire OFFICIELLE
    r = (area_acc / ao) if (ao and ao > 0) else np.nan          # accumulée/officielle
    QP = Q / P if np.isfinite(Q) else np.nan
    flags = []
    if np.isfinite(QP) and QP > 1.0: flags.append("Q>P")
    if np.isfinite(r) and (r > 2.0 or r < 0.5): flags.append("AIRE")
    if nm[sid] < 365: flags.append("PEU_OBS")
    rows.append((sid, ao, round(area_acc), n_up, round(r, 2), round(P), round(ET), round(Q), round(QP, 2), nm[sid], ";".join(flags)))
df = pd.DataFrame(rows, columns=["station","area_off","area_acc","n_up","acc/off","P","ET","Q","Q/P","nobs","flags"])
df = df.sort_values("acc/off")
pd.set_option("display.width", 220)
print(f"{len(df)} stations | aire reconstruite vs officielle\n")
print(df.to_string(index=False))
broken = df[df['flags'].str.contains("Q>P") | df['flags'].str.contains("AIRE")]
print(f"\nCASSÉES (Q>P ou aire off par >2x) : {len(broken)} -> {', '.join(broken.station.tolist()) if len(broken) else 'aucune'}")
print(f"acc/off : médian {df['acc/off'].median():.2f}  (1.0 = topologie reproduit l'aire officielle)")
broken.to_csv(".runs/slso/results/broken_stations.csv", index=False)
