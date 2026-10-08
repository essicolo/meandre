"""Ré-snapping des jauges au réseau OD HydroSHEDS.
Le snapping actuel accroche les jauges à des tronçons dont l'aire accumulée
sur-estime l'aire officielle de +26% (→ sur-production beta 1.3). Fix : pour
chaque jauge, choisir parmi les tronçons PROCHES celui dont l'aire accumulée
matche le mieux l'aire officielle. Écrit basin-resnap.duckdb (stations.node_idx
corrigé). CPU seulement.
"""
import os, sys, shutil
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np, duckdb
from collections import defaultdict, deque
from scipy.spatial import cKDTree
from meandre.data.basin_cache import BasinCache

SRC = ".runs/slso-od/data/basin.duckdb"
DST = ".runs/slso-od/data/basin-resnap.duckdb"
RADIUS_KM = 12.0     # rayon de recherche de tronçons candidats
MAX_LOG_RATIO = 0.5  # rejeter si aucun candidat dans e^±0.5 (~0.6-1.65×) de l'aire off

h = BasinCache(SRC).load(device="cpu")
nc = h["node_coords"].numpy(); terr = h["territorial"]
al = None
for nm in ["area_km2_local", "area_km2_physical", "drainage_area_km2"]:
    if hasattr(terr, nm): al = getattr(terr, nm).numpy(); break
n = len(nc)
ei = h["graph"].edge_index.numpy()
up = defaultdict(list); indeg = defaultdict(int)
for s, d in zip(ei[0], ei[1]):
    up[int(d)].append(int(s)); indeg[int(s)] += 0
# aire accumulée par passe topologique (feuilles -> exutoire)
# ordre : traiter un nœud quand tous ses amont sont faits. On fait un simple
# parcours mémoïsé (le réseau est un arbre, pas de cycle).
acc = np.full(n, np.nan)
def accum(node):
    stack = [node]; order = []
    seen = set()
    while stack:
        x = stack.pop()
        if x in seen: continue
        seen.add(x); order.append(x)
        for u in up[x]:
            if u not in seen: stack.append(u)
    for x in reversed(order):
        acc[x] = al[x] + sum(acc[u] for u in up[x])
    return acc[node]
for node in range(n):
    if np.isnan(acc[node]): accum(node)

cosl = np.cos(np.radians(nc[:, 1].mean()))
tree = cKDTree(np.c_[nc[:, 0] * cosl, nc[:, 1]])

c = duckdb.connect(SRC, read_only=True)
st = c.execute("SELECT station_id, node_idx, lon, lat, drainage_area_km2 FROM stations").fetchdf()
c.close()

remap = {}; before = []; after = []; moved = 0
for _, r in st.iterrows():
    if not (r.drainage_area_km2 and r.drainage_area_km2 > 0):
        continue
    idx = tree.query_ball_point([r.lon * cosl, r.lat], RADIUS_KM / 111.0)
    if not idx:
        continue
    idx = np.array(idx)
    lr = np.abs(np.log(np.clip(acc[idx], 1e-6, None) / r.drainage_area_km2))
    j = idx[np.argmin(lr)]
    if lr.min() > MAX_LOG_RATIO:   # aucun bon candidat : on garde l'actuel
        continue
    before.append(acc[int(r.node_idx)] / r.drainage_area_km2)
    after.append(acc[j] / r.drainage_area_km2)
    if j != int(r.node_idx): moved += 1
    remap[r.station_id] = int(j)

before = np.array(before); after = np.array(after)
print(f"{len(remap)} stations ré-snappées | {moved} déplacées")
print(f"ratio aire méd : AVANT {np.median(before):.2f} -> APRÈS {np.median(after):.2f}")
print(f"mismatch>1.4|<0.7 : AVANT {((before>1.4)|(before<0.7)).sum()} -> APRÈS {((after>1.4)|(after<0.7)).sum()}")

# écrire la base corrigée
shutil.copy(SRC, DST)
con = duckdb.connect(DST)
for sid, nidx in remap.items():
    con.execute("UPDATE stations SET node_idx = ? WHERE station_id = ?", [nidx, sid])
con.close()
print(f"[ok] écrit {DST}")
