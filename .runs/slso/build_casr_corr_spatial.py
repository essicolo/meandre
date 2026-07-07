"""Calage volume SPATIAL du forçage CaSR corrigé : bilan d'eau PAR SOUS-BASSIN jaugé.
Le calage global (1147 mm/an partout) ignore la variabilité spatiale du bilan : la
décomposition per-station montre une dispersion beta 0.69..1.69 sur le held-out.
Ici, pour chaque station, cible locale = lame observée (mm/an, PÉRIODE TRAIN seulement,
aucune donnée held-out) + ETR 450. Chaque nœud reçoit la cible de son PLUS PETIT
sous-bassin jaugé englobant (emboîtement respecté) ; hors couverture : global 1147.
Facteur borné [0.75, 1.30] pour ne pas absorber régulation/prélèvements dans P.
Entrée : forcing-casr-corr.nc (déjà dé-crachiné + jour-local). Sortie : forcing-casr-corr2.nc.
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, xarray as xr, duckdb, torch
from meandre.data.basin_cache import BasinCache

DB = ".runs/slso/data/slso.duckdb"
SRC = os.environ.get("SRC", "D:/meandre-data/slso/forcing-casr-corr.nc")
OUT = ".runs/slso/data/forcing-casr-corr2.nc"
ET_MM = 450.0; GLOB = 1147.0; TRAIN_END = "2021-12-31"
BOUNDS = (0.75, 1.30)

h = BasinCache(DB).load(device="cpu")
g = h["graph"]; n_nodes = h["n_nodes"]
ei = g.edge_index.numpy()  # [2, E], src -> dst (amont -> aval)
c = duckdb.connect(DB, read_only=True)
stations = c.execute("SELECT station_id, node_idx, drainage_area_km2 FROM stations").fetchdf()
obs = c.execute(f"SELECT station_id, AVG(discharge) q FROM observations WHERE date <= '{TRAIN_END}' GROUP BY station_id").fetchdf()
c.close()
st = stations.merge(obs, on="station_id").dropna(subset=["q", "drainage_area_km2"])
st["lame_mm"] = st.q * 31_557_600.0 / (st.drainage_area_km2 * 1e6) * 1000.0
st["target"] = st.lame_mm + ET_MM
print(f"{len(st)} stations avec obs train | lame {st.lame_mm.min():.0f}..{st.lame_mm.max():.0f} mm/an | cible P {st.target.min():.0f}..{st.target.max():.0f}")

# ensemble amont de chaque jauge (BFS inverse sur edge_index)
up_of = {int(d): [] for d in range(n_nodes)}
for s, d in ei.T: up_of[int(d)].append(int(s))
def upstream_set(node):
    seen = {node}; stack = [node]
    while stack:
        for u in up_of[stack.pop()]:
            if u not in seen: seen.add(u); stack.append(u)
    return seen

# plus petit bassin jaugé englobant par nœud
node_target = np.full(n_nodes, GLOB); node_basin_size = np.full(n_nodes, np.inf)
for _, row in st.sort_values("drainage_area_km2", ascending=False).iterrows():
    us = upstream_set(int(row.node_idx)); sz = len(us)
    for n in us:
        if sz < node_basin_size[n]: node_basin_size[n] = sz; node_target[n] = row.target
cov = np.isfinite(node_basin_size).sum() if np.isfinite(node_basin_size).any() else 0
in_gauged = node_basin_size < np.inf
print(f"nœuds couverts par un bassin jaugé : {in_gauged.sum()}/{n_nodes} ({in_gauged.mean()*100:.0f}%)")

ds = xr.open_dataset(SRC); F = ds["forcing"].values.copy(); t = pd.to_datetime(ds["time"].values); VARS = list(ds["var"].values.astype(str)); ds.close()
P = F[:, :, 0]
cur = P.mean(axis=0) * 365.25  # mm/an par nœud
scale = np.clip(node_target / cur, *BOUNDS)
print(f"facteurs : min {scale.min():.3f} | med {np.median(scale):.3f} | max {scale.max():.3f} | aux bornes {((scale<=BOUNDS[0]+1e-9)|(scale>=BOUNDS[1]-1e-9)).mean()*100:.0f}%")
F[:, :, 0] = (P * scale[None, :]).astype(np.float32)
print(f"P final : moyenne domaine {F[:, :, 0].mean()*365.25:.0f} mm/an (avant {cur.mean():.0f})")
if os.path.exists(OUT): os.remove(OUT)
xr.Dataset({"forcing": (("time", "node", "var"), F.astype(np.float32))},
           coords={"time": t, "node": np.arange(n_nodes), "var": VARS}).to_netcdf(OUT)
print(f"[ok] {OUT}")
