"""PLAFOND d'un UH de versant. ZERO run.
La génération brute (non routée) a r=0.38 vs obs. Question : un hydrogramme de
versant LINÉAIRE (Nash : cascade de n réservoirs, constante k) appliqué a cette
génération suffit-il a atteindre le r d'Hydrotel (~0.82) ?
  - oui (best_r monte vers 0.8) -> le levier est l'UH de versant (façonner la
    génération avant le canal). Fix clair et borné.
  - non (best_r plafonne bas) -> la PHASE de la génération est fausse, aucun
    lissage linéaire ne sauve : c'est la dynamique de la colonne (infiltration/
    saturation/fonte) qu'il faut revoir.

Balaye une grille (n, k) par station, garde le meilleur r. Compare au routé actuel
et a Hydrotel.
  python .runs/slso/diag_uh_ceiling.py
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd
import duckdb
import xarray as xr
from collections import deque
from math import factorial
from meandre.data.basin_cache import BasinCache

DB = ".runs/slso/data/slso.duckdb"
PARQUET = ".runs/slso/results/reach-physitel-hydrotel-overnight.parquet"
FIELDS = ".runs/slso/results/fields-physitel-hydrotel-overnight.nc"
HYDRO = "Z:/Atlas_hydro/SRH/DEBITS_SIM/A20_HYDREP_QCMERI_XXX_DEBITJ_HIS_XXX_XXX_XXX_XXX_XXX_XXX_HYD_LN24HA_GCQ_HC_04032025.nc"
T0, T1 = "2022-01-01", "2024-12-31"

# grille UH Nash : n reservoirs, k jours
NS = [1, 2, 3]
KS = [0.5, 1, 1.5, 2, 3, 4, 6, 9, 12]
LMAX = 60

def nash_kernel(n, k):
    t = np.arange(0, LMAX)
    h = t**(n-1) * np.exp(-t/k) / (k**n * factorial(n-1))
    return h / h.sum()

def conv(x, h):
    return np.convolve(x, h, mode="full")[:len(x)]

def r_(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 30 or a[m].std() < 1e-9 or b[m].std() < 1e-9: return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])

h = BasinCache(DB).load(device="cpu"); g = h["graph"]; ei = g.edge_index.numpy(); n = h["n_nodes"]
area = h["territorial"].get_physical("area_km2_local").numpy()
pred = [[] for _ in range(n)]
for s, d in zip(ei[0], ei[1]): pred[int(d)].append(int(s))
def upstream(node):
    seen={node}; q=deque([node])
    while q:
        u=q.popleft()
        for p in pred[u]:
            if p not in seen: seen.add(p); q.append(p)
    return np.array(sorted(seen))

ds = xr.open_dataset(FIELDS); lat = ds["lateral_mm"].sel(time=slice(T0, T1))
tf = pd.to_datetime(lat["time"].values).normalize(); latv = lat.values; ds.close()
qspec = latv * area[None, :] / 86.4

ds = xr.open_dataset(HYDRO); sidh = ds["station_id"].values.astype(str)
idx = np.where(np.char.startswith(sidh, "SLSO"))[0]
lonh, lath = ds["lon"].values[idx], ds["lat"].values[idx]
dis = ds["Dis"].isel(station=idx).sel(time=slice(T0, T1)); th = pd.to_datetime(dis["time"].values).normalize()
disv = dis.values; ds.close()

c = duckdb.connect(DB, read_only=True)
st = c.execute("SELECT station_id, node_idx, lon, lat, drainage_area_km2 FROM stations").fetchdf()
obs = c.execute(f"SELECT station_id, date, discharge FROM observations WHERE date>='{T0}' AND date<='{T1}'").fetchdf()
c.close()
obs["date"] = pd.to_datetime(obs["date"]).dt.normalize()
sim = duckdb.sql(f"SELECT date, reach_id, Q_sim_m3s FROM '{PARQUET}' WHERE date>='{T0}' AND date<='{T1}'").df()
sim["date"] = pd.to_datetime(sim["date"]).dt.normalize()

kernels = [(nn, kk, nash_kernel(nn, kk)) for nn in NS for kk in KS]
rows = []
for _, s in st.iterrows():
    sid, nidx = s["station_id"], int(s["node_idx"])
    o = obs[obs["station_id"] == sid][["date", "discharge"]]
    if len(o) < 30: continue
    up = upstream(nidx)
    qu_full = pd.DataFrame({"date": tf, "qu": qspec[:, up].sum(axis=1)})
    qr = sim[sim["reach_id"] == nidx+1][["date", "Q_sim_m3s"]].rename(columns={"Q_sim_m3s": "qr"})
    jh = int(np.argmin((lonh-s.lon)**2 + (lath-s.lat)**2))
    qh = pd.DataFrame({"date": th, "qh": disv[jh]})
    m = o.merge(qu_full, on="date").merge(qr, on="date").merge(qh, on="date")
    if len(m) < 60: continue
    qo = m["discharge"].to_numpy(); qu = m["qu"].to_numpy()
    r_raw = r_(qu, qo)
    best_r, best_nk = -2, None
    for nn, kk, h_ in kernels:
        rr = r_(conv(qu, h_), qo)
        if rr is not None and rr > best_r: best_r, best_nk = rr, (nn, kk)
    rows.append((sid, s.drainage_area_km2, r_raw, r_(m["qr"].to_numpy(), qo), best_r, best_nk[0], best_nk[1], r_(m["qh"].to_numpy(), qo)))

df = pd.DataFrame(rows, columns=["station","area","r_gen","r_routed","r_uhbest","uh_n","uh_k","r_hydrotel"]).dropna()
df = df.sort_values("area")
pd.set_option("display.float_format", lambda x: f"{x:6.2f}"); pd.set_option("display.width", 200)
print(f"\n=== PLAFOND UH versant : r génération brute -> meilleur UH Nash, vs routé actuel vs Hydrotel ===")
print(df.to_string(index=False))
print(f"\nr_gen median      {df.r_gen.median():.3f}")
print(f"r_routed median   {df.r_routed.median():.3f}  (Muskingum actuel)")
print(f"r_uhbest median   {df.r_uhbest.median():.3f}  (meilleur UH Nash sur la génération)")
print(f"r_hydrotel median {df.r_hydrotel.median():.3f}")
print(f"k median du meilleur UH : {df.uh_k.median():.1f} j  (n median {df.uh_n.median():.0f})")
print(f"\nLecture: r_uhbest ~ r_hydrotel -> UH versant SUFFIT (lever clair). "
      f"r_uhbest << r_hydrotel -> phase génération fausse (colonne).")
