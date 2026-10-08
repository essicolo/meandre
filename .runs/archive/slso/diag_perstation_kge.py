"""KGE par station sur le TEST held-out (2022-2024), depuis la sortie deja sauvee
du run de nuit (reach parquet) vs obs (slso.duckdb). ZERO run.
Dit si la mediane 0.48 = quelques stations catastrophiques ou mediocrite uniforme,
et decompose r/beta/gamma par station.

  python .runs/slso/diag_perstation_kge.py
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd
import duckdb

DB = ".runs/slso/data/slso.duckdb"
PARQUET = ".runs/slso/results/reach-physitel-hydrotel-overnight.parquet"
T0, T1 = "2022-01-01", "2024-12-31"

c = duckdb.connect(DB, read_only=True)
stations = c.execute("SELECT station_id, node_idx, drainage_area_km2 FROM stations").fetchdf()
obs = c.execute(f"SELECT station_id, date, discharge FROM observations WHERE date>='{T0}' AND date<='{T1}'").fetchdf()
c.close()

# mapping reach_id : on teste si reach_id == node_idx
rids = duckdb.sql(f"SELECT DISTINCT reach_id FROM '{PARQUET}'").df()["reach_id"].to_numpy()
print(f"reach_id : {rids.min()}..{rids.max()} (n={len(rids)})  | node_idx stations : {stations.node_idx.min()}..{stations.node_idx.max()}")

sim = duckdb.sql(f"SELECT date, reach_id, Q_sim_m3s FROM '{PARQUET}' WHERE date>='{T0}' AND date<='{T1}'").df()
sim["date"] = pd.to_datetime(sim["date"]).dt.normalize()
obs["date"] = pd.to_datetime(obs["date"]).dt.normalize()
print(f"obs stations: {obs.station_id.nunique()}  dates sim {sim.date.min()}..{sim.date.max()}  obs {obs.date.min()}..{obs.date.max()}")

def kge(qs, qo):
    m = np.isfinite(qs) & np.isfinite(qo)
    qs, qo = qs[m], qo[m]
    if len(qs) < 30 or qo.std() < 1e-9 or qs.std() < 1e-9:
        return (np.nan,)*4
    r = np.corrcoef(qs, qo)[0, 1]
    beta = qs.mean() / qo.mean()
    gamma = (qs.std()/qs.mean()) / (qo.std()/qo.mean())
    k = 1 - np.sqrt((r-1)**2 + (beta-1)**2 + (gamma-1)**2)
    return k, r, beta, gamma

rows = []
for _, s in stations.iterrows():
    sid, nidx, area = s["station_id"], int(s["node_idx"]), s["drainage_area_km2"]
    o = obs[obs["station_id"] == sid][["date", "discharge"]]
    if len(o) < 30:
        continue
    q = sim[sim["reach_id"] == nidx + 1][["date", "Q_sim_m3s"]]   # reach_id 1-indexe
    if len(q) == 0:
        continue
    mrg = pd.merge(o, q, on="date", how="inner")
    if len(mrg) < 30:
        continue
    k, r, b, g = kge(mrg["Q_sim_m3s"].to_numpy(), mrg["discharge"].to_numpy())
    rows.append((sid, area, len(mrg), k, r, b, g))

df = pd.DataFrame(rows, columns=["station", "area_km2", "n", "kge", "r", "beta", "gamma"]).dropna(subset=["kge"])
df = df.sort_values("kge")
pd.set_option("display.float_format", lambda x: f"{x:7.3f}")
print(f"\n=== KGE par station (test {T0}..{T1}), {len(df)} stations ===")
print(df.to_string(index=False))
print(f"\nmediane KGE {df.kge.median():.3f}  moyenne {df.kge.mean():.3f}")
print(f"KGE<0 : {(df.kge<0).sum()}   0-0.5 : {((df.kge>=0)&(df.kge<0.5)).sum()}   >0.5 : {(df.kge>=0.5).sum()}")
print(f"r mediane {df.r.median():.3f}  beta mediane {df.beta.median():.3f}  gamma mediane {df.gamma.median():.3f}")
print(f"\n5 pires : {list(df.head(5).station)}")
print(f"correlation KGE vs aire (log) : {np.corrcoef(np.log(df.area_km2.clip(1)), df.kge)[0,1]:.2f}")
