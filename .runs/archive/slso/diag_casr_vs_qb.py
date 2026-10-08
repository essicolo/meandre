"""Diagnostic du déficit r CaSR vs quebec.zarr : uniforme (qualité forçage) ou
concentré (ancrage station) ? Et le surplus d'eau CaSR est-il nival (DJF+MAM,
signature sous-captation neige des jauges) ? Compare eval_qb.npz / eval_casr2.npz
par station vs obs, + bilan P saisonnier des deux forçages.
  python .runs/slso/diag_casr_vs_qb.py
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, xarray as xr, duckdb

DB = ".runs/slso/data/slso.duckdb"
qb = np.load(".runs/slso/results/eval_qb.npz", allow_pickle=True)
cs = np.load(".runs/slso/results/eval_casr2.npz", allow_pickle=True)
dates = pd.to_datetime(qb["dates"].astype(str)).normalize()
Qqb, Qcs = qb["Q"], cs["Q"]
T0, T1 = "2022-01-01", "2024-12-31"   # test held-out seul

c = duckdb.connect(DB, read_only=True)
st = c.execute("SELECT station_id,node_idx,lon,lat FROM stations").fetchdf()
obs = c.execute(f"SELECT station_id,date,discharge FROM observations WHERE date>='{T0}' AND date<='{T1}'").fetchdf()
c.close()
obs["date"] = pd.to_datetime(obs["date"]).dt.normalize()

def r_of(sim_col, o):
    qm = pd.DataFrame({"date": dates, "qm": sim_col})
    mo = o.merge(qm, on="date")
    if len(mo) < 60: return None
    s, ob = mo["qm"].to_numpy(), mo["discharge"].to_numpy()
    m = np.isfinite(s) & np.isfinite(ob) & (ob >= 0)
    if m.sum() < 60: return None
    return float(np.corrcoef(s[m], ob[m])[0, 1]), float(s[m].mean()/ob[m].mean())

rows = []
for _, s in st.iterrows():
    o = obs[obs["station_id"] == s.station_id][["date", "discharge"]]
    if len(o) < 60: continue
    a = r_of(Qqb[:, int(s.node_idx)], o); b = r_of(Qcs[:, int(s.node_idx)], o)
    if a and b:
        rows.append((s.station_id, a[0], b[0], b[0]-a[0], a[1], b[1]))
df = pd.DataFrame(rows, columns=["station", "r_qb", "r_casr", "dr", "beta_qb", "beta_casr"])
print(f"\n{len(df)} stations comparables (test 2022-2024)")
print(f"r médian   : qb {df.r_qb.median():.3f}  casr {df.r_casr.median():.3f}  Δ {df.dr.median():+.3f}")
print(f"Δr signe   : casr meilleur {int((df.dr>0).sum())}/{len(df)}, pire {int((df.dr<0).sum())}/{len(df)}")
print(f"Δr quartiles (0.1/0.25/0.5/0.75/0.9) : {[round(x,3) for x in df.dr.quantile([0.1,0.25,0.5,0.75,0.9])]}")
print(f"beta médian : qb {df.beta_qb.median():.3f}  casr {df.beta_casr.median():.3f}")
print("\n5 stations où CaSR PERD le plus en r :")
print(df.sort_values("dr").head(5).round(3).to_string(index=False))
print("\n5 stations où CaSR GAGNE le plus en r :")
print(df.sort_values("dr").tail(5).round(3).to_string(index=False))

def seasonal_P(forc):
    ds = xr.open_dataset(forc); t = pd.to_datetime(ds["time"].values)
    sl = (t >= pd.Timestamp(T0)) & (t <= pd.Timestamp(T1))
    P = ds["forcing"].values[sl][..., 0].mean(axis=1); tt = t[sl]; ds.close()
    s = pd.Series(P, index=tt)
    by = lambda mm: s[s.index.month.isin(mm)].mean()*365
    return s.mean()*365, by([12,1,2]), by([3,4,5]), by([6,7,8]), by([9,10,11])
pq = seasonal_P(".runs/slso/data/forcing.nc")
pc = seasonal_P(".runs/slso/data/forcing-casr.nc")
print("\nBilan P (mm/an) | annuel  DJF  MAM  JJA  SON")
print(f"  quebec.zarr : {pq[0]:6.0f} {pq[1]:5.0f} {pq[2]:5.0f} {pq[3]:5.0f} {pq[4]:5.0f}")
print(f"  CaSR        : {pc[0]:6.0f} {pc[1]:5.0f} {pc[2]:5.0f} {pc[3]:5.0f} {pc[4]:5.0f}")
print(f"  CaSR - qb   : {pc[0]-pq[0]:6.0f} {pc[1]-pq[1]:5.0f} {pc[2]-pq[2]:5.0f} {pc[3]-pq[3]:5.0f} {pc[4]-pq[4]:5.0f}")
print(f"  surplus hiver+printemps (DJF+MAM) = {(pc[1]-pq[1])+(pc[2]-pq[2]):.0f} mm/an sur {pc[0]-pq[0]:.0f} total")
