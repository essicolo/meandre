"""Analyse de LAG : l'écart de r meandre vs Hydrotel est-il un RETARD (célérité/
routage) ou une décorrélation/lissage (diffusion/forçage) ? ZERO run.
Pour chaque station : lag optimal (corr croisée sur ±15 j) de meandre/obs et
Hydrotel/obs, r a lag 0 vs r au meilleur lag, et ratio d'amplitude (std_sim/std_obs).
  - meandre lag >> 0 et r remonte fort au best-lag -> RETARD = célérité/routage.
  - lag ~0 mais r bas -> pas un retard : lissage/diffusion ou forçage.

  python .runs/slso/diag_lag.py
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd
import duckdb
import xarray as xr

DB = ".runs/slso/data/slso.duckdb"
PARQUET = ".runs/slso/results/reach-physitel-hydrotel-overnight.parquet"
HYDRO = "Z:/Atlas_hydro/SRH/DEBITS_SIM/A20_HYDREP_QCMERI_XXX_DEBITJ_HIS_XXX_XXX_XXX_XXX_XXX_XXX_HYD_LN24HA_GCQ_HC_04032025.nc"
T0, T1 = "2022-01-01", "2024-12-31"
LAGS = range(-3, 16)   # j ; positif = sim en retard sur obs (on avance sim)

def best_lag(qsim, qobs):
    """r a lag 0, meilleur lag (sim decale de +k = sim[t] compare a obs[t+k]) et r."""
    n = len(qobs)
    r0 = np.corrcoef(qsim, qobs)[0, 1]
    best = (-2, 0)
    for k in LAGS:
        if k >= 0:
            a, b = qsim[:n-k] if k else qsim, qobs[k:] if k else qobs
        else:
            a, b = qsim[-k:], qobs[:n+k]
        if len(a) < 30 or a.std() < 1e-9 or b.std() < 1e-9:
            continue
        r = np.corrcoef(a, b)[0, 1]
        if r > best[0]:
            best = (r, k)
    return r0, best[1], best[0]

c = duckdb.connect(DB, read_only=True)
st = c.execute("SELECT station_id, node_idx, lon, lat, drainage_area_km2 FROM stations").fetchdf()
obs = c.execute(f"SELECT station_id, date, discharge FROM observations WHERE date>='{T0}' AND date<='{T1}'").fetchdf()
c.close()
obs["date"] = pd.to_datetime(obs["date"]).dt.normalize()
sim = duckdb.sql(f"SELECT date, reach_id, Q_sim_m3s FROM '{PARQUET}' WHERE date>='{T0}' AND date<='{T1}'").df()
sim["date"] = pd.to_datetime(sim["date"]).dt.normalize()

ds = xr.open_dataset(HYDRO)
sidh = ds["station_id"].values.astype(str); idx = np.where(np.char.startswith(sidh, "SLSO"))[0]
lonh, lath, areah = ds["lon"].values[idx], ds["lat"].values[idx], ds["drainage_area"].values[idx]
dis = ds["Dis"].isel(station=idx).sel(time=slice(T0, T1)); th = pd.to_datetime(dis["time"].values).normalize()
disv = dis.values; ds.close()

rows = []
for _, s in st.iterrows():
    sid, nidx = s["station_id"], int(s["node_idx"])
    o = obs[obs["station_id"] == sid][["date", "discharge"]]
    if len(o) < 30: continue
    j = int(np.argmin((lonh-s.lon)**2 + (lath-s.lat)**2 + 0*areah))
    qh = pd.DataFrame({"date": th, "qh": disv[j]})
    qm = sim[sim["reach_id"] == nidx+1][["date", "Q_sim_m3s"]].rename(columns={"Q_sim_m3s": "qm"})
    m = o.merge(qh, on="date").merge(qm, on="date")
    if len(m) < 60: continue
    qo = m["discharge"].to_numpy(); qmm = m["qm"].to_numpy(); qhh = m["qh"].to_numpy()
    r0m, lagm, rbm = best_lag(qmm, qo)
    r0h, lagh, rbh = best_lag(qhh, qo)
    amp_m = qmm.std()/qo.std(); amp_h = qhh.std()/qo.std()
    rows.append((sid, r0m, lagm, rbm, r0h, lagh, rbh, amp_m, amp_h))

df = pd.DataFrame(rows, columns=["station","r0_M","lag_M","rbest_M","r0_H","lag_H","rbest_H","amp_M","amp_H"])
pd.set_option("display.float_format", lambda x: f"{x:6.2f}"); pd.set_option("display.width", 200)
print(f"\n=== LAG (j, +=sim en retard) & r a lag0 vs best-lag, test {T0}..{T1}, {len(df)} stations ===")
print(df.sort_values("lag_M", ascending=False).to_string(index=False))
print(f"\n                       meandre   Hydrotel")
print(f"lag median (j)        {df.lag_M.median():7.1f}  {df.lag_H.median():7.1f}")
print(f"r median lag0         {df.r0_M.median():7.3f}  {df.r0_H.median():7.3f}")
print(f"r median best-lag     {df.rbest_M.median():7.3f}  {df.rbest_H.median():7.3f}")
print(f"gain r du lag         {(df.rbest_M-df.r0_M).median():7.3f}  {(df.rbest_H-df.r0_H).median():7.3f}")
print(f"amplitude std/obs med {df.amp_M.median():7.3f}  {df.amp_H.median():7.3f}")
print(f"\nstations meandre lag>=2j : {(df.lag_M>=2).sum()}/{len(df)}")
print("Lecture: lag_M grand + gain r fort -> RETARD (celerite/routage). lag~0 + r bas -> lissage/forcage.")
