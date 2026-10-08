"""Phasage du FRESHET : date du pic de crue printaniere, obs vs meandre vs Hydrotel,
par station et par annee. ZERO run. Dit si meandre est en AVANCE ou en RETARD.
  python .runs/slso/diag_freshet_timing.py
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd, duckdb, xarray as xr

DB=".runs/slso/data/slso.duckdb"; PARQUET=".runs/slso/results/reach-physitel-hydrotel-overnight.parquet"
HYDRO="Z:/Atlas_hydro/SRH/DEBITS_SIM/A20_HYDREP_QCMERI_XXX_DEBITJ_HIS_XXX_XXX_XXX_XXX_XXX_XXX_HYD_LN24HA_GCQ_HC_04032025.nc"
T0,T1="2022-01-01","2024-12-31"; YEARS=[2022,2023,2024]

c=duckdb.connect(DB,read_only=True)
st=c.execute("SELECT station_id,node_idx,lon,lat FROM stations").fetchdf()
obs=c.execute(f"SELECT station_id,date,discharge FROM observations WHERE date>='{T0}' AND date<='{T1}'").fetchdf()
c.close()
obs["date"]=pd.to_datetime(obs["date"]).dt.normalize()
sim=duckdb.sql(f"SELECT date,reach_id,Q_sim_m3s FROM '{PARQUET}' WHERE date>='{T0}' AND date<='{T1}'").df()
sim["date"]=pd.to_datetime(sim["date"]).dt.normalize()
ds=xr.open_dataset(HYDRO); sidh=ds["station_id"].values.astype(str); idx=np.where(np.char.startswith(sidh,"SLSO"))[0]
lonh,lath=ds["lon"].values[idx],ds["lat"].values[idx]
dis=ds["Dis"].isel(station=idx).sel(time=slice(T0,T1)); th=pd.to_datetime(dis["time"].values).normalize(); disv=dis.values; ds.close()

def peak_doy(dates, q, year):
    win=(dates>=pd.Timestamp(year,2,15))&(dates<=pd.Timestamp(year,6,30))
    if win.sum()<30 or not np.isfinite(q[win]).any(): return np.nan
    qq=q[win].copy(); dd=dates[win]
    i=int(np.nanargmax(qq))
    return dd.iloc[i].dayofyear if hasattr(dd,'iloc') else dd[i].dayofyear

offM,offH=[],[]
for _,s in st.iterrows():
    o=obs[obs["station_id"]==s.station_id][["date","discharge"]]
    if len(o)<60: continue
    jh=int(np.argmin((lonh-s.lon)**2+(lath-s.lat)**2))
    qh=pd.DataFrame({"date":th,"qh":disv[jh]})
    qm=sim[sim["reach_id"]==int(s.node_idx)+1][["date","Q_sim_m3s"]].rename(columns={"Q_sim_m3s":"qm"})
    m=o.merge(qh,on="date").merge(qm,on="date").sort_values("date").reset_index(drop=True)
    if len(m)<60: continue
    for y in YEARS:
        po=peak_doy(m["date"],m["discharge"].to_numpy(),y)
        pm=peak_doy(m["date"],m["qm"].to_numpy(),y)
        ph=peak_doy(m["date"],m["qh"].to_numpy(),y)
        if np.isfinite(po) and np.isfinite(pm): offM.append(pm-po)
        if np.isfinite(po) and np.isfinite(ph): offH.append(ph-po)

offM=np.array(offM); offH=np.array(offH)
print(f"\n=== Décalage du PIC de freshet (jours, + = en retard sur obs), {len(offM)} station-annees ===")
print(f"          meandre   Hydrotel")
print(f"median    {np.median(offM):+7.1f}  {np.median(offH):+7.1f}")
print(f"moyenne   {np.mean(offM):+7.1f}  {np.mean(offH):+7.1f}")
print(f"|decalage|>7j : meandre {np.mean(np.abs(offM)>7)*100:.0f}%   Hydrotel {np.mean(np.abs(offH)>7)*100:.0f}%")
print(f"\ndistribution meandre (jours) : {np.percentile(offM,[10,25,50,75,90]).round(1)}")
print(f"distribution Hydrotel (jours): {np.percentile(offH,[10,25,50,75,90]).round(1)}")
