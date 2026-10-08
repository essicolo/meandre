"""r par SAISON, meandre vs Hydrotel vs obs. ZERO run.
La génération mal phasée se concentre-t-elle au printemps (freshet/fonte neige) ?
DJF=hiver, MAM=printemps (freshet), JJA=ete, SON=automne.
  python .runs/slso/diag_seasonal_r.py
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd, duckdb, xarray as xr

DB = ".runs/slso/data/slso.duckdb"
PARQUET = ".runs/slso/results/reach-physitel-hydrotel-overnight.parquet"
HYDRO = "Z:/Atlas_hydro/SRH/DEBITS_SIM/A20_HYDREP_QCMERI_XXX_DEBITJ_HIS_XXX_XXX_XXX_XXX_XXX_XXX_HYD_LN24HA_GCQ_HC_04032025.nc"
T0, T1 = "2022-01-01", "2024-12-31"
SEAS = {"DJF":[12,1,2], "MAM":[3,4,5], "JJA":[6,7,8], "SON":[9,10,11]}

def r_(a,b):
    m=np.isfinite(a)&np.isfinite(b)
    if m.sum()<20 or a[m].std()<1e-9 or b[m].std()<1e-9: return np.nan
    return float(np.corrcoef(a[m],b[m])[0,1])

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

rec={s:{"M":[], "H":[]} for s in SEAS}
for _,s in st.iterrows():
    o=obs[obs["station_id"]==s.station_id][["date","discharge"]]
    if len(o)<30: continue
    jh=int(np.argmin((lonh-s.lon)**2+(lath-s.lat)**2))
    qh=pd.DataFrame({"date":th,"qh":disv[jh]})
    qm=sim[sim["reach_id"]==int(s.node_idx)+1][["date","Q_sim_m3s"]].rename(columns={"Q_sim_m3s":"qm"})
    m=o.merge(qh,on="date").merge(qm,on="date")
    if len(m)<60: continue
    mo=m["date"].dt.month
    for sname,months in SEAS.items():
        sel=mo.isin(months).to_numpy()
        if sel.sum()<20: continue
        rec[sname]["M"].append(r_(m["qm"].to_numpy()[sel], m["discharge"].to_numpy()[sel]))
        rec[sname]["H"].append(r_(m["qh"].to_numpy()[sel], m["discharge"].to_numpy()[sel]))

print(f"\n=== r MEDIAN par saison (30 stations), test {T0}..{T1} ===")
print(f"{'saison':8s} {'meandre':>8s} {'Hydrotel':>9s} {'ecart':>7s}")
for sname in ["DJF","MAM","JJA","SON"]:
    M=np.nanmedian(rec[sname]["M"]); H=np.nanmedian(rec[sname]["H"])
    print(f"{sname:8s} {M:8.3f} {H:9.3f} {H-M:7.3f}")
print("\nMAM=printemps/freshet. Gros ecart en MAM -> fonte neige mal phasee.")
