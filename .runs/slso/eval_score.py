"""Score une sortie eval_<interv>.npz en tete-a-tete vs Hydrotel + obs.
KGE/r/beta/gamma median (test 2022-2024), r par saison, decalage du pic de freshet.
  python .runs/slso/eval_score.py <interv> [interv2 ...]
Tourne sur Windows (acces Z: Hydrotel + slso.duckdb).
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd, duckdb, xarray as xr

DB = ".runs/slso/data/slso.duckdb"
HYDRO = "Z:/Atlas_hydro/SRH/DEBITS_SIM/A20_HYDREP_QCMERI_XXX_DEBITJ_HIS_XXX_XXX_XXX_XXX_XXX_XXX_HYD_LN24HA_GCQ_HC_04032025.nc"
T0, T1 = "2022-01-01", "2024-12-31"
SEAS = {"DJF":[12,1,2], "MAM":[3,4,5], "JJA":[6,7,8], "SON":[9,10,11]}

def kge(qs, qo):
    m=np.isfinite(qs)&np.isfinite(qo); qs,qo=qs[m],qo[m]
    if len(qs)<30 or qo.std()<1e-9 or qs.std()<1e-9: return (np.nan,)*4
    r=np.corrcoef(qs,qo)[0,1]; b=qs.mean()/qo.mean(); g=(qs.std()/qs.mean())/(qo.std()/qo.mean())
    return 1-np.sqrt((r-1)**2+(b-1)**2+(g-1)**2), r, b, g

def peak_doy(dates, q, year):
    win=(dates>=pd.Timestamp(year,2,15))&(dates<=pd.Timestamp(year,6,30))
    if win.sum()<30 or not np.isfinite(q[win.to_numpy()]).any(): return np.nan
    return dates[win].iloc[int(np.nanargmax(q[win.to_numpy()]))].dayofyear

c=duckdb.connect(DB,read_only=True)
st=c.execute("SELECT station_id,node_idx,lon,lat FROM stations").fetchdf()
obs=c.execute(f"SELECT station_id,date,discharge FROM observations WHERE date>='{T0}' AND date<='{T1}'").fetchdf()
c.close()
obs["date"]=pd.to_datetime(obs["date"]).dt.normalize()
ds=xr.open_dataset(HYDRO); sidh=ds["station_id"].values.astype(str); idx=np.where(np.char.startswith(sidh,"SLSO"))[0]
lonh,lath=ds["lon"].values[idx],ds["lat"].values[idx]
dis=ds["Dis"].isel(station=idx).sel(time=slice(T0,T1)); th=pd.to_datetime(dis["time"].values).normalize(); disv=dis.values; ds.close()

print(f"{'interv':10s} {'KGE':>6s} {'r':>6s} {'beta':>6s} {'gamma':>6s} | {'rDJF':>5s} {'rMAM':>5s} {'rJJA':>5s} {'rSON':>5s} | {'freshet_dj':>10s}")
def score(interv, Q, dates):
    dd=pd.to_datetime(dates).normalize()
    sel=(dd>=pd.Timestamp(T0))&(dd<=pd.Timestamp(T1))
    dd=dd[sel]; Q=Q[sel]
    KG,R,B,G,off=[],[],[],[],[]
    seas={s:[] for s in SEAS}
    for _,s in st.iterrows():
        o=obs[obs["station_id"]==s.station_id][["date","discharge"]]
        if len(o)<30: continue
        qm=pd.DataFrame({"date":dd,"qm":Q[:,int(s.node_idx)]})
        mo=o.merge(qm,on="date")
        if len(mo)<60: continue
        qo=mo["discharge"].to_numpy(); qq=mo["qm"].to_numpy()
        k,r,b,g=kge(qq,qo)
        if np.isfinite(k): KG.append(k); R.append(r); B.append(b); G.append(g)
        mon=mo["date"].dt.month
        for sn,mm in SEAS.items():
            ss=mon.isin(mm).to_numpy()
            if ss.sum()>=20:
                rr=kge(qq[ss],qo[ss])[1]
                if np.isfinite(rr): seas[sn].append(rr)
        for y in [2022,2023,2024]:
            po=peak_doy(mo["date"],qo,y); pm=peak_doy(mo["date"],qq,y)
            if np.isfinite(po) and np.isfinite(pm): off.append(pm-po)
    md=lambda x: np.nanmedian(x) if len(x) else np.nan
    print(f"{interv:10s} {md(KG):6.3f} {md(R):6.3f} {md(B):6.3f} {md(G):6.3f} | "
          f"{md(seas['DJF']):5.2f} {md(seas['MAM']):5.2f} {md(seas['JJA']):5.2f} {md(seas['SON']):5.2f} | "
          f"{md(off):+6.1f} (mean {np.nanmean(off):+.1f})")

# reference Hydrotel
score("HYDROTEL", disv.T, th.values)
for interv in sys.argv[1:]:
    p=f".runs/slso/results/eval_{interv}.npz"
    if not os.path.exists(p):
        print(f"{interv:10s} (npz absent)"); continue
    z=np.load(p, allow_pickle=True)
    score(interv, z["Q"], z["dates"])
