import sys; sys.stdout.reconfigure(encoding="utf-8")
import tomllib, math, numpy as np, torch, pandas as pd, xarray as xr
from meandre.data.basin_cache import BasinCache
from meandre.model import HydroModel
from meandre.utils.state import HydroState
cfg=tomllib.load(open('.runs/slso-od/config/slso-od-hs-latent.toml','rb'))
dev='cuda' if torch.cuda.is_available() else 'cpu'
cache=BasinCache('.runs/slso-od/data/basin_hydrosheds.duckdb'); h=cache.load(device=dev); n=h['n_nodes']
ds=xr.open_dataset(cfg['paths']['forcing_cache']); tms=pd.to_datetime(ds['time'].values); ff=ds['forcing'].values.astype('float32'); ds.close()
w0=int(np.searchsorted(tms,np.datetime64('2017-01-01'))); win=tms[w0:]
fc=torch.from_numpy(ff[w0:]).to(dev); doy=torch.tensor(win.dayofyear.values,dtype=torch.long,device=dev)
wd=cache.load_withdrawals(str(win[0].date()),str(win[-1].date()),device=dev)
obs=cache.load_observations(date_start='2000-01-01',date_end='2024-12-31',min_valid_days=365)
st=sorted(set(obs['station_node_map'].values())); qo=obs['discharge'][w0:][:,st]
vmask=(win>=pd.Timestamp('2019-01-01'))&(win<=pd.Timestamp('2021-12-31')); vi=np.where(vmask)[0]
ck=torch.load('.runs/slso-od/checkpoints/best-hs-latent.pt',map_location='cpu',weights_only=False)
kw=dict(ck['init_kwargs']); kw['n_coord_freqs']=8
m=HydroModel(**kw).to(dev); m.load('.runs/slso-od/checkpoints/best-hs-latent.pt'); m.temperature=None
m.routing.routing_mode='operator-lagged'; m.eval()
with torch.no_grad():
    Q,_=m.simulate(forcing=fc,initial_state=HydroState.zeros(n,device=dev),graph=h['graph'],
                   node_coords=h['node_coords'],territorial=h['territorial'],withdrawals=wd,day_of_year=doy)
Q=Q.cpu().numpy()
def pr_tol(sim,o,vi,tol):
    o=o[vi]; sim_full=sim
    msk=~np.isnan(o); 
    if msk.sum()<50: return np.nan
    thr=np.quantile(o[msk],0.99); 
    days=vi[(o>=thr)&msk]
    rr=[]
    for d in days:
        ob=qo_full=o  # placeholder
    return None
# simpler: per station
res0,res2=[],[]
for j,ni in enumerate(st):
    o=qo[:,j]; sim=Q[:,ni]
    ov=o[vi]; msk=~np.isnan(ov)
    if msk.sum()<50: continue
    thr=np.quantile(ov[msk],0.99)
    hi=vi[(ov>=thr)&msk]
    if len(hi)<3: continue
    obd=o[hi]
    s0=sim[hi]
    s2=np.array([sim[max(0,d-2):d+3].max() for d in hi])
    res0.append(np.nanmean(s0)/np.nanmean(obd))
    res2.append(np.nanmean(s2)/np.nanmean(obd))
print(f"peak_ratio exact (jour pile)     = {np.nanmedian(res0):.3f}")
print(f"peak_ratio ±2 jours (tolérance)  = {np.nanmedian(res2):.3f}")
print("=> si ±2j >> exact : TIMING ; sinon : MAGNITUDE")
