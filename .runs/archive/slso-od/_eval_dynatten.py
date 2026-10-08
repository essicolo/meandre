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
vm=(win>=pd.Timestamp('2019-01-01'))&(win<=pd.Timestamp('2021-12-31')); vi=torch.tensor(np.where(vm)[0],device=dev)
ck=torch.load('.runs/slso-od/checkpoints/best-hs-latent.pt',map_location='cpu',weights_only=False)
kw=dict(ck['init_kwargs']); kw['n_coord_freqs']=8
m=HydroModel(**kw).to(dev); m.load('.runs/slso-od/checkpoints/best-hs-latent.pt'); m.temperature=None
m.routing.routing_mode='operator-lagged'; m.eval()
o=qo[vi.cpu().numpy()]
def kge(s,oo):
    mk=~np.isnan(oo); s_,o_=s[mk],oo[mk]
    if len(o_)<30 or o_.std()<1e-9 or s_.std()<1e-9: return np.nan
    r=np.corrcoef(s_,o_)[0,1]; return 1-math.sqrt((r-1)**2+(s_.mean()/o_.mean()-1)**2+((s_.std()/s_.mean())/(o_.std()/o_.mean())-1)**2)
def pr(s,oo):
    mk=~np.isnan(oo); oo=oo[mk]
    if len(oo)<50: return np.nan
    s=s[mk]; hi=oo>=np.quantile(oo,0.99)
    if hi.sum()<3 or oo[hi].mean()<1e-9: return np.nan
    return s[hi].mean()/oo[hi].mean()
@torch.no_grad()
def run(da,beta=2.0,qref=0.05):
    m.routing.dynamic_atten=da; m.routing.da_beta=beta; m.routing.da_qref_specific=qref; m.routing._op_state=None
    Q,_=m.simulate(forcing=fc,initial_state=HydroState.zeros(n,device=dev),graph=h['graph'],node_coords=h['node_coords'],territorial=h['territorial'],withdrawals=wd,day_of_year=doy)
    s=Q[vi][:,st].cpu().numpy()
    return np.nanmedian([kge(s[:,j],o[:,j]) for j in range(s.shape[1])]),np.nanmedian([pr(s[:,j],o[:,j]) for j in range(s.shape[1])])
print(f"{'config':>26} {'kge_med':>9} {'peak_ratio':>11}")
for nm,da,b,q in [("OFF (Muskingum statique)",False,2,.05),("dyn beta=2 qref=0.05",True,2,.05),("dyn beta=1 qref=0.05",True,1,.05),("dyn beta=2 qref=0.02",True,2,.02),("dyn beta=3 qref=0.02",True,3,.02)]:
    km,prr=run(da,b,q); print(f"{nm:>26} {km:9.3f} {prr:11.3f}")
