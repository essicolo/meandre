import sys; sys.stdout.reconfigure(encoding="utf-8")
import tomllib, math, numpy as np, torch, pandas as pd, xarray as xr, duckdb
from meandre.data.basin_cache import BasinCache
from meandre.model import HydroModel
from meandre.utils.state import HydroState
cfg=tomllib.load(open('.runs/slso-od/config/slso-od-hs-latent.toml','rb'))
dev='cuda' if torch.cuda.is_available() else 'cpu'
DB='.runs/slso-od/data/basin_hydrosheds.duckdb'
cache=BasinCache(DB); h=cache.load(device=dev); n=h['n_nodes']
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
    Q,_,diag=m.simulate(forcing=fc,initial_state=HydroState.zeros(n,device=dev),graph=h['graph'],
                        node_coords=h['node_coords'],territorial=h['territorial'],withdrawals=wd,
                        day_of_year=doy,return_diagnostics=True)
    sp=m.spatial_encoder(h['node_coords'], h['territorial'].to_tensor())
qlat=diag.q_lateral.cpu().numpy()           # (T,N) m³/s local
# temps de parcours PHYSIQUE = longueur tronçon / célérité (m/s) -> jours
import duckdb as _d
_c=_d.connect(DB,read_only=True); _e=_c.execute('SELECT src,dst,edge_attr_0 FROM edges').fetchall(); _c.close()
length_m=np.full(n, 3000.0)  # défaut 3 km
for s_,d_,L_ in _e: length_m[int(s_)]=float(L_)
CELERITY=1.0  # m/s
Kdays=(length_m/CELERITY)/86400.0
T=qlat.shape[0]
# topologie : src -> dst (aval)
con=duckdb.connect(DB,read_only=True); edges=con.execute("SELECT src,dst FROM edges").fetchall(); con.close()
parents={}  # node -> dst (aval direct)
children={} # dst -> [src...]
for s,d in edges:
    parents[int(s)]=int(d); children.setdefault(int(d),[]).append(int(s))
# ordre topo (amont -> aval) : tri par profondeur depuis racines
order=h['graph'].topo_order.cpu().numpy() if hasattr(h['graph'],'topo_order') else None
if order is None:
    # fallback : Kahn sur children
    import collections
    indeg={i:0 for i in range(n)}
    for s,d in edges: indeg[d]+=1
    q=collections.deque([i for i in range(n) if indeg[i]==0]); order=[]
    while q:
        x=q.popleft(); order.append(x)
        d=parents.get(x)
        if d is not None:
            indeg[d]-=1
            if indeg[d]==0: q.append(d)
    order=np.array(order)
def kge(sim,o):
    ov=o[vi]; sv=sim[vi]; mk=~np.isnan(ov)
    s_,o_=sv[mk],ov[mk]
    if len(o_)<30 or o_.std()<1e-9 or s_.std()<1e-9: return np.nan
    r=np.corrcoef(s_,o_)[0,1]
    return 1.0-math.sqrt((r-1)**2+(s_.mean()/o_.mean()-1)**2+((s_.std()/s_.mean())/(o_.std()/o_.mean())-1)**2)
def peak_ratio(sim,o):
    ov=o[vi]; mk=~np.isnan(ov)
    if mk.sum()<50: return np.nan
    thr=np.quantile(ov[mk],0.99); hi=vi[(ov>=thr)&mk]
    if len(hi)<3: return np.nan
    return np.nanmean(sim[hi])/np.nanmean(o[hi])
# Advection + délai entier par tronçon, zéro atténuation
Qadv=np.zeros((T,n),dtype=np.float64)
lag=np.clip(np.round(Kdays).astype(int),0,30)
for node in order:
    acc=qlat[:,node].astype(np.float64).copy()
    for c in children.get(node,[]):
        L=lag[c]
        if L>0:
            shifted=np.zeros(T); shifted[L:]=Qadv[:-L,c]
        else:
            shifted=Qadv[:,c]
        acc+=shifted
    Qadv[:,node]=acc

def nash_kernel(tp, M=20):
    # hydrogramme unitaire gamma (Nash), temps de pointe tp jours, n=2 réservoirs
    if tp<=0.05: 
        k=np.zeros(M); k[0]=1.0; return k
    n=2.0; k=tp/(n-1) if n>1 else tp
    t=np.arange(M)
    g=(t/k)**(n-1)*np.exp(-t/k)
    return g/g.sum()

def route_adv(qlat_in):
    Qa=np.zeros((T,n))
    for node in order:
        acc=qlat_in[:,node].astype(np.float64).copy()
        for c in children.get(node,[]):
            L=lag[c]
            sh=np.zeros(T)
            if L>0: sh[L:]=Qa[:-L,c]
            else: sh=Qa[:,c]
            acc+=sh
        Qa[:,node]=acc
    return Qa

Qm=Q.cpu().numpy()
print(f"{'config':>26} {'kge_med':>9} {'peak_ratio':>11}")
print(f"{'MUSKINGUM (référence)':>26} {np.nanmedian([kge(Qm[:,ni],qo[:,j]) for j,ni in enumerate(st)]):9.3f} {np.nanmedian([peak_ratio(Qm[:,ni],qo[:,j]) for j,ni in enumerate(st)]):11.3f}")
for tp in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
    ker=nash_kernel(tp)
    # convolution causale par nœud
    qs=np.zeros_like(qlat)
    for i,w in enumerate(ker):
        if w>0:
            qs[i:,:]+=w*qlat[:T-i,:]
    Qa=route_adv(qs)
    km=np.nanmedian([kge(Qa[:,ni],qo[:,j]) for j,ni in enumerate(st)])
    pr=np.nanmedian([peak_ratio(Qa[:,ni],qo[:,j]) for j,ni in enumerate(st)])
    print(f"{'versant UH tp='+str(tp)+'j':>26} {km:9.3f} {pr:11.3f}")
