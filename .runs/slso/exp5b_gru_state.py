"""EXP-5b : GRU résidu nourri de PROXIES D'ÉTAT (API sol + swe neige + gel). sur physique GELÉE (test hybride honnête).
Q_physique (baseline) est fixe. Un petit GRU nourri de [Q_phys, P, Tmean, doy-sin/cos]
apprend une correction MULTIPLICATIVE bornée. Entraîné 2004-2021, JUGÉ sur held-out
2022-24 (non stationnaire). Régularisé fort, capacité faible : on veut savoir si un
résidu appris généralise ou sur-apprend. Ne touche PAS la physique. CPU.
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, duckdb, xarray as xr, torch, torch.nn as nn

DEV = "cpu"
TRAIN0, TRAIN1 = "2004-01-01", "2019-12-31"
VAL0, VAL1 = "2020-01-01", "2021-12-31"     # early-stop
TEST0, TEST1 = "2022-01-01", "2024-12-31"   # held-out AVEUGLE

# physique gelée : Q_sim par tronçon (reach parquet, reach_id = node_idx+1)
rp = pd.read_parquet(".runs/slso/results/reach-physitel-hydrotel-qzarr.parquet")
rp["date"] = pd.to_datetime(rp["date"]).dt.normalize()
c = duckdb.connect(".runs/slso/data/slso.duckdb", read_only=True)
st = c.execute("SELECT station_id, node_idx FROM stations").fetchdf()
obs = c.execute("SELECT station_id,date,discharge FROM observations").fetchdf()
c.close(); obs["date"] = pd.to_datetime(obs["date"]).dt.normalize()
ds = xr.open_dataset(".runs/slso/data/forcing.nc")  # quebec.zarr (physique dessus)
ft = pd.to_datetime(ds["time"].values).normalize()
F = ds["forcing"].values; ds.close()  # (T,N,6) P,Tmin,Tmax,...

def kge(s, o):
    m = np.isfinite(s) & np.isfinite(o) & (o >= 0)
    if m.sum() < 60: return np.nan
    s, o = s[m], o[m]
    if s.std() == 0 or o.std() == 0 or o.mean() == 0: return np.nan
    return 1 - np.sqrt((np.corrcoef(s, o)[0,1]-1)**2 + (s.mean()/o.mean()-1)**2
                       + ((s.std()/s.mean())/(o.std()/o.mean())-1)**2)

# construire les séquences par station
fdate = pd.DataFrame({"date": ft, "i": np.arange(len(ft))})
samples = []
for _, s in st.iterrows():
    ni = int(s.node_idx)
    q = rp[rp.reach_id == ni + 1][["date", "Q_sim_m3s"]]
    o = obs[obs.station_id == s.station_id][["date", "discharge"]]
    m = q.merge(o, on="date").merge(fdate, on="date").sort_values("date")
    if len(m) < 2000: continue
    ii = m.i.to_numpy()
    P = F[ii, ni, 0]; Tm = 0.5*(F[ii, ni, 1] + F[ii, ni, 2])
    doy = m.date.dt.dayofyear.to_numpy()
    # PROXIES D'ÉTAT PHYSIQUE (vision GRU nourri de la physique) :
    # API = humidité du sol (précip antécédente à décroissance exp), swe = bucket neige
    api = np.zeros_like(P); swe = np.zeros_like(P); a=0.0; sw=0.0
    for k in range(len(P)):
        a = 0.92*a + P[k]; api[k]=a           # API décroissance 0.92/j (~12j)
        if Tm[k] < 0: sw += P[k]              # accumulation neige
        else: sw = max(0.0, sw - 3.0*Tm[k])   # fonte degré-jour
        swe[k]=sw
    froz = (Tm < 0).astype("f4")
    feat = np.stack([np.log1p(m.Q_sim_m3s.to_numpy()), np.log1p(P), Tm/20.0,
                     np.sin(2*np.pi*doy/365), np.cos(2*np.pi*doy/365),
                     np.log1p(api), np.log1p(swe), froz], 1).astype("f4")  # +3 proxies d'état
    samples.append(dict(sid=s.station_id, date=m.date.values,
                        feat=feat, qphys=m.Q_sim_m3s.to_numpy().astype("f4"),
                        qobs=m.discharge.to_numpy().astype("f4")))
print(f"{len(samples)} stations, features {samples[0]['feat'].shape[1]}")

class ResGRU(nn.Module):
    def __init__(self, nf, hid=16):
        super().__init__()
        self.gru = nn.GRU(nf, hid, batch_first=True)
        self.head = nn.Linear(hid, 1)
    def forward(self, x):
        h, _ = self.gru(x)
        # correction multiplicative bornée ±30% : q_corr = q_phys * (1 + 0.3*tanh)
        return 0.3 * torch.tanh(self.head(h)).squeeze(-1)

torch.manual_seed(0)
net = ResGRU(samples[0]["feat"].shape[1]).to(DEV)
opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-3)

def split(dts, a, b): return (dts >= np.datetime64(a)) & (dts <= np.datetime64(b))
best_val = -9; best_state = None
for ep in range(60):
    net.train(); tot = 0
    for s in samples:
        tr = split(s["date"], TRAIN0, TRAIN1)
        if tr.sum() < 200: continue
        x = torch.from_numpy(s["feat"][tr]).unsqueeze(0).to(DEV)
        qp = torch.from_numpy(s["qphys"][tr]).to(DEV)
        qo = torch.from_numpy(s["qobs"][tr]).to(DEV)
        d = net(x).squeeze(0)
        qc = qp * (1.0 + d)
        mask = torch.isfinite(qo) & (qo >= 0)
        loss = ((torch.log1p(qc[mask].clamp(min=0)) - torch.log1p(qo[mask]))**2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
    # early-stop sur val
    net.eval(); vk = []
    with torch.no_grad():
        for s in samples:
            v = split(s["date"], VAL0, VAL1)
            if v.sum() < 60: continue
            x = torch.from_numpy(s["feat"][v]).unsqueeze(0)
            qc = (torch.from_numpy(s["qphys"][v]) * (1 + net(x).squeeze(0))).numpy()
            vk.append(kge(qc, s["qobs"][v]))
    mvk = np.nanmedian(vk)
    if mvk > best_val: best_val = mvk; best_state = {k: v.clone() for k, v in net.state_dict().items()}
    if ep % 10 == 0: print(f"ep{ep} loss {tot/len(samples):.4f} val_kge_med {mvk:.4f}")

net.load_state_dict(best_state); net.eval()
# HELD-OUT : comparer Q_corrigé vs Q_physique
kc, kp = [], []
with torch.no_grad():
    for s in samples:
        te = split(s["date"], TEST0, TEST1)
        if te.sum() < 60: continue
        x = torch.from_numpy(s["feat"][te]).unsqueeze(0)
        qc = (torch.from_numpy(s["qphys"][te]) * (1 + net(x).squeeze(0))).numpy()
        kc.append(kge(qc, s["qobs"][te])); kp.append(kge(s["qphys"][te], s["qobs"][te]))
kc, kp = np.array(kc), np.array(kp)
print(f"\n=== HELD-OUT 2022-24 ({np.isfinite(kc).sum()} stations) ===")
print(f"Q PHYSIQUE seule   : KGE médian {np.nanmedian(kp):.4f}")
print(f"Q + GRU résidu     : KGE médian {np.nanmedian(kc):.4f}  ({np.nanmedian(kc)-np.nanmedian(kp):+.4f})")
print(f"stations améliorées : {(kc>kp).sum()}/{np.isfinite(kc).sum()}")
print(">>> si GRU > physique : le résidu appris GÉNÉRALISE. Sinon : sur-apprentissage (physique pure plus robuste).")
