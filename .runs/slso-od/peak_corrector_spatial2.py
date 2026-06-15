"""Transfert SPATIAL du correcteur de pics : validation croisée 5 plis sur les
stations. Pour chaque pli, le correcteur est entraîné sur 80 % des stations et
évalué sur les 20 % JAMAIS vues. Si le peak_ratio des stations held-out monte
quand même, la règle pluie->boost est régionalisable (pas de la mémorisation
par station). C'est le test de déploiement réel (stations non jaugées).

  python .runs/slso-od/peak_corrector_spatial.py [config] [checkpoint]
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
import math, tomllib
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import xarray as xr

from meandre.data.basin_cache import BasinCache
from meandre.model import HydroModel
from meandre.utils.state import HydroState

CFG = sys.argv[1] if len(sys.argv) > 1 else ".runs/slso-od/config/slso-od-mini-latent.toml"
CKPT = sys.argv[2] if len(sys.argv) > 2 else ".runs/slso-od/checkpoints/best-mini-latent.pt"
WIN_START = "2019-01-01"
TRAIN = ("2020-01-01", "2021-12-31")
VAL = ("2022-01-01", "2022-12-31")
W_PEAK = float(sys.argv[3]) if len(sys.argv) > 3 else 8.0
W_REG, N_EPOCH, LR, K_FOLD = 1e-3, 400, 1e-2, 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = tomllib.load(open(CFG, "rb"))
DB = (".runs/slso-od/" + cfg["paths"]["basin_db"]) if not cfg["paths"]["basin_db"].startswith("/") else cfg["paths"]["basin_db"]
DS, DE = cfg["temporal"]["date_start"], cfg["temporal"]["date_end"]
cache = BasinCache(DB); h = cache.load(device=device); n_nodes = h["n_nodes"]
ds = xr.open_dataset(cfg["paths"]["forcing_cache"])
times = pd.to_datetime(ds["time"].values); ff = ds["forcing"].values.astype(np.float32); ds.close()
w0 = int(np.searchsorted(times, np.datetime64(WIN_START))); win = times[w0:]
fc = torch.from_numpy(ff[w0:]).to(device)
doy = torch.tensor(win.dayofyear.values, dtype=torch.long, device=device)
wd = cache.load_withdrawals(str(win[0].date()), str(win[-1].date()), device=device)
obs = cache.load_observations(date_start=DS, date_end=DE, min_valid_days=365)
st = sorted(set(obs["station_node_map"].values()))
q_obs = torch.from_numpy(obs["discharge"][w0:][:, st]).to(device)

def idx(a, b):
    mm = (win >= pd.Timestamp(a)) & (win <= pd.Timestamp(b)); return torch.tensor(np.where(mm)[0], device=device)
tr_i, va_i = idx(*TRAIN), idx(*VAL)

def kge_np(s, o):
    mm = ~np.isnan(o); s, o = s[mm], o[mm]
    if len(o) < 30 or o.std() < 1e-9 or s.std() < 1e-9: return np.nan
    r = np.corrcoef(s, o)[0, 1]
    return 1.0 - math.sqrt((r-1)**2 + (s.mean()/o.mean()-1)**2 + ((s.std()/s.mean())/(o.std()/o.mean())-1)**2)
def pr_np(s, o):
    mm = ~np.isnan(o); o = o[mm]
    if len(o) < 40: return np.nan
    s = s[mm]; hi = o >= np.quantile(o, 0.99)
    if hi.sum() < 2 or o[hi].mean() < 1e-9: return np.nan
    return s[hi].mean()/o[hi].mean()

_ck = torch.load(CKPT, map_location="cpu", weights_only=False)
kw = dict(_ck["init_kwargs"]); kw["n_coord_freqs"] = cfg["model"].get("n_coord_freqs", 6)
m = HydroModel(**kw).to(device); m.load(CKPT); m.temperature = None
m.routing.routing_mode = cfg["model"].get("routing_mode", "operator-lagged"); m.eval()
with torch.no_grad():
    Q, _ = m.simulate(forcing=fc, initial_state=HydroState.zeros(n_nodes, device=device),
                      graph=h["graph"], node_coords=h["node_coords"], territorial=h["territorial"],
                      withdrawals=wd, day_of_year=doy)
Qs = Q[:, st].clamp(min=0.0); P = fc[:, st, 0].clamp(min=0.0); T, S = Qs.shape

def crollmean(x, k):
    xp = torch.cat([x[:1].expand(k-1, -1), x], 0); return torch.stack([xp[i:i+T] for i in range(k)], 0).mean(0)
def crollsum(x, k):
    xp = torch.cat([torch.zeros(k-1, S, device=device), x], 0); return torch.stack([xp[i:i+T] for i in range(k)], 0).sum(0)
q_base = crollmean(Qs, 30) + 1e-3
dyn = torch.stack([
    torch.log1p(Qs), torch.log1p(P), torch.log1p(crollsum(P, 3)),
    Qs / q_base, torch.relu(Qs - torch.cat([Qs[:1], Qs[:-1]], 0)),
], dim=-1)                                            # (T, S, n_dyn)
# Descripteurs STATIQUES par nœud (territorial) -> module la magnitude par site
terr = h["territorial"].to_tensor().to(device)[st]   # (S, n_terr)
terr = (terr - terr.mean(0)) / (terr.std(0) + 1e-6)
stat = terr.unsqueeze(0).expand(T, S, terr.shape[-1])  # (T, S, n_terr)
feats_raw = torch.cat([dyn, stat], dim=-1)            # (T, S, F)
F = feats_raw.shape[-1]

class PeakCorrector(nn.Module):
    def __init__(self, n_feat, hidden=16):
        super().__init__()
        self.gru = nn.GRU(n_feat, hidden, batch_first=True); self.out = nn.Linear(hidden, 1)
        nn.init.zeros_(self.out.weight); nn.init.constant_(self.out.bias, -4.0)
    def forward(self, x):
        z, _ = self.gru(x); return torch.nn.functional.softplus(self.out(z)).squeeze(-1)

# folds spatiaux déterministes
g = torch.Generator().manual_seed(0)
perm = torch.randperm(S, generator=g).tolist()
folds = [perm[i::K_FOLD] for i in range(K_FOLD)]

q_cor_ho = Qs.clone()      # série corrigée, remplie sur chaque held-out
obs_t = q_obs.T; qs_t = Qs.T; valid = ~torch.isnan(obs_t)

for kf in range(K_FOLD):
    ho = sorted(folds[kf]); tr = sorted(set(range(S)) - set(ho))
    fmu = feats_raw[tr_i][:, tr].reshape(-1, F).mean(0); fsd = feats_raw[tr_i][:, tr].reshape(-1, F).std(0) + 1e-6
    feats = (feats_raw - fmu) / fsd
    x_in = feats.permute(1, 0, 2).contiguous()        # (S, T, F)
    cor = PeakCorrector(F).to(device); opt = torch.optim.Adam(cor.parameters(), lr=LR)
    o_tr = obs_t[:, tr_i]
    p90 = torch.nanquantile(torch.where(valid[:, tr_i], o_tr, torch.nan), 0.90, dim=1, keepdim=True)
    tr_mask = torch.zeros(S, T, dtype=torch.bool, device=device); tr_mask[:, tr_i] = True
    st_mask = torch.zeros(S, 1, dtype=torch.bool, device=device); st_mask[tr] = True
    w_t = torch.where(obs_t > p90, W_PEAK, 1.0) * (valid & tr_mask & st_mask).float()
    for ep in range(N_EPOCH):
        opt.zero_grad()
        gain = cor(x_in); q_cor = qs_t * (1.0 + gain)
        loss = (w_t * (q_cor - torch.nan_to_num(obs_t)) ** 2).sum() / (w_t.sum() + 1e-6) + W_REG * (gain ** 2).mean()
        loss.backward(); opt.step()
    cor.eval()
    with torch.no_grad():
        gain = cor(x_in)
    qc = (qs_t * (1.0 + gain)).T                       # (T, S)
    for j in ho:
        q_cor_ho[:, j] = qc[:, j]                       # prédiction OUT-OF-FOLD

def med(win_i, sim, cols=None):
    cols = range(S) if cols is None else cols
    s = sim[win_i].cpu().numpy(); o = q_obs[win_i].cpu().numpy()
    ks = [kge_np(s[:, j], o[:, j]) for j in cols]; ps = [pr_np(s[:, j], o[:, j]) for j in cols]
    return np.nanmedian(ks), np.nanmedian(ps)

print(f"\nTransfert SPATIAL ({K_FOLD} plis, stations held-out out-of-fold)")
print(f"{'':>26} | {'train kge_med':>13} {'pr':>6} | {'val kge_med':>11} {'pr':>6}")
for name, sim in [("baseline (gain=0)", Qs), ("+ correcteur (held-out)", q_cor_ho)]:
    tk, tp = med(tr_i, sim); vk, vp = med(va_i, sim)
    print(f"{name:>26} | {tk:13.3f} {tp:6.3f} | {vk:11.3f} {vp:6.3f}", flush=True)
