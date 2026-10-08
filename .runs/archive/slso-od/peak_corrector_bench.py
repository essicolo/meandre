"""Correcteur de sortie ciblé pics — backbone gelé, tête entraînée en post-hoc.

Idée (cf. Frame et al. 2021, post-processeur LSTM sur la physique) : la physique
génère le bon VOLUME de crue (peak_ratio génération ≈ 0.997) et le bon TIMING
(Q routé déjà aligné). Le déficit est une atténuation systématique de magnitude.
Un correcteur qui agit sur le Q ROUTÉ — pas sur le routage — peut rajouter la
magnitude manquante SANS toucher au timing (il ne décale rien) ni casser l'étiage
(gain multiplicatif NON-NÉGATIF : il ne peut qu'ajouter de l'eau, jamais en
retirer ; à bas débit Q_sim est petit donc le boost absolu est petit).

  Q_corr_t = Q_sim_t * (1 + gain_t),  gain_t = softplus(GRU_causal(features)_t) >= 0

Piloté par la pluie : le GRU apprend « pluie forte récente + Q_sim qui monte vite
=> Q_sim sous-estime systématiquement => booste ». Règle qui transfère via les
features de forçage, donc régionalisable plus tard.

Test décisif : entraîné sur la fenêtre train (2020-2021) aux stations jaugées,
évalué kge_med + peak_ratio sur train ET val (2022). Si val s'améliore aussi,
c'est une correction systématique réelle, pas de la mémorisation.

  python .runs/slso-od/peak_corrector_bench.py <config.toml> <checkpoint.pt>
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
W_PEAK = 8.0        # poids des pas où obs > p90 station, dans la MSE
W_REG = 1e-3        # L2 sur le gain (garde la correction minimale)
N_EPOCH = 400
LR = 1e-2

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
q_obs = torch.from_numpy(obs["discharge"][w0:][:, st]).to(device)   # (T, S)

def idx(a, b):
    m = (win >= pd.Timestamp(a)) & (win <= pd.Timestamp(b))
    return torch.tensor(np.where(m)[0], device=device)
tr_i, va_i = idx(*TRAIN), idx(*VAL)

def kge_np(sim, o):
    m = ~np.isnan(o); s, o = sim[m], o[m]
    if len(o) < 30 or o.std() < 1e-9 or s.std() < 1e-9: return np.nan
    r = np.corrcoef(s, o)[0, 1]
    return 1.0 - math.sqrt((r-1)**2 + (s.mean()/o.mean()-1)**2 + ((s.std()/s.mean())/(o.std()/o.mean())-1)**2)

def pr_np(sim, o):
    m = ~np.isnan(o); o = o[m]
    if len(o) < 40: return np.nan
    s = sim[m]; hi = o >= np.quantile(o, 0.99)
    if hi.sum() < 2 or o[hi].mean() < 1e-9: return np.nan
    return s[hi].mean()/o[hi].mean()

# ── 1. Backbone gelé : une passe simulate ────────────────────────────────
_ck = torch.load(CKPT, map_location="cpu", weights_only=False)
kw = dict(_ck["init_kwargs"]); kw["n_coord_freqs"] = cfg["model"].get("n_coord_freqs", 6)
m = HydroModel(**kw).to(device); m.load(CKPT); m.temperature = None
m.routing.routing_mode = cfg["model"].get("routing_mode", "operator-lagged"); m.eval()
with torch.no_grad():
    Q, _ = m.simulate(forcing=fc, initial_state=HydroState.zeros(n_nodes, device=device),
                      graph=h["graph"], node_coords=h["node_coords"], territorial=h["territorial"],
                      withdrawals=wd, day_of_year=doy)
Qs = Q[:, st].clamp(min=0.0)                          # (T, S) Q_sim aux jauges
P = fc[:, st, 0].clamp(min=0.0)                       # (T, S) pluie locale
T, S = Qs.shape

# ── 2. Features causales par (jauge, t) ──────────────────────────────────
def causal_rollmean(x, k):
    # moyenne glissante causale sur k pas (incluant t)
    xp = torch.cat([x[:1].expand(k-1, -1), x], 0)
    return torch.stack([xp[i:i+T] for i in range(k)], 0).mean(0)
def causal_rollsum(x, k):
    xp = torch.cat([torch.zeros(k-1, S, device=device), x], 0)
    return torch.stack([xp[i:i+T] for i in range(k)], 0).sum(0)

q_base = causal_rollmean(Qs, 30) + 1e-3
feats = torch.stack([
    torch.log1p(Qs),                                 # niveau
    torch.log1p(P),                                  # pluie du jour
    torch.log1p(causal_rollsum(P, 3)),               # cumul 3j (orage)
    Qs / q_base,                                      # débit relatif (montée)
    torch.relu(Qs - torch.cat([Qs[:1], Qs[:-1]], 0)),# d+Q (rising limb)
], dim=-1)                                            # (T, S, F)
fmu = feats[tr_i].reshape(-1, feats.shape[-1]).mean(0)
fsd = feats[tr_i].reshape(-1, feats.shape[-1]).std(0) + 1e-6
feats = (feats - fmu) / fsd

# ── 3. Tête correctrice : GRU causal, nœuds = batch ──────────────────────
class PeakCorrector(nn.Module):
    def __init__(self, n_feat, hidden=16):
        super().__init__()
        self.gru = nn.GRU(n_feat, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 1)
        nn.init.zeros_(self.out.weight); nn.init.constant_(self.out.bias, -4.0)  # gain≈0 au départ
    def forward(self, x):                            # x: (S, T, F)
        z, _ = self.gru(x)
        return torch.nn.functional.softplus(self.out(z)).squeeze(-1)  # (S, T) >=0

cor = PeakCorrector(feats.shape[-1]).to(device)
opt = torch.optim.Adam(cor.parameters(), lr=LR)
x_in = feats.permute(1, 0, 2).contiguous()           # (S, T, F)
obs_t = q_obs.T                                       # (S, T)
qs_t = Qs.T                                           # (S, T)
valid = ~torch.isnan(obs_t)
# poids pics : par station, seuil p90 sur la fenêtre train
o_tr = obs_t[:, tr_i]
p90 = torch.nanquantile(torch.where(valid[:, tr_i], o_tr, torch.nan), 0.90, dim=1, keepdim=True)
tr_mask = torch.zeros(S, T, dtype=torch.bool, device=device); tr_mask[:, tr_i] = True
w_t = torch.where(obs_t > p90, W_PEAK, 1.0) * (valid & tr_mask).float()

for ep in range(N_EPOCH):
    opt.zero_grad()
    gain = cor(x_in)                                 # (S, T)
    q_cor = qs_t * (1.0 + gain)
    err2 = (q_cor - torch.nan_to_num(obs_t)) ** 2
    loss = (w_t * err2).sum() / (w_t.sum() + 1e-6) + W_REG * (gain ** 2).mean()
    loss.backward(); opt.step()
    if ep % 100 == 0 or ep == N_EPOCH - 1:
        print(f"  ep {ep:4d}  loss={loss.item():.4f}  gain_mean={gain.mean().item():.3f}  gain_peakdays={gain[obs_t>p90].mean().item():.3f}", flush=True)

# ── 4. Éval 2 fenêtres : baseline (gain=0) vs corrigé ────────────────────
cor.eval()
with torch.no_grad():
    gain = cor(x_in)                                 # (S, T)
q_cor_full = (qs_t * (1.0 + gain)).T                 # (T, S)

def med(win_i, sim):
    s = sim[win_i].cpu().numpy(); o = q_obs[win_i].cpu().numpy()
    ks = [kge_np(s[:, j], o[:, j]) for j in range(S)]
    ps = [pr_np(s[:, j], o[:, j]) for j in range(S)]
    return np.nanmedian(ks), np.nanmedian(ps)

print(f"\n{'':>22} | {'train kge_med':>13} {'pr':>6} | {'val kge_med':>11} {'pr':>6}")
for name, sim in [("baseline (gain=0)", Qs), ("+ correcteur pics", q_cor_full)]:
    tk, tp = med(tr_i, sim); vk, vp = med(va_i, sim)
    print(f"{name:>22} | {tk:13.3f} {tp:6.3f} | {vk:11.3f} {vp:6.3f}", flush=True)
