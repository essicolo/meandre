"""Balayage du paysage de perte le long de K_sat couche 1 (forward only).

Teste l'hypothèse d'Essi (2026-06-14) : van Genuchten se comporte comme un
robinet ouvert/fermé. Un K_sat élevé draine la couche 1 trop vite, elle
n'atteint jamais la porosité -> pas de ruissellement par saturation. Pour
générer des pics il faudrait baisser K_sat, mais en chemin la récession se
dégrade AVANT que les pics compensent -> barrière non-convexe, le gradient
reste piégé dans le bassin K_sat haut.

Méthode : on prend le modèle entraîné (best-vsafull), on neutralise le VSA
(vsa_b -> grand, f_sat = Se^b -> 0) pour isoler la PHYSIQUE van Genuchten
seule, puis on multiplie K_sat_1 par un facteur décroissant et on recalcule
KGE médian, peak_ratio et la fraction ruisselée sur la fenêtre de validation.
Forward only, aucun entraînement.

  python .runs/slso-od/scan_ksat_landscape.py
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
import tomllib
import numpy as np
import torch
import pandas as pd
import xarray as xr

from meandre.data.basin_cache import BasinCache
from meandre.model import HydroModel
from meandre.utils.state import HydroState

CFG_PATH = ".runs/slso-od/config/slso-od-vsafull.toml"
CKPT = ".runs/slso-od/checkpoints/best-vsafull.pt"
# Fenêtre forward : 2 ans de spinup + période de validation (métriques sur val).
WIN_START = "2017-01-01"
VAL_START = "2019-01-01"
VAL_END = "2021-12-31"
FACTORS = [1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02]

cfg = tomllib.load(open(CFG_PATH, "rb"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATE_START = cfg["temporal"]["date_start"]
DATE_END = cfg["temporal"]["date_end"]

cache = BasinCache(".runs/slso-od/data/basin.duckdb")
h = cache.load(device=device)
n_nodes = h["n_nodes"]

# Forçage plein-période puis fenêtre.
ds = xr.open_dataset(cfg["paths"]["forcing_cache"])
all_times = pd.to_datetime(ds["time"].values)
forcing_full = ds["forcing"].values.astype(np.float32)
ds.close()

w0 = int(np.searchsorted(all_times, np.datetime64(WIN_START)))
w1 = len(all_times)
win_times = all_times[w0:w1]
fc = torch.from_numpy(forcing_full[w0:w1]).to(device)
doy = torch.tensor(win_times.dayofyear.values, dtype=torch.long, device=device)
wd = cache.load_withdrawals(str(win_times[0].date()), str(win_times[-1].date()), device=device)

# Observations alignées sur le plein-période, slicées sur la fenêtre.
obs = cache.load_observations(date_start=DATE_START, date_end=DATE_END, min_valid_days=365)
station_node_map = obs["station_node_map"]
station_indices = sorted(set(station_node_map.values()))
discharge_full = obs["discharge"]  # (T_full, N_all)
q_obs_win = torch.from_numpy(discharge_full[w0:w1][:, station_indices]).to(device)

# Masque de la période de validation dans la fenêtre.
val_mask = (win_times >= pd.Timestamp(VAL_START)) & (win_times <= pd.Timestamp(VAL_END))
val_idx = torch.tensor(np.where(val_mask)[0], device=device)

print(f"n_nodes={n_nodes}  fenêtre {win_times[0].date()}..{win_times[-1].date()} "
      f"({len(win_times)} j)  val {VAL_START}..{VAL_END} ({int(val_mask.sum())} j)  "
      f"stations={len(station_indices)}  device={device}", flush=True)

# Modèle entraîné.
_ck = torch.load(CKPT, map_location="cpu", weights_only=False)
_kw = dict(_ck["init_kwargs"])
# init_kwargs ne sérialise pas n_coord_freqs (=None) : le reprendre du config
# sinon le buffer coord_enc.freqs n'a pas la bonne taille (8 vs défaut 6).
_kw["n_coord_freqs"] = cfg["model"].get("n_coord_freqs", 6)
m = HydroModel(**_kw).to(device)
m.load(CKPT)
m.temperature = None
m.routing.routing_mode = cfg["model"].get("routing_mode", "operator-lagged")
m.eval()

# Hook : on intercepte la sortie de l'encodeur spatial pour (1) neutraliser le
# VSA (vsa_b grand) et (2) multiplier K_sat_1 par le facteur courant.
_orig_fwd = m.spatial_encoder.forward
STATE = {"factor": 1.0, "kill_vsa": True}

def _patched(*a, **k):
    sp = _orig_fwd(*a, **k)
    if STATE["kill_vsa"] and hasattr(sp, "vsa_b"):
        sp.vsa_b = torch.full_like(sp.vsa_b, 50.0)  # Se^50 -> 0 : VSA éteint
    sp.K_sat_1 = sp.K_sat_1 * STATE["factor"]
    return sp

m.spatial_encoder.forward = _patched


def kge(sim, obs):
    msk = ~np.isnan(obs)
    s, o = sim[msk], obs[msk]
    if len(o) < 30 or o.std() < 1e-9 or s.std() < 1e-9:
        return np.nan
    r = np.corrcoef(s, o)[0, 1]
    beta = s.mean() / o.mean()
    gamma = (s.std() / s.mean()) / (o.std() / o.mean())
    return 1.0 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)


def peak_ratio(sim, obs):
    msk = ~np.isnan(obs)
    o = obs[msk]
    if len(o) < 50:
        return np.nan
    s = sim[msk]
    thr = np.quantile(o, 0.99)
    hi = o >= thr
    if hi.sum() < 3 or o[hi].mean() < 1e-9:
        return np.nan
    return s[hi].mean() / o[hi].mean()


@torch.no_grad()
def run(factor, kill_vsa):
    STATE["factor"] = factor
    STATE["kill_vsa"] = kill_vsa
    Q, _ = m.simulate(
        forcing=fc, initial_state=HydroState.zeros(n_nodes, device=device),
        graph=h["graph"], node_coords=h["node_coords"], territorial=h["territorial"],
        withdrawals=wd, day_of_year=doy,
    )
    sim = Q[val_idx][:, station_indices].cpu().numpy()  # (T_val, n_st)
    o = q_obs_win[val_idx].cpu().numpy()
    kges, prs = [], []
    for j in range(sim.shape[1]):
        kges.append(kge(sim[:, j], o[:, j]))
        prs.append(peak_ratio(sim[:, j], o[:, j]))
    # Fraction ruisselée approximée par le volume total simulé à l'exutoire-stations.
    vol = float(np.nansum(sim))
    return (np.nanmedian(kges), np.nanmedian(prs), vol)


print("\n=== van Genuchten seul (VSA neutralisé) : balayage K_sat_1 ===", flush=True)
print(f"{'facteur':>8} {'K_sat_x':>8} {'kge_med':>9} {'peak_ratio':>11} {'vol_rel':>9}", flush=True)
vol0 = None
for f in FACTORS:
    km, pr, vol = run(f, kill_vsa=True)
    if vol0 is None:
        vol0 = vol
    print(f"{f:8.2f} {f:8.2f} {km:9.3f} {pr:11.3f} {vol/vol0:9.3f}", flush=True)

# Référence : modèle tel qu'entraîné (VSA actif), facteur 1.0.
km, pr, vol = run(1.0, kill_vsa=False)
print(f"\nréférence VSA actif, facteur 1.0 : kge_med={km:.3f} peak_ratio={pr:.3f}", flush=True)
print("SCAN_DONE", flush=True)
