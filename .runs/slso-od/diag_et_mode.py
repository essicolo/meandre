"""Bilan d'ET comparé Penman-Monteith vs McGuinness sur le mini-banc, MÊME
checkpoint clone, forward seul (pas d'entraînement). Mesure si l'ET RÉELLE
sol-limitée de McGuinness descend vers l'obs (~584 mm/an) là où le PM
sur-évapore (~680), avant de payer un run d'entraînement complet.

Sortie : ET/P domaine (pondéré aire), ETP potentiel, et le coefficient de
ruissellement de GÉNÉRATION (q_lateral / P) aux jauges, pour les deux modes.

  python .runs/slso-od/diag_et_mode.py
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
import tomllib
import numpy as np
import torch
import pandas as pd
import xarray as xr
import duckdb

from meandre.data.basin_cache import BasinCache
from meandre.model import HydroModel
from meandre.utils.state import HydroState

CFG = ".runs/slso-od/config/slso-od-mini-clone.toml"
CKPT = ".runs/slso-od/checkpoints/best-mini-clone.pt"
WIN_START, VAL_START, VAL_END = "2019-01-01", "2022-01-01", "2022-12-31"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = tomllib.load(open(CFG, "rb"))
DB = ".runs/slso-od/" + cfg["paths"]["basin_db"]
cache = BasinCache(DB); h = cache.load(device=device); n_nodes = h["n_nodes"]
ds = xr.open_dataset(cfg["paths"]["forcing_cache"])
times = pd.to_datetime(ds["time"].values); ff = ds["forcing"].values.astype(np.float32); ds.close()
w0 = int(np.searchsorted(times, np.datetime64(WIN_START))); win = times[w0:]
fc = torch.from_numpy(ff[w0:]).to(device)
doy = torch.tensor(win.dayofyear.values, dtype=torch.long, device=device)
wd = cache.load_withdrawals(str(win[0].date()), str(win[-1].date()), device=device)

# Aire locale par nœud (pondération domaine) + map jauges.
con = duckdb.connect(DB, read_only=True)
area = np.array([r[0] for r in con.execute(
    "SELECT area_km2_local FROM territorial ORDER BY node_idx").fetchall()], dtype=np.float64)
con.close()
DS, DE = cfg["temporal"]["date_start"], cfg["temporal"]["date_end"]
obs = cache.load_observations(date_start=DS, date_end=DE, min_valid_days=365)
st = sorted(set(obs["station_node_map"].values()))

vi = np.where((win >= pd.Timestamp(VAL_START)) & (win <= pd.Timestamp(VAL_END)))[0]
nyr_all = len(win) / 365.25
nyr_val = len(vi) / 365.25

_ck = torch.load(CKPT, map_location="cpu", weights_only=False)
kw = dict(_ck["init_kwargs"]); kw["n_coord_freqs"] = cfg["model"].get("n_coord_freqs", 6)
m = HydroModel(**kw).to(device); m.load(CKPT); m.temperature = None
m.routing.routing_mode = cfg["model"].get("routing_mode", "operator-lagged"); m.eval()

P_node = fc[:, :, 0].cpu().numpy()  # (T, N) mm/j

def run(et_mode):
    m.vertical_column.et.et_mode = et_mode
    with torch.no_grad():
        Q, _, diag = m.simulate(
            forcing=fc, initial_state=HydroState.zeros(n_nodes, device=device),
            graph=h["graph"], node_coords=h["node_coords"], territorial=h["territorial"],
            withdrawals=wd, day_of_year=doy, return_diagnostics=True)
    etr = diag.etr.cpu().numpy(); etp = diag.etp.cpu().numpy(); ql = diag.q_lateral  # ql en m3/s
    return etr, etp

def wmean_annual(x, sl, nyr):
    # x (T, N) mm/j -> moyenne domaine pondérée aire, mm/an
    return (x[sl].sum(0) * area).sum() / area.sum() / nyr

print(f"checkpoint {CKPT}  | banc mini {n_nodes} nœuds | fenêtre {WIN_START}..{win[-1].date()}")
print(f"{'mode':>12} | {'P':>6} | {'ETP_pot':>8} | {'ETR':>6} | {'ET/P':>5} | {'ETR_val22':>9} | {'ET/P_val':>8}")
P_all = wmean_annual(P_node, slice(None), nyr_all)
P_val = wmean_annual(P_node, vi, nyr_val)
for mode in ("penman", "mcguinness"):
    etr, etp = run(mode)
    ETP = wmean_annual(etp, slice(None), nyr_all)
    ETR = wmean_annual(etr, slice(None), nyr_all)
    ETR_v = wmean_annual(etr, vi, nyr_val)
    print(f"{mode:>12} | {P_all:6.0f} | {ETP:8.0f} | {ETR:6.0f} | {ETR/P_all:5.2f} "
          f"| {ETR_v:9.0f} | {ETR_v/P_val:8.2f}")
print(f"\n  P domaine = {P_all:.0f} mm/an ({P_val:.0f} val22) ; obs implique ET/P ~0.57 (RC ~0.43)")
print("  CIBLE : ETR descend de ~680 (PM) vers ~584 (obs) ?  DONE")
