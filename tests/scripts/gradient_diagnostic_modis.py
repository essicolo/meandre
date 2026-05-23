"""Diagnostic d'identifiabilité spatiale : compare les gradients par nœud
avec et sans le terme NLL_ET (MOD16A2 ETR).

Hypothèse à tester
------------------
Sans MODIS : ∂L/∂params(nœud_i) = 0 si le nœud i n'est PAS en amont d'une
             station jaugée → le NeRF ne peut pas différencier les paramètres
             de ces nœuds, ils restent à l'init.
Avec MODIS : ∂L_ET/∂params(nœud_i) ≠ 0 pour tout i où MODIS observe un pixel
             → signal de gradient distribué, identifiabilité spatiale possible.

Sortie
------
Statistiques par nœud :
  - |∂L_Q/∂K_c|   norm du gradient sous KGE/MSE/log_NSE seul
  - |∂L_ET/∂K_c|  norm du gradient sous NLL gaussien sur l'ETR
  - Histogramme séparé pour nœuds JAUGÉS (en amont d'une station) vs NON-JAUGÉS

Si MODIS aide → |∂L_ET| > 0 partout, surtout pour les nœuds non-jaugés.

Usage
-----
    python tests/scripts/gradient_diagnostic_modis.py
        .runs/slso/config/slso-kendall-gal-v2.toml
"""
import os
import sys
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[2])

import tomllib
import numpy as np
import pandas as pd
import torch

from meandre.utils.paths import run_dir_from_config, resolve_run_path

CFG_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 \
    else Path(".runs/slso/config/slso-kendall-gal-v2.toml")

with open(CFG_PATH, "rb") as f:
    cfg = tomllib.load(f)

RUN_DIR = run_dir_from_config(CFG_PATH)
def _p(k): return resolve_run_path(cfg["paths"][k], RUN_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Config  : {CFG_PATH}")
print(f"Device  : {device}")

# ── Load basin + ETR observations ───────────────────────────────────────────
from meandre.data.basin_cache import BasinCache
cache = BasinCache(_p("basin_db"))
hydro = cache.load(device=device)
n_nodes = hydro["n_nodes"]
node_coords = hydro["node_coords"]
territorial = hydro["territorial"]
graph = hydro["graph"]

DATE_START = cfg["temporal"]["date_start"]
DATE_END = cfg["temporal"]["date_end"]

if not cache.has_modis_et():
    print("ERREUR : table modis_et absente — lancer ingest_remote_sensing.py d'abord")
    sys.exit(1)

et_obs_full = cache.load_modis_et(DATE_START, DATE_END, device=device)
print(f"Nodes   : {n_nodes}")
print(f"et_obs  : {tuple(et_obs_full.shape)}  valid={(~et_obs_full.isnan()).sum().item():,}")

# ── Identify gauged vs ungauged nodes ───────────────────────────────────────
obs = cache.load_observations(DATE_START, DATE_END, min_valid_days=365)
station_node_idx = sorted(obs["station_node_map"].values())
print(f"Stations: {len(station_node_idx)} jaugées")

# Trace upstream nodes from each station (nœuds qui contribuent au débit jaugé)
gauged_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)

# Pour la première analyse : "jaugé" = nœud station directement
# (l'upstream tracing demanderait le graphe topo, simplification ici)
for sid in station_node_idx:
    if sid < n_nodes:
        gauged_mask[sid] = True

# Plus utilement : "jaugé" = nœud dont les paramètres affectent au moins une station via routing
# Simplification : marquer tous les nœuds qui sont aux N stations ET leurs voisins immédiats
print(f"Nœuds jaugés (stations directes) : {gauged_mask.sum().item()}")
print(f"Nœuds non-jaugés                 : {(~gauged_mask).sum().item()}")

# ── Build model fresh from literature init ──────────────────────────────────
from meandre.model import HydroModel
from meandre.utils.state import HydroState
from meandre.data.gridded_forcing import extract_forcing
import pandas as pd

print("\nBuilding fresh model (literature init)…")
N_FORCING = cfg["model"]["n_forcing"]
CONTEXT_WINDOW = cfg["model"]["context_window"]
RESIDUAL_HISTORY = cfg["model"]["residual_history"]
MAX_TRAVEL_DAYS = cfg["model"]["max_travel_days"]

model = HydroModel(
    n_nodes=n_nodes,
    n_forcing=N_FORCING,
    context_window=CONTEXT_WINDOW,
    residual_history=RESIDUAL_HISTORY,
    max_travel_days=MAX_TRAVEL_DAYS,
    use_temporal=False,
    use_residual=False,
    use_travel_time_attn=False,
    n_territorial_features=territorial.to_tensor().shape[1],
).to(device)
model.spatial_encoder.init_from_literature({})

# ── Load forcing for a short diagnostic window ──────────────────────────────
forcing = extract_forcing(
    zarr_path=str(_p("weather_grid")),
    node_coords=node_coords,
    node_elev=None,
    date_start=DATE_START,
    date_end=DATE_END,
    cache_nc=_p("forcing_cache"),
    device=device,
)
withdrawals = cache.load_withdrawals(DATE_START, DATE_END, device=device)
dates = pd.date_range(DATE_START, DATE_END, freq="D")
doy = torch.tensor([d.dayofyear for d in dates], dtype=torch.long, device=device)

# Window : 1 année (2001) — suffisant pour gradient diagnostic
T_START = int((pd.Timestamp("2001-01-01") - pd.Timestamp(DATE_START)).days)
T_LEN = 365
diag_slice = slice(T_START, T_START + T_LEN)

print(f"Diagnostic window: {DATE_START} +{T_START}..{T_START+T_LEN} days")

# ── Forward pass with diagnostics ───────────────────────────────────────────
state0 = HydroState.zeros(n_nodes, device=device)
result = model.simulate(
    forcing=forcing[diag_slice],
    initial_state=state0,
    graph=graph,
    node_coords=node_coords,
    territorial=territorial,
    withdrawals=withdrawals,
    day_of_year=doy[diag_slice],
    return_diagnostics=True,
)
if isinstance(result, tuple) and len(result) == 3:
    Q_sim, final_state, diag = result
else:
    print("ERREUR : simulate n'a pas retourné de diagnostics")
    sys.exit(1)

et_sim = diag.etr  # (T, n_nodes) mm/jour
et_obs = et_obs_full[diag_slice]  # (T, n_nodes)
print(f"Q_sim   : {tuple(Q_sim.shape)}")
print(f"et_sim  : {tuple(et_sim.shape)}")
print(f"et_obs  : {tuple(et_obs.shape)}  valid={(~et_obs.isnan()).sum().item():,}")

# Observed discharge (for L_Q)
q_obs_full = torch.tensor(obs["discharge"], dtype=torch.float32, device=device)
q_obs = q_obs_full[diag_slice]   # (T, n_stations_loaded)
# Map to nodes via station_node_map → (T, n_nodes) with NaN ailleurs
q_obs_nodes = torch.full((T_LEN, n_nodes), float("nan"), device=device)
for col, sid in enumerate(sorted(obs["station_node_map"].keys())):
    ni = obs["station_node_map"][sid]
    if ni < n_nodes and col < q_obs.shape[1]:
        q_obs_nodes[:, ni] = q_obs[:, ni] if ni < q_obs.shape[1] else q_obs[:, col]

# ── Compute losses ──────────────────────────────────────────────────────────
# L_Q = MSE on Q where observed (à approximer pour ce diag)
valid_q = ~q_obs_nodes.isnan()
if valid_q.any():
    L_Q = ((Q_sim - q_obs_nodes)[valid_q] ** 2).mean()
else:
    print("Aucune obs Q dans la fenêtre — utilise MSE arbitraire")
    L_Q = (Q_sim ** 2).mean() * 1e-6

# L_ET = MSE on ET where observed
valid_et = ~et_obs.isnan()
n_valid_et = valid_et.sum().item()
if n_valid_et > 0:
    L_ET = ((et_sim - et_obs)[valid_et] ** 2).mean()
    print(f"L_ET observations valides : {n_valid_et:,}")
else:
    print("Aucune obs ETR dans la fenêtre — diagnostic impossible")
    sys.exit(1)

print(f"L_Q  = {L_Q.item():.4f}")
print(f"L_ET = {L_ET.item():.4f}")

# ── Gradient computation per node ──────────────────────────────────────────
# We compute ∂L/∂spatial_params(node) — params are the output of the NeRF.
# To get one gradient per node, we need to call autograd on a per-node scalar.
# Efficient trick : sum over a known parameter (K_c) and use autograd.grad
# with `is_grads_batched=False` then aggregate.

# Easier : compute gradient w.r.t. the spatial_encoder output (n_nodes, N_PARAMS)
# and look at the magnitude per node.

# Force the spatial encoder to retain its output for backward
sp = model.spatial_encoder(node_coords, territorial.to_tensor())
sp_raw = sp.to_tensor()  # (n_nodes, 36)
# We need to re-simulate using sp_raw as the source — but model.simulate
# internally calls spatial_encoder. For a clean per-node gradient, we
# differentiate L w.r.t. fc_out.bias which has dimension N_PARAMS but
# is shared — won't give per-node info. So we go via model parameters.

# Pragmatic shortcut : compute ∂L/∂K_c at each node by inserting K_c
# explicitly as a learnable per-node tensor. Without that, autograd gives
# the chain rule through fc_out.weight × Fourier(coords) which mixes nodes.

# For this diagnostic, we use a DIFFERENT approach :
# We perturb K_c at each node individually (finite differences via vmap)
# and observe the loss change. Too expensive for 2889 nodes.

# Best : differentiate w.r.t. spatial_encoder output and use the Jacobian
# (sp_raw → loss). The Jacobian row for node i tells us how much L changes
# with sp[i] (the params for node i alone).

# Concretely, we add a per-node perturbation as a learnable tensor,
# re-simulate, and compute gradient w.r.t. that perturbation.

print("\n── Re-simulate with per-node K_c perturbation tensor ──")
# Add perturbation as a (n_nodes,) parameter, initially zero
K_c_perturb = torch.zeros(n_nodes, device=device, requires_grad=True)

# Monkey-patch the spatial encoder to add the perturbation to K_c output
# (simple, no model surgery needed for a diagnostic)
orig_forward = model.spatial_encoder.forward
def perturbed_forward(node_coords, territorial_tensor):
    out = orig_forward(node_coords, territorial_tensor)
    # SpatialParams has K_c at index — add perturbation
    out.K_c = out.K_c + K_c_perturb
    return out
model.spatial_encoder.forward = perturbed_forward

# Re-simulate with perturbation in graph
state0 = HydroState.zeros(n_nodes, device=device)
result2 = model.simulate(
    forcing=forcing[diag_slice],
    initial_state=state0,
    graph=graph,
    node_coords=node_coords,
    territorial=territorial,
    withdrawals=withdrawals,
    day_of_year=doy[diag_slice],
    return_diagnostics=True,
)
Q_sim2, _, diag2 = result2
et_sim2 = diag2.etr

# ── Compute two losses with the perturbed model ────────────────────────
if valid_q.any():
    L_Q2 = ((Q_sim2 - q_obs_nodes)[valid_q] ** 2).mean()
else:
    L_Q2 = (Q_sim2 ** 2).mean() * 1e-6

L_ET2 = ((et_sim2 - et_obs)[valid_et] ** 2).mean()

# Gradients per node via K_c_perturb
grad_Q = torch.autograd.grad(L_Q2, K_c_perturb, retain_graph=True)[0].abs()
grad_ET = torch.autograd.grad(L_ET2, K_c_perturb, retain_graph=False)[0].abs()

# ── Statistics by group ────────────────────────────────────────────────
def _stats(g, mask, label):
    sub = g[mask].cpu().numpy()
    n_nonzero = (sub > 1e-12).sum()
    return {
        "label": label,
        "n": len(sub),
        "n_nonzero": int(n_nonzero),
        "frac_nonzero": float(n_nonzero / max(len(sub), 1)),
        "mean": float(np.mean(sub)),
        "median": float(np.median(sub)),
        "p90": float(np.quantile(sub, 0.90)) if len(sub) else 0.0,
    }

print()
print("=" * 76)
print("RÉSULTATS DU DIAGNOSTIC")
print("=" * 76)

for name, grad in [("L_Q  (KGE/MSE seul)", grad_Q),
                   ("L_ET (NLL MODIS ETR)", grad_ET)]:
    print(f"\n{name} :")
    print(f"  norme totale : {grad.sum().item():.4e}")
    g_st = _stats(grad, gauged_mask, "JAUGÉS  ")
    g_un = _stats(grad, ~gauged_mask, "NON-JAUGÉS")
    for s in [g_st, g_un]:
        print(f"  {s['label']} (n={s['n']:>4}) : "
              f"non-nuls={s['n_nonzero']:>4} ({s['frac_nonzero']:>5.1%}), "
              f"mean={s['mean']:.2e}, p90={s['p90']:.2e}")

# Final verdict
g_Q_un = grad_Q[~gauged_mask].abs()
g_ET_un = grad_ET[~gauged_mask].abs()
ratio = g_ET_un.mean() / (g_Q_un.mean() + 1e-15)

print("\n" + "=" * 76)
print("VERDICT")
print("=" * 76)
print(f"Ratio |∂L_ET/∂K_c| / |∂L_Q/∂K_c| sur nœuds NON-JAUGÉS : {ratio.item():.1f}×")
if ratio > 10:
    print("→ MODIS ETR apporte un signal DRAMATIQUEMENT plus fort sur les nœuds aveugles à Q")
elif ratio > 2:
    print("→ MODIS ETR apporte un signal significatif sur les nœuds non-jaugés")
else:
    print("→ Gain marginal ; vérifier la couverture MODIS et le nombre d'obs valides")
