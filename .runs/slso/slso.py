# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#   kernelspec:
#     display_name: meandre
#     language: python
#     name: meandre
# ---

# %% [markdown]
"""
# SLSO — Simulation

## Configuration

Le modèle Physitel est importé pour enregistrer ses informations minimales dans un fichier `{region}.duckdb`. La météo est extraite du Zarr de référence pour les coordonnées des tronçons, puis mise en cache dans un NetCDF pour accélérer les runs suivants.
"""

# %%
from pathlib import Path
import os
import sys
# Sortie en UTF-8 : sous Windows, stdout redirigé vers un fichier est en cp1252
# et plante sur les caractères non-latin1 (ex. → dans les messages autopilote).
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass
# Repo root is .runs/slso/ → repo root = parents[2]
os.chdir(Path(__file__).resolve().parents[2])
import tomllib

from meandre.utils.paths import run_dir_from_config, resolve_run_path

# ── Load config ───────────────────────────────────────────────────────────
CFG_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".runs/slso/config/slso.toml")
with open(CFG_PATH, "rb") as f:
    cfg = tomllib.load(f)

# Paths in TOML are resolved relative to the run directory (parent of config/).
# Absolute paths (e.g. Windows-style C:/... for the zarr forcing grid) are kept as-is.
RUN_DIR = run_dir_from_config(CFG_PATH)
def _p(key: str) -> Path:
    return resolve_run_path(cfg["paths"][key], RUN_DIR)

BASIN_DB       = _p("basin_db")
ZARR_PATH      = _p("weather_grid")
FORCING_CACHE  = _p("forcing_cache")
CHECKPOINT     = _p("checkpoint")
FIELDS_NC      = _p("fields_nc")
REACH_PARQUET  = _p("reach_parquet")
RUNS_DB        = _p("runs_db")

# Temporal window
DATE_START  = cfg["temporal"]["date_start"]
SPINUP_END  = cfg["temporal"]["spinup_end"]
TRAIN_START = cfg["temporal"]["train_start"]
TRAIN_END   = cfg["temporal"]["train_end"]
VAL_START   = cfg["temporal"]["val_start"]
VAL_END     = cfg["temporal"]["val_end"]
TEST_START  = cfg["temporal"].get("test_start")  # optional: held-out test
TEST_END    = cfg["temporal"].get("test_end")
DATE_END    = cfg["temporal"]["date_end"]

# Model
N_FORCING        = cfg["model"]["n_forcing"]
CONTEXT_WINDOW   = cfg["model"]["context_window"]
RESIDUAL_HISTORY = cfg["model"]["residual_history"]
MAX_TRAVEL_DAYS  = cfg["model"]["max_travel_days"]

# Training
N_EPOCHS    = cfg["training"]["n_epochs"]
LR          = cfg["training"]["lr"]
LR_FINETUNE = cfg["training"].get("lr_finetune", LR * 0.1)
LR_NEW_MULT = cfg["training"].get("lr_new_features_mult", None)
WEIGHT_DECAY = cfg["training"]["weight_decay"]
GRAD_CLIP   = cfg["training"]["grad_clip"]
WARM_START  = cfg["training"].get("warm_start", False)
WARM_START_FROM = cfg["training"].get("warm_start_from", None)
if WARM_START_FROM is not None:
    WARM_START_FROM = str(resolve_run_path(WARM_START_FROM, RUN_DIR))

ENABLE_TEMPORAL_EPOCH  = cfg["training"]["enable_temporal_epoch"]
ENABLE_RESIDUAL_EPOCH  = cfg["training"]["enable_residual_epoch"]
ENABLE_TRAVEL_EPOCH    = cfg["training"]["enable_travel_epoch"]
RESIDUAL_WARMUP_EPOCHS = cfg["training"].get("residual_warmup_epochs", 5)
TTA_WARMUP_EPOCHS      = cfg["training"].get("tta_warmup_epochs", 10)

print(f"Config loaded from {CFG_PATH}")
print(f"  Train: {TRAIN_START} – {TRAIN_END}")
print(f"  Val (dev): {VAL_START} – {VAL_END}")
if TEST_START and TEST_END:
    print(f"  Test (held-out): {TEST_START} – {TEST_END}")

# %% [markdown]
"""
Supprimer le cache DuckDB des runs précédents pour éviter les conflits de schéma après les évolutions récentes du code.
"""

# %%
import os
for f in [RUNS_DB, Path(str(RUNS_DB) + ".wal")]:
    if f.exists():
        os.remove(f)
        print(f"Removed {f}")

# %% [markdown]
"""
Ajouter les effets anthropiques (prélèvements et rejets) dans la BD.
"""

# %%
# %%
# Run once to import withdrawals into DuckDB
import duckdb
from meandre.data.basin_cache import BasinCache as _BC
_con = duckdb.connect(str(BASIN_DB), read_only=True)
_has_wd = "withdrawals" in _con.execute("SHOW TABLES").df()["name"].tolist()
_con.close()
if not _has_wd:
    _BC(BASIN_DB).import_withdrawals("notebooks/io-eau-meandre.parquet", site_col="site_id")


# %% [markdown]
"""
## Données statiques du bassin
Charge le graphe de la rivière, les indicateurs territoriaux et l'état hydrologique initial. Lors du premier run, convertit le projet PHYSITEL en cache DuckDB (`BASIN_DB`). Les runs suivants chargent rapidement depuis le cache.
"""


# %%
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from meandre.data.basin_cache import BasinCache

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

if not BASIN_DB.exists():
    raise FileNotFoundError(f"Basin cache not found: {BASIN_DB}. "
                            "Copy it or rebuild from PHYSITEL.")

cache = BasinCache(BASIN_DB)

hydro = cache.load(device=device)

graph         = hydro["graph"]
territorial   = hydro["territorial"]
node_coords   = hydro["node_coords"]
initial_state = hydro["initial_state"]
node_ids      = hydro["node_ids"]
n_nodes       = hydro["n_nodes"]

print(f"Nodes    : {n_nodes}")
print(f"Edges    : {graph.n_edges}")
print(f"Lakes    : {graph.is_lake.sum().item()}")
print(f"Lon range: {node_coords[:,0].min():.2f} to {node_coords[:,0].max():.2f}")
print(f"Lat range: {node_coords[:,1].min():.2f} to {node_coords[:,1].max():.2f}")

# %% [markdown]
"""
## Forçage météorologique
Extrait quotidiennement P / T_min / T_max du Zarr du Québec et en dérive R_n (Hargreaves–Samani), e_a (point de rosée ≈ T_min), u2 = 2 m/s. Le résultat est mis en cache dans `FORCING_CACHE` ; les runs suivants sautent la lecture du Zarr.
"""

# %%
import numpy as np
import xarray as xr
from meandre.data.gridded_forcing import extract_forcing

FORCING_CACHE.parent.mkdir(parents=True, exist_ok=True)

# [forcing] (optionnel) : surcharge R_n / e_a / u2 par ERA5-Land mesuré.
#   era5_vars = ["R_n", "e_a", "u2"]   # vide/absent = proxies dérivés de T
#   era5_cache = "data/era5"           # optionnel (défaut: <forcing_cache>/era5)
#   era5_bbox  = [-73.0, 44.5, -69.6, 47.7]  # optionnel (défaut: nœuds + marge)
_fcfg = cfg.get("forcing", {})
ERA5_VARS  = list(_fcfg.get("era5_vars", []))
ERA5_CACHE = resolve_run_path(_fcfg["era5_cache"], RUN_DIR) if _fcfg.get("era5_cache") else None
ERA5_BBOX  = tuple(_fcfg["era5_bbox"]) if _fcfg.get("era5_bbox") else None
if ERA5_VARS:
    print(f"[forcing] Surcharge ERA5-Land activée : {ERA5_VARS}")

forcing = extract_forcing(
    zarr_path   = ZARR_PATH,
    node_coords = node_coords,
    node_elev   = None,  # already computed in forcing cache
    date_start  = DATE_START,
    date_end    = DATE_END,
    cache_nc    = FORCING_CACHE,
    device      = device,
    era5_vars   = ERA5_VARS,
    era5_cache  = ERA5_CACHE,
    era5_bbox   = ERA5_BBOX,
)
# forcing: (T, N, 6)  columns = [P, T_min, T_max, R_n, u2, e_a]

# Recover dates array from the local forcing cache (chemin effectif = tag ERA5
# si surcharge active, sinon cache proxy).
from meandre.data.gridded_forcing import effective_cache_path
ds_time = xr.open_dataset(effective_cache_path(FORCING_CACHE, ERA5_VARS))
all_dates = ds_time.time.sel(time=slice(DATE_START, DATE_END)).values
ds_time.close()

print(f"Forcing shape : {tuple(forcing.shape)}")
print(f"Date range    : {all_dates[0]} to {all_dates[-1]}")

# %% [markdown]
"""
### Assurance rapide — forçage quotidien moyen sur le bassin
"""

# %%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

dates_pd = pd.DatetimeIndex(all_dates)

fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)

# Precipitation
pr_mean = forcing[:, :, 0].mean(dim=1).cpu().numpy()
axes[0].bar(dates_pd, pr_mean, width=1, color="steelblue", alpha=0.7)
axes[0].set_ylabel("P (mm/day)")
axes[0].set_title("Basin-mean daily precipitation")

# Temperature
tmin_mean = forcing[:, :, 1].mean(dim=1).cpu().numpy()
tmax_mean = forcing[:, :, 2].mean(dim=1).cpu().numpy()
axes[1].fill_between(dates_pd, tmin_mean, tmax_mean, alpha=0.4, color="tomato")
axes[1].plot(dates_pd, (tmin_mean + tmax_mean) / 2, color="tomato", lw=0.8)
axes[1].axhline(0, color="k", lw=0.5, ls="--")
axes[1].set_ylabel("T (°C)")
axes[1].set_title("Basin-mean T_min / T_max")

# Net radiation
rn_mean = forcing[:, :, 3].mean(dim=1).cpu().numpy()
axes[2].plot(dates_pd, rn_mean, color="goldenrod", lw=0.8)
axes[2].set_ylabel("R_n (MJ/m²/day)")
axes[2].set_title("Basin-mean net radiation (Hargreaves–Samani)")

axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.tight_layout()
#plt.show()

# %% [markdown]
"""
## Débit observé aux stations de jaugeage
Filtre la base de données hydrométrique du Québec pour les stations SLSO. Seules les stations avec au moins 365 jours valides dans la fenêtre de simulation sont conservées.
"""

# %%
obs = cache.load_observations(
    date_start    = DATE_START,
    date_end      = DATE_END,
    min_valid_days= 365,
)

station_node_map = obs["station_node_map"]   # {station_id: node_index}
station_indices  = sorted(set(station_node_map.values()))
n_stations       = len(station_indices)

station_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
for ni in station_indices:
    station_mask[ni] = True

# (T, n_stations) observed discharge — aligned with forcing
discharge_np  = obs["discharge"]                          # (T, N_all)
q_obs_tensor  = torch.from_numpy(discharge_np[:, station_indices]).to(device)

print(f"Stations retained: {n_stations}")
print(f"Observed Q shape : {q_obs_tensor.shape}")

# %% [markdown]
"""
### Localisation des stations de jaugeage
"""

# %%
import duckdb

sids = list(station_node_map.keys())
_con = duckdb.connect(str(BASIN_DB), read_only=True)
lons, lats, areas = [], [], []
for s in sids:
    row = _con.execute(
        "SELECT lon, lat, drainage_area_km2 FROM stations WHERE station_id = ?", [s]
    ).fetchone()
    lons.append(float(row[0]) if row else 0.0)
    lats.append(float(row[1]) if row else 0.0)
    areas.append(float(row[2]) if row else 0.0)
_con.close()

fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(
    node_coords[:, 0].cpu(), node_coords[:, 1].cpu(),
    s=2, c="lightgrey", zorder=1,
)
sc2 = ax.scatter(
    lons, lats, s=[max(10, a / 200) for a in areas],
    c="steelblue", edgecolors="k", linewidths=0.5, zorder=2,
)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title(f"SLSO network ({n_nodes} troncons) and {n_stations} gauging stations")
fig.tight_layout()
#plt.show()

# %% [markdown]
"""
## Indicateurs temporels et découpage des périodes
"""

# %%
def dates_to_slice(dates: np.ndarray, start: str, end: str) -> slice:
    days = dates.astype("datetime64[D]")
    s = int(np.searchsorted(days, np.datetime64(start, "D")))
    e = int(np.searchsorted(days, np.datetime64(end,   "D"), side="right"))
    return slice(s, e)

spinup_sl = dates_to_slice(all_dates, DATE_START,  SPINUP_END)
train_sl  = dates_to_slice(all_dates, TRAIN_START, TRAIN_END)
val_sl    = dates_to_slice(all_dates, VAL_START,   VAL_END)
spinup_steps = spinup_sl.stop   # steps before training period

doy = torch.tensor(
    [pd.Timestamp(d).day_of_year for d in all_dates],
    dtype=torch.long, device=device,
)

print(f"Spinup : {DATE_START} – {SPINUP_END}  ({spinup_sl.stop} steps)")
print(f"Train  : {TRAIN_START} – {TRAIN_END}  (steps {train_sl.start}:{train_sl.stop})")
print(f"Val    : {VAL_START} – {VAL_END}  (steps {val_sl.start}:{val_sl.stop})")

n_val = val_sl.stop - val_sl.start
print(f"Val period: {n_val} days")
print(f"q_obs_val shape: {q_obs_tensor[val_sl].shape}")
print(f"Non-NaN fraction: {(~q_obs_tensor[val_sl].isnan()).float().mean():.3f}")
print(f"Stations with any obs: {(~q_obs_tensor[val_sl].isnan()).any(dim=0).sum()}")

# %% [markdown]
"""
## Modèle HydroModel
"""

# %%
from meandre.model import HydroModel

DROPOUT = cfg["model"].get("dropout", 0.0)

# Read [soil] config: Z1 fixed, Z2/Z3 bounds + rain_hours bounds
soil_cfg = cfg.get("soil", {})
soil_z1 = soil_cfg.get("z1", 0.30)
soil_vsa_b = soil_cfg.get("vsa_b", 2.5)
soil_quickflow_reservoir = soil_cfg.get("quickflow_reservoir", False)
soil_quickflow_beta = soil_cfg.get("quickflow_beta", 0.5)
soil_separate_infil_capacity = soil_cfg.get("separate_infil_capacity", False)
soil_frozen_gate = soil_cfg.get("frozen_gate", False)
soil_bounds = {
    "z2_min":           soil_cfg.get("z2_min",          0.30),
    "z2_max":           soil_cfg.get("z2_max",          1.50),
    "z3_min":           soil_cfg.get("z3_min",          0.50),
    "z3_max":           soil_cfg.get("z3_max",          4.00),
    "rain_hours_min":   soil_cfg.get("rain_hours_min",  3.0),
    "rain_hours_max":   soil_cfg.get("rain_hours_max", 24.0),
}

model = HydroModel(
    n_nodes = n_nodes,
    n_territorial = territorial.n_features,
    n_forcing = N_FORCING,
    context_window = CONTEXT_WINDOW,
    residual_history = RESIDUAL_HISTORY,
    max_travel_time = MAX_TRAVEL_DAYS,
    use_temporal = cfg["training"].get("enable_temporal_epoch", 0) < 9999,
    use_residual = cfg["training"].get("enable_residual_epoch", 9999) < 9999,
    use_travel_time_attn = cfg["training"].get("enable_travel_epoch", 9999) < 9999,
    use_temperature = cfg.get("model", {}).get("use_temperature", True),
    dropout = DROPOUT,
    concrete_dropout = cfg["model"].get("concrete_dropout", False),
    concrete_init_p = cfg["model"].get("concrete_init_p", 0.05),
    param_mode = cfg["model"].get("param_mode", "nerf"),
    soil_z1 = soil_z1,
    soil_vsa_b = soil_vsa_b,
    soil_quickflow_reservoir = soil_quickflow_reservoir,
    soil_quickflow_beta = soil_quickflow_beta,
    soil_separate_infil_capacity = soil_separate_infil_capacity,
    soil_frozen_gate = soil_frozen_gate,
    soil_runoff_clean = soil_cfg.get("runoff_clean", False),
    soil_mode = soil_cfg.get("mode", "meandre"),
    soil_clone_substep = soil_cfg.get("clone_substep", 48),
    soil_clone_krec_init = soil_cfg.get("clone_krec_init", 1e-5),
    et_mode = cfg.get("et", {}).get("mode", "penman"),
    column_mode = cfg.get("model", {}).get("column_mode", "meandre"),
    column_theta_init_frac = cfg.get("model", {}).get("column_theta_init_frac", 0.9),
    use_frost_rankinen = cfg.get("model", {}).get("use_frost_rankinen", True),
    compile_soil = cfg.get("model", {}).get("compile_soil", False),
    compile_column = cfg.get("model", {}).get("compile_column", False),
    use_overland_uh = cfg.get("model", {}).get("use_overland_uh", False),
    use_hillslope_uh = cfg.get("model", {}).get("use_hillslope_uh", False),
    soil_bounds = soil_bounds,
    use_quantile_head = cfg.get("loss", {}).get("nll_distribution", "").lower() == "quantile",
    quantile_taus = tuple(cfg.get("loss", {}).get("quantile_taus", [0.05, 0.10, 0.25, 0.75, 0.90, 0.95])),
    use_mixture_head = cfg.get("loss", {}).get("nll_distribution", "").lower() == "mixture",
    mixture_n_components = int(cfg.get("model", {}).get("mixture_n_components", 10)),
    mixture_hidden = int(cfg.get("model", {}).get("mixture_hidden", 64)),
    use_contextual_quantile_head = cfg.get("loss", {}).get("nll_distribution", "").lower() == "contextual-quantile",
    cqh_n_features = int(cfg.get("model", {}).get("cqh_n_features", 45)),
    cqh_taus = tuple(cfg.get("model", {}).get("cqh_taus", [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])),
    cqh_hidden = int(cfg.get("model", {}).get("cqh_hidden", 64)),
    use_phenology_modulator = cfg.get("model", {}).get("use_phenology_modulator", False),
    routing_mode = cfg.get("model", {}).get("routing_mode", "level"),
    predict_lake_params = cfg.get("model", {}).get("predict_lake_params", False),
    n_coord_freqs = cfg.get("model", {}).get("n_coord_freqs", 6),
    use_latent_codes = cfg.get("model", {}).get("use_latent_codes", False),
    latent_dim = int(cfg.get("model", {}).get("latent_dim", 8)),
    latent_mode = cfg.get("model", {}).get("latent_mode", "additive"),
    routing_substeps = int(cfg.get("model", {}).get("routing_substeps", 2)),
    discharge_dependent_celerity = cfg.get("model", {}).get("discharge_dependent_celerity", False),
    dq_beta = float(cfg.get("model", {}).get("dq_beta", 0.4)),
    dq_qref_specific = float(cfg.get("model", {}).get("dq_qref_specific", 0.01)),
    pure_advection = cfg.get("model", {}).get("pure_advection", False),
    dynamic_atten = cfg.get("model", {}).get("dynamic_atten", False),
    da_beta = float(cfg.get("model", {}).get("da_beta", 2.0)),
    da_qref_specific = float(cfg.get("model", {}).get("da_qref_specific", 0.05)),
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params:,}")

# Warm-start from previous checkpoint if available
_ws_path = Path(WARM_START_FROM) if WARM_START_FROM else CHECKPOINT
if WARM_START and _ws_path.exists():
    model.load(str(_ws_path))
    LR = LR_FINETUNE
    print(f"Warm-start from {_ws_path} (lr={LR:.1e})")
    # NeRF anti-collapse: init_from_literature shrinks fc_out.weight 100× so
    # all nodes start with the same params.  If training never grew it back
    # (observed: std < 1% of range for all params), kick it up once so the
    # optimizer has gradient amplitude to work with.
    with torch.no_grad():
        w = model.spatial_encoder.fc_out.weight
        fro = w.norm().item()
        if fro < 0.3:
            w.mul_(5.0)
            print(f"  fc_out.weight collapsed (Frobenius={fro:.3f}) — "
                  f"rescaled 5× → {w.norm().item():.3f}")
else:
    # Initialise spatial parameters from public literature defaults
    # (Rawls 1982 soil hydraulics, FAO-56 ET, Chow 1959 Manning, etc.).
    # Without this, K_sat starts ~50x too high (0.5 vs 0.01 m/day for
    # loam/silt_loam) and the optimizer wastes dozens of epochs just
    # bringing soil hydraulics into a physically plausible range.
    literature_targets = cfg.get("literature_prior", cfg.get("hydrotel_prior"))
    model.spatial_encoder.init_from_literature(literature_targets)
    print("Training from scratch (literature-initialised spatial params)")

# Ancrage Hydrotel (REPRODUCE) : remplace le sol NeRF/littérature par la
# calibration Hydrotel par nœud (bv3c.csv + textures, agrégée UHRH→troncon).
# Optionnel ([soil].hydrotel_calib_dir) — point de départ, retiré pour découpler.
_calib_dir = cfg.get("soil", {}).get("hydrotel_calib_dir")
if _calib_dir and cfg.get("model", {}).get("column_mode") == "hydrotel":
    from meandre.data.hydrotel_calib import load_calibrated_soil
    _z1 = float(getattr(model.vertical_column, "z1", 0.15))
    _calib = load_calibrated_soil(_calib_dir, node_ids, _z1, device=device)
    model.vertical_column.set_calibrated_soil(_calib)
    print(f"[reproduce] sol ancre sur calibration Hydrotel : {_calib_dir}")

# Optional : freeze spatial encoder. Useful to isolate noise_head + temporal
# encoder training when the literature init is already good (cf. stfran case
# 2026-05-13 where cold-start gives val_kge=0.17 / β=0.98 and Adam overshoots).
if cfg["training"].get("freeze_spatial", False):
    for p in model.spatial_encoder.parameters():
        p.requires_grad = False
    n_frozen = sum(p.numel() for p in model.spatial_encoder.parameters())
    print(f"Spatial encoder FROZEN ({n_frozen:,} params)")

# Phase 2 probabilistic: freeze temporal encoder core (keep ConcreteDropout
# trainable for epistemic uncertainty). KGE stays at phase-1 level since
# the backbone is frozen — only sigma and dropout rate learn.
if cfg["training"].get("freeze_temporal", False):
    if model.temporal_encoder is None:
        # use_temporal=False : module GRU jamais instancié, rien à geler.
        print("Temporal encoder absent (use_temporal=False), freeze_temporal sans effet")
    else:
        for name, p in model.temporal_encoder.named_parameters():
            if "drop." not in name:  # ConcreteDropout stays trainable
                p.requires_grad = False
        n_frozen_t = sum(p.numel() for n, p in model.temporal_encoder.named_parameters()
                         if "drop." not in n and not p.requires_grad)
        n_train_drop = sum(p.numel() for n, p in model.temporal_encoder.named_parameters()
                           if "drop." in n and p.requires_grad)
        print(f"Temporal encoder FROZEN ({n_frozen_t:,} params), "
              f"ConcreteDropout trainable ({n_train_drop} params)")

# Freeze vertical column + routing (deterministic backbone).
if cfg["training"].get("freeze_backbone", False):
    for p in model.vertical_column.parameters():
        p.requires_grad = False
    for p in model.routing.parameters():
        p.requires_grad = False
    n_frozen_v = sum(p.numel() for p in model.vertical_column.parameters())
    n_frozen_r = sum(p.numel() for p in model.routing.parameters())
    print(f"Vertical column FROZEN ({n_frozen_v:,} params), "
          f"Routing FROZEN ({n_frozen_r:,} params)")

# %% [markdown]
"""
## Prélèvements et rejets
Chargement des données de prélèvement/rejet depuis le DuckDB. Les données mensuelles sont étalées uniformément sur chaque jour du mois. Convention : positif = eau ajoutée (rejet), négatif = eau retirée (pompage).
"""

# %%
withdrawals = cache.load_withdrawals(
    date_start=DATE_START, date_end=DATE_END, device=device,
)
n_wd_active = (withdrawals.net.abs() > 0).any(dim=0).sum().item()
print(f"Withdrawals: {n_wd_active} active nodes")
print(f"Net mean: {withdrawals.net.mean():.4f} m³/s")
print(f"Range: [{withdrawals.net.min():.3f}, {withdrawals.net.max():.3f}] m³/s")

# ── MODIS ETR (conditionnel — auto-fetch si absent) ────────────────────────
# Activé si w_nll_et > 0 dans le TOML. Si la table modis_et n'est pas
# dans le DuckDB, fetch automatique depuis Planetary Computer.
et_obs_tensor = None
_lcfg_early = cfg.get("loss", {})
_use_modis = _lcfg_early.get("w_nll_et", 0.0) > 0 or _lcfg_early.get("w_et", 0.0) > 0
if _use_modis:
    if not cache.has_modis_et():
        print("\n[MODIS] modis_et absent — téléchargement depuis Planetary Computer…")
        from meandre.data.modis_loader import fetch_modis_et
        import numpy as _np
        _nc = node_coords.cpu().numpy()
        _bbox = (float(_nc[:, 0].min()) - 0.1, float(_nc[:, 1].min()) - 0.1,
                 float(_nc[:, 0].max()) + 0.1, float(_nc[:, 1].max()) + 0.1)
        _df = fetch_modis_et(
            bbox=_bbox,
            date_start=DATE_START,
            date_end=DATE_END,
            node_coords=_nc,
            node_indices=_np.arange(len(_nc)),
        )
        if _df.empty:
            print("[MODIS] Avertissement : aucune donnée récupérée, w_nll_et ignoré")
            _use_modis = False
        else:
            cache.import_modis_et(_df)
            print(f"[MODIS] {len(_df):,} lignes ingérées")

    if _use_modis:
        et_obs_tensor = cache.load_modis_et(DATE_START, DATE_END, device=device)
        _n_valid = (~torch.isnan(et_obs_tensor)).sum().item() if et_obs_tensor is not None else 0
        print(f"[MODIS] et_obs chargé : {_n_valid:,} observations valides / "
              f"{et_obs_tensor.numel():,} total")
else:
    print("[MODIS] w_nll_et = 0 — MODIS ETR non utilisé")

# ── GRACE TWS (conditionnel — si w_tws > 0) ────────────────────────────────
# Anomalie de stockage total mensuelle, placée au ~15 du mois dans une série
# journalière (NaN ailleurs) — même pattern sparse que et_obs.
tws_obs_tensor = None
if _lcfg_early.get("w_tws", 0.0) > 0:
    import duckdb as _ddb, pandas as _pd
    _con = _ddb.connect(str(BASIN_DB), read_only=True)
    if "grace_tws" in [t[0] for t in _con.execute("show tables").fetchall()]:
        _g = _con.execute("select date, tws_mm from grace_tws where quality_ok = true order by date").fetchdf()
        _con.close()
        _adates = _pd.to_datetime(all_dates).normalize()
        _tws = torch.full((len(_adates),), float("nan"), device=device)
        for _dt, _val in zip(_pd.to_datetime(_g["date"]), _g["tws_mm"].values):
            _target = _pd.Timestamp(year=_dt.year, month=_dt.month, day=15)
            _dd = np.abs((_adates - _target).days.values)
            _idx = int(_dd.argmin())
            if _dd[_idx] <= 20:
                _tws[_idx] = float(_val)
        tws_obs_tensor = _tws
        print(f"[GRACE] tws_obs chargé : {int((~torch.isnan(_tws)).sum())} mois valides")
    else:
        _con.close()
        print("[GRACE] table grace_tws absente — w_tws ignoré")

# %% [markdown]
"""
## Entraînement
La boucle d'entraînement est configurée pour activer progressivement les différentes composantes du modèle : contexte temporel, correcteur de résidu et attention sur le temps de parcours. Les checkpoints sont enregistrés dans `CHECKPOINT` et les métriques de validation sont suivies dans `RUNS_DB` via `RunLogger`.
"""

# %%
from meandre.training.loss import HydroLoss
from meandre.training.trainer import Trainer, TrainingConfig, TrainingData
from meandre.training.run_logger import RunLogger
from meandre.utils.state import HydroState

# Build TrainingData — q_obs sliced so that [:n_period] covers the right rows
def make_data(period_sl: slice) -> TrainingData:
    return TrainingData(
        forcing = forcing,
        q_obs = q_obs_tensor[period_sl.start:],
        station_mask = station_mask,
        station_idx = torch.tensor(station_indices, device=device),
        graph = graph,
        node_coords = node_coords,
        territorial = territorial,
        withdrawals = withdrawals,
        day_of_year = doy,
        train_slice = period_sl,
        val_slice = period_sl,
    )

# Slice et_obs / tws_obs to align with the q_obs start offset used in make_data
_et_train = et_obs_tensor[train_sl.start:] if et_obs_tensor is not None else None
_et_val = et_obs_tensor[val_sl.start:] if et_obs_tensor is not None else None
_tws_train = tws_obs_tensor[train_sl.start:] if tws_obs_tensor is not None else None
_tws_val = tws_obs_tensor[val_sl.start:] if tws_obs_tensor is not None else None

# IHI : précalculer les indices une fois sur le forcing complet, normaliser
# z-score, sliciter aux stations. Utilisés par ContextualQuantileHead.
_ihi_full = None
if cfg.get("loss", {}).get("nll_distribution", "").lower() == "contextual-quantile":
    from meandre.temporal.indices import compute_all_indices
    print("[IHI] Compute hydrométéo indices sur full forcing...")
    with torch.no_grad():
        _idx = compute_all_indices(forcing, doy)
    # Empilement (T, N, 5) avec GDD, API, SPI, FN, SWE_proxy
    _ihi_keys = ["gdd_cum", "api_30", "spi_30", "frost_number_90", "swe_proxy"]
    _ihi_stack = torch.stack([_idx[k] for k in _ihi_keys], dim=-1)            # (T, N, 5)
    # Slice aux stations
    _station_idx_t = torch.tensor(station_indices, device=device)
    _ihi_st = _ihi_stack[:, _station_idx_t, :]                                # (T, n_st, 5)
    # Z-score normalization globale par indice (stable, comparable au fit_head)
    _ihi_mean = _ihi_st.mean(dim=(0, 1), keepdim=True)
    _ihi_std = _ihi_st.std(dim=(0, 1), keepdim=True) + 1e-6
    _ihi_full = ((_ihi_st - _ihi_mean) / _ihi_std).contiguous()
    print(f"[IHI] indices shape : {_ihi_full.shape}  (T={_ihi_full.shape[0]}, n_st={_ihi_full.shape[1]}, 5)")
_ihi_train = _ihi_full[train_sl.start:] if _ihi_full is not None else None
_ihi_val = _ihi_full[val_sl.start:] if _ihi_full is not None else None
train_data = TrainingData(
    forcing = forcing,
    q_obs = q_obs_tensor[train_sl.start:],
    station_mask = station_mask,
    station_idx = torch.tensor(station_indices, device=device),
    graph = graph,
    node_coords = node_coords,
    territorial = territorial,
    withdrawals = withdrawals,
    day_of_year = doy,
    train_slice = train_sl,
    val_slice = train_sl,
    et_obs = _et_train,
    tws_obs = _tws_train,
    indices_ihi = _ihi_train,
)
val_data = TrainingData(
    forcing = forcing,
    q_obs = q_obs_tensor[val_sl.start:],
    station_mask = station_mask,
    station_idx = torch.tensor(station_indices, device=device),
    graph = graph,
    node_coords = node_coords,
    territorial = territorial,
    withdrawals = withdrawals,
    day_of_year = doy,
    train_slice = val_sl,
    val_slice = val_sl,
    et_obs = _et_val,
    tws_obs = _tws_val,
    indices_ihi = _ihi_val,
)

# Station weights : two modes, switched by config `[training].station_weight_mode`.
#   "sqrt_area" : sqrt(area) clamped [50, 500] — favorise les grands bassins
#   "uniform"   : poids égal pour chaque station — favorise les petits bassins
# Default = "uniform" (les NSE catastrophiques observés sur petits bassins en
# 2026-05-20 venaient du sous-poids "sqrt_area"). Cf. project_meandre_probabilistic_results.
idx_to_sid = {v: k for k, v in station_node_map.items()}
sids_ordered = [idx_to_sid[ni] for ni in station_indices]
areas_ordered = [areas[sids.index(s)] for s in sids_ordered]
_station_weight_mode = cfg["training"].get("station_weight_mode", "uniform")
if _station_weight_mode == "sqrt_area":
    station_areas = torch.sqrt(torch.clamp(
        torch.tensor(areas_ordered, dtype=torch.float32, device=device),
        min=50.0, max=500.0,
    ))
else:
    station_areas = torch.ones(n_stations, dtype=torch.float32, device=device)
print(f"station_weight_mode = {_station_weight_mode}")

# Variance of observed discharge per station — normalises MSE so each station
# contributes equally to the loss, regardless of its magnitude.  This is
# equivalent to (1-NSE) per station but chunk-safe (the denominator is fixed).
station_var = torch.zeros(n_stations, dtype=torch.float32, device=device)
for i in range(n_stations):
    mask = ~torch.isnan(q_obs_tensor[:, i])
    if mask.sum() > 30:
        station_var[i] = q_obs_tensor[mask, i].var()
    else:
        station_var[i] = 1.0  # fallback for stations with few observations

print(f"station_var range: {station_var.min():.1f} - {station_var.max():.1f}")

# Seuil pics par station : Q_p75 climato sur la période d'entraînement.
# Utilisé par peak_weighted_mse_loss si w_peak > 0. Empêche le peak-shaving
# diagnostiqué dans le PIT val 2019-2021 (surplus u > 0.75 = pics sous-prédits).
peak_threshold = torch.zeros(n_stations, dtype=torch.float32, device=device)
q_train_slice = q_obs_tensor[train_sl]
for i in range(n_stations):
    mask = ~torch.isnan(q_train_slice[:, i])
    if mask.sum() > 30:
        peak_threshold[i] = torch.quantile(q_train_slice[mask, i], 0.75)
    else:
        peak_threshold[i] = float("inf")  # désactive si pas assez de données
n_finite = (peak_threshold < float("inf")).sum().item()
print(f"peak_threshold (Q_p75 train) range: {peak_threshold[peak_threshold < float('inf')].min():.2f} - "
      f"{peak_threshold[peak_threshold < float('inf')].max():.2f}  ({n_finite}/{n_stations} stations)")

lcfg = cfg["loss"]

# ── Distribution probabiliste : "normal" | "log-normal" | "box-cox" ─────
# Mappe vers lambda Box-Cox (1 = linéaire/normal, 0 = log-normal, 0.3 standard hydro)
_dist = str(lcfg.get("nll_distribution", "normal")).lower()
if _dist == "normal":
    _nll_lambda = 1.0
elif _dist == "log-normal":
    _nll_lambda = 0.0
elif _dist == "box-cox":
    _nll_lambda = float(lcfg.get("nll_box_cox_lambda", 0.3))
elif _dist == "student-t":
    # Student-t en espace Box-Cox (queues lourdes + variance stabilisée).
    # ν appris via noise_head.log_df. λ configurable, défaut 0.3 (standard hydro).
    _nll_lambda = float(lcfg.get("nll_box_cox_lambda", 0.3))
elif _dist == "quantile":
    # Régression quantile (Phase 2 v2) : pas de NLL, pinball loss en m³/s.
    # λ ignoré (les quantiles vivent en linéaire). w_quantile pilote.
    _nll_lambda = 1.0
elif _dist == "mixture":
    # MDN (option 2b) : densité conditionnelle Σ_k π_k N(μ_k, σ_k²).
    # En linéaire (pas de Box-Cox). w_mixture pilote.
    _nll_lambda = 1.0
elif _dist == "contextual-quantile":
    # ContextualQuantileHead (IHI, Phase A) : K quantiles non-paramétriques
    # avec features enrichies (spatial_params, Q_sim, log Q_sim, indices IHI,
    # DOY sin/cos). Médiane libre. w_quantile pilote (pinball loss).
    _nll_lambda = 1.0
else:
    raise ValueError(
        f"nll_distribution invalide : {_dist!r} "
        "(attendu : 'normal', 'log-normal', 'box-cox', 'student-t', 'quantile', 'mixture', "
        "'contextual-quantile')"
    )
print(f"[loss] nll_distribution = {_dist}  (lambda Box-Cox = {_nll_lambda})")

loss_fn = HydroLoss(
    w_nse=lcfg["w_nse"], w_kge=lcfg["w_kge"], w_pbias=lcfg["w_pbias"],
    w_mse=lcfg["w_mse"], w_nrmse=lcfg["w_nrmse"],
    w_log_nse=lcfg["w_log_nse"], w_log_mse=lcfg["w_log_mse"],
    w_nll=lcfg.get("w_nll", 0.0),
    w_nll_et=lcfg.get("w_nll_et", 0.0),
    w_nll_swe=lcfg.get("w_nll_swe", 0.0),
    w_et=lcfg.get("w_et", 0.0),
    w_snow=lcfg.get("w_snow", 0.0),
    w_tws=lcfg.get("w_tws", 0.0),
    w_quantile=lcfg.get("w_quantile", 0.0),
    w_mixture=lcfg.get("w_mixture", 0.0),
    w_peak=lcfg.get("w_peak", 0.0),
    nll_distribution=_dist,
    w_flatness=lcfg.get("w_flatness", 0.0),
    nll_lambda=_nll_lambda,
    flatness_n_bins=int(lcfg.get("flatness_n_bins", 21)),
    flatness_bandwidth=float(lcfg.get("flatness_bandwidth", 0.02)),
    w_physics=lcfg["w_physics"], w_residual=lcfg["w_residual"],
    per_station=True, station_weights=station_areas,
    station_var=station_var,
    peak_threshold=peak_threshold if lcfg.get("w_peak", 0.0) > 0 else None,
)

tcfg = cfg["training"]

train_cfg = TrainingConfig(
    lr = LR,
    weight_decay = WEIGHT_DECAY,
    grad_clip = GRAD_CLIP,
    n_epochs = N_EPOCHS,
    spinup_steps = spinup_steps,
    warm_spinup_steps = 90,  # after epoch 0, reuse cached state + 90-day warm spinup
    tbptt_steps = tcfg["tbptt_steps"],
    chunk_steps = tcfg["chunk_steps"],
    val_every = tcfg["val_every"],
    enable_temporal_context_epoch = ENABLE_TEMPORAL_EPOCH,
    enable_residual_corrector_epoch = ENABLE_RESIDUAL_EPOCH,
    residual_warmup_epochs = RESIDUAL_WARMUP_EPOCHS,
    enable_travel_time_attn_epoch = ENABLE_TRAVEL_EPOCH,
    tta_warmup_epochs = TTA_WARMUP_EPOCHS,
    patience = tcfg["patience"],
    best_metric = tcfg.get("best_metric", "nse"),
    best_metric_tolerance = tcfg.get("best_metric_tolerance", 0.005),
    warmup_epochs = cfg["training"].get("warmup_epochs", 0 if WARM_START else 5),
    lr_new_features_mult = LR_NEW_MULT if WARM_START else None,
    compile_modules = tcfg.get("compile_modules", False),
    w_prior = tcfg.get("w_prior", 0.0),
    w_diversity = tcfg.get("w_diversity", 0.0),
    diversity_cv_target = tcfg.get("diversity_cv_target", 0.12),
    w_latent_reg = tcfg.get("w_latent_reg", 1e-3),
    latent_lr_mult = tcfg.get("latent_lr_mult", 50.0),
    w_boundary = tcfg.get("w_boundary", 0.0),
    w_sigma_anchor = tcfg.get("w_sigma_anchor", 0.0),
    sigma_anchor_target_a = tcfg.get("sigma_anchor_target_a", -3.0),
    sigma_anchor_target_b = tcfg.get("sigma_anchor_target_b", None),
    sigma_anchor_var_weight = tcfg.get("sigma_anchor_var_weight", 0.0),
    w_concrete_kl = tcfg.get("w_concrete_kl", 0.0),
    eta_min_factor = tcfg.get("eta_min_factor", 0.01),
    # Autopilot
    autopilot = tcfg.get("autopilot", False),
    autopilot_grace_epochs = tcfg.get("autopilot_grace_epochs", 0),
    autopilot_beta_threshold = tcfg.get("autopilot_beta_threshold", 0.15),
    autopilot_beta_penalty = tcfg.get("autopilot_beta_penalty", 0.005),
    autopilot_gamma_threshold = tcfg.get("autopilot_gamma_threshold", 0.20),
    autopilot_gamma_penalty = tcfg.get("autopilot_gamma_penalty", 0.003),
    autopilot_lr_patience = tcfg.get("autopilot_lr_patience", 8),
    autopilot_lr_factor = tcfg.get("autopilot_lr_factor", 0.5),
    autopilot_lr_min = tcfg.get("autopilot_lr_min", 1e-6),
    autopilot_restart_regression = tcfg.get("autopilot_restart_regression", 0.05),
    autopilot_restart_max = tcfg.get("autopilot_restart_max", 3),
    autopilot_restart_min_no_improve = tcfg.get("autopilot_restart_min_no_improve", 3),
    autopilot_activate_residual_at_kge = tcfg.get("autopilot_activate_residual_at_kge", None),
    autopilot_activate_tta_at_kge = tcfg.get("autopilot_activate_tta_at_kge", None),
    autopilot_unfreeze_spatial_epoch = tcfg.get("autopilot_unfreeze_spatial_epoch", None),
    autopilot_unfreeze_spatial_min_kge = tcfg.get("autopilot_unfreeze_spatial_min_kge", None),
    autopilot_unfreeze_spatial_lr_factor = tcfg.get("autopilot_unfreeze_spatial_lr_factor", 0.05),
    autopilot_nll = tcfg.get("autopilot_nll", False),
    autopilot_nll_initial_kge = tcfg.get("autopilot_nll_initial_kge", None),
    autopilot_nll_max_regression = tcfg.get("autopilot_nll_max_regression", 0.05),
    autopilot_nll_ramp_rate = tcfg.get("autopilot_nll_ramp_rate", 1.5),
    autopilot_nll_max = tcfg.get("autopilot_nll_max", 0.5),
    autopilot_nll_min = tcfg.get("autopilot_nll_min", 0.001),
    # Kendall-Gal auto phase 1→2 — trigger params in [training], overrides in [phase2]
    kendall_gal_auto = tcfg.get("kendall_gal_auto", False),
    kga_phase1_kge_threshold = tcfg.get("kga_phase1_kge_threshold", 0.85),
    kga_phase1_plateau_patience = tcfg.get("kga_phase1_plateau_patience", 15),
    kga_phase1_min_epochs = tcfg.get("kga_phase1_min_epochs", 5),
    kga_phase2_reset_no_improve = tcfg.get("kga_phase2_reset_no_improve", True),
    kga_phase2_freeze_spatial = (cfg.get("phase2") or {}).get("freeze_spatial", True),
    kga_phase2_freeze_temporal = (cfg.get("phase2") or {}).get("freeze_temporal", True),
    kga_phase2_freeze_backbone = (cfg.get("phase2") or {}).get("freeze_backbone", True),
    kga_phase2_best_metric = (cfg.get("phase2") or {}).get("best_metric", "nll"),
    kga_phase2_lr = (cfg.get("phase2") or {}).get("lr", None),
    kga_phase2_loss_weights = (cfg.get("phase2") or {}).get("loss", None),
)

if train_cfg.autopilot:
    nll_info = ""
    if train_cfg.autopilot_nll:
        nll_info = f", NLL_auto(ramp×{train_cfg.autopilot_nll_ramp_rate}, max_reg={train_cfg.autopilot_nll_max_regression:.0%})"
    print(f"  Autopilot ON -- beta_thr={train_cfg.autopilot_beta_threshold}, "
          f"gamma_thr={train_cfg.autopilot_gamma_threshold}, "
          f"LR_patience={train_cfg.autopilot_lr_patience}, "
          f"grace={train_cfg.autopilot_grace_epochs}{nll_info}")

CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)

run_logger = RunLogger(RUNS_DB)

trainer = Trainer(
    model = model,
    loss_fn = loss_fn,
    train_data = train_data,
    val_data = val_data,
    config = train_cfg,
    run_name = f"slso_{TRAIN_START[:4]}_{TRAIN_END[:4]}",
    run_logger = run_logger,
    checkpoint_path = str(CHECKPOINT),
)
# Mode éval seule : MEANDRE_EVAL_ONLY=1 saute l'entraînement et va directement
# aux blocs d'évaluation (test held-out + couverture) avec le checkpoint sauvé.
import os as _os
EVAL_ONLY = _os.environ.get("MEANDRE_EVAL_ONLY") == "1" or cfg["training"].get("eval_only", False)
if EVAL_ONLY:
    print("[eval-only] entraînement sauté — évaluation du checkpoint existant")
else:
    trainer.fit()

# %% [markdown]
"""
## Held-out TEST evaluation (truly unseen)
Charge le best.pt sélectionné sur la période dev, l'évalue sur le test
(jamais vu pendant training). Métrique honnête de performance.
"""

# %%
if TEST_START and TEST_END:
    model.load(str(CHECKPOINT))
    model.eval()
    test_sl = dates_to_slice(all_dates, TEST_START, TEST_END)
    print(f"\n{'='*72}")
    print(f"  HELD-OUT TEST : {TEST_START} -> {TEST_END}  (steps {test_sl.start}:{test_sl.stop})")
    print(f"{'='*72}")
    # Simulate full forcing, then slice test period
    with torch.no_grad():
        Q_test_full, _ = model.simulate(
            forcing=forcing, initial_state=HydroState.zeros(n_nodes, device=device),
            graph=graph, node_coords=node_coords, territorial=territorial,
            withdrawals=withdrawals, day_of_year=doy,
        )
    Q_test = Q_test_full[test_sl.start:test_sl.stop, station_mask].cpu()
    q_obs_test = q_obs_tensor[test_sl.start:test_sl.stop].cpu()
    n_test = min(Q_test.shape[0], q_obs_test.shape[0])
    Q_test = Q_test[:n_test]; q_obs_test = q_obs_test[:n_test]
    from meandre.utils.metrics import kge as _kge_fn, kge_components as _kgec, nse as _nse
    test_kges = []
    for s in range(Q_test.shape[1]):
        v = ~torch.isnan(q_obs_test[:, s]) & ~torch.isnan(Q_test[:, s])
        if v.sum() < 30:
            continue
        test_kges.append(float(_kge_fn(q_obs_test[v, s], Q_test[v, s])))
    import numpy as _np
    test_kges = _np.array(test_kges)
    # Pooled
    qo_flat = q_obs_test.reshape(-1); qs_flat = Q_test.reshape(-1)
    v = ~torch.isnan(qo_flat) & ~torch.isnan(qs_flat)
    pooled_kge = float(_kge_fn(qo_flat[v], qs_flat[v]))
    print(f"  Test KGE pooled        : {pooled_kge:.4f}")
    print(f"  Test KGE per-station median : {_np.median(test_kges):.4f}")
    print(f"  Test KGE per-station mean   : {test_kges.mean():.4f}")
    print(f"  Stations with KGE > 0.5: {(test_kges > 0.5).sum()}/{len(test_kges)}")
    print(f"  Stations with KGE < 0  : {(test_kges < 0).sum()}/{len(test_kges)}")

    # ── Couverture probabiliste Box-Cox sur le test (noise head) ────────────
    if hasattr(model, "noise_head") and model.noise_head is not None:
        from meandre.utils.noise_head import SpatialNoiseHead
        from meandre.training.loss import box_cox as _bc, gaussian_nll_loss as _gnll
        _lam = (cfg["loss"].get("nll_box_cox_lambda", 0.3)
                if cfg["loss"].get("nll_distribution") == "box-cox" else 1.0)
        with torch.no_grad():
            if isinstance(model.noise_head, SpatialNoiseHead):
                _sp = model.spatial_encoder(node_coords, territorial.to_tensor())
                _ls_full = model.noise_head(_sp.to_tensor(), Q_test_full.detach())
            else:
                _ls_full = model.noise_head(Q_test_full.detach())
        _ls = _ls_full[test_sl.start:test_sl.stop, station_mask].cpu()[:n_test]  # (n_test, n_st)
        _sig = _ls.exp()
        _qo_t = _bc(q_obs_test, _lam) if _lam != 1.0 else q_obs_test
        _qs_t = _bc(Q_test, _lam) if _lam != 1.0 else Q_test
        _valid = ~torch.isnan(_qo_t) & ~torch.isnan(_qs_t)
        print(f"  -- Couverture Box-Cox(lam={_lam}) sur le test --")
        for _lvl, _z in [(50, 0.674), (90, 1.645)]:
            _lo = _qs_t - _z * _sig
            _hi = _qs_t + _z * _sig
            _inint = (_qo_t >= _lo) & (_qo_t <= _hi)
            _cov = (_inint & _valid).sum().float() / _valid.sum().float()
            print(f"  Test cov_{_lvl}: {float(_cov):.4f}  (cible {_lvl/100:.2f})")
        _nll = _gnll(q_obs_test[_valid], Q_test[_valid], _ls[_valid], lam=_lam)
        print(f"  Test NLL (box-cox): {float(_nll):.4f}   σ_mean: {float(_sig[_valid].mean()):.3f}")

# %% [markdown]
"""
## Full-period simulation
Run the model over the complete window using the best checkpoint.
"""

# %%
model.load(str(CHECKPOINT))
model.eval()
print(f"temporal={model.use_temporal}, residual={model.use_residual}, tta={model.routing.use_tta}")

with torch.no_grad():
    Q_sim, _ = model.simulate(
        forcing       = forcing,
        initial_state = initial_state,
        graph         = graph,
        node_coords   = node_coords,
        territorial   = territorial,
        withdrawals   = withdrawals,
        day_of_year   = doy,
    )

print(f"Q_sim shape: {tuple(Q_sim.shape)}")  # (T, N)

# %% [markdown]
"""
### Deep diagnostics — water balance internals
"""

# %%
# Re-run with diagnostics to get intermediate fluxes
with torch.no_grad():
    Q_sim_d, _, diag = model.simulate(
        forcing       = forcing,
        initial_state = initial_state,
        graph         = graph,
        node_coords   = node_coords,
        territorial   = territorial,
        withdrawals   = withdrawals,
        day_of_year   = doy,
        return_diagnostics = True,
    )

# Also get spatial params
with torch.no_grad():
    sp = model.spatial_encoder(node_coords, territorial.to_tensor())

print("=" * 60)
print("SPATIAL PARAMETERS (median across nodes)")
print("=" * 60)
print(f"  K_sat_1:   {sp.K_sat_1.median():.4f} m/day ({sp.K_sat_1.median()*1000:.0f} mm/day)")
print(f"  K_sat_2:   {sp.K_sat_2.median():.4f} m/day")
print(f"  K_sat_3:   {sp.K_sat_3.median():.4f} m/day")
print(f"  porosity_1: {sp.porosity_1.median():.3f}")
print(f"  theta_fc_1: {sp.theta_fc_1.median():.3f}")
print(f"  theta_wp_1: {sp.theta_wp_1.median():.3f}")
print(f"  C_f:        {sp.C_f.median():.2f} mm/C/day (range {sp.C_f.min():.2f}-{sp.C_f.max():.2f})")
print(f"  T_melt:     {sp.T_melt.median():.2f} C (range {sp.T_melt.min():.2f}-{sp.T_melt.max():.2f})")
print(f"  T_snow:     {sp.T_snow.median():.2f} C")
print(f"  frost_alpha: {sp.frost_alpha.median():.3f}")
print(f"  f_wetland:  {sp.f_wetland.median():.3f}")
print(f"  interception: {sp.interception_capacity.median():.2f} mm")
print(f"  k_gw:       {sp.k_gw.median():.4f} /day")
print(f"  T_gw:       {sp.T_gw.median():.1f} C")
print(f"  K_atm:      {sp.K_atm.median():.3f} /day")

print(f"\n  K_musk: {sp.K_musk_hours.min():.1f}-{sp.K_musk_hours.median():.1f}-{sp.K_musk_hours.max():.1f} hours")
print(f"  x_musk: {sp.x_musk.median():.3f}")
print(f"  vg_n: {sp.vg_n.median():.3f}")
print(f"  f_vert: L1={sp.f_vert_1.median():.3f}  L2={sp.f_vert_2.median():.3f}  L3={sp.f_vert_3.median():.3f}")

print("\n" + "=" * 60)
print("WATER BALANCE (basin-mean mm/day)")
print("=" * 60)
P_mean = forcing[:, :, 0].mean(dim=1).cpu()    # (T,) precip
etp    = diag.etp.mean(dim=1).cpu()
etr    = diag.etr.mean(dim=1).cpu()
snowmelt_d = diag.snowmelt.mean(dim=1).cpu()
lat_mm = diag.lateral_mm.mean(dim=1).cpu()
q_bf   = diag.q_baseflow.mean(dim=1).cpu()

print(f"  Precipitation: {P_mean.mean():.2f} mm/day")
print(f"  ETP:           {etp.mean():.2f} mm/day")
print(f"  ETR:           {etr.mean():.2f} mm/day")
print(f"  Snowmelt:      {snowmelt_d.mean():.2f} mm/day")
print(f"  Lateral inflow:{lat_mm.mean():.2f} mm/day")
print(f"  Q_baseflow:    {q_bf.mean():.2f} mm/day")
print(f"  Runoff ratio:  {lat_mm.mean() / (P_mean.mean() + 1e-8):.2f}")

# Seasonal breakdown
months = np.array([pd.Timestamp(d).month for d in all_dates])
print("\n  --- Seasonal means (mm/day) ---")
for label, mos in [("Winter DJF", [12,1,2]), ("Spring MAM", [3,4,5]),
                   ("Summer JJA", [6,7,8]), ("Autumn SON", [9,10,11])]:
    mask = np.isin(months, mos)
    if mask.sum() == 0:
        continue
    print(f"  {label}: P={P_mean[mask].mean():.1f}  ETR={etr[mask].mean():.1f}  "
          f"melt={snowmelt_d[mask].mean():.1f}  lateral={lat_mm[mask].mean():.1f}  "
          f"baseflow={q_bf[mask].mean():.2f}")

print("\n" + "=" * 60)
print("ROUTING CHECK")
print("=" * 60)
area_local_t = territorial.get_physical("area_km2_local")
if area_local_t is not None:
    area_local = area_local_t.cpu()
    lat_m3s = (diag.lateral_mm.cpu() * 1e-3 * area_local.unsqueeze(0) * 1e6 / 86400.0)
    total_lat = lat_m3s.sum(dim=1)
    for sid in ["023402"]:
        if sid in station_node_map:
            ni = station_node_map[sid]
            col = station_indices.index(ni)
            q_s = Q_sim_d[:, ni].cpu()
            q_lat_up = lat_m3s[:, ni]
            print(f"  Station {sid} (node {ni}):")
            print(f"    Mean Q_sim:      {q_s.mean():.1f} m³/s")
            print(f"    Mean local lat:  {q_lat_up.mean():.2f} m³/s")
            print(f"    Total basin lat: {total_lat.mean():.1f} m³/s")
            n_upstream = (graph.edge_index[1] == ni).sum() if graph.n_edges > 0 else 0
            print(f"    Direct upstream reaches: {n_upstream}")

print("\n" + "=" * 60)
print("FLOW ACCUMULATION CHECK")
print("=" * 60)
for sid in list(station_node_map.keys())[:6]:
    ni = station_node_map[sid]
    upstream = set()
    queue = [ni]
    edge_src = graph.edge_index[0].cpu().numpy() if graph.n_edges > 0 else np.array([])
    edge_dst = graph.edge_index[1].cpu().numpy() if graph.n_edges > 0 else np.array([])
    while queue:
        node = queue.pop(0)
        for s, d in zip(edge_src, edge_dst):
            if d == node and s not in upstream:
                upstream.add(s)
                queue.append(s)
    upstream.add(ni)
    upstream_list = sorted(upstream)
    cum_lat = lat_m3s[:, upstream_list].sum(dim=1).mean()
    q_sim_station = Q_sim_d[:, ni].cpu().mean()
    q_obs_mean = q_obs_tensor[:, station_indices.index(ni)].cpu()
    q_obs_mean = q_obs_mean[~q_obs_mean.isnan()].mean()
    area_obs = areas[sids.index(sid)] if sid in sids else 0
    area_sim = area_local[upstream_list].sum() if area_local_t is not None else 0
    print(f"  {sid}: {len(upstream)} upstream nodes, "
          f"area_sim={area_sim:.0f} km², area_obs={area_obs:.0f} km², "
          f"lat_sum={cum_lat:.1f} m³/s, Q_sim={q_sim_station:.1f} m³/s, "
          f"Q_obs={q_obs_mean:.1f} m³/s")

# %% [markdown]
"""
## Save results
"""


# %%
FIELDS_NC.parent.mkdir(parents=True, exist_ok=True)
REACH_PARQUET.parent.mkdir(parents=True, exist_ok=True)

# ── Raster fields (time, node) → NetCDF ──────────────────────────────────
field_vars = {
    "etp":        ("mm/day", "Potential evapotranspiration (Penman-Monteith)"),
    "etr":        ("mm/day", "Actual evapotranspiration (canopy + soil)"),
    "recharge":   ("mm/day", "Deep drainage from soil L3 into aquifer"),
    "q_baseflow": ("mm/day", "Groundwater baseflow (aquifer output)"),
    "snowmelt":   ("mm/day", "Snow melt flux"),
    "lateral_mm": ("mm/day", "Lateral inflow to routing (surface + interflow + baseflow)"),
}
data_vars = {}
for name, (unit, long_name) in field_vars.items():
    arr = getattr(diag, name).cpu().numpy().astype("float32")
    data_vars[name] = xr.DataArray(
        arr, dims=["time", "node"],
        attrs={"units": unit, "long_name": long_name},
    )

ds_fields = xr.Dataset(
    data_vars,
    coords={"time": all_dates, "node": np.array(node_ids, dtype="int32")},
    attrs={
        "title": f"SLSO meandre fields {DATE_START} to {DATE_END}",
        "model": "HydroModel",
    },
)
ds_fields.to_netcdf(FIELDS_NC)
print(f"Fields saved to {FIELDS_NC}  ({len(field_vars)} variables)")

# ── Reach-level tabular data → Parquet ────────────────────────────────────
# All reaches (troncons): Q_sim and T_water at every node, every timestep.
Q_np = Q_sim.cpu().numpy().astype("float32")
reach_data = {
    "date":      np.repeat(all_dates, n_nodes),
    "reach_id":  np.tile(np.array(node_ids, dtype="int32"), len(all_dates)),
    "Q_sim_m3s": Q_np.ravel(),
}
if diag.T_water is not None:
    reach_data["T_water_C"] = diag.T_water.cpu().numpy().astype("float32").ravel()

df_reach = pd.DataFrame(reach_data)
df_reach.to_parquet(REACH_PARQUET, index=False)
print(f"Reach saved to {REACH_PARQUET}  ({len(df_reach)} rows, {n_nodes} reaches)")

# %% [markdown]
"""
## Evaluation at gauging stations
### Global metrics
"""

# %%
from meandre.utils.metrics import nse, kge, pbias

q_sim_gauged = Q_sim[:, station_mask].cpu()
q_obs_gauged = q_obs_tensor[:len(Q_sim), :].cpu()

results = []
for i, sid in enumerate(sids):
    q_o = q_obs_gauged[:, i]
    q_s = q_sim_gauged[:, station_indices.index(station_node_map[sid])]
    valid = ~torch.isnan(q_o)
    if valid.sum() < 30:
        continue
    q_o_v, q_s_v = q_o[valid], q_s[valid]
    results.append({
        "station": sid,
        "node":    station_node_map[sid],
        "NSE":     float(nse(q_o_v, q_s_v)),
        "KGE":     float(kge(q_o_v, q_s_v)),
        "PBIAS":   float(pbias(q_o_v, q_s_v)),
        "n_days":  int(valid.sum()),
    })

import pandas as pd
df_metrics = pd.DataFrame(results).sort_values("NSE", ascending=False)
print(df_metrics.to_string(index=False, float_format="{:.3f}".format))

# %% [markdown]
"""
### Hydrographs
Top stations by drainage area
"""

# %%
top_sids = sorted(sids, key=lambda s: -areas[sids.index(s)])[:6]
fig, axes = plt.subplots(len(top_sids), 1, figsize=(12, 3 * len(top_sids)), sharex=True)
if len(top_sids) == 1:
    axes = [axes]

for ax, sid in zip(axes, top_sids):
    ni   = station_node_map[sid]
    col  = station_indices.index(ni)
    q_o  = q_obs_gauged[:, col].numpy()
    q_s  = q_sim_gauged[:, col].numpy()
    valid= ~np.isnan(q_o)

    ax.plot(dates_pd, q_o, color="steelblue", lw=0.8, label="Observed", alpha=0.8)
    ax.plot(dates_pd, q_s, color="tomato",    lw=0.8, label="Simulated", alpha=0.8)
    ax.set_ylabel("Q (m³/s)")
    ax.set_title(f"Station {sid}  (troncon {ni})")
    ax.legend(loc="upper right", fontsize=8)

axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.tight_layout()
fig.savefig(FIELDS_NC.parent / "hydrographs.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
"""
### NSE distribution across stations
"""

# %%
fig, ax = plt.subplots(figsize=(7, 4))
nse_vals = df_metrics["NSE"].values
ax.hist(nse_vals, bins=20, color="steelblue", edgecolor="k", alpha=0.7)
ax.axvline(np.median(nse_vals), color="tomato", lw=1.5, ls="--",
           label=f"Median NSE = {np.median(nse_vals):.2f}")
ax.set_xlabel("NSE")
ax.set_ylabel("Number of stations")
ax.set_title(f"NSE distribution — SLSO {DATE_START} to {DATE_END}")
ax.legend()
fig.tight_layout()
fig.savefig(FIELDS_NC.parent / "nse_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
"""
### Diagnostic — volume bias, peak timing, and flow components
"""

# %%
diag_rows = []
for i, sid in enumerate(sids):
    ni = station_node_map[sid]
    col = station_indices.index(ni)
    q_o = q_obs_gauged[:, col]
    q_s = q_sim_gauged[:, col]
    valid = ~torch.isnan(q_o)
    if valid.sum() < 30:
        continue
    q_o_v = q_o[valid].numpy()
    q_s_v = q_s[valid].numpy()

    vol_ratio = q_s_v.sum() / (q_o_v.sum() + 1e-8)
    peak_obs = q_o_v.max()
    peak_sim = q_s_v.max()
    peak_ratio = peak_sim / (peak_obs + 1e-8)

    peak_obs_idx = q_o_v.argmax()
    lo = max(0, peak_obs_idx - 30)
    hi = min(len(q_s_v), peak_obs_idx + 31)
    peak_sim_idx = lo + q_s_v[lo:hi].argmax()
    peak_lag = int(peak_sim_idx - peak_obs_idx)

    q_o_sorted = np.sort(q_o_v)
    q_s_sorted = np.sort(q_s_v)
    n = len(q_o_v)
    bfi_obs = q_o_sorted[int(0.1 * n)] / (q_o_sorted[int(0.5 * n)] + 1e-8)
    bfi_sim = q_s_sorted[int(0.1 * n)] / (q_s_sorted[int(0.5 * n)] + 1e-8)

    diag_rows.append({
        "station": sid,
        "area_km2": areas[sids.index(sid)],
        "NSE": float(nse(torch.tensor(q_o_v), torch.tensor(q_s_v))),
        "vol_ratio": vol_ratio,
        "peak_ratio": peak_ratio,
        "peak_lag_d": peak_lag,
        "bfi_obs": bfi_obs,
        "bfi_sim": bfi_sim,
    })

df_diag = pd.DataFrame(diag_rows).sort_values("area_km2", ascending=False)
print(df_diag.to_string(index=False, float_format="{:.2f}".format))

print("\n── Summary ──")
print(f"Median vol_ratio : {df_diag['vol_ratio'].median():.2f}  (1.0 = perfect)")
print(f"Median peak_ratio: {df_diag['peak_ratio'].median():.2f}  (1.0 = perfect)")
print(f"Median peak_lag  : {df_diag['peak_lag_d'].median():.0f} days")
print(f"Median NSE       : {df_diag['NSE'].median():.3f}")
print(f"Mean bfi_obs     : {df_diag['bfi_obs'].mean():.2f}")
print(f"Mean bfi_sim     : {df_diag['bfi_sim'].mean():.2f}  (higher = too much baseflow)")

# %%