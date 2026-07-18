# CLAUDE.md — meandre

## What is meandre?

Differentiable end-to-end neural-physics hybrid hydrological model in PyTorch.
Reimagines Hydrotel (INRS-ETE) as a fully differentiable pipeline:
**Spatial Encoder (NeRF + z_n latents) -> Hydrotel Column (faithful physics clone) -> Routing (Muskingum-Cunge, operator mode) -> Quantile head -> Loss**

All operations are vectorised over n_nodes (river reaches/troncons).
The GRU temporal encoder and residual corrector are LEGACY (inactive, kept for checkpoint compatibility).

## Language

The codebase is in English. Comments, config, and conversation with the developer are in French.

## Code style

- **Pas d'alignement multi-espaces** sur les colonnes (listes de tuples, dicts, assignations, arguments). Un seul espace après la virgule ou le `=`. L'alignement visuel rend chaque édition pénible (il faut reformater toutes les autres lignes pour garder l'alignement) et casse les diffs. Préférer toujours `("a", 1), ("bb", 22)` à `("a",  1), ("bb", 22)`.

## Key architecture decisions

- **NeRF spatial params**: `meandre/spatial/field_network.py` maps (lon, lat, territorial_features) -> 37 hydrological parameters per node via an MLP with Fourier positional encoding (isotropic haversine projection — raw degrees caused NeRF collapse). Constraints via sigmoid/softplus. Optional per-node additive latent codes `z_n` (`use_latent_codes`) = best deterministic recipe.
- **Vertical column**: `meandre/vertical/hydrotel_column.py` orchestrates the FAITHFUL clones from `hydrotel_clone/` (ports of Hydrotel C++ 4.3.6, each validated per-UHRH to the decimal against the binary): snow degree-day modified -> frost Rankinen -> ETP (mcguinness | linacre regional | penman | oudin) -> BV3C2 soil (3 layers) -> wetland -> optional restituting aquifer / hillslope Nash UH.
- **Regional anchors** (Quebec scale-up): `[et].mode="linacre"` + `linacre_project_dir` loads the per-UHRH optimized ETP multiplier; `[snow].melt_project_dir` loads calibrated melt rates AND thresholds. LAW OF ANCHORS: anchor scalar processes, NEVER freeze the soil field (`[soil].hydrotel_calib_dir` consistently breaks the model — the NeRF needs soil freedom to compensate structural divergences).
- **Routing**: Muskingum-Cunge; `routing_mode = "operator"` (triangular solve, ~25-30x faster) is the default for training. Lakes with learned k/beta (NeRF). Message passing along topological sort.
- **Probabilistic**: quantile head K=6 (offsets from median, median = Q_sim so KGE is preserved), pinball loss on FROZEN backbone (`nll_distribution = "quantile"`, warm_start + freeze_*, `best_metric = "nll"`). Supersedes ParamNoise/ConcreteDropout/sigma head.
- **Multi-objective**: MODIS ET (`w_et`, auto-fetched per region) + GRACE TWS (`w_tws`) de-collapse the vertical partition.

## Literature-based parameter initialization

`SpatialFieldNetwork.init_from_literature()` biases `fc_out` so parameters start at public literature defaults (Rawls 1982 soil hydraulics, Hock 2003 snow, Chow 1959 Manning, FAO-56 ET) for temperate forested loam/silt_loam.
Without this, K_sat starts ~50x too high (0.5 vs 0.01 m/day), making convergence slow.

## Running

```bash
# Install
uv sync

# SLSO training (from repo root); the config picks forcing + recipe
python .runs/slso/slso.py .runs/slso/config/slso-casr-zn.toml

# Eval-only on the saved checkpoint (held-out + quantile coverage)
MEANDRE_EVAL_ONLY=1 python .runs/slso/slso.py <config.toml>

# Quebec regions: build + train + compare to the 6-member Hydrotel ensemble
python .runs/quebec/build_regions.py GASP
python .runs/quebec/build_forcing_region.py GASP
python .runs/slso/slso.py .runs/quebec/config/gasp-v7.toml
python .runs/quebec/eval_regions.py GASP

# Hydrotel C++ reference run (validation harnesses), via WSL
# (requires hydro/station.sth — an empty one is accepted)
wsl /mnt/c/.../GitHub/hydrotel/gcc/hydrotel MONT.csv

# Tests
pytest tests/ -x -q
```

## Project layout

```
meandre/
  data/           # Basin cache (DuckDB), PHYSITEL loader, forcing, regional
                  # calib loaders (hydrotel_calib: soil/linacre/melt), MODIS, GRACE
  spatial/        # NeRF field network, positional encoding, latent codes
  temporal/       # LEGACY (GRU, residual corrector) — inactive
  vertical/       # HydrotelColumn + ET modes + spatial melt
  routing/        # Graph, kinematic/operator routing, message passing, lakes
  training/       # Trainer, loss (incl. pinball), autopilot
  utils/          # State, metrics, quantile head
  model.py        # HydroModel — top-level orchestrator
hydrotel_clone/   # Faithful C++ ports + validate_*.py harnesses (per-UHRH vs binary)
.runs/slso/       # SLSO case: slso.py (training script), config/, data builders
.runs/quebec/     # Quebec scale-up: 15 regions, fleet + eval scripts
reports/          # experiment_log.md (ALL runs + verdicts), RAPPORT_QUEBEC.md
tests/            # Mirrors meandre/ structure
```

## Config

TOML configs in `.runs/slso/config/` and `.runs/quebec/config/`. Key sections:
- `[paths]`: basin_db, forcing_cache, checkpoint, fields_nc, reach_parquet
- `[model]`: use_latent_codes, spatial_melt, melt_factor_scale (legacy scalar; ignored on warm-start and when spatial_melt), n_forcing, routing/lakes
- `[et]`: mode + linacre_project_dir ; `[snow]`: melt_project_dir ; `[soil]`: hydrotel_calib_dir (do not use — see law of anchors)
- `[training]`: lr, epochs, chunk_steps, tbptt_steps, best_metric (kge_median; "nll" in quantile mode), warm_start(_from), freeze_*
- `[loss]`: MSE/log-MSE/PBIAS/peak weights, w_et/w_tws, nll_distribution/quantile_taus
- `[literature_prior]`: optional overrides for init_from_literature() targets

## Training safeguards

- **Divergence guard**: rollback to best checkpoint if loss > 3x EMA (max 3 rollbacks)
- **Autopilot**: LR plateau cuts + smart restart gated on regression AND beta/gamma drift
- **Gradient clipping**: clip_grad_norm = 1.0
- **Truncated BPTT**: detach every `tbptt_steps` (default 365)
- **Chunked gradient accumulation**: `chunk_steps` (default 180) fits 8GB VRAM
- **Warm spinup**: after epoch 0, only re-run last 90 days from cached state

## Important numerical details

- Soil psi clamp: -100 m (prevents d(psi)/d(Se) explosion as Se->0)
- Muskingum x bounded [0.01, 0.49], K bounded [4, 48] hours
- Scaling factors (sf1, sf2, sf3) cap extraction so theta stays in [0, porosity]
- spatial_melt scale = clamp(C_f/4.5, 0.15, 1.8) applied to class melt factors

## Common pitfalls

- Don't use chunk_steps > 0 with NSE/KGE loss (they need full-sequence stats). Use MSE + log-MSE + PBIAS for chunk-safe training.
- Warm-start changes LR to lr_finetune and skips warmup; `melt_factor_scale` is automatically IGNORED on warm-start (double-application bug, fixed 2026-07-13).
- In quantile mode the held-out block prints BOTH sigma-head coverages (obsolete) and "(quantile)" coverages — only the quantile lines are meaningful.
- reach_parquet `reach_id` is 1-indexed: compare with node_idx + 1.
- Dev metrics are selection metrics; only held-out 2022-2024 counts, against the FULL 6-member Hydrotel ensemble (posttraitement_{LN24HA,MG24Hx}.zarr) on common stations/days.
- Kill background fleets by killing the PARENT loop, then verify; a surviving bash loop silently relaunches trainings.
- The `physical_prior_loss` targets should be consistent with `init_from_literature()` defaults.
