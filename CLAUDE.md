# CLAUDE.md — meandre

## What is meandre?

Differentiable end-to-end neural-physics hybrid hydrological model in PyTorch.
Reimagines Hydrotel (INRS-ETE) as a fully differentiable pipeline:
**Spatial Encoder (NeRF) -> Temporal Context (GRU) -> Vertical Column (Physics) -> Residual Corrector -> Routing (Muskingum-Cunge) -> Loss**

All operations are vectorised over n_nodes (river reaches/troncons).

## Language

The codebase is in English. Comments, config, and conversation with the developer are in French.

## Key architecture decisions

- **NeRF spatial params**: `meandre/spatial/field_network.py` maps (lon, lat, territorial_features) -> 32 hydrological parameters per node via an MLP with Fourier positional encoding. Constraints via sigmoid/softplus ensure physical plausibility.
- **Vertical column**: `meandre/vertical/column.py` chains Snow -> Frost -> Interception -> ET (Penman-Monteith) -> Soil (3-layer van Genuchten BV3C2) -> Wetland -> Aquifer. All differentiable.
- **Soil module**: `meandre/vertical/soil.py` uses van Genuchten K(theta) and psi(theta). Interflow is Hydrotel-inspired (slope-dependent). Matric potential clamped to -100 m to prevent gradient explosion.
- **Routing**: Muskingum-Cunge with 4 sub-steps (`meandre/routing/kinematic.py`), message passing along topological sort (`meandre/routing/message_passing.py`).
- **Residual corrector**: 2-layer GRU that learns systematic physics errors. Gate initialized to ~5% (physics-first). Zero-sum projection on soil layers.
- **Curriculum training**: Temporal context enabled first, residual corrector at epoch 10, travel-time attention disabled by default.

## Literature-based parameter initialization

`SpatialFieldNetwork.init_from_literature()` biases `fc_out` so parameters start at public literature defaults (Rawls 1982 soil hydraulics, Hock 2003 snow, Chow 1959 Manning, FAO-56 ET) for temperate forested loam/silt_loam.
Without this, K_sat starts ~50x too high (0.5 vs 0.01 m/day), making convergence slow.
The previous name `init_from_hydrotel` is kept as a deprecated alias.

## Running

```bash
# Install
uv sync

# Run SLSO training (from repo root)
python notebooks/slso/slso.py                        # uses slso.toml
python notebooks/slso/slso.py notebooks/slso/config/slso-sub.toml  # subset config

# Tests
pytest tests/ -x -q
```

## Project layout

```
meandre/
  data/           # Basin cache (DuckDB), forcing extraction, PHYSITEL loader
  spatial/        # NeRF field network, positional encoding, concrete dropout
  temporal/       # GRU context encoder, residual corrector
  vertical/       # Snow, frost, interception, ET, soil, wetland, aquifer
  routing/        # Graph, kinematic wave, message passing, lake, temperature
  training/       # Trainer, loss, scheduler, run logger
  utils/          # State, metrics, differentiable helpers
  model.py        # HydroModel — top-level orchestrator
notebooks/slso/   # SLSO basin config, training script, results
tests/            # Mirrors meandre/ structure
```

## Config

TOML config files in `notebooks/slso/config/`. Key sections:
- `[model]`: param_mode ("nerf" or "static"), context_window, dropout
- `[training]`: lr, epochs, curriculum epochs, chunk_steps, tbptt_steps
- `[loss]`: weights for MSE, log-MSE, PBIAS, physics closure, residual reg
- `[literature_prior]`: optional overrides for init_from_literature() targets

## Training safeguards

- **Divergence guard**: Rollback to best checkpoint if loss > 3x EMA (max 3 rollbacks)
- **Gradient clipping**: clip_grad_norm = 1.0
- **Truncated BPTT**: detach every `tbptt_steps` (default 365)
- **Chunked gradient accumulation**: `chunk_steps` (default 180) to fit in 8GB VRAM
- **Warm spinup**: After epoch 0, only re-run last 90 days from cached state

## Important numerical details

- van Genuchten psi clamp: -100 m (prevents d(psi)/d(Se) explosion as Se->0)
- Residual corrector state history clamped to [-50, 500] mm
- Muskingum x bounded [0.01, 0.49], K bounded [4, 48] hours
- Scaling factors (sf1, sf2, sf3) cap extraction so theta stays in [0, porosity]

## Common pitfalls

- Don't use chunk_steps > 0 with NSE/KGE loss (they need full-sequence stats). Use MSE + log-MSE + PBIAS for chunk-safe training.
- Warm-start changes LR to lr_finetune and skips warmup. Set `warm_start = false` and delete checkpoint to retrain from scratch.
- The `physical_prior_loss` targets should be consistent with `init_from_literature()` defaults.
