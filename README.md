# meandre

Differentiable end-to-end hydrological model in PyTorch.  Reimagines Hydrotel
(INRS-ETE) as a fully differentiable spatio-temporal pipeline trained by
gradient descent on observed streamflow.

## What it does

```
NeRF spatial encoder   →  Temporal context (GRU)  →  Vertical column (physics)
        ↓                          ↓                          ↓
  per-node hydraulic              30-day                 Snow → Frost →
  parameters from              meteorological            Interception → ET (FAO-56)
  (lon, lat, soil, …)            memory                  → Soil (van Genuchten BV3C)
                                                         → Wetland → Aquifer
                                                                ↓
                                  Routing (Muskingum-Cunge, message passing)
                                                                ↓
                                                              Q(t, n)
                                                                ↓
                                                  Loss: KGE + log-MSE + PBIAS
                                                        + physics closure
                                                        + Concrete Dropout KL
                                                        + soft prior on params
```

All operations are vectorised over `n_nodes` (river reaches).  Spatial
parameters are produced by an MLP with Fourier positional encoding from
node coordinates and territorial features (soil texture, land cover, slope,
…), so the model **generalises geographically**: a single trained model
covers any basin in the domain.

![](docs/model_architecture.png)

## Why

Operational hydrological models in Quebec (Hydrotel, Raven, GR4J) require
manual calibration per basin and produce point estimates.  meandre:

* **Calibrates by gradient descent** rather than DDS / SCE-UA — converges in
  hours on GPU vs days/weeks of CPU runtime.
* **Generalises across basins** through the NeRF spatial encoder — one
  trained model for the whole Province de Québec instead of one calibration
  per watershed.
* **Quantifies uncertainty** via *Concrete Dropout* (Gal, Hron & Kendall
  2017) — the per-layer dropout rate is learned to calibrate the ensemble
  spread, replacing classical multi-model approaches (Hydrotel + Raven +
  GR4J) by N forward passes through the same model.
* **Is differentiable end-to-end**, so gradients flow from observed Q back
  to soil hydraulics, snow physics, ET, and routing parameters.

## Status

Tested on the SLSO basin (Saint-Laurent Sud-Ouest, 2889 reaches, 30 stations):

| Metric on dev (2019-2021) | meandre best.pt |
|-----|-----|
| pooled KGE | 0.87 |
| per-station median KGE | 0.78 |
| β (volume) | 0.96 |
| γ (variance) | 0.91 |
| r (timing) | 0.92 |
| KGE_log (baseflow) | 0.92 |

Comparable to per-basin-calibrated Hydrotel.  No manual tuning involved.
Test on held-out 2022-2024 evaluated separately by `eval_test.py`.

## Repository layout

```
meandre/
  data/         Basin cache (DuckDB), forcing extraction, withdrawals loader
  spatial/      NeRF field network, Fourier positional encoding, Concrete Dropout
  temporal/     GRU context encoder, residual corrector
  vertical/     Snow, frost, interception, ET, soil (3-layer van Genuchten),
                wetland, aquifer
  routing/      RiverGraph, Muskingum-Cunge kinematic wave, message passing,
                lake (Newton-Raphson), stream temperature, withdrawals
  training/     Trainer, HydroLoss, scheduler, run logger, MC uncertainty
  utils/        HydroState, metrics (KGE, NSE, log-NSE, PBIAS, …)
  model.py      HydroModel — top-level orchestrator

notebooks/slso/ SLSO basin config, training script, diagnostics, results
docs/           Architecture diagrams, basin DB schema
tests/          Mirrors meandre/ structure
```

## Quickstart

```bash
# Install
uv sync

# Run SLSO training (from repo root) — uses notebooks/slso/config/slso.toml
python notebooks/slso/slso.py

# Held-out test evaluation on best.pt (CPU, no GPU conflict)
python eval_test.py

# Full diagnostics (water balance, hydrographs, KGE maps,
#                  MC uncertainty ensemble with Talagrand + PICP)
quarto render notebooks/slso/diagnostics.qmd

# Generate MC Dropout uncertainty ensemble (NetCDF)
python mc_uncertainty.py
```

## Configuration

Per-basin TOML at `notebooks/slso/config/slso.toml`:

* `[paths]` — basin DuckDB, forcing zarr/nc, checkpoint
* `[temporal]` — train / dev / test split (rigorous, no leakage)
* `[model]` — NeRF settings, Concrete Dropout, residual history
* `[soil]` — z1 (fixed) + per-node bounds for z2, z3, rain_hours
* `[training]` — lr, epochs, chunk_steps (for gradient accumulation),
                 best_metric, prior weight `w_prior`, curriculum epochs
* `[loss]` — weighted combination of KGE / log-MSE / PBIAS / physics-closure

## Documentation

* [Basin DB schema](docs/basin_db_schema.md) — DuckDB tables
* [Architecture diagram](docs/model_architecture.png) — full pipeline
* [Workflow](docs/workflow.drawio) — data flow PHYSITEL → bassin.duckdb → training
* [HESS reference](docs/hess-29-2811-2025.pdf) — published context

## Key design choices

* **NeRF parameters with Fourier features** — continuous spatial fields,
  no spatial discontinuities at sub-basin boundaries.
* **3-layer van Genuchten soil** — depths `z2, z3` learned per node,
  `z1` fixed (configurable).  K(θ) and ψ(θ) used directly; ψ clamped to
  -100 m to prevent gradient explosion as Se → 0.
* **Newton-Raphson lake module** — replaces explicit Euler that was
  mass-non-conservative (528 % residual fixed).
* **Eagleson sub-daily infiltration excess** — `rain_hours` per node
  controls effective intensity.
* **Cold content snow physics** — prevents mid-winter melt artefacts.
* **Withdrawals** — surface and groundwater pumping/return flow injected
  per node from monthly site-level parquet (`io-eau-meandre.parquet`),
  snapped to nearest reach.
* **Soft prior regularization** — log-space L2 toward Hydrotel literature
  defaults (Rawls 1982, Hock 2003, FAO-56) prevents overfitting toward
  unphysical parameter regions.

## Training safeguards

* Truncated BPTT every 365 days
* Chunked gradient accumulation every 180 days (fits 8 GB VRAM)
* Divergence rollback if loss > 3× EMA (max 3 rollbacks)
* Cached end-of-train state for fast `_val_epoch` (saves ≈ 50 min/epoch)
* `_val_epoch` does continuous spinup → train → val forward pass for
  honest metric (no protocol shortcut that skipped train period)

## Known limitations

* Concrete Dropout `length_scale` default of `1.0` (was `1e-2` and led to
  `p → 0` collapse).  After re-training the dispersion should be
  calibrated to PICP ≈ 80 %.
* Residual corrector currently disabled (`enable_residual_epoch = 9999`)
  pending redesign — gate initialization and noise injection at
  activation cause loss spikes.
* Travel-time attention disabled (`enable_travel_epoch = 9999`) for the
  same warm-start instability reason.

## Citation / context

Builds on Hydrotel (Fortin et al., INRS-ETE) and recent neural-physics
hybrid literature.  See `docs/hess-29-2811-2025.pdf` for the published
reference informing the approach.
