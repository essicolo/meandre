# meandre

> Research prototype, not yet an operational tool. `main` is a periodic snapshot of the development branch. Results below are those of the provincial model frozen in October 2026.

meandre is a differentiable hydrological model written in PyTorch. It reimplements the physics of Hydrotel (INRS-ETE) and learns its parameters by gradient descent on observed streamflow. A neural field predicts the parameters of every river reach from its location and landscape, so the model also runs on ungauged reaches.

## What it does

```
Spatial parameter field              Hydrotel column (faithful port)
(lon, lat, landscape)          →     snow → frost → evapotranspiration
43 parameters per reach              → 3-layer soil → wetlands → aquifer
                                              ↓
                       Routing: Muskingum-Cunge solved as one linear
                       operator, lakes with learned storage-outflow
                                              ↓
                       Q_sim(t, reach) + quantile head (6 quantiles,
                       median = Q_sim, so deterministic skill is kept)
                                              ↓
                       Loss: KGE, volume bias, log errors, low-flow
                       terms, MOD16 evapotranspiration
```

All operations are vectorised over the river reaches. The vertical physics is a line-by-line port of Hydrotel C++ 4.3.6, validated to the decimal against the compiled binary on thousands of computational units. Departures from Hydrotel are documented options, for instance an unconfined aquifer that returns baseflow, a mass-conserving wetland exchange and a viscosity correction of soil conductivity. See [`docs/architecture.md`](docs/architecture.md) for the module-level description.

## Why

Operational hydrological modelling in Quebec relies on Hydrotel calibrated region by region. The operational reference is in fact an ensemble of six calibrations whose regional rankings disagree. meandre:

* calibrates the whole province at once, in about ten hours on a single 8 GB GPU;
* learns one continuous parameter field instead of several discrete calibrations;
* quantifies uncertainty with a quantile head trained on the frozen deterministic model;
* keeps Hydrotel's physics where it was validated, and departs from it only through documented options.

## Current results

One model covers the province: the 15 PHYSITEL territories form a single domain of 27 834 reaches. It is forced by the raw CaSR v3.2 reanalysis (ECCC), trained on 2001-2018 and selected on 2019-2021. The figures below are for 2022-2024, a period used neither for training nor for selection.

| Measure, 2022-2024 | Value |
|---|---|
| Stations with at least one year of observations | 143 |
| Median KGE per station | 0.70 |
| Interquartile range of KGE per station | 0.60 to 0.81 |
| Share of stations with KGE above 0.5 | 88 % |

Low flows and small streams, with a typical low flow under 0.5 m³/s, remain the hardest cases. Their low-flow error is about 1.7 times that of the other stations.

The quantile head was trained on 2001-2018 with the deterministic model frozen. Its median is the deterministic discharge, to within 3e-7 in relative terms. On 2022-2024, over the same 143 stations:

| Nominal interval | Share of observations inside |
|---|---|
| 50 % | 51 % |
| 80 % | 80 % |
| 90 % | 90 % |

The 90 % interval covers between 88 % (summer) and 94 % (autumn) of the observations depending on the season. Its median width is 2.2 times the median discharge. Coverage is lowest on three of the five North Shore territories, which have few and partly regulated stations (76 to 86 %).

## Forcing

The model is forced by raw CaSR v3.2, with no correction. Hydrotel's soil calibration compensates the biases of its own kriged weather, so it is not imported. meandre learns soil and evapotranspiration on the forcing it actually receives. Weather preprocessing is a separate task, outside this repository. The forcing is never set in a configuration file and must be given at launch.

## Repository layout

```
meandre/
  data/         basin caches (DuckDB), PHYSITEL loader, forcing, MODIS, GRACE,
                snow, groundwater levels, withdrawals, soil texture
  spatial/      parameter field, Fourier encoding, latent codes
  vertical/     HydrotelColumn and the processes added to it
  routing/      river graph, Muskingum-Cunge operator, lakes, withdrawals
  training/     trainer, loss terms (including pinball), autopilot
  utils/        state, metrics, quantile head, configuration loader
  model.py      HydroModel, the top-level orchestrator
hydrotel_clone/ ports of the Hydrotel C++ physics and their validation harnesses
.runs/quebec/   provincial pilot (etl_run.py), configurations, evaluation scripts
.runs/slso/     the original single-basin test case
reports/        hypothesis register and experiment log (in French)
docs/           architecture, basin database schema
tests/          unit tests, mirroring meandre/
```

## Quickstart

```bash
uv sync

# Train the provincial model from scratch (about 10 h on an 8 GB GPU)
ETL_CONFIG=.runs/quebec/config/modele-provincial-2026-10.toml \
ETL_REGION=quebec JOINT_FX_SUFFIX=-casr-brut \
python .runs/quebec/etl_run.py

# Forward pass on a frozen checkpoint, no training
ETL_EPOCHS=0 ETL_WARM_FROM=<checkpoint.pt> <same variables> python .runs/quebec/etl_run.py

# Quantile head on the frozen model
ETL_QUANTILE=1 ETL_WARM_FROM=<checkpoint.pt> <same variables> python .runs/quebec/etl_run.py

# Tests
pytest tests/ -x -q
```

Data paths are read from `MEANDRE_DATA` and related variables (`meandre/utils/paths.py`).

## Configuration

A model is defined by a TOML file. Its `[recette]` section sets the variables read by the pilot; a variable already set in the environment takes precedence over the file, and the file over the code default. Each checkpoint writes the configuration that produced it to `<name>.recette.toml`, readable without torch. The frozen provincial model is described by `.runs/quebec/config/modele-provincial-2026-10.toml`.

The other sections carry the physics: `[soil]` declares processes per layer, `[et]` the evapotranspiration formula, `[phenology]` the seasonal cycle of vegetation, `[loss]` the loss weights, `[training]` the optimiser settings.

## Design rules

* Validate any ported physics against its C++ reference on the smallest unit before integrating it.
* Answer a question with the shortest test that can answer it, such as a forward pass, a water balance audit or a gradient probe, before any training.
* Judge models on the same loss, over the same period, on years never used for training or selection.
* Fix a systematic seasonal error in the physics, not by tuning a parameter or a weight.
* Judge auxiliary observations on what they make identifiable, not on the KGE they bring.

## Known limitations

* Low flows and small streams remain the largest share of the loss.
* Regulated rivers carry the signature of reservoir operations, which are not modelled.
* GPU computations are not bit-reproducible: two runs with the same seed differ by up to about 0.01 in loss.

## Context

Builds on Hydrotel (Fortin et al., INRS-ETE, C++ 4.3.6) and the literature on hybrid neural and physical models. Forcing: CaSR v3.2 (ECCC). Auxiliary observations: MOD16A2GF evapotranspiration, NEISIM snow water equivalent, groundwater levels from the provincial monitoring network. License: GNU GPL v3.
