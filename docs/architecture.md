# Meandre — Model Architecture

Two views of the model:
- **Overview** (FR) — high-level dataflow for stakeholders / presentations
- **Detailed** (EN) — module-level breakdown for developers

Both diagrams are Mermaid blocks: they render natively on GitHub, GitLab, and most Markdown viewers (VS Code requires the *Markdown Preview Mermaid Support* extension).

---

## Overview / Vue d'ensemble

```mermaid
flowchart LR
    Met["<b>Météo</b><br/>P, T, radiation,<br/>vent, humidité"]:::input
    Pay["<b>Paysage</b><br/>occupation du sol,<br/>pentes, aires"]:::input
    Net["<b>Réseau</b><br/>topologie, lacs,<br/>tronçons"]:::input
    Pre["<b>Prélèvements</b><br/>m³/s par tronçon"]:::input
    Qobs["<b>Q_obs</b><br/>débits observés<br/>aux stations"]:::input
    Aux["<b>Obs. auxiliaires</b><br/>ET (MODIS)<br/>TWS (GRACE)"]:::input

    Pay --> SE["<b>Encodeur spatial</b><br/>NeRF Fourier<br/>→ 36 paramètres / tronçon"]:::neural
    Met --> TE["<b>Encodeur temporel</b><br/>GRU<br/>→ contexte (16D)"]:::neural

    subgraph VC["Colonne verticale (physique)"]
        direction TB
        Snow["<b>Neige</b><br/>accumulation / fonte"]:::physics
        ETm["<b>Évapotranspiration</b><br/>Penman-Monteith FAO-56"]:::physics
        Sol["<b>Sol</b> (3 couches)<br/>van Genuchten + Darcy"]:::physics
        Aqu["<b>Milieux humides + Aquifère</b><br/>réservoirs linéaires"]:::physics
        Snow --> ETm --> Sol --> Aqu
    end

    SE --> VC
    Met --> VC
    VC --> Lat["<b>Apport latéral</b><br/>ruissellement + interflow<br/>+ débit de base"]:::flow

    Net --> Topo["<b>Balayage topologique</b><br/>Kahn / scatter_add"]:::routing
    Pre --> Wd["<b>Prélèvements / rejets</b><br/>net_W appliqué à Q"]:::routing
    Lat --> Topo
    Wd --> Topo

    Topo --> Musk["<b>Muskingum-Cunge</b><br/>rivières (K, x appris)"]:::routing
    Topo --> Lake["<b>Module lac</b><br/>bilan + loi puissance"]:::routing
    Topo --> Therm["<b>Thermie</b><br/>advection H = Q·T<br/>+ échange atmosphérique"]:::routing

    Musk --> Qsim["<b>Q_sim</b><br/>débit simulé (m³/s)"]:::output
    Lake --> Qsim
    Therm --> Teau["<b>T_eau</b><br/>température (°C)"]:::output

    Qsim --> PH["<b>Tête probabiliste</b> (phase 2)<br/>ContextualQuantileHead<br/>K quantiles non-paramétriques<br/>médiane libre"]:::neural
    SE --> PH
    TE --> PH

    PH --> Loss["<b>Fonction de perte</b><br/>phase 1 : KGE + log-MSE + PBIAS<br/>+ ET (MODIS) + TWS (GRACE) + bilan<br/>phase 2 : NLL ou quantile (CRPS)"]:::loss
    Qobs --> Loss
    Aux --> Loss

    classDef input fill:#d5e8d4,stroke:#82b366
    classDef neural fill:#e1d5e7,stroke:#9673a6
    classDef physics fill:#b3d9ff,stroke:#6c8ebf
    classDef flow fill:#dae8fc,stroke:#6c8ebf
    classDef routing fill:#ffe6cc,stroke:#d79b00
    classDef output fill:#fff2cc,stroke:#d6b656
    classDef loss fill:#f8cecc,stroke:#b85450
```

---

## Detailed architecture

```mermaid
flowchart TB
    %% ─── Inputs ─────────────────────────────────────────────
    Forcing["<b>Forcing</b><br/>(T, N, 6)"]:::input
    Coords["<b>Coords</b><br/>(N, 2)"]:::input
    Terr["<b>Territorial</b><br/>(N, 17)"]:::input
    Graph["<b>RiverGraph</b>"]:::input
    State0["<b>Initial State</b><br/>(N, 9)"]:::input
    Withdraw["<b>Withdrawals</b><br/>(T, N, 5)"]:::input
    DOY["<b>Day of Year</b>"]:::input
    Qobs["<b>Q_obs</b><br/>(T, n_stations)"]:::input

    %% ─── Spatial encoder ────────────────────────────────────
    subgraph SE["<b>SpatialFieldNetwork</b>"]
        direction TB
        Fourier["Fourier encoding<br/>(lon, lat) → 26D"]:::neural
        Concat["concat(enc, territorial)<br/>→ 43D"]:::hybrid
        MLP["MLP fc1 → fc2 → fc_out<br/>43→256→256→36 (skip)"]:::neural
        ParamNoise["<b>ParamNoise</b><br/>⚠ LEGACY — replaced by the<br/>predictive probabilistic head"]:::disabled
        Constr["_apply_constraints<br/>scaled-tanh, log-normal"]:::neural
        SP["<b>SpatialParams</b><br/>(N, 36)"]:::hybrid

        Fourier --> Concat --> MLP --> Constr --> SP
        ParamNoise -.-> Constr
    end

    %% ─── Temporal encoder ───────────────────────────────────
    subgraph TE["<b>TemporalContextEncoder</b>"]
        direction TB
        TInputProj["input_proj 6→64<br/>+ DOY encoding"]:::neural
        TGRU["GRU 64→64<br/>(Concrete Dropout — legacy)"]:::neural
        TOut["LayerNorm + 64→16<br/>→ context (T, N, 16)"]:::neural
        TInputProj --> TGRU --> TOut
    end

    Coords --> Fourier
    Terr --> Concat
    Forcing --> TInputProj
    DOY --> TInputProj

    %% ─── Vertical column (physics) ──────────────────────────
    subgraph VC["<b>Vertical Column</b> (per timestep, vectorised over N)"]
        direction TB
        Enrich["Enriched forcing<br/>(N, 22)"]:::flow
        Snow["<b>1. Snow</b><br/>rain/snow split, melt"]:::physics
        Frost["<b>2. Frost</b><br/>K_sat × frost_factor"]:::physics
        Inter["<b>3. Interception</b><br/>canopy → throughfall"]:::physics
        ETm["<b>4. Evapotranspiration</b><br/>Penman-Monteith FAO-56"]:::physics
        Sol["<b>5. Soil</b> (3 layers)<br/>van Genuchten + Darcy<br/>runoff / interflow / baseflow"]:::physics
        Wet["<b>6. Wetland</b><br/>power-law"]:::physics
        Aqu["<b>7. Aquifer</b><br/>linear reservoir"]:::physics
        LatOut["<b>lateral_inflow</b><br/>(N,) mm/day"]:::flow

        Enrich --> Snow --> Frost --> Inter --> ETm --> Sol --> Wet --> Aqu --> LatOut
    end

    SP --> VC
    TOut -.-> Enrich

    %% ─── State corrections ──────────────────────────────────
    subgraph SC["<b>State Corrections</b>"]
        Resid["<b>Residual Corrector</b><br/>GRU over state history<br/>state' = physics + gate·δ<br/>⚠ DISABLED — pending redesign"]:::disabled
        Noise["<b>AR(1) State Noise</b><br/>η_t = ρ·η_{t-1} + σ·ε_t<br/>⚠ LEGACY — breaks mass conservation"]:::disabled
    end
    Sol -.-> Resid
    Resid -.-> Sol

    %% ─── Routing ────────────────────────────────────────────
    LatOut --> Conv["mm/day → m³/s"]:::flow

    subgraph RT["<b>Routing</b>"]
        direction TB
        Topo["<b>Topological sweep</b><br/>Kahn levels + scatter_add<br/>− net withdrawals"]:::routing
        Musk["<b>Muskingum-Cunge</b><br/>rivers (K, x learned)"]:::routing
        Lake["<b>Lake module</b><br/>power-law"]:::routing
        TTA["<b>Travel-Time Attention</b><br/>⚠ DISABLED"]:::disabled
        Therm["<b>Stream Temperature</b><br/>advection + atmospheric"]:::routing
        Topo --> Musk
        Topo --> Lake
        Topo --> Therm
        TTA -.-> Topo
    end

    Graph --> Topo
    Withdraw --> Topo
    Conv --> Topo

    %% ─── Outputs + loss ─────────────────────────────────────
    Musk --> Qsim["<b>Q_sim</b><br/>(T, N) m³/s"]:::output
    Lake --> Qsim
    Therm --> Twater["<b>T_water</b><br/>(T, N) °C"]:::output

    %% ─── Probabilistic head (phase 2) ───────────────────────
    subgraph PH["<b>Probabilistic Head</b> (phase 2 — frozen backbone)"]
        direction TB
        Noiseh["<b>SpatialNoiseHead</b> (legacy)<br/>log σ(t,n) = a(n) + b(n)·log|Q|<br/>Gaussian · Student-t · Box-Cox"]:::disabled
        Quanth["<b>QuantileHead</b> (legacy)<br/>K monotone offsets from μ<br/>(median = μ; CRPS / pinball)"]:::disabled
        CtxQ["<b>ContextualQuantileHead</b> ✅<br/>K=7 quantiles non-paramétriques<br/>médiane libre + features GRU<br/>δ²≤0.06, cov_90=0.90"]:::neural
    end
    SP --> PH
    TOut --> PH
    Qsim --> PH

    Qsim --> Loss["<b>Loss</b><br/>skill scores + multi-obj + regularizers<br/>(see table below)"]:::loss
    PH --> Loss
    Qobs --> Loss
    ETobs["<b>ET_obs</b><br/>(MODIS)"]:::input --> Loss
    TWSobs["<b>TWS_obs</b><br/>(GRACE)"]:::input --> Loss

    State0 --> VC

    classDef input fill:#d5e8d4,stroke:#82b366
    classDef neural fill:#f8cecc,stroke:#b85450
    classDef hybrid fill:#e1d5e7,stroke:#9673a6
    classDef physics fill:#b3d9ff,stroke:#6c8ebf
    classDef flow fill:#dae8fc,stroke:#6c8ebf
    classDef routing fill:#ffe6cc,stroke:#d79b00
    classDef output fill:#d5e8d4,stroke:#82b366
    classDef loss fill:#f8cecc,stroke:#b85450
    classDef disabled fill:#eeeeee,stroke:#999999,stroke-dasharray:5 5,color:#666
```

**Other outputs** (not shown above to keep the diagram compact):
- `final_state` — `HydroState (N, 9)` for warm restart
- `diagnostics` — ETP, ETR, snowmelt, q_baseflow, q_upstream, T_water (per timestep)

---

## Module status

| Module | Status | Notes |
|---|---|---|
| Snow / Frost / Interception / ET / Soil / Wetland / Aquifer | ✅ Active | core physics chain |
| Routing (Muskingum-Cunge, Lake, Stream Temperature) | ✅ Active | |
| SpatialFieldNetwork (NeRF) | ✅ Active | Fourier + MLP → 36 params; constrained by ET/TWS multi-obj |
| TemporalContextEncoder (GRU) | ✅ Active | Concrete Dropout now legacy (Position B) |
| Probabilistic head (NoiseHead / QuantileHead) | ✅ Active | phase 2, frozen backbone — heteroscedastic σ or quantiles |
| Multi-objective (MODIS ET, GRACE TWS) | ✅ Active | phase 1 — decollapses `f_vert`, identifiability |
| Withdrawals | ✅ Active | rebuilt 2026-05-01 from `io-eau-meandre.parquet` |
| Residual Corrector | ⚠️ **Disabled** | pending redesign — gate never trained, noise injection at activation |
| Travel-Time Attention | ⚠️ **Disabled** | random-init weights crash forward; needs warmup gate |
| ParamNoise (Position B) | ⚠️ **Legacy** | ensemble stack abandoned 2026-05-11 → predictive probabilistic head |
| Concrete Dropout (Position B) | ⚠️ **Legacy** | idem |
| AR(1) State Noise | ⚠️ **Legacy** | breaks mass conservation |

## Probabilistic prediction (two-phase)

The ensemble-style "Position B" stack (ParamNoise + Concrete Dropout, sampled at inference) was **abandoned on 2026-05-11** in favour of a **predictive probabilistic head** trained in two phases:

1. **Phase 1 — backbone (deterministic, multi-objective).** Train physics + spatial/temporal encoders against discharge **and** auxiliary observations: MODIS ET (`w_et`) and GRACE TWS (`w_tws`). The multi-objective signal lifts the equifinality that otherwise collapses the deep-soil flux `f_vert`, restoring parameter identifiability.
2. **Phase 2 — uncertainty (frozen backbone).** Freeze the backbone and train *only* a probabilistic head on top of `μ = Q_sim`:
   - **`SpatialNoiseHead`** — per-node `log σ(t,n) = a(n) + b(n)·log(|Q|+ε)`, fit by heteroscedastic NLL (`w_nll`). Distribution selectable: Gaussian, **Student-t** (heavy tails, learned ν), Box-Cox or log-normal.
   - **`QuantileHead`** — per-node monotone offsets `δ_τ` from the median (`μ`), so `q_τ = μ + δ_τ`; fit by pinball / CRPS (`w_quantile`). The median stays `= μ`, preserving the phase-1 KGE.

## Loss components

Weights are config-driven (TOML `[loss]`); below are representative values from the SLSO runs.

**Phase 1 — backbone (`slso.toml` + multi-obj overrides):**

| Term | Weight | Chunk-safe | Purpose |
|---|---:|:---:|---|
| `w_kge` | 1.0 | ⚠️ approx | targets β, r, γ directly |
| `w_log_mse` | 0.3 | ✅ | baseflow emphasis |
| `w_pbias` | 0.1 | ✅ | volumetric balance |
| `w_mse` | 0.1 | ✅ | overall fit |
| `w_et` | 0.1 | ✅ | MODIS ET match (multi-obj) |
| `w_tws` | 0.3 | ✅ | GRACE TWS, z-scored (multi-obj) — drives `f_vert` |
| `w_physics` | 0.01 | ✅ | water-balance closure (P − ET − Q − ΔS) |
| `w_residual` | 0.01 | ✅ | L2 on corrector gate (kept small) |
| `w_nse`, `w_log_nse`, `w_nrmse` | 0.0 | ❌ | NOT chunk-safe — disabled when chunk_steps > 0 |

**Phase 2 — probabilistic (frozen backbone), pick one driver:**

| Term | Weight | Purpose |
|---|---:|---|
| `w_nll` | 1.0 | heteroscedastic NLL (`nll_distribution` = normal / student-t / box-cox) |
| `w_quantile` | 1.0 | multi-τ pinball / CRPS (alternative to NLL) |
| `w_nll_et`, `w_nll_swe` | 0.0 | optional NLL on ET / SWE auxiliary channels |
| `w_peak` | 0.0 | optional peak weighting (climatological Q_p75 threshold) |

## Learned parameters summary

- **SpatialFieldNetwork MLP** weights → 36 params per node (soil hydraulics ×12, snow ×3, ET ×2, routing ×2, etc.)
- **TemporalContextEncoder** GRU + projections (Concrete Dropout `logit_p` — legacy)
- **SpatialNoiseHead** per-node `(a, b)` MLP + `log_df` (ν, Student-t) *(phase 2)*
- **QuantileHead** per-node `(a_τ, b_τ)` MLP *(phase 2)*
- **ParamNoise** `log_sigma` per param *(legacy)*
- **StateResidualCorrector** GRU + gate logits *(disabled)*
- **TravelTimeAttention** Q/K/V projections *(disabled)*
- **CorrelatedStateNoise** ρ, σ per state variable *(legacy)*
