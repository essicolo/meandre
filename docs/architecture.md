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

    Pay --> SE["<b>Encodeur spatial</b><br/>NeRF Fourier + ParamNoise<br/>→ 36 paramètres / tronçon"]:::neural
    Met --> TE["<b>Encodeur temporel</b><br/>GRU + Concrete Dropout<br/>→ contexte (16D)"]:::neural

    subgraph VC["Colonne verticale (physique)"]
        direction TB
        Snow["<b>Neige</b><br/>accumulation / fonte"]:::physics
        ETm["<b>Évapotranspiration</b><br/>Penman-Monteith FAO-56"]:::physics
        Sol["<b>Sol</b> (3 couches)<br/>van Genuchten + Darcy"]:::physics
        Aqu["<b>Milieux humides + Aquifère</b><br/>réservoirs linéaires"]:::physics
        Snow --> ETm --> Sol --> Aqu
    end

    SE --> VC
    TE --> VC
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

    Qsim --> Loss["<b>Fonction de perte</b><br/>KGE + log-MSE + PBIAS<br/>+ bilan hydrique<br/>+ KL Concrete + KL ParamNoise"]:::loss
    Qobs --> Loss

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
        ParamNoise["<b>ParamNoise</b><br/>raw + σ·ε (logit-space)<br/>mass-conserving"]:::neural
        Constr["_apply_constraints<br/>scaled-tanh, log-normal"]:::neural
        SP["<b>SpatialParams</b><br/>(N, 36)"]:::hybrid

        Fourier --> Concat --> MLP --> ParamNoise --> Constr --> SP
    end

    %% ─── Temporal encoder ───────────────────────────────────
    subgraph TE["<b>TemporalContextEncoder</b>"]
        direction TB
        TInputProj["input_proj 6→64<br/>+ DOY encoding"]:::neural
        TGRU["GRU 64→64<br/><b>+ Concrete Dropout</b>"]:::neural
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
    TOut --> Enrich

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

    Qsim --> Loss["<b>Loss</b><br/>skill scores + regularizers<br/>(see table below)"]:::loss
    Qobs --> Loss

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
| SpatialFieldNetwork (NeRF) | ✅ Active | with ParamNoise (Position B) |
| TemporalContextEncoder (GRU) | ✅ Active | with Concrete Dropout (Position B) |
| Withdrawals | ✅ Active | rebuilt 2026-05-01 from `io-eau-meandre.parquet` |
| Residual Corrector | ⚠️ **Disabled** | pending redesign — gate never trained, noise injection at activation |
| Travel-Time Attention | ⚠️ **Disabled** | random-init weights crash forward; needs warmup gate |
| AR(1) State Noise | ⚠️ **Legacy** | breaks mass conservation; replaced by ParamNoise |

## Position B uncertainty stack

Two complementary noise injections, both differentiable, combined at inference:

- **ParamNoise** on `SpatialFieldNetwork`: Gaussian σ injected on `fc_out` logits BEFORE constraints → mass-conserving (constraints bound perturbed params within physical ranges). Each ensemble member = a coherent bounded param set, analogous to an alternative Hydrotel calibration.
- **Concrete Dropout** on `TemporalContextEncoder`: learned dropout rate (Gal et al. 2017) over the GRU context → epistemic uncertainty on the meteorological-context interpretation.

At inference, [scripts/mc_uncertainty.py](../scripts/mc_uncertainty.py) combines `frozen_param_noise(model, seed) ∘ frozen_dropout(model, seed)` to sample N coherent ensemble members.

## Loss components (current weights)

| Term | Weight | Chunk-safe | Purpose |
|---|---:|:---:|---|
| `w_kge` | 1.0 | ⚠️ approx | targets β, r, γ directly |
| `w_log_mse` | 0.3 | ✅ | baseflow emphasis |
| `w_pbias` | 0.1 | ✅ | volumetric balance |
| `w_mse` | 0.1 | ✅ | overall fit |
| `w_physics` | 0.01 | ✅ | water-balance closure (P − ET − Q − ΔS) |
| `w_prior` | 0.005 | ✅ | log-space pull toward literature defaults |
| `w_param_noise_kl` | 0.1 | ✅ | log_sigma → log(target=0.05) |
| `w_concrete_kl` | 0.1 | ✅ | Gal 2017 KL |
| `w_residual` | 0.01 | ✅ | L2 on corrector gate (kept small) |
| `w_nse`, `w_log_nse`, `w_nrmse` | 0.0 | ❌ | NOT chunk-safe — disabled when chunk_steps > 0 |

## Learned parameters summary

- **SpatialFieldNetwork MLP** weights → 36 params per node (soil hydraulics ×12, snow ×3, ET ×2, routing ×2, etc.)
- **ParamNoise** `log_sigma` (per param)
- **TemporalContextEncoder** GRU + projections + Concrete Dropout `logit_p`
- **StateResidualCorrector** GRU + gate logits *(disabled)*
- **TravelTimeAttention** Q/K/V projections *(disabled)*
- **CorrelatedStateNoise** ρ, σ per state variable *(legacy)*
