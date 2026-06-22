"""Top-level HydroModel model class.

Orchestrates: spatial encoder -> temporal context -> vertical column ->
              residual corrector -> routing -> loss.

The simulate() method runs one full forward pass over n_timesteps days.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from meandre.routing.dam import DamData
from meandre.routing.graph import RiverGraph
from meandre.routing.message_passing import RoutingLayer
from meandre.routing.temperature import StreamTemperatureModule
from meandre.routing.withdrawals import WithdrawalData
from meandre.spatial.field_network import SpatialFieldNetwork, SpatialParams
from meandre.spatial.territorial import TerritorialFeatures
from meandre.temporal.context_encoder import TemporalContextEncoder
from meandre.temporal.residual_corrector import StateResidualCorrector
from meandre.temporal.ring_buffer import OutflowRingBuffer
from meandre.temporal.state_noise import CorrelatedStateNoise
from meandre.utils.diagnostics import SimDiagnostics
from meandre.utils.noise_head import HeteroscedasticNoiseHead, SpatialNoiseHead
from meandre.utils.state import HydroState
from meandre.vertical.column import VerticalColumn


class HydroModel(nn.Module):
    """End-to-end differentiable hydrological model.

    Parameters
    ----------
    n_nodes : int
        Number of graph nodes (subbasins/reaches).
    n_forcing : int
        Number of raw forcing variables (default 6: P, Tmin, Tmax, Rn, u2, ea).
    context_window : int
        Number of past days for temporal attention.
    residual_history : int
        Number of past states for residual corrector.
    max_travel_time : int
        Ring buffer depth = max travel time in the network (days).
    use_temporal : bool
        Enable temporal context encoder (Phase 2+).
    use_residual : bool
        Enable state residual corrector (Phase 3+).
    use_travel_time_attn : bool
        Enable travel-time attention in routing (Phase 4+).
    use_state_noise : bool
        Enable AR(1) correlated state noise for ensemble generation.
        Disabled by default; activate for ensemble / UQ runs.
    use_temperature : bool
        Enable stream temperature tracking. When True, heat loads are
        propagated through the routing network alongside discharge.
    n_state_vars : int
        Number of state variables for the residual corrector.
        Use None to auto-detect from HydroState.N_VARS (default).
        Old checkpoints may have 7; new models have 9.
    """

    def __init__(
        self,
        n_nodes: int,
        n_forcing: int = 6,
        context_window: int = 60,
        residual_history: int = 14,
        max_travel_time: int = 30,
        use_temporal: bool = True,
        use_residual: bool = True,
        use_travel_time_attn: bool = True,
        use_state_noise: bool = False,
        use_temperature: bool = True,
        n_territorial: int = 17,
        n_state_vars: int | None = None,
        dropout: float = 0.0,
        concrete_dropout: bool = False,
        concrete_init_p: float = 0.05,
        param_mode: str = "nerf",
        clamp_min: float = -50.0,
        clamp_max: float = 500.0,
        soil_z1: float = 0.30,
        soil_vsa_b: float = 2.5,
        soil_quickflow_reservoir: bool = False,
        soil_quickflow_beta: float = 0.5,
        soil_separate_infil_capacity: bool = False,
        soil_frozen_gate: bool = False,
        soil_runoff_clean: bool = False,
        soil_mode: str = "meandre",
        soil_clone_substep: int = 48,
        soil_clone_krec_init: float = 1e-5,
        et_mode: str = "penman",
        column_mode: str = "meandre",   # "meandre" | "hydrotel" (colonne fidèle clonée)
        column_theta_init_frac: float = 0.9,  # theta init = frac·thetas (init Hydrotel validé) en mode hydrotel ; 0 = garder la theta du cache
        use_frost_rankinen: bool = True,
        compile_soil: bool = False,   # mode hydrotel : torch.compile du sol seul
        compile_column: bool = False,   # mode hydrotel : torch.compile de TOUT le pas (snow+gel+ET+sol)
        use_overland_uh: bool = False,
        use_hillslope_uh: bool = False,
        soil_bounds: dict | None = None,
        use_quantile_head: bool = False,
        quantile_taus: tuple[float, ...] = (0.05, 0.10, 0.25, 0.75, 0.90, 0.95),
        use_mixture_head: bool = False,
        mixture_n_components: int = 10,
        mixture_hidden: int = 64,
        # ContextualQuantileHead (IHI) — médiane libre + features riches
        use_contextual_quantile_head: bool = False,
        cqh_n_features: int = 45,
        cqh_taus: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95),
        cqh_hidden: int = 64,
        # PhenologyModulator (IHI Phase B étape 1) : K_c modulé par GDD
        use_phenology_modulator: bool = False,
        # Routage : "level" (balayage par niveau, historique), "operator"
        # (solve triangulaire par étages de lacs, sémantique identique) ou
        # "operator-lagged" (lacs sur stockage de la veille, un seul solve).
        routing_mode: str = "level",
        # Params de lac (k_lake, beta) appris spatialement par le NeRF plutôt
        # que scalaires globaux du LakeModule. Opt-in (change l'archi NeRF).
        predict_lake_params: bool = False,
        # Nombre de bandes de fréquences Fourier pour l'encodage (lon,lat). Les
        # coords sont projetées isotropes + normalisées [-1,1] (cf. _project_coords).
        n_coord_freqs: int = 6,
        use_latent_codes: bool = False,
        latent_dim: int = 8,
        latent_mode: str = "additive",
        routing_substeps: int = 2,
        discharge_dependent_celerity: bool = False,
        dq_beta: float = 0.4,
        dq_qref_specific: float = 0.01,
        pure_advection: bool = False,
        dynamic_atten: bool = False,
        da_beta: float = 2.0,
        da_qref_specific: float = 0.05,
    ) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.n_forcing = n_forcing
        self.context_window = context_window
        self.residual_history = residual_history
        self.max_travel_time = max_travel_time
        self.use_temporal = use_temporal
        self.use_residual = use_residual

        # Modules
        self.spatial_encoder = SpatialFieldNetwork(
            n_territorial=n_territorial,
            n_coord_freqs=n_coord_freqs,
            dropout=dropout,
            param_mode=param_mode,
            soil_bounds=soil_bounds,
            predict_lake_params=predict_lake_params,
            n_nodes=n_nodes,
            use_latent_codes=use_latent_codes,
            latent_dim=latent_dim,
            latent_mode=latent_mode,
        )
        # Store for SoilModule init via VerticalColumn
        self._soil_z1 = soil_z1

        n_context = 16 if use_temporal else 0
        self.temporal_encoder = TemporalContextEncoder(
            n_forcing=n_forcing,
            window=context_window,
            n_context_out=n_context,
            concrete_dropout=concrete_dropout,
            concrete_init_p=concrete_init_p,
        ) if use_temporal else None

        self.vertical_column = VerticalColumn(
            soil_z1=soil_z1, soil_vsa_b=soil_vsa_b,
            soil_quickflow_reservoir=soil_quickflow_reservoir,
            soil_quickflow_beta=soil_quickflow_beta,
            soil_separate_infil_capacity=soil_separate_infil_capacity,
            soil_frozen_gate=soil_frozen_gate,
            soil_runoff_clean=soil_runoff_clean,
            soil_mode=soil_mode,
            soil_clone_substep=soil_clone_substep,
            soil_clone_krec_init=soil_clone_krec_init,
            use_overland_uh=use_overland_uh,
            use_hillslope_uh=use_hillslope_uh,
            et_mode=et_mode,
        )
        # Colonne verticale FIDÈLE Hydrotel (Phase A) : remplace VerticalColumn,
        # présente la même interface (column_step → ColumnOutput).
        self.column_mode = str(column_mode)
        self.column_theta_init_frac = float(column_theta_init_frac)
        if self.column_mode == "hydrotel":
            from meandre.vertical.hydrotel_column import HydrotelColumn
            self.vertical_column = HydrotelColumn(
                et_mode=(et_mode if et_mode in ("mcguinness", "hydro_quebec", "penman") else "mcguinness"),
                use_frost=use_frost_rankinen,
                compile_soil=bool(compile_soil),
                compile_column=bool(compile_column),
            )

        _n_state = n_state_vars if n_state_vars is not None else HydroState.N_VARS
        self.residual_corrector = StateResidualCorrector(
            n_state_vars=_n_state,
            history=residual_history,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
        ) if use_residual else None
        self.routing = RoutingLayer(
            use_travel_time_attention=use_travel_time_attn,
            max_tau_days=max_travel_time,
            routing_mode=routing_mode,
            routing_substeps=routing_substeps,
        )
        # Célérité de canal dépendante du débit (onde cinématique, cf Hydrotel).
        self.routing.dq_celerity = bool(discharge_dependent_celerity)
        self.routing.dq_beta = float(dq_beta)
        self.routing.dq_qref_specific = float(dq_qref_specific)
        self.routing.pure_advection = bool(pure_advection)
        self.routing.dynamic_atten = bool(dynamic_atten)
        self.routing.da_beta = float(da_beta)
        self.routing.da_qref_specific = float(da_qref_specific)

        # Optional AR(1) state noise for ensemble / UQ runs (Phase 5+)
        self.state_noise: CorrelatedStateNoise | None = (
            CorrelatedStateNoise(n_state_vars=4) if use_state_noise else None
        )

        # Stream temperature module
        self.temperature: StreamTemperatureModule | None = (
            StreamTemperatureModule() if use_temperature else None
        )

        # Heteroscedastic noise heads: predict log σ from each observable
        # magnitude for the probabilistic (Gaussian NLL) training loss.
        # SpatialNoiseHead conditions on NeRF spatial params for per-node
        # uncertainty; HeteroscedasticNoiseHead is the 2-scalar global fallback.
        self.noise_head: SpatialNoiseHead | HeteroscedasticNoiseHead = (
            SpatialNoiseHead(n_spatial_params=SpatialParams.N_PARAMS)
        )
        self.noise_head_et = HeteroscedasticNoiseHead()    # ET (mm/jour)
        self.noise_head_swe = HeteroscedasticNoiseHead()   # SWE (mm)
        # Tête quantile optionnelle (Phase 2 v2) : offsets monotones depuis μ.
        self.use_quantile_head = use_quantile_head
        if use_quantile_head:
            from meandre.utils.quantile_head import QuantileHead
            self.quantile_head = QuantileHead(
                n_spatial_params=SpatialParams.N_PARAMS, taus=tuple(quantile_taus),
            )
        # Mixture Density Network (option 2b) : densité conditionnelle non-paramétrique
        self.use_mixture_head = use_mixture_head
        self.mixture_n_components = mixture_n_components
        self.mixture_hidden = mixture_hidden
        if use_mixture_head:
            from meandre.utils.mixture_density_head import MixtureDensityHead
            self.mixture_head = MixtureDensityHead(
                n_features=SpatialParams.N_PARAMS,
                n_components=mixture_n_components,
                hidden=mixture_hidden,
            )
        # ContextualQuantileHead (IHI) : K quantiles non-paramétriques + médiane libre
        # + features {spatial_params, Q_sim, log Q_sim, indices hydrométéo, DOY}
        self.use_contextual_quantile_head = use_contextual_quantile_head
        self.cqh_n_features = cqh_n_features
        self.cqh_taus = cqh_taus
        self.cqh_hidden = cqh_hidden
        if use_contextual_quantile_head:
            from meandre.utils.contextual_quantile_head import ContextualQuantileHead
            self.contextual_quantile_head = ContextualQuantileHead(
                n_features=cqh_n_features,
                taus=tuple(cqh_taus),
                hidden=cqh_hidden,
            )
        # PhenologyModulator (IHI Phase B étape 1) : K_c modulé par GDD cumulé
        self.use_phenology_modulator = use_phenology_modulator
        if use_phenology_modulator:
            from meandre.temporal.phenology_modulator import PhenologyModulator
            self.phenology_modulator = PhenologyModulator()

        # Muskingum K and x are now per-node spatial params from the NeRF
        # (SpatialParams.K_musk_hours and SpatialParams.x_musk).

    def simulate(
        self,
        forcing: Tensor,
        initial_state: HydroState,
        graph: RiverGraph,
        node_coords: Tensor,
        territorial: TerritorialFeatures,
        withdrawals: WithdrawalData,
        day_of_year: Tensor,
        inject_noise: bool = False,
        dam_data: DamData | None = None,
        h_context: Tensor | None = None,
        tbptt_steps: int = 0,
        return_diagnostics: bool = False,
    ) -> tuple[Tensor, HydroState] | tuple[Tensor, HydroState, SimDiagnostics]:
        """Full forward pass: scan over timesteps.

        Args:
            forcing:      (n_timesteps, n_nodes, n_forcing)
            initial_state: HydroState at t=0
            graph:        RiverGraph
            node_coords:  (n_nodes, 2) [lon, lat]
            territorial:  TerritorialFeatures
            withdrawals:  WithdrawalData
            day_of_year:  (n_timesteps,) integer 1-366
            inject_noise: If True and state_noise module is present, add AR(1)
                          correlated noise to soil/SWE states each timestep.
                          Use True for ensemble generation, False for training.
            dam_data:     Optional DamData with per-timestep forced releases for
                          regulated reservoir nodes.  None = all lakes unregulated.
            h_context:    Optional GRU hidden state from a previous simulate()
                          call (e.g. spinup).  Pass model._last_h_context after
                          spinup to carry temporal memory into training.
            tbptt_steps:  Truncated BPTT interval in timesteps.  State and
                          Q_out_prev are detached every tbptt_steps steps,
                          limiting backward-pass depth to tbptt_steps instead
                          of the full sequence length.  0 = no truncation.
                          Recommended: 90 (one season) for long sequences.
            return_diagnostics: If True, return a SimDiagnostics object as a
                          third element with per-node, per-timestep fluxes:
                          etp, etr, snowmelt, lateral_mm (mm/day),
                          q_lateral, q_upstream (m³/s).
                          Disabled by default to keep memory low during training.
        Returns:
            Q_sim:        (n_timesteps, n_nodes) simulated discharge (m3/s)
            final_state:  HydroState at t=n_timesteps
            diagnostics:  SimDiagnostics (only when return_diagnostics=True)
        """
        n_timesteps = forcing.shape[0]
        state = initial_state

        # Static spatial params (computed once per simulate call).
        spatial_params = self.spatial_encoder(
            node_coords, territorial.to_tensor(),
        )
        # Clone BV3C2 : charge une fois les fractions d'occupation brutes par nœud.
        if getattr(self.vertical_column, "soil_mode", "meandre") == "clone" \
                and getattr(self.vertical_column, "_clone_static", "x") is None:
            self.vertical_column.set_clone_static(territorial)
        # Latitude par nœud pour l'ETP McGuinness (et_mode="mcguinness").
        # node_coords = [lon, lat] ; statique, posée une fois par simulate.
        self.vertical_column._node_lat = node_coords[:, 1]
        # Colonne fidèle Hydrotel : assemble params (NeRF→clone) + init état riche.
        if getattr(self, "column_mode", "meandre") == "hydrotel":
            # Cale la theta initiale sur l'init Hydrotel validé (0.9·thetas) au lieu
            # de la valeur du cache (0.3, trop sèche → gros réservoir à remplir avant
            # de produire = déficit de volume). frac=0 garde la theta du cache.
            frac = getattr(self, "column_theta_init_frac", 0.9)
            if frac > 0.0:
                import dataclasses
                state = dataclasses.replace(
                    state,
                    theta1=frac * spatial_params.porosity_1,
                    theta2=frac * spatial_params.porosity_2,
                    theta3=frac * spatial_params.porosity_3)
            self.vertical_column.setup_simulate(
                spatial_params, territorial, node_coords, state)
        K_musk = spatial_params.K_musk_hours * 3600.0  # hours → seconds
        x_musk = spatial_params.x_musk

        # Params de lac par nœud (NeRF) — constants dans le temps, posés sur la
        # couche de routage avant la boucle. None si la tête n'est pas activée.
        if getattr(self.spatial_encoder, "predict_lake_params", False):
            k_lake_n, beta_lake_n = self.spatial_encoder.lake_params(
                node_coords, territorial.to_tensor(),
            )
            self.routing._lake_k = k_lake_n
            self.routing._lake_beta = beta_lake_n
        else:
            self.routing._lake_k = None
            self.routing._lake_beta = None

        Q_all: list[Tensor] = []
        T_water_all: list[Tensor] = []
        state_buffer: list[Tensor] = []
        Q_out_prev = torch.zeros(self.n_nodes, device=forcing.device)
        # Diagnostic accumulators (only allocated when requested)
        diag_lists: dict[str, list[Tensor]] = (
            {k: [] for k in ("etp", "etr", "snowmelt", "lateral_mm",
                             "q_lateral", "q_upstream", "recharge",
                             "q_baseflow", "T_water", "swe",
                             "theta1", "theta2", "theta3",
                             "s_gw", "canopy", "wetland")}
            if return_diagnostics else {}
        )
        outflow_buffer = OutflowRingBuffer(
            self.n_nodes, depth=self.max_travel_time, device=forcing.device
        )

        # Lake storage state (m3); only allocated when lake nodes exist
        has_lakes = bool(graph.is_lake.any())
        lake_storage: Tensor | None = (
            torch.zeros(self.n_nodes, device=forcing.device) if has_lakes else None
        )
        # Use physical (un-normalised) drainage area for lake routing.
        # territorial.drainage_area_km2 may be z-score normalised (negative
        # values possible), which breaks the depth = S/area computation in
        # LakeModule.  Fall back to 1 km² per node if not available.
        if has_lakes:
            if territorial.area_km2_physical is not None:
                area_km2: Tensor | None = territorial.area_km2_physical.to(forcing.device)
            else:
                area_km2 = torch.ones(self.n_nodes, device=forcing.device)
        else:
            area_km2 = None

        # Local sub-watershed area for lateral inflow (mm/day → m³/s) conversion.
        # Must NOT use cumulative drainage area here — that would inflate q_lat
        # at downstream nodes by orders of magnitude.
        area_km2_local: Tensor | None = (
            territorial.area_km2_local.to(forcing.device)
            if territorial.area_km2_local is not None
            else None
        )

        # AR(1) state noise (for ensemble generation)
        noise: Tensor | None = None
        if inject_noise and self.state_noise is not None:
            noise = self.state_noise.init_noise(self.n_nodes, forcing.device)

        # Precompute temporal context for ALL timesteps before the loop.
        # One chunked GRU pass (O(T·N·d²)) instead of T separate calls.
        all_context: Tensor | None = None
        if self.use_temporal and self.temporal_encoder is not None:
            all_context, h_context = self.temporal_encoder.encode_sequence(
                forcing, day_of_year, h0=h_context
            )

        do_temp = self.temperature is not None

        # ── Main simulation loop (interleaved vertical + routing) ──────
        for t in range(n_timesteps):
            # Truncated BPTT: detach state and Q_out at chunk boundaries
            if tbptt_steps > 0 and t > 0 and t % tbptt_steps == 0:
                state = state.detach()
                Q_out_prev = Q_out_prev.detach()
                if getattr(self, "column_mode", "meandre") == "hydrotel":
                    self.vertical_column.detach_aux()

            # 1. Temporal context (indexed from pre-computed tensor)
            if all_context is not None:
                enriched = torch.cat([forcing[t], all_context[t]], dim=-1)
            else:
                enriched = forcing[t]

            # 2. Vertical balance (returns ColumnOutput)
            # Convert groundwater withdrawal (m³/s) → mm/day using local area.
            gw_w_mm: Tensor | None = None
            if area_km2_local is not None:
                gw_m3s = withdrawals.gw_withdrawal(t)
                if gw_m3s.abs().sum() > 0:
                    # mm/day = m³/s * 86400 s/day / (km² * 1e6 m²/km² / 1e3 mm/m)
                    #       = m³/s * 86.4 / km²
                    gw_w_mm = gw_m3s * 86.4 / torch.clamp(area_km2_local, min=1e-3)

            # Update GDD cumulé pour PhenologyModulator (IHI Phase B étape 1).
            # Reset chaque 1er janvier, cumule relu(T_mean - 10°C) sinon.
            if getattr(self, "use_phenology_modulator", False):
                from meandre.temporal.phenology_modulator import update_gdd_cum
                _T_mean_t = 0.5 * (forcing[t, :, 1] + forcing[t, :, 2])    # (T_min + T_max)/2
                _doy_val = day_of_year[t] if day_of_year is not None else None
                _doy_int = int(_doy_val.item()) if _doy_val is not None else 0
                state = HydroState(
                    theta1=state.theta1, theta2=state.theta2, theta3=state.theta3,
                    swe=state.swe, t_soil=state.t_soil,
                    canopy_storage=state.canopy_storage,
                    wetland_storage=state.wetland_storage,
                    S_gw=state.S_gw, T_water=state.T_water,
                    cold_content=state.cold_content,
                    gdd_cum=update_gdd_cum(state.gdd_cum, _T_mean_t, _doy_int),
                )

            if getattr(self, "column_mode", "meandre") == "hydrotel":
                vc_out = self.vertical_column.column_step(
                    enriched, state,
                    doy=day_of_year[t] if day_of_year is not None else None,
                    return_diagnostics=return_diagnostics,
                )
            else:
                vc_out = self.vertical_column(
                    enriched, state, spatial_params,
                    return_diagnostics=return_diagnostics,
                    gw_withdrawal_mm=gw_w_mm,
                    doy=day_of_year[t] if day_of_year is not None else None,
                    phenology_modulator=(
                        self.phenology_modulator
                        if getattr(self, "use_phenology_modulator", False) else None
                    ),
                )
            physics_state = vc_out.state

            # 3. State residual correction
            state_buffer.append(physics_state.to_tensor().detach())
            if len(state_buffer) > self.residual_history:
                state_buffer.pop(0)
            if self.use_residual and self.residual_corrector is not None and len(state_buffer) >= 2:
                history = torch.stack(state_buffer, dim=1)
                corrected_tensor = self.residual_corrector(
                    history, physics_state.to_tensor()
                )
                state = HydroState.from_tensor(corrected_tensor)
            else:
                state = physics_state

            # 3b. Inject AR(1) correlated noise
            if noise is not None and self.state_noise is not None:
                noise = self.state_noise.step(noise)
                state = HydroState(
                    theta1=torch.clamp(state.theta1 + noise[:, 0], min=0.0),
                    theta2=torch.clamp(state.theta2 + noise[:, 1], min=0.0),
                    theta3=torch.clamp(state.theta3 + noise[:, 2], min=0.0),
                    swe=torch.clamp(state.swe + noise[:, 3], min=0.0),
                    t_soil=state.t_soil,
                    canopy_storage=state.canopy_storage,
                    wetland_storage=state.wetland_storage,
                    S_gw=state.S_gw,
                    T_water=state.T_water,
                )

            lateral_inflow = vc_out.lateral_inflow

            # 4. Temperature lateral heat load
            H_lateral = None
            T_air_t = None
            R_n_t = None
            if do_temp:
                T_air_t = 0.5 * (forcing[t, :, 1] + forcing[t, :, 2])
                R_n_t = forcing[t, :, 3]
                _area = area_km2_local if area_km2_local is not None else (
                    area_km2 if area_km2 is not None else
                    torch.ones(self.n_nodes, device=forcing.device)
                )
                H_lateral, _ = self.temperature.lateral_heat_load(
                    T_air_t,
                    vc_out.snowmelt,
                    vc_out.Q_baseflow,
                    lateral_inflow,
                    spatial_params.T_gw,
                    _area,
                )

            # 5. Routing
            Q_out, lake_storage, T_water_t = self.routing(
                lateral_inflow, graph, Q_out_prev,
                outflow_buffer, withdrawals, t,
                K_musk, x_musk,
                lake_storage=lake_storage,
                area_km2=area_km2,
                dam_data=dam_data,
                area_km2_local=area_km2_local,
                H_lateral=H_lateral,
                T_air=T_air_t,
                R_n=R_n_t,
                K_atm=spatial_params.K_atm if do_temp else None,
            )

            outflow_buffer.push(Q_out)
            Q_out_prev = Q_out
            Q_all.append(Q_out)
            if do_temp and T_water_t is not None:
                T_water_all.append(T_water_t)

            # Collect diagnostics
            if return_diagnostics:
                diag_lists["etp"].append(vc_out.diag["etp"])
                diag_lists["etr"].append(vc_out.diag["etr"])
                diag_lists["snowmelt"].append(vc_out.diag["snowmelt"])
                diag_lists["lateral_mm"].append(vc_out.diag["lateral_mm"])
                diag_lists["recharge"].append(vc_out.recharge)
                diag_lists["q_baseflow"].append(vc_out.Q_baseflow)
                conv_area = area_km2_local if area_km2_local is not None else area_km2
                q_lat_m3s = (
                    lateral_inflow * 1e-3 * conv_area * 1e6 / 86400.0
                    if conv_area is not None else lateral_inflow
                )
                q_up = torch.zeros(self.n_nodes, device=forcing.device)
                if graph.n_edges > 0:
                    q_up.scatter_add_(
                        0, graph.edge_index[1],
                        Q_out[graph.edge_index[0]],
                    )
                diag_lists["q_lateral"].append(q_lat_m3s)
                diag_lists["q_upstream"].append(q_up)
                diag_lists["swe"].append(state.swe)
                diag_lists["theta1"].append(state.theta1)
                diag_lists["theta2"].append(state.theta2)
                diag_lists["theta3"].append(state.theta3)
                diag_lists["s_gw"].append(state.S_gw)
                diag_lists["canopy"].append(state.canopy_storage)
                diag_lists["wetland"].append(state.wetland_storage)
                diag_lists["T_water"].append(
                    T_water_t if T_water_t is not None
                    else torch.full((self.n_nodes,), float('nan'), device=forcing.device)
                )

        # Store final GRU hidden state for optional carryover to next call
        self._last_h_context: Tensor | None = (
            h_context.detach() if h_context is not None else None
        )

        # Empty sequence (e.g. zero-length spinup): return zeros
        if n_timesteps == 0:
            Q_sim = torch.zeros(0, self.n_nodes, device=forcing.device)
            if not return_diagnostics:
                return Q_sim, state
            z = torch.zeros(0, self.n_nodes, device=forcing.device)
            return Q_sim, state, SimDiagnostics(
                etp=z, etr=z, snowmelt=z, lateral_mm=z,
                recharge=z, q_baseflow=z, q_lateral=z, q_upstream=z,
            )

        Q_sim = torch.stack(Q_all, dim=0)

        if not return_diagnostics:
            return Q_sim, state

        diagnostics = SimDiagnostics(
            etp=torch.stack(diag_lists["etp"], dim=0),
            etr=torch.stack(diag_lists["etr"], dim=0),
            snowmelt=torch.stack(diag_lists["snowmelt"], dim=0),
            lateral_mm=torch.stack(diag_lists["lateral_mm"], dim=0),
            q_lateral=torch.stack(diag_lists["q_lateral"], dim=0),
            q_upstream=torch.stack(diag_lists["q_upstream"], dim=0),
            recharge=torch.stack(diag_lists["recharge"], dim=0),
            q_baseflow=torch.stack(diag_lists["q_baseflow"], dim=0),
            swe=torch.stack(diag_lists["swe"], dim=0),
            theta1=torch.stack(diag_lists["theta1"], dim=0),
            theta2=torch.stack(diag_lists["theta2"], dim=0),
            theta3=torch.stack(diag_lists["theta3"], dim=0),
            s_gw=torch.stack(diag_lists["s_gw"], dim=0),
            canopy=torch.stack(diag_lists["canopy"], dim=0),
            wetland=torch.stack(diag_lists["wetland"], dim=0),
            T_water=torch.stack(diag_lists["T_water"], dim=0),
        )
        return Q_sim, state, diagnostics

    # ---- Uncertainty regularisation ----

    # ---- Persistence ----

    def save(self, path: str | Path) -> None:
        torch.save({
            "state_dict": self.state_dict(),
            "use_temporal": self.temporal_encoder is not None,
            "use_residual": self.residual_corrector is not None,
            "use_tta": hasattr(self.routing, "tta") and self.routing.tta is not None,
            "init_kwargs": {
                "n_nodes": self.n_nodes,
                "n_forcing": self.n_forcing,
                "context_window": self.context_window,
                "residual_history": self.residual_history,
                "max_travel_time": self.max_travel_time,
                # Store module EXISTENCE (not runtime flag) so from_checkpoint
                # creates matching architecture.  Runtime flags are restored
                # separately via use_temporal/use_residual/use_tta keys.
                "use_temporal": self.temporal_encoder is not None,
                "use_residual": self.residual_corrector is not None,
                "use_travel_time_attn": hasattr(self.routing, "tta") and self.routing.tta is not None,
                "use_state_noise": self.state_noise is not None,
                "use_temperature": self.temperature is not None,
                "n_territorial": self.spatial_encoder.n_territorial,
                "dropout": self.spatial_encoder.drop1.p if hasattr(self.spatial_encoder, "drop1") else 0.0,
                "n_state_vars": self.residual_corrector.gru.input_size if self.residual_corrector is not None else HydroState.N_VARS,
                "clamp_min": self.residual_corrector.clamp_min if self.residual_corrector is not None else -50.0,
                "clamp_max": self.residual_corrector.clamp_max if self.residual_corrector is not None else 500.0,
                "concrete_dropout": (
                    self.temporal_encoder.drop.__class__.__name__ == "ConcreteDropout"
                    if self.temporal_encoder is not None else False
                ),
                "concrete_init_p": (
                    self.temporal_encoder.drop.p.detach().item()
                    if self.temporal_encoder is not None
                    and hasattr(self.temporal_encoder.drop, "p")
                    else 0.05
                ),
                "use_quantile_head": getattr(self, "use_quantile_head", False),
                "quantile_taus": (
                    tuple(self.quantile_head.taus)
                    if getattr(self, "use_quantile_head", False)
                    else (0.05, 0.10, 0.25, 0.75, 0.90, 0.95)
                ),
                "use_mixture_head": getattr(self, "use_mixture_head", False),
                "mixture_n_components": getattr(self, "mixture_n_components", 10),
                "mixture_hidden": getattr(self, "mixture_hidden", 64),
                "use_contextual_quantile_head": getattr(self, "use_contextual_quantile_head", False),
                "cqh_n_features": getattr(self, "cqh_n_features", 45),
                "cqh_taus": (tuple(self.contextual_quantile_head.taus)
                             if getattr(self, "use_contextual_quantile_head", False)
                             else (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)),
                "cqh_hidden": getattr(self, "cqh_hidden", 64),
                "use_phenology_modulator": getattr(self, "use_phenology_modulator", False),
                "routing_mode": getattr(self.routing, "routing_mode", "level"),
                "routing_substeps": getattr(self.routing.muskingum, "n_substeps", 2),
                "discharge_dependent_celerity": getattr(self.routing, "dq_celerity", False),
                "dq_beta": getattr(self.routing, "dq_beta", 0.4),
                "dq_qref_specific": getattr(self.routing, "dq_qref_specific", 0.01),
                "pure_advection": getattr(self.routing, "pure_advection", False),
                "dynamic_atten": getattr(self.routing, "dynamic_atten", False),
                "da_beta": getattr(self.routing, "da_beta", 2.0),
                "da_qref_specific": getattr(self.routing, "da_qref_specific", 0.05),
                "predict_lake_params": getattr(self.spatial_encoder, "predict_lake_params", False),
                "use_latent_codes": getattr(self.spatial_encoder, "use_latent_codes", False),
                "latent_dim": getattr(self.spatial_encoder, "latent_dim", 8) or 8,
                "latent_mode": getattr(self.spatial_encoder, "latent_mode", "additive"),
                "soil_quickflow_reservoir": getattr(self.vertical_column.soil, "use_quickflow_reservoir", False),
                "soil_quickflow_beta": getattr(self.vertical_column.soil, "quickflow_beta", 0.5),
                "soil_separate_infil_capacity": getattr(self.vertical_column.soil, "use_separate_infil_capacity", False),
                "soil_frozen_gate": getattr(self.vertical_column.soil, "use_frozen_gate", False),
                "soil_runoff_clean": getattr(self.vertical_column.soil, "runoff_clean", False),
                "soil_mode": getattr(self.vertical_column, "soil_mode", "meandre"),
                "soil_clone_substep": getattr(getattr(self.vertical_column, "soil_clone", None), "n_substep", 48),
                "soil_clone_krec_init": 1e-5,  # init scalaire seul ; cl_krec_raw appris est dans le state_dict
                "et_mode": getattr(getattr(self.vertical_column, "et", None), "et_mode", "penman"),
                "column_mode": getattr(self, "column_mode", "meandre"),
                "column_theta_init_frac": getattr(self, "column_theta_init_frac", 0.9),
                "use_overland_uh": getattr(self.vertical_column, "use_overland_uh", False),
                "use_hillslope_uh": getattr(self.vertical_column, "use_hillslope_uh", False),
            },
        }, path)

    def load(self, path: str | Path) -> None:
        """Load weights and module flags into this model instance in-place."""
        device = next(self.parameters()).device
        checkpoint = torch.load(path, map_location=device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            sd = checkpoint["state_dict"]
            # Backward compatibility: pad fc_out if old checkpoint had fewer params
            fc_out_key = "spatial_encoder.fc_out.weight"
            if fc_out_key in sd:
                old_n = sd[fc_out_key].shape[0]
                new_n = self.spatial_encoder.fc_out.out_features
                if old_n < new_n:
                    # Pad weight and bias with zeros for new params
                    pad_w = torch.zeros(new_n - old_n, sd[fc_out_key].shape[1], device=device)
                    sd[fc_out_key] = torch.cat([sd[fc_out_key], pad_w], dim=0)
                    bias_key = "spatial_encoder.fc_out.bias"
                    if bias_key in sd:
                        pad_b = torch.zeros(new_n - old_n, device=device)
                        sd[bias_key] = torch.cat([sd[bias_key], pad_b], dim=0)

            # Backward compatibility: pad fc1/fc2 if territorial features grew.
            # Use small Kaiming-scaled init (not zeros) so new features have
            # a non-zero gradient signal from the start.
            fc1_key = "spatial_encoder.fc1.weight"
            if fc1_key in sd:
                old_in = sd[fc1_key].shape[1]
                new_in = self.spatial_encoder.fc1.in_features
                if old_in < new_in:
                    delta = new_in - old_in
                    fan_in = new_in  # full input dim after padding
                    std = (2.0 / fan_in) ** 0.5 * 0.1  # Kaiming * 0.1 scale
                    # fc1: small random init for new input columns
                    sd[fc1_key] = torch.cat([
                        sd[fc1_key],
                        torch.randn(sd[fc1_key].shape[0], delta, device=device) * std,
                    ], dim=1)
                    # fc2 has a skip connection: hidden + in_dim
                    fc2_key = "spatial_encoder.fc2.weight"
                    if fc2_key in sd:
                        fan_in2 = sd[fc2_key].shape[1] + delta
                        std2 = (2.0 / fan_in2) ** 0.5 * 0.1
                        sd[fc2_key] = torch.cat([
                            sd[fc2_key],
                            torch.randn(sd[fc2_key].shape[0], delta, device=device) * std2,
                        ], dim=1)
                    # Tag layers that received new features for discriminative LR
                    self._padded_layers = {"spatial_encoder.fc1", "spatial_encoder.fc2"}
                elif old_in > new_in:
                    # Features were removed (e.g. depth_to_bedrock_m excluded).
                    # Truncate trailing columns — safe when the removed feature
                    # was constant (zero) so the corresponding weights are ~0.
                    sd[fc1_key] = sd[fc1_key][:, :new_in]
                    fc2_key = "spatial_encoder.fc2.weight"
                    if fc2_key in sd:
                        delta = old_in - new_in
                        sd[fc2_key] = sd[fc2_key][:, :sd[fc2_key].shape[1] - delta]

            # Remove legacy params now handled by SpatialParams
            for legacy_key in ("log_K_musk", "logit_x_musk",
                               "vertical_column.soil.log_k_interflow"):
                sd.pop(legacy_key, None)

            # Backward compatibility: pad residual_corrector state dict if
            # n_state_vars grew (e.g. when cold_content was added: 9 → 10).
            # Drop incompatible-shaped corrector weights — they'll re-init.
            current_n_vars = HydroState.N_VARS
            corrector_keys = [k for k in sd if k.startswith("residual_corrector.")]
            for k in corrector_keys:
                w = sd[k]
                # Check if any dim equals an old N_VARS value (7, 8, 9) and
                # mismatches current
                if w.dim() >= 1 and w.shape[0] in (7, 8, 9) and w.shape[0] != current_n_vars:
                    sd.pop(k)
                elif w.dim() >= 2 and w.shape[1] in (7, 8, 9) and w.shape[1] != current_n_vars:
                    sd.pop(k)

            # Drop any key whose shape mismatches the current model (e.g. when
            # N_PARAMS grew 36→37 with vsa_b: noise_head/quantile heads change
            # input dim). strict=False ignores missing/unexpected keys but NOT
            # size mismatches → filter them so cross-architecture warm-start works.
            own = self.state_dict()
            mism = [k for k, v in sd.items() if k in own and own[k].shape != v.shape]
            if mism:
                print(f"[load] {len(mism)} clés de forme incompatible ignorées "
                      f"(warm-start cross-archi) : {mism[:4]}{'...' if len(mism) > 4 else ''}", flush=True)
                for k in mism:
                    sd.pop(k)

            self.load_state_dict(sd, strict=False)
            self.use_temporal = checkpoint.get("use_temporal", False)
            self.use_residual = checkpoint.get("use_residual", False)
            if hasattr(self.routing, "use_tta"):
                self.routing.use_tta = checkpoint.get("use_tta", False)
        else:
            # Legacy checkpoint: raw state_dict
            self.load_state_dict(checkpoint, strict=False)

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, **kwargs
    ) -> "HydroModel":
        """Reconstruct a HydroModel instance from a checkpoint.

        If the checkpoint contains ``init_kwargs`` (saved by newer ``save()``),
        the model is fully self-contained and no extra kwargs are needed.
        Any explicit ``**kwargs`` override the stored values.
        """
        checkpoint = torch.load(path, map_location="cpu")
        if isinstance(checkpoint, dict) and "init_kwargs" in checkpoint:
            stored = checkpoint["init_kwargs"]
            stored.update(kwargs)  # caller overrides win
            model = cls(**stored)
        else:
            model = cls(**kwargs)
        model.load(path)
        return model