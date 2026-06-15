"""Vertical column orchestrator — chains all vertical balance modules.

Processes one timestep for all nodes simultaneously (vectorised).
Input:  enriched forcing (raw + temporal context), current HydroState, SpatialParams
Output: ColumnOutput with lateral inflow, updated state, and component fluxes
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from meandre.spatial.field_network import SpatialParams
from meandre.temporal.temporal_modulator import TemporalModulator
from meandre.utils.state import HydroState
from meandre.vertical.aquifer import AquiferModule
from meandre.vertical.evapotranspiration import ETModule
from meandre.vertical.frost import FrostModule
from meandre.vertical.interception import InterceptionModule
from meandre.vertical.snow import SnowModule
from meandre.vertical.soil import SoilModule
from meandre.vertical.wetland import WetlandModule


@dataclass
class ColumnOutput:
    """Output from VerticalColumn.forward().

    Always returned (no conditional tuple unpacking).
    """

    lateral_inflow: Tensor   # (n_nodes,) mm/day — total input to routing
    state: HydroState        # updated state after one day
    snowmelt: Tensor         # (n_nodes,) mm/day — for temperature module
    recharge: Tensor         # (n_nodes,) mm/day — soil L3 drainage (pre-aquifer)
    Q_baseflow: Tensor       # (n_nodes,) mm/day — GW contribution (isolated)
    diag: dict | None        # per-node diagnostic tensors, or None


class VerticalColumn(nn.Module):
    """Chain: Snow -> Frost -> Interception -> ET -> Soil -> Wetland -> Aquifer.

    All modules are applied per-node and vectorised over the full graph.
    """

    # Order: rain_hours, interception_capacity, C_f.
    # Pas K_c (déjà phenology hardcodée), pas f_root (contrainte softmax).
    MODULATED = ("rain_hours", "interception_capacity", "C_f")

    def __init__(self, soil_z1: float = 0.30, soil_vsa_b: float = 2.5,
                 soil_quickflow_reservoir: bool = False,
                 soil_quickflow_beta: float = 0.5,
                 soil_separate_infil_capacity: bool = False,
                 soil_frozen_gate: bool = False,
                 soil_mode: str = "meandre",
                 use_hillslope_uh: bool = False) -> None:
        super().__init__()
        self.snow = SnowModule()
        self.frost = FrostModule()
        self.interception = InterceptionModule()
        self.et = ETModule()
        self.soil = SoilModule(z1=soil_z1, vsa_b=soil_vsa_b,
                               use_quickflow_reservoir=soil_quickflow_reservoir,
                               quickflow_beta=soil_quickflow_beta,
                               use_separate_infil_capacity=soil_separate_infil_capacity,
                               use_frozen_gate=soil_frozen_gate)
        # ── Mode SOL FIDÈLE Hydrotel (ronde d'équivalence 2026-06-15) ──────
        # soil_mode="hydrotel" : remplace le bilan vertical de méandre (van
        # Genuchten + VSA + partition softmax + aquifère) par le BV3C2 EXACT
        # d'Hydrotel (Campbell, hortonien plafonné Ks, portes gel/saturation,
        # interflow pente, baseflow krec). Le baseflow krec étant interne au sol
        # fidèle, l'aquifère est COURT-CIRCUITÉ en mode hydrotel. Les params de
        # forme Campbell (b, psis par couche) et krec, absents de SpatialParams,
        # sont GLOBAUX apprenables, init calibrés (silt_loam/loam, bv3c.csv).
        # ks/thetas/thetacc/thetapf viennent du NeRF (round structurel) ; ils
        # seront injectés calibrés à la validation PHYSITEL finale.
        self.soil_mode = str(soil_mode)
        if self.soil_mode == "hydrotel":
            from meandre.vertical.bv3c_hydrotel import BV3CHydrotel, SOIL_TEXTURES, KREC_DEFAULT
            import math as _m
            self.soil_hydrotel = BV3CHydrotel(n_substeps_max=48)
            tx = (SOIL_TEXTURES["silt_loam"], SOIL_TEXTURES["loam"], SOIL_TEXTURES["loam"])
            for i, t in enumerate(tx, start=1):
                # b = 1/lambda, psis (m) — globaux apprenables (log pour positivité).
                setattr(self, f"bv_log_b{i}", nn.Parameter(torch.tensor(_m.log(1.0 / t["lam"]))))
                setattr(self, f"bv_log_psis{i}", nn.Parameter(torch.tensor(_m.log(t["psis"]))))
            self.bv_log_krec = nn.Parameter(torch.tensor(_m.log(KREC_DEFAULT)))
        self.wetland = WetlandModule()
        self.aquifer = AquiferModule()
        # Hydrogramme unitaire de VERSANT (cascade de Nash à 2 réservoirs).
        # Lisse le ruissellement RAPIDE par étalement des temps de parcours de
        # versant, AVANT le canal. C'est le lissage d'Hydrotel (onde cinématique
        # de versant) qui préserve les pics, contrairement à l'atténuation du
        # Muskingum qui les rabote. Diagnostic 2026-06-15 : le lissage doit être
        # sur le versant, pas sur le canal. k apprenable (jours), init ~1 jour
        # (optimum offline). État : uh_s1, uh_s2 par nœud.
        self.use_hillslope_uh = bool(use_hillslope_uh)
        if self.use_hillslope_uh:
            import math as _m
            # DEUX hydrogrammes séparés (fidèle à Hydrotel) : surface POINTUE
            # (k court → pic préservé), interflow LARGE (k long → douceur kge).
            self.log_uh_k_surf = nn.Parameter(torch.tensor(_m.log(0.3)))   # ~0.3 j
            self.log_uh_k_inter = nn.Parameter(torch.tensor(_m.log(2.5)))  # ~2.5 j
        self.temporal_modulator = TemporalModulator(n_modulated=len(self.MODULATED))

    def forward(
        self,
        forcing: Tensor,
        state: HydroState,
        params: SpatialParams,
        return_diagnostics: bool = False,
        gw_withdrawal_mm: Tensor | None = None,
        doy: Tensor | int | None = None,
        phenology_modulator: "PhenologyModulator | None" = None,
    ) -> ColumnOutput:
        """
        Args:
            forcing: (n_nodes, n_forcing_vars)
                Columns (in order): P, T_min, T_max, R_n, u2, e_a
                Optional context appended after first 6 columns.
            state:   Current HydroState (n_nodes,) per variable.
            params:  SpatialParams per node from spatial field network.
            return_diagnostics: If True, populate diag dict with
                intermediate fluxes: etp, etr, snowmelt, lateral_mm.
        Returns:
            ColumnOutput with lateral_inflow, new_state, snowmelt,
            Q_baseflow, and optional diagnostics.
        """
        P      = forcing[:, 0]
        T_min  = forcing[:, 1]
        T_max  = forcing[:, 2]
        R_n    = forcing[:, 3]
        u2     = forcing[:, 4]
        e_a    = forcing[:, 5]
        T_air  = 0.5 * (T_min + T_max)

        # ── Temporal modulation: rain_hours, interception_capacity, C_f ──
        # Si doy fourni, applique cycle saisonnier learnable + réactivité P/T.
        # Sinon, modulator = 1 (pas de modulation).
        if doy is not None:
            mod = self.temporal_modulator(
                doy if isinstance(doy, Tensor) else torch.tensor(float(doy)),
                P, T_air,
            )  # (n_nodes, 3)
            rain_hours_eff = getattr(params, "rain_hours", None)
            if rain_hours_eff is not None:
                rain_hours_eff = rain_hours_eff * mod[:, 0]
            interception_cap_eff = params.interception_capacity * mod[:, 1]
            C_f_eff = params.C_f * mod[:, 2]
        else:
            rain_hours_eff = getattr(params, "rain_hours", None)
            interception_cap_eff = params.interception_capacity
            C_f_eff = params.C_f

        # 1. Snow (with cold content tracking)
        P_eff, swe_new, cold_content_new = self.snow(
            P, T_air, state.swe,
            C_f_eff, params.T_melt, params.T_snow,
            cold_content=state.cold_content,
        )

        # 2. Frost: update soil temperature and reduce K_sat for all 3 layers
        K_sat_1_eff, t_soil_new = self.frost(
            T_air, state.t_soil, params.K_sat_1, params.frost_alpha, params.alpha_T
        )
        # Apply same frost factor to layers 2 and 3
        frost_factor = K_sat_1_eff / (params.K_sat_1 + 1e-8)
        K_sat_2_eff = params.K_sat_2 * frost_factor
        K_sat_3_eff = params.K_sat_3 * frost_factor
        # État de gel pour la porte gel du sol : ~1 quand K_sat effondré (gelé),
        # ~0 dégelé. Atténué par la neige au sol (la fonte sur sol gelé ruisselle,
        # mais un manteau épais découple le sol de l'air → moins gelé en surface).
        frozen_frac = torch.clamp(1.0 - frost_factor, 0.0, 1.0)

        # 3. Interception: compute ETP first for canopy evap.
        # Apply crop/calibration coefficient K_c (Hydrotel-style multiplier),
        # modulated by a phenology factor: warm + no snow → growing season,
        # cold OR snow on ground → dormant. Prevents over-evaporation in spring
        # when the soil is saturated by snowmelt but the canopy isn't yet active.
        K_c_base = getattr(params, "K_c", None)
        # Modulation phénologique : 3 modes possibles selon disponibilité.
        if phenology_modulator is not None and K_c_base is not None:
            # IHI Phase B : modulateur appris sur GDD cumulé (4 params nommés)
            # Update state.gdd_cum déjà fait dans simulate() avant cet appel.
            K_c_eff = phenology_modulator(state.gdd_cum, K_c_base)
            # Multiplier par "snow suppression" : pas de transpiration sous neige
            # (préservation du garde-fou physique de la version hardcodée)
            snow_suppression = torch.exp(-state.swe / 10.0)
            K_c_eff = K_c_eff * snow_suppression
        else:
            # Fallback hardcoded (rétrocompat) : 0.3 (dormant) → 1.0 (full growing).
            # T_air > 5°C threshold ; SWE > 5-10 mm supprime la transpiration.
            phenology = torch.sigmoid((T_air - 5.0) / 2.0) * torch.exp(-state.swe / 10.0)
            season_modulator = 0.3 + 0.7 * phenology
            if K_c_base is not None:
                K_c_eff = K_c_base * season_modulator
            else:
                K_c_eff = season_modulator  # bare phenology when no K_c

        ETP_approx = self.et.penman_monteith(T_min, T_max, R_n, u2, e_a) * K_c_eff
        P_thru, E_canopy, canopy_new = self.interception(
            P_eff, ETP_approx, state.canopy_storage, interception_cap_eff
        )

        # 4. ET per layer (updated after canopy)
        ET1, ET2, ET3, _ = self.et(
            T_min, T_max, R_n, u2, e_a,
            state.theta1, state.theta2, state.theta3,
            params.theta_wp_1, params.theta_wp_2, params.theta_wp_3,
            params.theta_fc_1, params.theta_fc_2, params.theta_fc_3,
            params.f_root_1,  params.f_root_2,   params.f_root_3,
            E_canopy,
            K_c=K_c_eff,
        )

        # 5. Soil balance (frost-modified K_sat for layer 1)
        Q_baseflow_faithful = None
        if self.soil_mode == "hydrotel":
            # ── Sol BV3C2 fidèle Hydrotel (baseflow interne, aquifère bypass) ──
            z2v = getattr(params, 'Z2', None); z3v = getattr(params, 'Z3', None)
            if z2v is None: z2v = torch.full_like(state.theta1, self.soil.z2_default)
            if z3v is None: z3v = torch.full_like(state.theta1, self.soil.z3_default)
            gp = lambda name: torch.exp(getattr(self, name))
            p_bv = {
                "z1": torch.full_like(state.theta1, self.soil.z1), "z2": z2v, "z3": z3v,
                "ks1": K_sat_1_eff / 24.0, "ks2": K_sat_2_eff / 24.0, "ks3": K_sat_3_eff / 24.0,
                "b1": gp("bv_log_b1"), "b2": gp("bv_log_b2"), "b3": gp("bv_log_b3"),
                "psis1": gp("bv_log_psis1"), "psis2": gp("bv_log_psis2"), "psis3": gp("bv_log_psis3"),
                "thetas1": params.porosity_1, "thetas2": params.porosity_2, "thetas3": params.porosity_3,
                "slope": torch.full_like(state.theta1, 0.04),   # TODO per-nœud via territorial
                "krec": gp("bv_log_krec"), "coef_recharge": torch.zeros_like(state.theta1),
            }
            frozen_bool = t_soil_new < 0.0
            runoff_f, interflow_f, base_f, rech_f, (theta1_new, theta2_new, theta3_new), _ = self.soil_hydrotel(
                state.theta1, state.theta2, state.theta3, P_thru, ET1 + ET2 + ET3,
                frozen_bool, state.swe, p_bv,
            )
            R_surface = runoff_f; interflow = interflow_f
            recharge = torch.zeros_like(state.theta1)
            S_uz_new = getattr(state, 'S_uz', None)
            Q_baseflow_faithful = base_f + rech_f
        else:
            theta1_new, theta2_new, theta3_new, R_surface, interflow, recharge, S_uz_new = self.soil(
                P_thru, ET1, ET2, ET3,
                state.theta1, state.theta2, state.theta3,
                K_sat_1_eff, K_sat_2_eff, K_sat_3_eff,
                params.porosity_1, params.porosity_2, params.porosity_3,
                params.theta_fc_1, params.theta_fc_2, params.theta_fc_3,
                params.theta_wp_1, params.theta_wp_2, params.theta_wp_3,
                f_vert_1=params.f_vert_1,
                f_vert_2=params.f_vert_2,
                f_vert_3=params.f_vert_3,
                vg_n=getattr(params, 'vg_n', None),
                z2=getattr(params, 'Z2', None),
                z3=getattr(params, 'Z3', None),
                rain_hours=rain_hours_eff,
                vsa_b=getattr(params, 'vsa_b', None),
                S_uz=getattr(state, 'S_uz', None),
                frozen_frac=frozen_frac,
            )

        # 6. Wetland
        Q_wetland, R_direct, wetland_new = self.wetland(
            R_surface, state.wetland_storage, params.f_wetland,
        )

        # 7. Aquifer: intercept soil recharge, delay through GW storage
        # Groundwater withdrawals act directly on S_gw (not on stream Q).
        # Mode hydrotel : baseflow krec interne au sol fidèle → aquifère bypass.
        if self.soil_mode == "hydrotel":
            Q_baseflow = Q_baseflow_faithful
            S_gw_new = state.S_gw
        else:
            Q_baseflow, S_gw_new = self.aquifer(
                recharge, state.S_gw, params.k_gw,
                gw_withdrawal=gw_withdrawal_mm,
            )

        # Hydrogrammes de versant SÉPARÉS (cascades de Nash) : surface POINTUE
        # (préserve le pic), interflow LARGE (douceur jour-à-jour pour le kge).
        # Le lissage vient de l'étalement des temps de parcours, PAS d'une
        # atténuation → les pics sont préservés. Baseflow direct (aquifère lisse).
        if self.use_hillslope_uh:
            def nash(inflow, s1, s2, log_k):
                if s1 is None: s1 = torch.zeros_like(inflow)
                if s2 is None: s2 = torch.zeros_like(inflow)
                k = torch.nn.functional.softplus(log_k) + 0.05      # jours
                a = 1.0 - torch.exp(-1.0 / k)                       # relâché/jour
                s1n = s1 + inflow; o1 = s1n * a; s1_new = s1n - o1
                s2n = s2 + o1;     o2 = s2n * a; s2_new = s2n - o2
                return o2, s1_new, s2_new
            surf_in = R_direct + Q_wetland
            surf_out, uh_s1_new, uh_s2_new = nash(
                surf_in, getattr(state, "uh_s1", None), getattr(state, "uh_s2", None),
                self.log_uh_k_surf)
            inter_out, uh_s3_new, uh_s4_new = nash(
                interflow, getattr(state, "uh_s3", None), getattr(state, "uh_s4", None),
                self.log_uh_k_inter)
            lateral_inflow = surf_out + inter_out + Q_baseflow
        else:
            uh_s1_new = getattr(state, "uh_s1", None)
            uh_s2_new = getattr(state, "uh_s2", None)
            uh_s3_new = getattr(state, "uh_s3", None)
            uh_s4_new = getattr(state, "uh_s4", None)
            lateral_inflow = R_direct + Q_wetland + interflow + Q_baseflow  # mm/day

        # Snowmelt (always computed — needed for temperature module)
        snowmelt = torch.clamp(state.swe - swe_new, min=0.0)

        new_state = HydroState(
            theta1=theta1_new,
            theta2=theta2_new,
            theta3=theta3_new,
            swe=swe_new,
            t_soil=t_soil_new,
            canopy_storage=canopy_new,
            wetland_storage=wetland_new,
            S_gw=S_gw_new,
            T_water=state.T_water,  # updated later by routing temperature
            cold_content=cold_content_new,
            gdd_cum=state.gdd_cum,  # préservé (mis à jour dans simulate avant cet appel)
            S_uz=S_uz_new,
            uh_s1=uh_s1_new,
            uh_s2=uh_s2_new,
            uh_s3=uh_s3_new,
            uh_s4=uh_s4_new,
        )

        diag = None
        if return_diagnostics:
            diag = {
                "etp":        ETP_approx,
                "etr":        E_canopy + ET1 + ET2 + ET3,
                "snowmelt":   snowmelt,
                "lateral_mm": lateral_inflow,
                "recharge":   recharge,
                "q_baseflow": Q_baseflow,
            }

        return ColumnOutput(
            lateral_inflow=lateral_inflow,
            state=new_state,
            snowmelt=snowmelt,
            recharge=recharge,
            Q_baseflow=Q_baseflow,
            diag=diag,
        )
