"""3-layer soil water balance module (BV3C2-inspired).

Three soil layers:
    Layer 1  0-30 cm    surface, evaporation, saturation-excess runoff
    Layer 2  30-100 cm  root zone, transpiration
    Layer 3  100-200 cm deep storage, groundwater recharge

State:  theta1, theta2, theta3 (volumetric water content, m3/m3)
Fluxes: q12, q23, q_recharge (inter-layer Darcy flow, m/day)
        R_surface (saturation-excess runoff, mm/day)

Inter-layer fluxes use van Genuchten hydraulic functions — both K(theta) and
psi(theta) are smooth, so gradients flow through the soil column.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from meandre.utils.differentiable import soft_relu, soft_threshold


# Layer thicknesses (m)
Z1, Z2, Z3 = 0.30, 0.70, 1.00  # 0-30, 30-100, 100-200 cm


class SoilModule(nn.Module):
    """Differentiable 3-layer soil water balance.

    van Genuchten functions are used for K(theta) and psi(theta) — both
    smooth and differentiable everywhere.
    """

    def __init__(
        self,
        z1: float = Z1,
        z2_default: float = Z2,
        z3_default: float = Z3,
        sharpness: float = 50.0,
        vg_alpha: float = 1.0,
        use_infiltration_excess: bool = True,
        default_rain_hours: float = 6.0,
        vsa_b: float = 2.5,
        use_quickflow_reservoir: bool = False,
        quickflow_beta: float = 0.5,
        use_separate_infil_capacity: bool = False,
    ) -> None:
        super().__init__()
        # Z1 (root zone surface) reste fixe — sémantique stable (~30cm).
        # Z2, Z3 deviennent per-node via forward() args; defaults pour fallback.
        self.z1 = float(z1)
        self.z2_default = float(z2_default)
        self.z3_default = float(z3_default)
        self.sharpness = sharpness
        self.vg_alpha = vg_alpha
        self.use_infiltration_excess = use_infiltration_excess
        self.default_rain_hours = default_rain_hours
        self.vsa_b = float(vsa_b)

        # ── Réservoir supérieur à seuil (HBV-EC : K0/UZL/K1) ──────────────
        # Le chaînon manquant pour les pics (cf. inspection Raven 2026-06-14) :
        # l'interflow linéaire actuel est lisse, jamais flashy. On le route à
        # travers un stock S_uz à DEUX sorties : vidange lente Q1 = K1·S_uz
        # toujours, et bouffée rapide Q0 = K0·softplus(β(S_uz−UZL))/β qui ne
        # s'ouvre qu'au-dessus du seuil UZL. Le seuil crée la crue. Trois
        # paramètres GLOBAUX apprenables (pas NeRF pour la PoC : ne casse pas
        # les checkpoints 37-params, donc warm-start possible depuis le best
        # multiobj → attribution propre de l'effet réservoir seul). Promotion
        # en champ NeRF par nœud prévue au refactor RavenTorch une fois prouvé.
        self.use_quickflow_reservoir = bool(use_quickflow_reservoir)
        self.quickflow_beta = float(quickflow_beta)
        # Bornes (1/jour pour K, mm pour UZL) — gamme HBV-light boréale.
        self._qf_bounds = {"k0": (0.1, 0.9), "k1": (0.01, 0.4), "uzl": (2.0, 60.0)}
        # Fraction de la drainage verticale RAPIDE de la couche 1 (q_vert_1, qui
        # part normalement en profondeur et se fait avaler) détournée vers le
        # réservoir supérieur. C'est la moitié GÉNÉRATION du mécanisme HBV (la
        # partition beta qui alimente le réservoir), sans laquelle le réservoir
        # est affamé (PoC 2026-06-14 : interflow seul ne franchit jamais UZL).
        self._qf_bounds["frac"] = (0.05, 0.95)
        if self.use_quickflow_reservoir:
            # raw paramétrés via sigmoïde → init littérature K0=0.4, K1=0.1, UZL=20.
            self.k0_uz_raw = nn.Parameter(self._inv_bounded(0.4, *self._qf_bounds["k0"]))
            self.k1_uz_raw = nn.Parameter(self._inv_bounded(0.1, *self._qf_bounds["k1"]))
            self.uzl_raw = nn.Parameter(self._inv_bounded(20.0, *self._qf_bounds["uzl"]))
            self.qf_frac_raw = nn.Parameter(self._inv_bounded(0.5, *self._qf_bounds["frac"]))

        # ── Capacité d'infiltration de surface découplée (Horton) ─────────
        # infil_ratio ∈ (0.05, 1.0) multiplie K_sat_1 dans le terme de Horton
        # SEULEMENT (pas le drainage). Init 0.5 : Horton actif dès le départ
        # (gradient fort sur chaque jour de pluie, contrairement au seuil du
        # réservoir qui gelait). L'optimiseur le baisse si les pics aident, le
        # remonte si le volume casse — sans jamais toucher la récession.
        self.use_separate_infil_capacity = bool(use_separate_infil_capacity)
        self._infil_bounds = (0.05, 1.0)
        if self.use_separate_infil_capacity:
            # Init à 0.30 = optimum mesuré par le scan infil_ratio (2026-06-14 :
            # kge_med 0.711 + peak_ratio 0.707, +5% volume seulement). On démarre
            # à l'optimum car la surface kge est plate autour et un seul param
            # global bouge trop lentement pour y descendre depuis 0.5 (le null).
            self.infil_ratio_raw = nn.Parameter(self._inv_bounded(0.30, *self._infil_bounds))

    @staticmethod
    def _inv_bounded(value: float, lo: float, hi: float) -> Tensor:
        """Logit inverse pour qu'un raw donné redonne `value` après sigmoïde."""
        import math
        frac = min(max((value - lo) / (hi - lo), 1e-4), 1.0 - 1e-4)
        return torch.tensor(math.log(frac / (1.0 - frac)))

    def _quickflow_params(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """(K0, K1, UZL, frac) bornés depuis les raw apprenables."""
        def b(raw, lo, hi):
            return lo + (hi - lo) * torch.sigmoid(raw)
        return (b(self.k0_uz_raw, *self._qf_bounds["k0"]),
                b(self.k1_uz_raw, *self._qf_bounds["k1"]),
                b(self.uzl_raw, *self._qf_bounds["uzl"]),
                b(self.qf_frac_raw, *self._qf_bounds["frac"]))

    def _effective_saturation(
        self, theta: Tensor, theta_r: Tensor, porosity: Tensor
    ) -> Tensor:
        """S_e in [0, 1] — clamped to avoid gradient singularities in vG."""
        denom = porosity - theta_r + 1e-6
        Se = (theta - theta_r) / denom
        # Wider bounds prevent Se.pow(-1/m) from producing 10^17 gradients
        # (at Se=1e-4, m=0.23, gradient ~ 10^21). At Se=0.01, gradient is
        # bounded to ~10^8 — still large but within float32 range.
        return torch.clamp(Se, 0.01, 0.99)

    def _K(
        self, theta: Tensor, K_sat: Tensor, theta_r: Tensor, porosity: Tensor,
        vg_n: Tensor | None = None,
    ) -> Tensor:
        """van Genuchten unsaturated hydraulic conductivity K(theta)."""
        Se = self._effective_saturation(theta, theta_r, porosity)
        n = vg_n if vg_n is not None else 1.5
        m = 1.0 - 1.0 / n
        # Clamp inner term to prevent (small)^(negative) gradient divergence
        # near Se=1 where (1-Se^(1/m))→0 and m-1<0 makes gradient explode.
        Se_pow = torch.clamp(Se.pow(1.0 / m), max=1.0 - 1e-6)
        inner = 1.0 - Se_pow
        K = K_sat * Se**0.5 * (1.0 - inner ** m) ** 2
        return K

    def _psi(
        self, theta: Tensor, theta_r: Tensor, porosity: Tensor,
        vg_n: Tensor | None = None,
    ) -> Tensor:
        """van Genuchten matric potential psi(theta) in m (negative = tension).

        Computed in log-space to prevent gradient explosion from Se.pow(-1/m).
        Direct computation: gradient of Se.pow(-1/m) is (-1/m)*Se^(-1/m-1)
        which reaches 10^21 at Se=1e-4, m=0.23 — float32 overflow → NaN.
        Log-space: log(Se^(-1/m)) = (-1/m)*log(Se), gradient is (-1/m)*(1/Se)
        which stays bounded (max ~433 at Se=0.01, m=0.23).
        """
        Se = self._effective_saturation(theta, theta_r, porosity)
        n = vg_n if vg_n is not None else 1.5
        m = 1.0 - 1.0 / n
        import math
        # Log-space computation: Se^(-1/m) = exp((-1/m) * log(Se))
        # This avoids the catastrophic gradient of pow with negative exponent.
        log_Se = torch.log(torch.clamp(Se, min=1e-6))
        log_Se_inv_m = (-1.0 / m) * log_Se
        # Clamp before exp to prevent overflow (exp(40) ≈ 2.4e17 < float32 max)
        log_Se_inv_m = torch.clamp(log_Se_inv_m, max=math.log(1e8))
        Se_inv_m = torch.exp(log_Se_inv_m)
        # (Se_inv_m - 1) is always > 0 for Se < 1. Compute pow in log-space:
        # psi = -(1/alpha) * (Se_inv_m - 1)^(1/n)
        # log|psi| = -log(alpha) + (1/n)*log(Se_inv_m - 1)
        arg = Se_inv_m - 1.0  # > 0 since Se < 1
        log_arg = torch.log(torch.clamp(arg, min=1e-20))
        log_psi = -math.log(self.vg_alpha) + (1.0 / n) * log_arg
        psi = -torch.exp(torch.clamp(log_psi, max=math.log(100.0)))
        return torch.clamp(psi, min=-100.0)

    def _partition_drainage(
        self,
        theta: Tensor,
        theta_fc: Tensor,
        porosity: Tensor,
        K_sat: Tensor,
        z_layer: Tensor | float,
        f_vert: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Excess-water drainage above field capacity, partitioned vert/lat.

        Replaces the legacy ``_interflow`` + ``krec`` competition (2026-05-11).
        The total drainage rate uses K_sat as the single rate driver and the
        Hydrotel nonlinearity ``(1 + excess_frac)`` for faster drainage near
        saturation. The NeRF-learned ``f_vert ∈ (0, 1)`` splits it between:

            q_vert = total × f_vert         (down to next layer / aquifer)
            q_lat  = total × (1 - f_vert)   (lateral to stream)

        f_vert + (1-f_vert) = 1 (mass conservation), so the model cannot
        cheat by inflating one direction while shrinking the other — the
        equifinality of slope_factor × k_interflow ↔ K_sat is broken.

        All quantities in m/day.
        """
        excess = soft_relu(theta - theta_fc, self.sharpness)
        excess_frac = excess / (porosity - theta_fc + 1e-6)
        excess_frac = torch.clamp(excess_frac, max=1.0)
        # Rate per unit depth: K_sat × (theta - theta_fc) × (1 + excess_frac).
        # Hydrotel-style nonlinearity: faster drainage near saturation.
        total_drainage = K_sat * excess * (1.0 + excess_frac)
        q_vert = total_drainage * f_vert
        q_lat = total_drainage * (1.0 - f_vert)
        return q_vert, q_lat

    def _darcy_flux(
        self,
        theta_up: Tensor,
        theta_dn: Tensor,
        K_sat_up: Tensor,
        theta_r_up: Tensor,
        porosity_up: Tensor,
        theta_r_dn: Tensor,
        porosity_dn: Tensor,
        dz: Tensor | float,
        vg_n: Tensor | None = None,
    ) -> Tensor:
        """Gravity + matric potential Darcy flux from upper to lower layer (m/day)."""
        K_up = self._K(theta_up, K_sat_up, theta_r_up, porosity_up, vg_n=vg_n)
        psi_up = self._psi(theta_up, theta_r_up, porosity_up, vg_n=vg_n)
        psi_dn = self._psi(theta_dn, theta_r_dn, porosity_dn, vg_n=vg_n)
        gradient = (psi_up - psi_dn) / dz + 1.0  # head gradient + gravity unit gradient
        flux = K_up * gradient
        # Allow both downward (positive) and upward (negative) flux
        # Upward flux is important for capillary rise
        return flux

    def forward(
        self,
        P_eff: Tensor,
        ET1: Tensor,
        ET2: Tensor,
        ET3: Tensor,
        theta1: Tensor,
        theta2: Tensor,
        theta3: Tensor,
        K_sat_1: Tensor,
        K_sat_2: Tensor,
        K_sat_3: Tensor,
        porosity_1: Tensor,
        porosity_2: Tensor,
        porosity_3: Tensor,
        theta_fc_1: Tensor,
        theta_fc_2: Tensor,
        theta_fc_3: Tensor,
        theta_wp_1: Tensor,
        theta_wp_2: Tensor,
        theta_wp_3: Tensor,
        f_vert_1: Tensor,
        f_vert_2: Tensor,
        f_vert_3: Tensor,
        vg_n: Tensor | None = None,
        z2: Tensor | None = None,
        z3: Tensor | None = None,
        rain_hours: Tensor | None = None,
        vsa_b: Tensor | None = None,
        S_uz: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """One-day soil water balance update.

        Args:
            z2, z3: per-node layer thicknesses (m). If None, falls back to
                    self.z2_default / self.z3_default scalars.
            rain_hours: per-node assumed storm duration (hours) for sub-daily
                    intensity estimation. If None, falls back to scalar default.
            S_uz: (n_nodes,) upper-zone fast-reservoir storage (mm) carried
                    from the previous step. If None, starts at zero.
        Returns:
            theta1_new, theta2_new, theta3_new: updated moisture content
            R_surface: (n_nodes,) saturation-excess runoff (mm/day)
            interflow: (n_nodes,) lateral subsurface flow to stream (mm/day);
                    when the quickflow reservoir is on, this is the reservoir's
                    released flow Q0 (threshold burst) + Q1 (slow drain)
            baseflow: (n_nodes,) deep drainage from layer 3 (mm/day)
            S_uz_new: (n_nodes,) updated upper-zone storage (mm)
        """
        # Z1 fixed (root zone surface), Z2/Z3 per-node if provided
        z1 = self.z1
        if z2 is None:
            z2 = torch.full_like(theta1, self.z2_default)
        if z3 is None:
            z3 = torch.full_like(theta1, self.z3_default)
        if rain_hours is None:
            rain_hours = torch.full_like(theta1, self.default_rain_hours)

        # Convert mm/day to m/day for consistency with z (m)
        P_m = P_eff * 1e-3
        ET1_m = ET1 * 1e-3
        ET2_m = ET2 * 1e-3
        ET3_m = ET3 * 1e-3

        # ── Infiltration-excess runoff (Eagleson 1978) ──────────────────
        # Capacité d'infiltration de SURFACE découplée du K_sat de DRAINAGE
        # (2026-06-14). Ce sont deux propriétés physiques distinctes :
        # l'infiltrabilité de surface (croûtage, scellage, macropores) limite
        # l'entrée de la pluie, le K_sat de van Genuchten décrit le drainage du
        # profil. Les confondre entremêle pics et récession (scan K_sat). On
        # multiplie K_sat_1 par un facteur appris infil_ratio ∈ (0.05,1) :
        # garde la variation spatiale + le couplage au gel, ajoute un seul
        # bouton de scellage. infil_ratio=1 ⇒ comportement actuel (équivalence).
        R_infilt_excess = torch.zeros_like(P_m)
        if self.use_infiltration_excess:
            P_mm = P_eff  # mm/day
            infil_factor = 1.0
            if self.use_separate_infil_capacity:
                lo, hi = self._infil_bounds
                infil_factor = lo + (hi - lo) * torch.sigmoid(self.infil_ratio_raw)
            K_sat_mmh = K_sat_1 * 1000.0 / 24.0 * infil_factor  # m/day → mm/h
            mean_intensity = P_mm / torch.clamp(rain_hours, min=0.5)  # mm/h
            ratio = K_sat_mmh / (mean_intensity + 1e-3)
            frac_excess = torch.exp(-ratio)
            R_infilt_excess = P_mm * frac_excess * 1e-3  # m/day
            R_infilt_excess = torch.where(
                P_mm > 1.0, R_infilt_excess, torch.zeros_like(R_infilt_excess)
            )
            P_m = P_m - R_infilt_excess

        # ── Ruissellement par aire-source-variable (VSA, type VIC/TOPMODEL) ──
        # Une fraction f_sat de la surface est saturée et ruisselle DIRECTEMENT,
        # avant que le seau de la couche 1 ne se remplisse. La fraction croît
        # avec l'humidité de surface : f_sat = Se_1^vsa_b. Capture le fait que
        # les zones humides (freshet, sol près de la nappe, orages successifs)
        # génèrent du ruissellement rapide que le modèle à seau seul rate — le
        # déficit de génération de crue diagnostiqué 2026-06-13 (60 % de l'eau
        # d'orage absorbée en stockage). Conserve la masse : R_sat_vsa quitte
        # P_m, le reste continue vers l'infiltration / saturation pleine.
        Se_1 = torch.clamp(
            (theta1 - theta_wp_1) / (porosity_1 - theta_wp_1 + 1e-6), 0.0, 1.0
        )
        vsa_b_eff = vsa_b if vsa_b is not None else self.vsa_b
        f_sat = Se_1 ** vsa_b_eff
        R_sat_vsa = f_sat * torch.clamp(P_m, min=0.0)
        P_m = P_m - R_sat_vsa

        # Saturation-excess runoff from layer 1 (smooth)
        excess_1 = soft_relu(
            theta1 + P_m / z1 - porosity_1, self.sharpness
        ) * z1
        P_infiltrated = P_m - excess_1

        # Inter-layer Darcy fluxes
        q12 = self._darcy_flux(
            theta1, theta2, K_sat_1, theta_wp_1, porosity_1,
            theta_wp_2, porosity_2, dz=(z1 + z2) / 2, vg_n=vg_n,
        )
        q23 = self._darcy_flux(
            theta2, theta3, K_sat_2, theta_wp_2, porosity_2,
            theta_wp_3, porosity_3, dz=(z2 + z3) / 2, vg_n=vg_n,
        )
        # Softmax-partition drainage above field capacity (replaces legacy
        # _interflow + krec triplet, 2026-05-11). One rate driver per layer
        # (K_sat × excess), one partition fraction per layer (f_vert ∈ (0,1)).
        # Mass-conserving: vertical + lateral = 1, no equifinality cheat.
        q_vert_1, q_inter_1 = self._partition_drainage(
            theta1, theta_fc_1, porosity_1, K_sat_1, z1, f_vert_1,
        )
        q_vert_2, q_inter_2 = self._partition_drainage(
            theta2, theta_fc_2, porosity_2, K_sat_2, z2, f_vert_2,
        )
        q_recharge, q_inter_3 = self._partition_drainage(
            theta3, theta_fc_3, porosity_3, K_sat_3, z3, f_vert_3,
        )

        # ── Détournement HBV (moitié GÉNÉRATION du quickflow) ─────────────
        # Une fraction qf_frac de la drainage verticale RAPIDE de la couche 1
        # (q_vert_1, qui partirait en profondeur et se ferait avaler) est
        # détournée vers le réservoir supérieur S_uz. Sans ça le réservoir est
        # affamé (PoC 2026-06-14). Le reste (1−qf_frac) continue en profondeur.
        if self.use_quickflow_reservoir:
            _, _, _, qf_frac = self._quickflow_params()
            qf_uz = qf_frac * q_vert_1
            q_vert_1 = q_vert_1 - qf_uz
        else:
            qf_uz = torch.zeros_like(q_vert_1)

        # Add the partition's vertical component to Darcy flux (different
        # mechanisms — Darcy is gradient-driven, partition is gravity-fast
        # above FC). Total downward flux from a layer is their sum.
        q12 = q12 + q_vert_1
        q23 = q23 + q_vert_2

        # ── Water-balance-safe mass balance ──────────────────────────────
        # avail = water that can be EXTRACTED (above theta_wp).  Eau sous wp
        # est "bound water" — chimiquement liée au sol, non-extractible.
        # Cohérent avec q_recharge qui utilise déjà drainable_3 = (theta-wp)*z.
        drainable_1 = soft_relu(theta1 - theta_wp_1, self.sharpness) * z1
        avail_1 = drainable_1 + P_infiltrated
        pos_demand_1 = (torch.clamp(ET1_m, min=0.0) + torch.clamp(q12, min=0.0)
                        + torch.clamp(q_inter_1, min=0.0) + torch.clamp(qf_uz, min=0.0))
        safe_demand_1 = torch.clamp(pos_demand_1, min=1e-10)
        sf1 = torch.clamp(avail_1 / safe_demand_1, min=0.0, max=1.0)
        q12_s = torch.where(q12 > 0, q12 * sf1, q12)
        q_inter_1_s = q_inter_1 * sf1
        qf_uz_s = qf_uz * sf1

        theta1_raw = theta1 + (P_infiltrated - ET1_m * sf1 - q12_s - q_inter_1_s - qf_uz_s) / z1
        ov1 = soft_relu(theta1_raw - porosity_1, self.sharpness) * z1
        theta1_new = torch.clamp(theta1_raw - ov1 / z1, min=0.0, max=1.0)

        drainable_2 = soft_relu(theta2 - theta_wp_2, self.sharpness) * z2
        avail_2 = drainable_2 + torch.clamp(q12_s, min=0.0)
        pos_demand_2 = torch.clamp(ET2_m, min=0.0) + torch.clamp(q23, min=0.0) + torch.clamp(q_inter_2, min=0.0)
        safe_demand_2 = torch.clamp(pos_demand_2, min=1e-10)
        sf2 = torch.clamp(avail_2 / safe_demand_2, min=0.0, max=1.0)
        q23_s = torch.where(q23 > 0, q23 * sf2, q23)
        q_inter_2_s = q_inter_2 * sf2

        theta2_raw = theta2 + (q12_s - ET2_m * sf2 - q23_s - q_inter_2_s) / z2
        ov2 = soft_relu(theta2_raw - porosity_2, self.sharpness) * z2
        theta2_new = torch.clamp(theta2_raw - ov2 / z2, min=0.0, max=1.0)

        drainable_3_avail = soft_relu(theta3 - theta_wp_3, self.sharpness) * z3
        avail_3 = drainable_3_avail + torch.clamp(q23_s, min=0.0) + torch.clamp(ov2, min=0.0)
        pos_demand_3 = (torch.clamp(ET3_m, min=0.0)
                        + torch.clamp(q_recharge, min=0.0)
                        + torch.clamp(q_inter_3, min=0.0))
        safe_demand_3 = torch.clamp(pos_demand_3, min=1e-10)
        sf3 = torch.clamp(avail_3 / safe_demand_3, min=0.0, max=1.0)
        q_recharge_s = torch.where(q_recharge > 0, q_recharge * sf3, q_recharge)
        q_inter_3_s = q_inter_3 * sf3

        theta3_raw = theta3 + (q23_s + ov2 - ET3_m * sf3 - q_recharge_s - q_inter_3_s) / z3
        ov3 = soft_relu(theta3_raw - porosity_3, self.sharpness) * z3
        theta3_new = torch.clamp(theta3_raw - ov3 / z3, min=0.0, max=1.0)

        # Overflow cascade: ov2 → L3 (above), ov3 → recharge
        R_surface = (excess_1 + ov1 + R_infilt_excess + R_sat_vsa) * 1e3   # mm/day
        interflow = (q_inter_1_s + q_inter_2_s + q_inter_3_s) * 1e3  # mm/day
        baseflow = (q_recharge_s + ov3) * 1e3   # mm/day

        # ── Réservoir supérieur à seuil (HBV-EC K0/UZL/K1) ────────────────
        # Nourri par la drainage verticale rapide L1 détournée (qf_uz_s, la
        # moitié génération ci-dessus). Le stock S_uz relâche par deux sorties :
        # Q1 = K1·S_uz toujours, Q0 = K0·softplus(β(S_uz−UZL))/β seulement
        # au-dessus du seuil — la bouffée de crue. La relâche s'AJOUTE à
        # l'interflow direct (les deux vont au cours d'eau). Désactivable
        # (équivalence : off ⇒ comportement actuel, S_uz nul). Conserve la
        # masse : un facteur d'échelle borne Q0+Q1 ≤ eau disponible (idiome
        # sf1/sf2/sf3), donc S_uz ≥ 0 sans fuite : qf_in = Q0+Q1 + ΔS_uz.
        if self.use_quickflow_reservoir:
            if S_uz is None:
                S_uz = torch.zeros_like(theta1)
            k0, k1, uzl, _ = self._quickflow_params()
            inflow_uz = qf_uz_s * 1e3  # m/jour → mm/jour
            excess_uz = F.softplus(self.quickflow_beta * (S_uz - uzl)) / self.quickflow_beta
            Q0 = k0 * excess_uz
            Q1 = k1 * S_uz
            avail_uz = S_uz + inflow_uz
            release_raw = Q0 + Q1
            scale_uz = torch.clamp(avail_uz / (release_raw + 1e-9), max=1.0)
            Q0 = Q0 * scale_uz
            Q1 = Q1 * scale_uz
            S_uz_new = torch.clamp(avail_uz - Q0 - Q1, min=0.0)
            interflow = interflow + Q0 + Q1  # relâche AJOUTÉE à l'interflow direct
        else:
            S_uz_new = S_uz if S_uz is not None else torch.zeros_like(theta1)

        return theta1_new, theta2_new, theta3_new, R_surface, interflow, baseflow, S_uz_new
