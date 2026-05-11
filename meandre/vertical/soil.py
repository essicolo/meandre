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

    def _effective_saturation(
        self, theta: Tensor, theta_r: Tensor, porosity: Tensor
    ) -> Tensor:
        """S_e in [0, 1] — differentiable via soft_threshold on boundaries."""
        denom = porosity - theta_r + 1e-6
        Se = (theta - theta_r) / denom
        return torch.clamp(Se, 1e-4, 1.0 - 1e-6)

    def _K(
        self, theta: Tensor, K_sat: Tensor, theta_r: Tensor, porosity: Tensor,
        vg_n: Tensor | None = None,
    ) -> Tensor:
        """van Genuchten unsaturated hydraulic conductivity K(theta)."""
        Se = self._effective_saturation(theta, theta_r, porosity)
        n = vg_n if vg_n is not None else 1.5
        m = 1.0 - 1.0 / n
        K = K_sat * Se**0.5 * (1.0 - (1.0 - Se ** (1.0 / m)) ** m) ** 2
        return K

    def _psi(
        self, theta: Tensor, theta_r: Tensor, porosity: Tensor,
        vg_n: Tensor | None = None,
    ) -> Tensor:
        """van Genuchten matric potential psi(theta) in m (negative = tension)."""
        Se = self._effective_saturation(theta, theta_r, porosity)
        n = vg_n if vg_n is not None else 1.5
        m = 1.0 - 1.0 / n
        # Clamp the intermediate Se^(-1/m) to prevent float32 overflow
        # when m is small (vg_n near lower bound).  1e8 is well below
        # float32 max and already far beyond the psi=-100 m clamp below.
        Se_inv_m = torch.clamp(Se.pow(-1.0 / m), max=1e8)
        psi = -(1.0 / self.vg_alpha) * (Se_inv_m - 1.0).pow(1.0 / n)
        # Clamp to ~1 MPa (~100 m head) — prevents gradient explosion through
        # d(psi)/d(Se) ∝ Se^(-1/m - 1) which diverges as Se → 0.
        # -100 m is the permanent wilting point (~-1.5 MPa); beyond this,
        # plants can't extract water and drainage is negligible.
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
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """One-day soil water balance update.

        Args:
            z2, z3: per-node layer thicknesses (m). If None, falls back to
                    self.z2_default / self.z3_default scalars.
            rain_hours: per-node assumed storm duration (hours) for sub-daily
                    intensity estimation. If None, falls back to scalar default.
        Returns:
            theta1_new, theta2_new, theta3_new: updated moisture content
            R_surface: (n_nodes,) saturation-excess runoff (mm/day)
            interflow: (n_nodes,) lateral subsurface flow from layers 1+2 (mm/day)
            baseflow: (n_nodes,) deep drainage from layer 3 (mm/day)
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
        R_infilt_excess = torch.zeros_like(P_m)
        if self.use_infiltration_excess:
            P_mm = P_eff  # mm/day
            K_sat_mmh = K_sat_1 * 1000.0 / 24.0  # m/day → mm/h
            mean_intensity = P_mm / torch.clamp(rain_hours, min=0.5)  # mm/h
            ratio = K_sat_mmh / (mean_intensity + 1e-3)
            frac_excess = torch.exp(-ratio)
            R_infilt_excess = P_mm * frac_excess * 1e-3  # m/day
            R_infilt_excess = torch.where(
                P_mm > 1.0, R_infilt_excess, torch.zeros_like(R_infilt_excess)
            )
            P_m = P_m - R_infilt_excess

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
        pos_demand_1 = torch.clamp(ET1_m, min=0.0) + torch.clamp(q12, min=0.0) + torch.clamp(q_inter_1, min=0.0)
        sf1 = torch.where(
            pos_demand_1 > 1e-10,
            torch.clamp(avail_1 / pos_demand_1, min=0.0, max=1.0),
            torch.ones_like(avail_1),
        )
        q12_s = torch.where(q12 > 0, q12 * sf1, q12)
        q_inter_1_s = q_inter_1 * sf1

        theta1_raw = theta1 + (P_infiltrated - ET1_m * sf1 - q12_s - q_inter_1_s) / z1
        ov1 = soft_relu(theta1_raw - porosity_1, self.sharpness) * z1
        theta1_new = torch.clamp(theta1_raw - ov1 / z1, min=0.0, max=1.0)

        drainable_2 = soft_relu(theta2 - theta_wp_2, self.sharpness) * z2
        avail_2 = drainable_2 + torch.clamp(q12_s, min=0.0)
        pos_demand_2 = torch.clamp(ET2_m, min=0.0) + torch.clamp(q23, min=0.0) + torch.clamp(q_inter_2, min=0.0)
        sf2 = torch.where(
            pos_demand_2 > 1e-10,
            torch.clamp(avail_2 / pos_demand_2, min=0.0, max=1.0),
            torch.ones_like(avail_2),
        )
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
        sf3 = torch.where(
            pos_demand_3 > 1e-10,
            torch.clamp(avail_3 / pos_demand_3, min=0.0, max=1.0),
            torch.ones_like(avail_3),
        )
        q_recharge_s = torch.where(q_recharge > 0, q_recharge * sf3, q_recharge)
        q_inter_3_s = q_inter_3 * sf3

        theta3_raw = theta3 + (q23_s + ov2 - ET3_m * sf3 - q_recharge_s - q_inter_3_s) / z3
        ov3 = soft_relu(theta3_raw - porosity_3, self.sharpness) * z3
        theta3_new = torch.clamp(theta3_raw - ov3 / z3, min=0.0, max=1.0)

        # Overflow cascade: ov2 → L3 (above), ov3 → recharge
        R_surface = (excess_1 + ov1 + R_infilt_excess) * 1e3   # mm/day
        interflow = (q_inter_1_s + q_inter_2_s + q_inter_3_s) * 1e3  # mm/day
        baseflow = (q_recharge_s + ov3) * 1e3   # mm/day

        return theta1_new, theta2_new, theta3_new, R_surface, interflow, baseflow
