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
        z2: float = Z2,
        z3: float = Z3,
        sharpness: float = 50.0,
        # van Genuchten shape parameters (can be learned or fixed)
        vg_n: float = 1.5,
        vg_alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.z = torch.tensor([z1, z2, z3])  # (3,) layer thicknesses
        self.sharpness = sharpness
        self.vg_n = vg_n
        self.vg_m = 1.0 - 1.0 / vg_n
        self.vg_alpha = vg_alpha
        # Learnable interflow recession rate (1/day); init ~0.05/day
        self.log_k_interflow = nn.Parameter(torch.tensor(-3.0))

    @property
    def k_interflow(self) -> Tensor:
        # Cap at 0.1/day — drains drainable water in ~10 days at most.
        # Hydrotel typical range: 0.01-0.10 /day.
        return torch.clamp(torch.nn.functional.softplus(self.log_k_interflow), max=0.1)

    def _effective_saturation(
        self, theta: Tensor, theta_r: Tensor, porosity: Tensor
    ) -> Tensor:
        """S_e in [0, 1] — differentiable via soft_threshold on boundaries."""
        denom = porosity - theta_r + 1e-6
        Se = (theta - theta_r) / denom
        return torch.clamp(Se, 1e-6, 1.0 - 1e-6)

    def _K(
        self, theta: Tensor, K_sat: Tensor, theta_r: Tensor, porosity: Tensor
    ) -> Tensor:
        """van Genuchten unsaturated hydraulic conductivity K(theta)."""
        Se = self._effective_saturation(theta, theta_r, porosity)
        m = self.vg_m
        K = K_sat * Se**0.5 * (1.0 - (1.0 - Se ** (1.0 / m)) ** m) ** 2
        return K

    def _psi(
        self, theta: Tensor, theta_r: Tensor, porosity: Tensor
    ) -> Tensor:
        """van Genuchten matric potential psi(theta) in m (negative = tension)."""
        Se = self._effective_saturation(theta, theta_r, porosity)
        n = self.vg_n
        m = self.vg_m
        psi = -(1.0 / self.vg_alpha) * (Se ** (-1.0 / m) - 1.0) ** (1.0 / n)
        # Clamp to ~1 MPa (~100 m head) — prevents gradient explosion through
        # d(psi)/d(Se) ∝ Se^(-1/m - 1) which diverges as Se → 0.
        # -100 m is the permanent wilting point (~-1.5 MPa); beyond this,
        # plants can't extract water and drainage is negligible.
        return torch.clamp(psi, min=-100.0)

    def _interflow(
        self,
        theta: Tensor,
        theta_fc: Tensor,
        porosity: Tensor,
        z_layer: float,
        slope_factor: Tensor | None = None,
    ) -> Tensor:
        """Lateral subsurface drainage when moisture > field capacity (m/day).

        Hydrotel-inspired: q = k_interflow * slope_factor * excess_water * (1 + excess_frac).
        slope_factor acts like sin(atan(slope)) in Hydrotel's BV3C1 — steeper
        catchments produce more interflow.  Learned per-node from spatial network.
        """
        # Drainable water in layer (m)
        excess_water = soft_relu(theta - theta_fc, self.sharpness) * z_layer
        # Nonlinear recession: faster drainage at higher saturation
        excess_frac = soft_relu(theta - theta_fc, self.sharpness) / (
            porosity - theta_fc + 1e-6
        )
        excess_frac = torch.clamp(excess_frac, max=1.0)
        q = self.k_interflow * excess_water * (1.0 + excess_frac)
        # Slope amplification (Hydrotel-style)
        if slope_factor is not None:
            q = q * slope_factor
        return q

    def _darcy_flux(
        self,
        theta_up: Tensor,
        theta_dn: Tensor,
        K_sat_up: Tensor,
        theta_r_up: Tensor,
        porosity_up: Tensor,
        theta_r_dn: Tensor,
        porosity_dn: Tensor,
        dz: float,
    ) -> Tensor:
        """Gravity + matric potential Darcy flux from upper to lower layer (m/day)."""
        K_up = self._K(theta_up, K_sat_up, theta_r_up, porosity_up)
        psi_up = self._psi(theta_up, theta_r_up, porosity_up)
        psi_dn = self._psi(theta_dn, theta_r_dn, porosity_dn)
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
        slope_factor: Tensor | None = None,
        krec: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """One-day soil water balance update.

        Returns:
            theta1_new, theta2_new, theta3_new: updated moisture content
            R_surface: (n_nodes,) saturation-excess runoff (mm/day)
            interflow: (n_nodes,) lateral subsurface flow from layers 1+2 (mm/day)
            baseflow: (n_nodes,) deep drainage from layer 3 (mm/day)
        """
        # Convert mm/day to m/day for consistency with z (m)
        P_m = P_eff * 1e-3
        ET1_m = ET1 * 1e-3
        ET2_m = ET2 * 1e-3
        ET3_m = ET3 * 1e-3

        # Saturation-excess runoff from layer 1 (smooth)
        excess_1 = soft_relu(
            theta1 + P_m / Z1 - porosity_1, self.sharpness
        ) * Z1
        P_infiltrated = P_m - excess_1

        # Inter-layer Darcy fluxes
        q12 = self._darcy_flux(
            theta1, theta2, K_sat_1, theta_wp_1, porosity_1,
            theta_wp_2, porosity_2, dz=(Z1 + Z2) / 2,
        )
        q23 = self._darcy_flux(
            theta2, theta3, K_sat_2, theta_wp_2, porosity_2,
            theta_wp_3, porosity_3, dz=(Z2 + Z3) / 2,
        )
        # Baseflow: Hydrotel-style linear reservoir recession if krec provided,
        # otherwise free-drainage (Darcy K(theta3)).
        if krec is not None:
            # q_recharge = krec * z3 * (theta3 - theta_wp3) — only drainable water.
            # Water below wilting point is bound and cannot drain.
            drainable_3 = soft_relu(theta3 - theta_wp_3, self.sharpness)
            q_recharge = krec * Z3 * drainable_3
        else:
            # Free-drainage bottom BC: unit hydraulic gradient (gravity only).
            q_recharge = self._K(theta3, K_sat_3, theta_wp_3, porosity_3)

        # Interflow: lateral subsurface drainage when moisture exceeds field
        # capacity.  Learnable recession rate on drainable water storage.
        # Slope amplification (Hydrotel-style): steeper → more interflow.
        q_inter_1 = self._interflow(theta1, theta_fc_1, porosity_1, Z1, slope_factor)
        q_inter_2 = self._interflow(theta2, theta_fc_2, porosity_2, Z2, slope_factor)

        # ── Water-balance-safe mass balance ──────────────────────────────
        # Cap extractions per layer so theta stays in [0, porosity].
        # Overflow beyond porosity → surface runoff (no water lost).

        # Layer 1
        # Only positive extractions need scaling; negative demand means net
        # inflow (capillary rise > ET+interflow) — no scaling needed.
        avail_1 = theta1 * Z1 + P_infiltrated
        pos_demand_1 = torch.clamp(ET1_m, min=0.0) + torch.clamp(q12, min=0.0) + torch.clamp(q_inter_1, min=0.0)
        sf1 = torch.where(
            pos_demand_1 > 1e-10,
            torch.clamp(avail_1 / pos_demand_1, max=1.0),
            torch.ones_like(avail_1),
        )
        # Scale only positive fluxes; negative fluxes (inflows) pass through
        q12_s = torch.where(q12 > 0, q12 * sf1, q12)
        q_inter_1_s = q_inter_1 * sf1

        theta1_raw = theta1 + (P_infiltrated - ET1_m * sf1 - q12_s - q_inter_1_s) / Z1
        ov1 = soft_relu(theta1_raw - porosity_1, self.sharpness) * Z1
        theta1_new = torch.clamp(theta1_raw - ov1 / Z1, min=0.0, max=1.0)

        # Layer 2
        avail_2 = theta2 * Z2 + torch.clamp(q12_s, min=0.0)
        pos_demand_2 = torch.clamp(ET2_m, min=0.0) + torch.clamp(q23, min=0.0) + torch.clamp(q_inter_2, min=0.0)
        sf2 = torch.where(
            pos_demand_2 > 1e-10,
            torch.clamp(avail_2 / pos_demand_2, max=1.0),
            torch.ones_like(avail_2),
        )
        q23_s = torch.where(q23 > 0, q23 * sf2, q23)
        q_inter_2_s = q_inter_2 * sf2

        theta2_raw = theta2 + (q12_s - ET2_m * sf2 - q23_s - q_inter_2_s) / Z2
        ov2 = soft_relu(theta2_raw - porosity_2, self.sharpness) * Z2
        theta2_new = torch.clamp(theta2_raw - ov2 / Z2, min=0.0, max=1.0)

        # Layer 3
        avail_3 = theta3 * Z3 + torch.clamp(q23_s, min=0.0)
        pos_demand_3 = torch.clamp(ET3_m, min=0.0) + torch.clamp(q_recharge, min=0.0)
        sf3 = torch.where(
            pos_demand_3 > 1e-10,
            torch.clamp(avail_3 / pos_demand_3, max=1.0),
            torch.ones_like(avail_3),
        )
        q_recharge_s = torch.where(q_recharge > 0, q_recharge * sf3, q_recharge)

        theta3_raw = theta3 + (q23_s - ET3_m * sf3 - q_recharge_s) / Z3
        ov3 = soft_relu(theta3_raw - porosity_3, self.sharpness) * Z3
        theta3_new = torch.clamp(theta3_raw - ov3 / Z3, min=0.0, max=1.0)

        # All overflow → surface runoff (water balance closes)
        R_surface = (excess_1 + ov1 + ov2 + ov3) * 1e3  # mm/day
        interflow = (q_inter_1_s + q_inter_2_s) * 1e3    # mm/day
        baseflow = q_recharge_s * 1e3                     # mm/day

        return theta1_new, theta2_new, theta3_new, R_surface, interflow, baseflow
