"""Spatial field network — NeRF-inspired MLP mapping coordinates to parameters.

Takes (lon, lat, territorial_features) per node and returns spatially continuous
hydrological parameter fields. No rasters, no UHRH boundaries — just a learned
field conditioned on meaningful hydrological descriptors.

Architecture: MLP with SiLU activations and skip connections.
    input = Fourier(lon, lat) + territorial_features
    -> Linear(in, 256) -> SiLU -> skip -> Linear(256, 256) -> SiLU
    -> Linear(256, n_params)
    -> Softplus/Sigmoid output constraints for physical plausibility
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from meandre.spatial.positional_encoding import FourierPositionalEncoding


@dataclass
class SpatialParams:
    """Per-node hydrological parameters output by the spatial field network.

    All tensors have shape (n_nodes,).

    Soil
    ----
    K_sat_{1,2,3}   Saturated hydraulic conductivity per layer (m/day).
    porosity_{1,2,3} Total porosity (m3/m3).
    theta_fc_{1,2,3} Field capacity (m3/m3).
    theta_wp_{1,2,3} Wilting point (m3/m3).
    theta_sat_{1,2,3} Saturation moisture content (m3/m3 = porosity).
    f_root_{1,2,3}  Root fraction per layer.

    Snow
    ----
    C_f             Degree-day melt factor (mm/C/day).
    T_melt          Melting temperature threshold (C), near 0.
    T_snow          Rain/snow threshold temperature (C).

    Canopy
    ------
    interception_capacity  Maximum canopy storage (mm).

    Routing
    -------
    manning_n       Manning roughness coefficient.

    Frost
    -----
    frost_alpha     Frost K_sat reduction coefficient.
    """

    # Soil per layer (9 params x 3 layers)
    K_sat_1: Tensor; K_sat_2: Tensor; K_sat_3: Tensor
    porosity_1: Tensor; porosity_2: Tensor; porosity_3: Tensor
    theta_fc_1: Tensor; theta_fc_2: Tensor; theta_fc_3: Tensor
    theta_wp_1: Tensor; theta_wp_2: Tensor; theta_wp_3: Tensor
    f_root_1: Tensor; f_root_2: Tensor; f_root_3: Tensor
    # Snow
    C_f: Tensor
    T_melt: Tensor
    T_snow: Tensor
    # Canopy
    interception_capacity: Tensor
    # Routing
    manning_n: Tensor
    # Frost
    frost_alpha: Tensor
    # Wetland
    f_wetland: Tensor
    # Hydrotel-inspired: slope-dependent interflow and baseflow recession
    slope_factor: Tensor    # interflow amplification from topographic slope [0.01, 2.0]
    krec: Tensor            # baseflow recession coefficient (1/day) [0.001, 0.2]
    # Groundwater
    k_gw: Tensor            # aquifer recession coefficient (1/day) [0.001, 0.05]
    # Stream temperature
    T_gw: Tensor            # groundwater temperature (C) [3, 13]
    K_atm: Tensor           # atmospheric heat exchange coefficient (1/day) [0.05, 0.55]
    # Frost thermal lag
    alpha_T: Tensor         # soil thermal damping (1/day) [0.01, 0.15] — fitted per node

    N_PARAMS: ClassVar[int] = 28

    @classmethod
    def from_tensor(cls, x: Tensor) -> "SpatialParams":
        """Reconstruct from (n_nodes, N_PARAMS) tensor."""
        fields = [x[:, i] for i in range(cls.N_PARAMS)]
        return cls(*fields)


class SpatialFieldNetwork(nn.Module):
    """NeRF-style MLP: (coords + territorial features) -> hydrological params.

    Parameters
    ----------
    n_territorial : int
        Number of territorial indicator features (default 17).
    n_coord_freqs : int
        Number of Fourier frequency bands for (lon, lat) encoding.
    hidden : int
        Width of hidden layers.
    dropout : float
        MC Dropout rate for epistemic uncertainty (set > 0 to enable).
    param_mode : str
        "nerf" for spatially-varying parameters (~13k params)
        "static" for global parameters like Hydrotel (28 params)
    """

    def __init__(
        self,
        n_territorial: int = 17,
        n_coord_freqs: int = 6,
        hidden: int = 256,  # RESTORED to original 256 for full capacity!
        dropout: float = 0.0,
        param_mode: str = "nerf",
    ) -> None:
        super().__init__()
        self.n_territorial = n_territorial
        self.param_mode = param_mode

        if param_mode == "static":
            # Static mode: just 28 global learnable parameters
            self.static_params = nn.Parameter(torch.randn(SpatialParams.N_PARAMS) * 0.1)
        else:
            # NeRF mode: MLP mapping coordinates to parameters
            self.coord_enc = FourierPositionalEncoding(n_freqs=n_coord_freqs, include_input=True)
            coord_dim = self.coord_enc.out_dim(2)  # encoded (lon, lat)
            in_dim = coord_dim + n_territorial

            self.fc1 = nn.Linear(in_dim, hidden)
            self.fc2 = nn.Linear(hidden + in_dim, hidden)  # skip connection
            self.fc_out = nn.Linear(hidden, SpatialParams.N_PARAMS)
            self.act = nn.SiLU()
            self.drop = nn.Dropout(p=dropout)

    def forward(self, coords: Tensor, territorial: Tensor) -> SpatialParams:
        """
        Args:
            coords: (n_nodes, 2)  [lon, lat] in degrees, normalised.
            territorial: (n_nodes, n_territorial)
        Returns:
            SpatialParams with one value per node per parameter.
        """
        if self.param_mode == "static":
            # Static mode: same parameters for all nodes
            n_nodes = coords.shape[0]
            raw = self.static_params.unsqueeze(0).expand(n_nodes, -1)
        else:
            # NeRF mode: spatially-varying parameters
            enc = self.coord_enc(coords)          # (n_nodes, coord_dim)
            x0 = torch.cat([enc, territorial], dim=-1)  # (n_nodes, in_dim)

            h = self.drop(self.act(self.fc1(x0)))
            h = torch.cat([h, x0], dim=-1)       # skip connection
            h = self.drop(self.act(self.fc2(h)))
            raw = self.fc_out(h)                  # (n_nodes, N_PARAMS)

        return self._apply_constraints(raw)

    def _apply_constraints(self, raw: Tensor) -> SpatialParams:
        """Map raw network outputs to physically plausible ranges.

        Uses scaled-tanh parameterization: center + half_range * tanh(x * scale).
        The scale factor (0.3) means the network needs large raw values (~6) to
        reach 95% of the range. Weight decay pulls raw values toward 0, which maps
        to the center (physically reasonable default). This creates implicit
        regularization without explicit priors or hard sigmoid walls.

        Key constraint: theta_wp < theta_fc < porosity is enforced by
        parameterizing theta_fc and theta_wp as fractions of porosity.
        """
        import math

        def soft(x, center, half_range, scale=0.3):
            """Scaled-tanh: smoothly bounded around center ± half_range."""
            return center + half_range * torch.tanh(x * scale)

        cols = [raw[:, i] for i in range(SpatialParams.N_PARAMS)]
        i = 0

        constrained = []
        # K_sat (m/day): log-normal with per-layer centers decreasing with depth.
        log_centers = [math.log(0.5), math.log(0.1), math.log(0.02)]
        for layer in range(3):
            exponent = torch.clamp(cols[i] * 0.15 + log_centers[layer], min=-8.0, max=4.0)
            constrained.append(torch.exp(exponent))
            i += 1
        # porosity: center 0.40, range ±0.20 → [0.20, 0.60]
        porosities = []
        for _ in range(3):
            p = soft(cols[i], 0.40, 0.20)
            porosities.append(p)
            constrained.append(p)
            i += 1
        # theta_fc as fraction of porosity: center 0.575, range ±0.275 → [0.30, 0.85]
        # Guarantees theta_fc < porosity always
        theta_fcs = []
        for layer in range(3):
            fc_frac = soft(cols[i], 0.575, 0.275)
            theta_fc = porosities[layer] * fc_frac
            theta_fcs.append(theta_fc)
            constrained.append(theta_fc)
            i += 1
        # theta_wp as fraction of theta_fc: center 0.325, range ±0.275 → [0.05, 0.60]
        # Guarantees theta_wp < theta_fc always
        for layer in range(3):
            wp_frac = soft(cols[i], 0.325, 0.275)
            theta_wp = theta_fcs[layer] * wp_frac
            constrained.append(theta_wp)
            i += 1
        # f_root (0, 1), then softmax so sum = 1
        # Bias toward upper layers (50/30/20 split)
        f_roots_raw = torch.stack(cols[i:i+3], dim=-1)  # (n, 3)
        f_roots_raw = f_roots_raw * 0.3 + torch.tensor([1.0, 0.5, -0.5], device=raw.device)
        f_roots = torch.softmax(f_roots_raw, dim=-1)
        constrained.extend([f_roots[:, j] for j in range(3)])
        i += 3
        # C_f: center 4.25, range ±3.75 → [0.5, 8.0] mm/C/day
        constrained.append(soft(cols[i], 4.25, 3.75)); i += 1
        # T_melt: center 0.0, range ±1.0 → [-1, 1] C
        constrained.append(soft(cols[i], 0.0, 1.0)); i += 1
        # T_snow: center 1.0, range ±1.0 → [0, 2] C
        constrained.append(soft(cols[i], 1.0, 1.0)); i += 1
        # interception_capacity: center 1.5, range ±1.0 → [0.5, 2.5] mm
        constrained.append(soft(cols[i], 1.5, 1.0)); i += 1
        # manning_n: center 0.105, range ±0.095 → [0.01, 0.20]
        constrained.append(soft(cols[i], 0.105, 0.095)); i += 1
        # frost_alpha: center 0.5, range ±0.5 → [0.0, 1.0]
        constrained.append(soft(cols[i], 0.5, 0.5)); i += 1
        # f_wetland: center 0.05, range ±0.05 → [0.0, 0.10]
        constrained.append(soft(cols[i], 0.05, 0.05)); i += 1
        # slope_factor: center 0.30, range ±0.29 → [0.01, 0.59]
        # Hydrotel uses sin(atan(slope)): for 5% slope → 0.05, 20% → 0.20.
        # Center at 0.30 ≈ sin(atan(0.31)) for moderate Quebec boreal slopes.
        constrained.append(soft(cols[i], 0.30, 0.29)); i += 1
        # krec: baseflow recession (1/day), log-normal.
        # Center at log(0.003) ≈ 0.3%/day.  With Z3=1m and drainable=0.15,
        # recharge ≈ 0.003 * 1.0 * 0.15 * 1000 = 0.45 mm/day = 164 mm/yr.
        # Max exp(-3) ≈ 0.05/day.
        exponent = torch.clamp(cols[i] * 0.15 + math.log(0.003), min=-8.0, max=-3.0)
        constrained.append(torch.exp(exponent)); i += 1
        # k_gw: aquifer recession (1/day), log-normal.
        # Center at log(0.005) ≈ 0.5%/day.  Slower than krec so GW
        # provides a delayed baseflow signal.  Max exp(-2) ≈ 0.14/day.
        exponent = torch.clamp(cols[i] * 0.15 + math.log(0.005), min=-8.0, max=-2.0)
        constrained.append(torch.exp(exponent)); i += 1
        # T_gw: groundwater temperature (C): center 8.0, range ±5.0 → [3, 13]
        constrained.append(soft(cols[i], 8.0, 5.0)); i += 1
        # K_atm: atmospheric heat exchange (1/day): center 0.30, range ±0.25 → [0.05, 0.55]
        constrained.append(soft(cols[i], 0.30, 0.25)); i += 1
        # alpha_T: soil thermal damping (1/day): center 0.03, range ±0.02 → [0.01, 0.05]
        # 0.03/day ≈ 33-day lag; 0.01 ≈ 100-day lag (deep soil); 0.05 ≈ 20-day lag (shallow)
        constrained.append(soft(cols[i], 0.03, 0.02)); i += 1

        return SpatialParams.from_tensor(torch.stack(constrained, dim=-1))

    def boundary_regularization(self, coords: Tensor, territorial: Tensor) -> Tensor:
        """Penalize raw network outputs that push constrained params toward extremes.

        For sigmoid-constrained params: (2*sigmoid(raw) - 1)^4 — penalizes bounds.
        For exp-constrained params (K_sat): raw^2 — L2 prior toward center.
        Excludes f_root (softmax, no bounds to hit).
        """
        enc = self.coord_enc(coords)
        x0 = torch.cat([enc, territorial], dim=-1)
        h = torch.nn.functional.silu(self.fc1(x0))
        h = torch.cat([h, x0], dim=-1)
        h = torch.nn.functional.silu(self.fc2(h))
        raw = self.fc_out(h)  # (n_nodes, N_PARAMS)

        # Softplus params: L2 on raw → pulls toward center
        # K_sat (0-2), krec (23), k_gw (24)
        unbounded_cols = [0, 1, 2, 23, 24]
        unbounded_penalty = (raw[:, unbounded_cols] ** 2).mean()

        # Sigmoid-constrained columns: penalize saturation
        # Skip K_sat (0-2), f_root (12-14), krec (23), k_gw (24)
        sig_cols = list(range(3, 12)) + list(range(15, 23)) + [25, 26, 27]
        raw_sig = raw[:, sig_cols]
        sig = torch.sigmoid(raw_sig)
        sig_penalty = ((2.0 * sig - 1.0) ** 4).mean()

        return unbounded_penalty + sig_penalty

    def physical_prior_loss(self, params: SpatialParams) -> Tensor:
        """Soft L2 penalty pulling parameters toward physically reasonable values.

        Operates in physical space — penalizes parameter VALUES that deviate
        from hydrological literature defaults.  All terms are normalized so
        that a deviation of ~1 "standard deviation" from the prior gives a
        contribution of ~1.0.
        """
        import math
        device = params.K_sat_1.device
        loss = torch.tensor(0.0, device=device)

        # K_sat in log-space: penalize deviation from layer-appropriate defaults
        for k, target in [(params.K_sat_1, 0.5), (params.K_sat_2, 0.1), (params.K_sat_3, 0.02)]:
            loss = loss + ((torch.log(k + 1e-8) - math.log(target)) ** 2).mean()

        # Porosity: typical 0.35-0.45
        for p in [params.porosity_1, params.porosity_2, params.porosity_3]:
            loss = loss + ((p - 0.40) ** 2).mean() * 3.0

        # C_f: typical degree-day factor 2-4 mm/C/day
        loss = loss + ((params.C_f - 3.0) ** 2).mean() * 0.5

        # T_melt: should be near 0°C
        loss = loss + (params.T_melt ** 2).mean()

        # frost_alpha: typical 0.3-0.7
        loss = loss + ((params.frost_alpha - 0.5) ** 2).mean()

        # f_root: should concentrate in upper layers (50/30/20 split)
        loss = loss + ((params.f_root_1 - 0.50) ** 2).mean()
        loss = loss + ((params.f_root_2 - 0.30) ** 2).mean()
        loss = loss + ((params.f_root_3 - 0.20) ** 2).mean()

        # f_wetland: typical 0.05 for Quebec boreal
        loss = loss + ((params.f_wetland - 0.05) ** 2).mean()

        # k_gw: typical 1%/day recession
        loss = loss + ((torch.log(params.k_gw + 1e-8) - math.log(0.01)) ** 2).mean()

        # T_gw: typical ~8°C for Quebec groundwater
        loss = loss + ((params.T_gw - 8.0) ** 2).mean() * 0.1

        # K_atm: typical 0.3/day atmospheric exchange
        loss = loss + ((params.K_atm - 0.3) ** 2).mean()

        # alpha_T: typical 0.03/day (~33-day thermal lag)
        loss = loss + ((params.alpha_T - 0.03) ** 2).mean() * 100.0

        return loss
