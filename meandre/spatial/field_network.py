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

import math
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
    K_musk_hours    Muskingum travel time (hours) [4, 48].
    x_musk          Muskingum weighting factor [0, 0.5].

    Frost
    -----
    frost_alpha     Frost K_sat reduction coefficient.

    Soil physics
    ------------
    vg_n            van Genuchten n shape parameter [1.1, 2.7].
    k_interflow     Interflow recession rate (1/day) [0.005, 0.1].
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
    slope_factor: Tensor    # interflow amplification from topographic slope [0.01, 0.59]
    krec: Tensor            # baseflow recession coefficient (1/day) [0.001, 0.05]
    # Groundwater
    k_gw: Tensor            # aquifer recession coefficient (1/day) [0.001, 0.14]
    # Stream temperature
    T_gw: Tensor            # groundwater temperature (C) [3, 13]
    K_atm: Tensor           # atmospheric heat exchange coefficient (1/day) [0.05, 0.55]
    # Frost thermal lag
    alpha_T: Tensor         # soil thermal damping (1/day) [0.01, 0.05]
    # --- New params (E, F, G) ---
    vg_n: Tensor            # van Genuchten n shape parameter [1.3, 2.7]
    k_interflow: Tensor     # interflow recession rate (1/day) [0.005, 0.1]
    K_musk_hours: Tensor    # Muskingum travel time (hours) [4, 48]
    x_musk: Tensor          # Muskingum weighting factor [0.01, 0.49]
    # ETP scaling — équivalent au "coefficient multiplicatif" d'Hydrotel
    # (cf .par files SLSO MG24HS_2020 = 0.85 sur McGuinness, autres 0.5-1.0).
    K_c: Tensor             # ETP crop/calibration coefficient [0.3, 1.5]
    # Sub-daily storm duration (hours) for Eagleson infiltration excess.
    # Plus court = pluies intenses → plus de runoff. Borne configurable.
    rain_hours: Tensor      # default range [3, 24] h
    # Soil layer thicknesses (m). Z1 fixe (root zone), Z2 et Z3 learnables.
    # Permet adaptation au bouclier (sol mince) vs alluvions (sol profond).
    Z2: Tensor              # default range [0.30, 1.50] m
    Z3: Tensor              # default range [0.50, 4.00] m

    N_PARAMS: ClassVar[int] = 36

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
        Ignored when ``concrete_dropout=True``.
    concrete_dropout : bool
        Use Concrete Dropout (Gal et al., 2017) with learnable rates.
    concrete_init_p : float
        Initial dropout probability for Concrete Dropout layers.
    n_data : int
        Number of data points for Concrete Dropout regularisation scaling.
    param_mode : str
        "nerf" for spatially-varying parameters (~13k params)
        "static" for global parameters like Hydrotel (32 params)
    """

    def __init__(
        self,
        n_territorial: int = 17,
        n_coord_freqs: int = 6,
        hidden: int = 256,
        dropout: float = 0.0,
        concrete_dropout: bool = False,
        concrete_init_p: float = 0.1,
        n_data: int = 2889,
        param_mode: str = "nerf",
        soil_bounds: dict | None = None,
        param_noise: bool = False,
        param_noise_init_sigma: float = 0.05,
    ) -> None:
        super().__init__()
        self.n_territorial = n_territorial
        self.param_mode = param_mode
        # Soil bounds (configurable via toml [soil] section).
        # Z1 is fixed (passed to SoilModule directly), Z2/Z3 are learnable
        # within these bounds. rain_hours bounds also configurable.
        defaults = dict(
            z2_min=0.30, z2_max=1.50,
            z3_min=0.50, z3_max=4.00,
            rain_hours_min=3.0, rain_hours_max=24.0,
        )
        if soil_bounds:
            defaults.update(soil_bounds)
        self.soil_bounds = defaults

        if param_mode == "static":
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
            if concrete_dropout:
                from meandre.spatial.concrete_dropout import ConcreteDropout
                self.drop1 = ConcreteDropout(n_data=n_data, init_p=concrete_init_p)
                self.drop2 = ConcreteDropout(n_data=n_data, init_p=concrete_init_p)
            else:
                self.drop1 = nn.Dropout(p=dropout)
                self.drop2 = nn.Dropout(p=dropout)

            # ParamNoise: σ apprenable par paramètre dans l'espace logit.
            # Injecté APRÈS fc_out, AVANT _apply_constraints — préserve la
            # conservation de masse car les constraints garantissent les
            # bornes physiques.  Plus expressif que ConcreteDropout pour
            # générer un ensemble : chaque membre = un set de params bornés
            # cohérent (analogue à un calage Hydrotel alternatif).
            self.param_noise = param_noise
            if param_noise:
                self.param_log_sigma = nn.Parameter(
                    torch.full(
                        (SpatialParams.N_PARAMS,),
                        math.log(max(param_noise_init_sigma, 1e-6)),
                    )
                )

    def init_from_literature(self, targets: dict[str, float] | None = None) -> None:
        """Initialise fc_out bias so _apply_constraints produces literature defaults.

        Shrinks fc_out.weight so all nodes start with ~identical parameters,
        then the MLP learns spatial variation from there.  This avoids the
        cold-start problem where random init puts K_sat 50x too high.

        References for default values
        -----------------------------
        - Soil hydraulics (K_sat, porosity, theta_fc, theta_wp):
            Rawls et al. (1982). Estimation of soil water properties.
            Trans. ASAE 25(5):1316-1320.  Beven (2001) scale factor for
            sub-daily intensity → daily timestep.
        - Snow degree-day (C_f, T_melt, T_snow):
            Hock (2003). Temperature index melt modelling. J. Hydrol. 282.
        - Manning's n: Chow (1959) Open-Channel Hydraulics, table 5-6.
        - Reference ET (Penman-Monteith): Allen et al. (1998) FAO-56.
        - Muskingum (K, x): Chow et al. (1988) Applied Hydrology, ch. 9.

        Parameters
        ----------
        targets : dict, optional
            Mapping of parameter names to target physical values.
            Missing keys fall back to literature averages for temperate
            forested loam/silt_loam catchments.
        """
        import math

        if self.param_mode == "static":
            # For static mode, set raw params directly
            bias = self._literature_raw_vector(targets)
            self.static_params.data.copy_(bias)
            return

        # Shrink output weights so initial output ≈ bias only
        with torch.no_grad():
            self.fc_out.weight.mul_(0.01)
            bias = self._literature_raw_vector(targets)
            self.fc_out.bias.data.copy_(bias)

    # Backward compatibility alias (deprecated — use init_from_literature)
    init_from_hydrotel = init_from_literature

    def _literature_raw_vector(self, targets: dict[str, float] | None = None) -> Tensor:
        """Compute raw (pre-constraint) values that produce literature targets."""
        import math

        # Defaults: literature averages for temperate forested loam/silt_loam
        # K_sat: Rawls 1982 (cm/h × 24 = m/day), Beven 2001 sub-daily scale ×0.3
        #   loam:      0.0132 m/h × 24 = 0.317 m/day
        #   silt_loam: 0.0068 m/h × 24 = 0.163 m/day → moyenne ~0.24 m/day
        #   ×0.3 (Beven) → 0.080 m/day couche 1, décroissant avec profondeur
        d = {
            # K_sat effectif (m/day) — Rawls 1982 × Beven 2001 sub-daily factor
            "K_sat_1": 0.080, "K_sat_2": 0.040, "K_sat_3": 0.015,
            # Porosity — Rawls 1982: loam=0.434, silt_loam=0.486
            "porosity_1": 0.46, "porosity_2": 0.44, "porosity_3": 0.42,
            # theta_fc — Rawls 1982: loam=0.270, silt_loam=0.330
            "theta_fc_1": 0.30, "theta_fc_2": 0.30, "theta_fc_3": 0.28,
            # theta_wp — Rawls 1982: loam=0.117, silt_loam=0.133
            "theta_wp_1": 0.125, "theta_wp_2": 0.125, "theta_wp_3": 0.12,
            # Root fractions — typical temperate forest (shallow dominant)
            "f_root_1": 0.50, "f_root_2": 0.30, "f_root_3": 0.20,
            # Snow degree-day — Hock 2003, typical 4-5 mm/°C/day boreal
            "C_f": 4.5, "T_melt": -0.5, "T_snow": 1.0,
            # Canopy interception capacity (mm) — typical mixed forest
            "interception_capacity": 1.5,
            # Manning's n — Chow 1959, table 5-6, mixed natural channel
            "manning_n": 0.10,
            # Frost
            "frost_alpha": 0.50,
            # Wetland
            "f_wetland": 0.02,
            # Slope/recession — slope_factor et krec laissés aux centres de
            # contrainte (sigmoid midpoint).  Ce sont des paramètres effectifs
            # du modèle, pas des mesures physiques — le NeRF les apprend.
            "slope_factor": 0.30, "krec": 0.005,
            # Groundwater — recession ~50 jours (k_gw=0.02), réaliste pour
            # aquifères peu profonds tempérés. Auparavant 0.005 (140 jours).
            "k_gw": 0.02,
            # Stream temperature
            "T_gw": 6.0, "K_atm": 0.20,
            # Frost thermal lag
            "alpha_T": 0.03,
            # van Genuchten n — loam ~1.5
            "vg_n": 1.5,
            # Interflow — centre de contrainte log-normal (exp(log(0.03)) = 0.03).
            # Paramètre effectif du modèle, pas une mesure physique.
            "k_interflow": 0.03,
            # Muskingum
            "K_musk_hours": 24.0, "x_musk": 0.20,
            # ETP scaling — défaut 1.0 (FAO-56 reference comme Hydrotel PM).
            "K_c": 1.0,
            # Sub-daily storm duration — 12h par défaut (vs 6h hardcodé avant).
            # Plus réaliste pour pluies frontales QC (vs orages convectifs courts).
            "rain_hours": 12.0,
            # Soil layer thicknesses — Hydrotel BV3C standard
            "Z2": 0.70,
            "Z3": 1.00,
        }
        if targets:
            d.update(targets)

        def inv_bounded(val, lo, hi):
            """Inverse of lo + (hi-lo)*sigmoid(x) → logit."""
            frac = (val - lo) / (hi - lo)
            frac = max(1e-4, min(1.0 - 1e-4, frac))
            return math.log(frac / (1.0 - frac))

        raw = torch.zeros(SpatialParams.N_PARAMS)
        i = 0

        # K_sat: exp(clamp(raw*0.3 + log_center)) → raw = (log(target) - log_center) / 0.3
        log_centers = [math.log(0.5), math.log(0.1), math.log(0.02)]
        for layer, key in enumerate(["K_sat_1", "K_sat_2", "K_sat_3"]):
            raw[i] = (math.log(d[key]) - log_centers[layer]) / 0.3
            i += 1
        # porosity: bounded [0.20, 0.60]
        for key in ["porosity_1", "porosity_2", "porosity_3"]:
            raw[i] = inv_bounded(d[key], 0.20, 0.60)
            i += 1
        # theta_fc as fraction of porosity: bounded [0.30, 0.85]
        for layer, key in enumerate(["theta_fc_1", "theta_fc_2", "theta_fc_3"]):
            por_key = f"porosity_{layer+1}"
            fc_frac = d[key] / d[por_key]
            raw[i] = inv_bounded(fc_frac, 0.30, 0.85)
            i += 1
        # theta_wp as fraction of theta_fc: bounded [0.05, 0.60]
        for layer, key in enumerate(["theta_wp_1", "theta_wp_2", "theta_wp_3"]):
            fc_key = f"theta_fc_{layer+1}"
            wp_frac = d[key] / d[fc_key]
            raw[i] = inv_bounded(wp_frac, 0.05, 0.60)
            i += 1
        # f_root: softmax with bias [1.0, 0.5, -0.5], scaled *0.3
        # We want softmax(raw*0.3 + bias) ≈ [0.50, 0.30, 0.20]
        # Since bias already gives ~[50,30,20], raw ≈ 0 is fine
        for _ in range(3):
            raw[i] = 0.0
            i += 1
        # C_f: bounded [0.5, 8.0]
        raw[i] = inv_bounded(d["C_f"], 0.5, 8.0); i += 1
        # T_melt: bounded [-1, 1]
        raw[i] = inv_bounded(d["T_melt"], -1.0, 1.0); i += 1
        # T_snow: bounded [0, 2]
        raw[i] = inv_bounded(d["T_snow"], 0.0, 2.0); i += 1
        # interception_capacity: bounded [0.5, 2.5]
        raw[i] = inv_bounded(d["interception_capacity"], 0.5, 2.5); i += 1
        # manning_n: bounded [0.01, 0.20]
        raw[i] = inv_bounded(d["manning_n"], 0.01, 0.20); i += 1
        # frost_alpha: bounded [0.0, 1.0]
        raw[i] = inv_bounded(d["frost_alpha"], 0.0, 1.0); i += 1
        # f_wetland: bounded [0.0, 0.10]
        raw[i] = inv_bounded(d["f_wetland"], 0.0, 0.10); i += 1
        # slope_factor: bounded [0.01, 0.59]
        raw[i] = inv_bounded(d["slope_factor"], 0.01, 0.59); i += 1
        # krec: exp(clamp(raw*0.3 + log(0.005)))
        raw[i] = (math.log(d["krec"]) - math.log(0.005)) / 0.3; i += 1
        # k_gw: exp(clamp(raw*0.3 + log(0.02)))
        raw[i] = (math.log(d["k_gw"]) - math.log(0.02)) / 0.3; i += 1
        # T_gw: bounded [3, 13]
        raw[i] = inv_bounded(d["T_gw"], 3.0, 13.0); i += 1
        # K_atm: bounded [0.05, 0.55]
        raw[i] = inv_bounded(d["K_atm"], 0.05, 0.55); i += 1
        # alpha_T: bounded [0.01, 0.05]
        raw[i] = inv_bounded(d["alpha_T"], 0.01, 0.05); i += 1
        # vg_n: bounded [1.3, 2.7]
        raw[i] = inv_bounded(d["vg_n"], 1.3, 2.7); i += 1
        # k_interflow: exp(clamp(raw*0.3 + log(0.03)))
        raw[i] = (math.log(d["k_interflow"]) - math.log(0.03)) / 0.3; i += 1
        # K_musk_hours: bounded [4, 48]
        raw[i] = inv_bounded(d["K_musk_hours"], 4.0, 48.0); i += 1
        # x_musk: bounded [0.01, 0.49]
        raw[i] = inv_bounded(d["x_musk"], 0.01, 0.49); i += 1
        # K_c: bounded [0.3, 1.5]
        raw[i] = inv_bounded(d["K_c"], 0.3, 1.5); i += 1
        # rain_hours: bounded [rh_min, rh_max] from soil_bounds
        rh_min = self.soil_bounds["rain_hours_min"]
        rh_max = self.soil_bounds["rain_hours_max"]
        raw[i] = inv_bounded(d["rain_hours"], rh_min, rh_max); i += 1
        # Z2, Z3: bounded from soil_bounds
        raw[i] = inv_bounded(d["Z2"], self.soil_bounds["z2_min"], self.soil_bounds["z2_max"]); i += 1
        raw[i] = inv_bounded(d["Z3"], self.soil_bounds["z3_min"], self.soil_bounds["z3_max"]); i += 1

        return raw

    # Backward compatibility alias (deprecated — use _literature_raw_vector)
    _hydrotel_raw_vector = _literature_raw_vector

    def forward(
        self,
        coords: Tensor,
        territorial: Tensor,
        perturb_params: bool = False,
        param_noise_eps: Tensor | None = None,
    ) -> SpatialParams:
        """
        Args:
            coords: (n_nodes, 2)  [lon, lat] in degrees, normalised.
            territorial: (n_nodes, n_territorial)
            perturb_params: If True and self.param_noise is enabled, add
                Gaussian noise to fc_out logits (BEFORE constraints).
                Generates one ensemble member.  Mass conservation is
                preserved because constraints map perturbed logits back
                into physical bounds before they reach the physics.
            param_noise_eps: Optional pre-sampled (N_PARAMS,) or
                (n_nodes, N_PARAMS) noise tensor.  Use to freeze a noise
                realisation across a trajectory (ensemble member).  When
                None and ``perturb_params=True``, samples fresh ε each call.
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

            h = self.drop1(self.act(self.fc1(x0)))
            h = torch.cat([h, x0], dim=-1)       # skip connection
            h = self.drop2(self.act(self.fc2(h)))
            raw = self.fc_out(h)                  # (n_nodes, N_PARAMS)

        # Inject ParamNoise BEFORE constraints (preserves mass conservation).
        if perturb_params and getattr(self, "param_noise", False):
            sigma = self.param_log_sigma.exp().clamp(min=1e-4, max=1.0)
            if param_noise_eps is None:
                eps = torch.randn_like(raw)
            else:
                # Broadcast a (N_PARAMS,) eps across nodes for a frozen-mask member
                if param_noise_eps.dim() == 1:
                    eps = param_noise_eps.unsqueeze(0).expand_as(raw)
                else:
                    eps = param_noise_eps
            raw = raw + sigma * eps

        return self._apply_constraints(raw)

    def _apply_constraints(self, raw: Tensor) -> SpatialParams:
        """Map raw network outputs to physically plausible ranges.

        Uses sigmoid parameterization: lo + (hi-lo) * sigmoid(x).
        Max gradient at x=0 is (hi-lo)/4, which is much better than the old
        tanh(x*0.3) approach that had max gradient of 0.3*half_range.

        Key constraint: theta_wp < theta_fc < porosity is enforced by
        parameterizing theta_fc and theta_wp as fractions of porosity.
        """
        import math

        def bounded(x, lo, hi):
            """Sigmoid-bounded: lo + (hi-lo) * sigmoid(x). Max grad = (hi-lo)/4."""
            return lo + (hi - lo) * torch.sigmoid(x)

        cols = [raw[:, i] for i in range(SpatialParams.N_PARAMS)]
        i = 0

        constrained = []
        # K_sat (m/day): log-normal with per-layer centers decreasing with depth.
        log_centers = [math.log(0.5), math.log(0.1), math.log(0.02)]
        for layer in range(3):
            exponent = torch.clamp(cols[i] * 0.3 + log_centers[layer], min=-8.0, max=4.0)
            constrained.append(torch.exp(exponent))
            i += 1
        # porosity: [0.20, 0.60]
        porosities = []
        for _ in range(3):
            p = bounded(cols[i], 0.20, 0.60)
            porosities.append(p)
            constrained.append(p)
            i += 1
        # theta_fc as fraction of porosity: [0.30, 0.85]
        # Guarantees theta_fc < porosity always
        theta_fcs = []
        for layer in range(3):
            fc_frac = bounded(cols[i], 0.30, 0.85)
            theta_fc = porosities[layer] * fc_frac
            theta_fcs.append(theta_fc)
            constrained.append(theta_fc)
            i += 1
        # theta_wp as fraction of theta_fc: [0.05, 0.60]
        # Guarantees theta_wp < theta_fc always
        for layer in range(3):
            wp_frac = bounded(cols[i], 0.05, 0.60)
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
        # C_f: [0.5, 8.0] mm/C/day
        constrained.append(bounded(cols[i], 0.5, 8.0)); i += 1
        # T_melt: [-1, 1] C
        constrained.append(bounded(cols[i], -1.0, 1.0)); i += 1
        # T_snow: [0, 2] C
        constrained.append(bounded(cols[i], 0.0, 2.0)); i += 1
        # interception_capacity: [0.5, 2.5] mm
        constrained.append(bounded(cols[i], 0.5, 2.5)); i += 1
        # manning_n: [0.01, 0.20]
        constrained.append(bounded(cols[i], 0.01, 0.20)); i += 1
        # frost_alpha: [0.0, 1.0]
        constrained.append(bounded(cols[i], 0.0, 1.0)); i += 1
        # f_wetland: [0.0, 0.10]
        constrained.append(bounded(cols[i], 0.0, 0.10)); i += 1
        # slope_factor: [0.01, 0.59]
        constrained.append(bounded(cols[i], 0.01, 0.59)); i += 1
        # krec: soil L3 → aquifer drainage (1/day), log-normal.
        # Recentré sur 0.005 (vs 0.003) — Hydrotel typique pour bassin tempéré.
        exponent = torch.clamp(cols[i] * 0.3 + math.log(0.005), min=-8.0, max=-3.0)
        constrained.append(torch.exp(exponent)); i += 1
        # k_gw: aquifer recession (1/day), log-normal.
        # Recentré sur 0.02 (vs 0.005) — recession ~50 jours réaliste pour
        # aquifères peu profonds Beauce/Lévis (auparavant ~140 jours, trop lent).
        exponent = torch.clamp(cols[i] * 0.3 + math.log(0.02), min=-8.0, max=-2.0)
        constrained.append(torch.exp(exponent)); i += 1
        # T_gw: groundwater temperature (C): [3, 13]
        constrained.append(bounded(cols[i], 3.0, 13.0)); i += 1
        # K_atm: atmospheric heat exchange (1/day): [0.05, 0.55]
        constrained.append(bounded(cols[i], 0.05, 0.55)); i += 1
        # alpha_T: soil thermal damping (1/day): [0.01, 0.05]
        constrained.append(bounded(cols[i], 0.01, 0.05)); i += 1
        # --- New params ---
        # vg_n: van Genuchten n shape parameter [1.1, 2.7]
        # Clay ~1.1, loam ~1.5, sand ~2.7
        constrained.append(bounded(cols[i], 1.3, 2.7)); i += 1
        # k_interflow: interflow recession rate (1/day) [0.005, 0.1]
        # Log-normal for better gradient scaling
        exponent = torch.clamp(cols[i] * 0.3 + math.log(0.03), min=math.log(0.005), max=math.log(0.1))
        constrained.append(torch.exp(exponent)); i += 1
        # K_musk_hours: Muskingum travel time [4, 48] hours.
        # Stabilité numérique avec n_substeps=2 (sub_dt=12h) requiert K ≥ ~6h;
        # bound 4 conservé comme défaut. Bornes étendues à 0.5 nécessitent
        # n_substeps ≥ 24 (cf message_passing.py) — pas activé par défaut.
        constrained.append(bounded(cols[i], 4.0, 48.0)); i += 1
        # x_musk: Muskingum weighting factor [0.01, 0.49]
        constrained.append(bounded(cols[i], 0.01, 0.49)); i += 1
        # K_c: ETP scaling [0.3, 1.5]. Default ~1.0 (FAO-56 reference).
        constrained.append(bounded(cols[i], 0.3, 1.5)); i += 1
        # rain_hours: storm duration for Eagleson sub-daily intensity.
        # Configurable bounds (default [3, 24] h) — moins = pluies plus intenses.
        rh_min = self.soil_bounds["rain_hours_min"]
        rh_max = self.soil_bounds["rain_hours_max"]
        constrained.append(bounded(cols[i], rh_min, rh_max)); i += 1
        # Z2: layer 2 thickness (m). Default [0.30, 1.50] — root zone profonde.
        z2_min = self.soil_bounds["z2_min"]
        z2_max = self.soil_bounds["z2_max"]
        constrained.append(bounded(cols[i], z2_min, z2_max)); i += 1
        # Z3: layer 3 thickness (m). Default [0.50, 4.00] — sol profond.
        z3_min = self.soil_bounds["z3_min"]
        z3_max = self.soil_bounds["z3_max"]
        constrained.append(bounded(cols[i], z3_min, z3_max)); i += 1

        return SpatialParams.from_tensor(torch.stack(constrained, dim=-1))

    def concrete_kl(self) -> Tensor:
        """Sum of Concrete Dropout KL terms (0 if using standard dropout)."""
        from meandre.spatial.concrete_dropout import ConcreteDropout
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        if isinstance(getattr(self, "drop1", None), ConcreteDropout):
            total = total + self.drop1.regularization(self.fc1.weight)
        if isinstance(getattr(self, "drop2", None), ConcreteDropout):
            total = total + self.drop2.regularization(self.fc2.weight)
        return total

    def param_noise_kl(self, target_sigma: float = 0.05) -> Tensor:
        """L2 regulariser pulling param_log_sigma toward log(target_sigma).

        Without this, the data loss collapses sigma → 0 (deterministic ensemble)
        because noise hurts fit.  With it, sigma is pulled toward a calibrated
        target — typical hydrological parameter uncertainty in logit space.

        The target ≈ 0.05 corresponds to ~5% perturbation of constrained
        parameters (sigmoid output near the centre).  Tune empirically based
        on Talagrand histogram flatness post-training.
        """
        if not getattr(self, "param_noise", False):
            return torch.tensor(0.0, device=next(self.parameters()).device)
        log_target = math.log(target_sigma)
        return ((self.param_log_sigma - log_target) ** 2).mean()

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

        # Exp-constrained params: L2 on raw → pulls toward center
        # K_sat (0-2), krec (23), k_gw (24), k_interflow (29)
        unbounded_cols = [0, 1, 2, 23, 24, 29]
        unbounded_penalty = (raw[:, unbounded_cols] ** 2).mean()

        # Sigmoid-constrained columns: penalize saturation
        # Skip K_sat (0-2), f_root (12-14), krec (23), k_gw (24), k_interflow (29)
        sig_cols = (list(range(3, 12)) + list(range(15, 23))
                    + [25, 26, 27, 28, 30, 31])
        raw_sig = raw[:, sig_cols]
        sig = torch.sigmoid(raw_sig)
        sig_penalty = ((2.0 * sig - 1.0) ** 4).mean()

        return unbounded_penalty + sig_penalty

    def physical_prior_loss(self, params: SpatialParams) -> Tensor:
        """Soft L2 penalty pulling parameters toward physically reasonable values.

        Light-touch: only penalise extreme deviations, not normal variation.
        """
        import math
        device = params.K_sat_1.device
        loss = torch.tensor(0.0, device=device)

        # K_sat in log-space
        # Targets cohérents avec init_from_literature (m/day, effectif journalier)
        for k, target in [(params.K_sat_1, 0.08), (params.K_sat_2, 0.04), (params.K_sat_3, 0.015)]:
            loss = loss + ((torch.log(k + 1e-8) - math.log(target)) ** 2).mean() * 0.3

        # Porosity
        for p in [params.porosity_1, params.porosity_2, params.porosity_3]:
            loss = loss + ((p - 0.40) ** 2).mean()

        # C_f
        loss = loss + ((params.C_f - 3.0) ** 2).mean() * 0.3

        # T_melt: near 0°C
        loss = loss + (params.T_melt ** 2).mean() * 0.5

        # frost_alpha
        loss = loss + ((params.frost_alpha - 0.5) ** 2).mean() * 0.3

        # alpha_T: reduced from 100x to 1x
        loss = loss + ((params.alpha_T - 0.03) ** 2).mean()

        # vg_n: typical 1.5 for loam
        loss = loss + ((params.vg_n - 1.5) ** 2).mean() * 0.3

        # k_gw aquifer recession (1/day): target ~0.02 (recession ~50d).
        # Empêche la dérive vers k_gw trop bas (= aquifère qui sur-stocke
        # et libère l'eau avec des années de retard).
        loss = loss + ((torch.log(params.k_gw + 1e-8) - math.log(0.02)) ** 2).mean() * 0.3

        # krec drainage soil L3 → aquifer (1/day): target ~0.005.
        # Évite saturation contre la borne max si elle existe.
        loss = loss + ((torch.log(params.krec + 1e-8) - math.log(0.005)) ** 2).mean() * 0.2

        # K_c: target ~0.85 (Hydrotel SLSO McGuinness coefficient typical)
        if hasattr(params, 'K_c'):
            loss = loss + ((params.K_c - 0.85) ** 2).mean() * 0.2

        return loss
