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
    f_vert_{1,2,3}  Per-layer drainage partition (0, 1): fraction of excess
                    drainage going DOWN (to next layer; aquifer for L3).
                    1 - f_vert goes LATERALLY to the stream (interflow).
                    Replaces the legacy slope_factor + k_interflow + krec
                    triplet to break their equifinality coupling.
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
    # Drainage partition (replaces slope_factor + k_interflow + krec
    # competing for the same water budget — see softmax-partition design,
    # 2026-05-11). Each f_vert_i is the fraction of the layer's excess
    # drainage that goes *down* to the next layer (or aquifer for L3);
    # 1 - f_vert_i goes laterally to the stream as interflow.
    f_vert_1: Tensor        # partition layer 1: vertical fraction (0, 1)
    f_vert_3: Tensor        # partition layer 3: recharge fraction (0, 1)
    # Groundwater
    k_gw: Tensor            # aquifer recession coefficient (1/day) [0.001, 0.14]
    # Stream temperature
    T_gw: Tensor            # groundwater temperature (C) [3, 13]
    K_atm: Tensor           # atmospheric heat exchange coefficient (1/day) [0.05, 0.55]
    # Frost thermal lag
    alpha_T: Tensor         # soil thermal damping (1/day) [0.01, 0.05]
    # --- New params (E, F, G) ---
    vg_n: Tensor            # van Genuchten n shape parameter [1.3, 2.7]
    f_vert_2: Tensor        # partition layer 2: vertical fraction (0, 1)
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

    vsa_b: Tensor           # exposant aire-source-variable (ruissellement) [0.5, 5.0]

    N_PARAMS: ClassVar[int] = 37

    @classmethod
    def from_tensor(cls, x: Tensor) -> "SpatialParams":
        """Reconstruct from (n_nodes, N_PARAMS) tensor."""
        fields = [x[:, i] for i in range(cls.N_PARAMS)]
        return cls(*fields)

    def to_tensor(self) -> Tensor:
        """Stack all parameter fields into (n_nodes, N_PARAMS) tensor.

        Inverse of ``from_tensor``: column i corresponds to field i.
        """
        import dataclasses
        return torch.stack(
            [getattr(self, f.name) for f in dataclasses.fields(self)],
            dim=-1,
        )


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
        MC Dropout rate (set > 0 to enable standard nn.Dropout).
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
        param_mode: str = "nerf",
        soil_bounds: dict | None = None,
        predict_lake_params: bool = False,
        n_nodes: int | None = None,
        use_latent_codes: bool = False,
        latent_dim: int = 8,
        latent_mode: str = "additive",
    ) -> None:
        super().__init__()
        self.n_territorial = n_territorial
        self.param_mode = param_mode
        # Codes latents par nœud (effet aléatoire spatial, type auto-décodeur).
        # Le NeRF lie les paramètres aux features ; deux bassins aux features
        # semblables reçoivent des params semblables → pics moyennés (déficit vs
        # Hydrotel calibré par bassin). Un code latent z_n par nœud, concaténé
        # aux features, laisse chaque bassin DÉVIER pour caler ses propres pics ;
        # le shrinkage L2 (vers 0) = partial pooling : adapte aux jauges, retombe
        # au feature-mean ailleurs. Init 0 → départ identique au NeRF sans codes.
        # latent_mode :
        #   "additive" (défaut) — effet aléatoire MIXTE : raw = NeRF(features) +
        #     z_n, où z_n est un offset par nœud PAR PARAMÈTRE ajouté aux params
        #     bruts avant contraintes. C'est la vraie structure d'effet mixte
        #     (effet fixe = NeRF, effet aléatoire = z_n), chaque bassin dévie
        #     DIRECTEMENT ses params sans passer par le goulot du tronc.
        #   "input" — z_n (dim latent_dim) concaténé aux features en entrée du
        #     tronc (auto-décodeur). Indirect : le nudge est filtré par le tronc.
        # Shrinkage L2 (w_latent_reg) dans les deux cas = partial pooling.
        self.use_latent_codes = bool(use_latent_codes) and param_mode != "static"
        self.latent_mode = latent_mode
        self.latent_dim = int(latent_dim) if self.use_latent_codes else 0
        if self.use_latent_codes:
            if n_nodes is None:
                raise ValueError("use_latent_codes=True requiert n_nodes")
            n_z = SpatialParams.N_PARAMS if latent_mode == "additive" else self.latent_dim
            self.latent_codes = nn.Parameter(torch.zeros(n_nodes, n_z))
        # Tête de lac optionnelle : k_lake et beta par nœud (sortie séparée de
        # fc_out pour ne pas changer N_PARAMS=36 ni casser les checkpoints
        # existants). Câblée au LakeModule par HydroModel quand activée.
        self.predict_lake_params = predict_lake_params
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
            if predict_lake_params:
                self.fc_lake_static = nn.Parameter(torch.zeros(2))
        else:
            # NeRF mode: MLP mapping coordinates to parameters
            self.coord_enc = FourierPositionalEncoding(n_freqs=n_coord_freqs, include_input=True)
            coord_dim = self.coord_enc.out_dim(2)  # encoded (lon, lat)
            # Les codes ne grossissent l'entrée du tronc qu'en mode "input".
            _latent_in = self.latent_dim if (self.use_latent_codes and self.latent_mode == "input") else 0
            in_dim = coord_dim + n_territorial + _latent_in

            self.fc1 = nn.Linear(in_dim, hidden)
            self.fc2 = nn.Linear(hidden + in_dim, hidden)  # skip connection
            self.fc_out = nn.Linear(hidden, SpatialParams.N_PARAMS)
            self.act = nn.SiLU()
            self.drop1 = nn.Dropout(p=dropout)
            self.drop2 = nn.Dropout(p=dropout)
            if predict_lake_params:
                # 2 sorties : k_lake (log) et beta. Biais initialisé pour
                # reproduire les défauts globaux (k=1e-4, beta=1.5) au départ.
                self.fc_lake = nn.Linear(hidden, 2)
                nn.init.zeros_(self.fc_lake.weight)
                with torch.no_grad():
                    # inverse des bornes appliquées dans lake_params()
                    self.fc_lake.bias[0] = 0.0  # k_lake → centre log = 1e-4
                    self.fc_lake.bias[1] = 0.0  # beta  → centre = 1.5

    def init_from_literature(
        self,
        targets: dict[str, float] | None = None,
        weight_shrink: float = 0.1,
    ) -> None:
        """Initialise fc_out bias so _apply_constraints produces literature defaults.

        Shrinks fc_out.weight so all nodes start near identical parameters,
        then the MLP learns spatial variation from there.  This avoids the
        cold-start problem where random init puts K_sat 50x too high.

        Parameters
        ----------
        weight_shrink :
            Factor applied to ``fc_out.weight`` after Xavier init. Smaller =
            more uniform start (closer to literature targets on every node);
            larger = more spatial signal but more dispersion around the
            literature targets. Default ``0.1`` (was ``0.01`` historically —
            that legacy value made the NeRF effectively spatially constant
            and required impractically long training to break uniformity).
            At ``0.1`` the per-node deviation from literature is roughly ±5%
            of the bias scale at init.

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

        # Shrink output weights so initial output ≈ bias + small per-node deviation
        with torch.no_grad():
            self.fc_out.weight.mul_(weight_shrink)
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
            # Drainage partition per layer (softmax binary = sigmoid).
            # L1 root zone : moitié-moitié — interflow dominant si pente.
            # L2 transition : un peu plus vertical (percolation).
            # L3 deep : majoritairement recharge aquifère.
            "f_vert_1": 0.50, "f_vert_2": 0.60, "f_vert_3": 0.70,
            # Groundwater — recession ~50 jours (k_gw=0.02), réaliste pour
            # aquifères peu profonds tempérés. Auparavant 0.005 (140 jours).
            "k_gw": 0.02,
            # Stream temperature
            "T_gw": 6.0, "K_atm": 0.20,
            # Frost thermal lag
            "alpha_T": 0.03,
            # van Genuchten n — loam ~1.5
            "vg_n": 1.5,
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
            "vsa_b": 2.5,
        }
        if targets:
            d.update(targets)
        # Source unique de vérité : physical_prior_loss tire vers CES cibles
        # (résolues, overrides config inclus). Avant, le prior avait ses propres
        # constantes contradictoires avec l'init (K_c 0.85 vs init 1.0/0.6,
        # C_f 3.0 vs 4.5, porosity 0.40 uniforme vs 0.46/0.44/0.42), ce qui
        # créait un gradient uniforme synchronisant dès le premier pas (revue
        # 2026-07-01).
        self._prior_targets = dict(d)

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
        # f_vert_1: bounded [0, 1]
        raw[i] = inv_bounded(d["f_vert_1"], 0.0, 1.0); i += 1
        # f_vert_3: bounded [0, 1]
        raw[i] = inv_bounded(d["f_vert_3"], 0.0, 1.0); i += 1
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
        # f_vert_2: bounded [0, 1]
        raw[i] = inv_bounded(d["f_vert_2"], 0.0, 1.0); i += 1
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
        raw[i] = inv_bounded(d["vsa_b"], 0.5, 5.0); i += 1

        return raw

    # Backward compatibility alias (deprecated — use _literature_raw_vector)
    _hydrotel_raw_vector = _literature_raw_vector

    def forward(
        self,
        coords: Tensor,
        territorial: Tensor,
    ) -> SpatialParams:
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
            raw = self.fc_out(self._trunk(coords, territorial))
            if self.use_latent_codes and self.latent_mode == "additive":
                # Effet aléatoire : offset par nœud sur les params BRUTS (avant
                # contraintes). raw = effet_fixe(features) + effet_aléatoire(z_n).
                raw = raw + self.latent_codes

        return self._apply_constraints(raw)

    def latent_reg(self) -> Tensor:
        """Pénalité de shrinkage L2 des codes latents (partial pooling).

        Tire les z_n vers 0 : chaque bassin ne dévie du feature-mean que si ses
        données le justifient. Zéro si les codes sont désactivés.
        """
        if self.use_latent_codes:
            return self.latent_codes.pow(2).mean()
        return torch.zeros((), device=self.fc_out.weight.device)

    def _project_coords(self, coords: Tensor) -> Tensor:
        """Projette (lon, lat) en degrés vers des coordonnées ISOTROPES normalisées.

        Les degrés lon/lat ne sont pas isotropes : à la latitude φ, 1° de longitude
        ≈ cos(φ)·111 km contre ≈ 111 km pour 1° de latitude. Traiter (lon, lat)
        comme cartésien distord l'encodage Fourier (une « fréquence » en lon n'a pas
        la même longueur d'onde physique qu'en lat). On applique une projection
        équirectangulaire centrée sur la latitude médiane (haversine-cohérente pour
        un bassin régional) → km, puis on normalise par l'extent isotrope max
        (aspect préservé) → coords ∈ ~[-1, 1]. L'encodage opère alors sur des
        distances physiques réelles.
        """
        lon = coords[:, 0]
        lat = coords[:, 1]
        lat0 = lat.mean()
        x = (lon - lon.mean()) * torch.cos(torch.deg2rad(lat0)) * 111.32
        y = (lat - lat.mean()) * 110.574
        scale = torch.maximum(x.abs().max(), y.abs().max()).clamp(min=1e-6)
        return torch.stack([x / scale, y / scale], dim=-1)

    def _trunk(self, coords: Tensor, territorial: Tensor) -> Tensor:
        """Tronc NeRF partagé (fc1 → skip → fc2) → features cachées h."""
        enc = self.coord_enc(self._project_coords(coords))  # (n_nodes, coord_dim)
        feats = [enc, territorial]
        if self.use_latent_codes and self.latent_mode == "input":
            # z_n aligné sur l'ordre des nœuds (coords couvre tous les nœuds).
            feats.append(self.latent_codes)
        x0 = torch.cat(feats, dim=-1)  # (n_nodes, in_dim)
        h = self.drop1(self.act(self.fc1(x0)))
        h = torch.cat([h, x0], dim=-1)              # skip connection
        return self.drop2(self.act(self.fc2(h)))

    def lake_params(self, coords: Tensor, territorial: Tensor) -> tuple[Tensor, Tensor]:
        """Paramètres de lac par nœud (k_lake, beta), bornés physiquement.

        k_lake ∈ [1e-6, 1e-2] (log-uniforme, centre 1e-4), beta ∈ [1.0, 2.5]
        (centre 1.5, tarage type seuil). Requiert predict_lake_params=True.
        """
        if not self.predict_lake_params:
            raise RuntimeError("predict_lake_params=False : pas de tête de lac")
        if self.param_mode == "static":
            raw = self.fc_lake_static.unsqueeze(0).expand(coords.shape[0], -1)
        else:
            raw = self.fc_lake(self._trunk(coords, territorial))
        # k_lake : log-uniforme centré sur 1e-4 ; raw=0 → 1e-4
        log_k = torch.clamp(raw[:, 0] * 0.5 + math.log(1e-4), min=math.log(1e-6), max=math.log(1e-2))
        k_lake = torch.exp(log_k)
        # beta : [1.0, 2.5], centré à 1.5 pour raw=0. 1.0 + 1.5*s = 1.5 → s=1/3,
        # donc décalage logit(1/3) = -log(2).
        beta = 1.0 + 1.5 * torch.sigmoid(raw[:, 1] - math.log(2.0))
        return k_lake, beta

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
        # f_vert_1: partition layer 1 vertical/lateral, (0, 1)
        # Binary softmax = sigmoid. Init centred at 0.5 (no prior on direction).
        constrained.append(bounded(cols[i], 0.0, 1.0)); i += 1
        # f_vert_3: partition layer 3 recharge/lateral, (0, 1)
        # Init biased toward recharge (~0.7) for deep layer.
        constrained.append(bounded(cols[i], 0.0, 1.0)); i += 1
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
        # f_vert_2: partition layer 2 vertical/lateral, (0, 1)
        constrained.append(bounded(cols[i], 0.0, 1.0)); i += 1
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
        # vsa_b: exposant de l'aire-source-variable (ruissellement de crue).
        constrained.append(bounded(cols[i], 0.5, 5.0)); i += 1

        return SpatialParams.from_tensor(torch.stack(constrained, dim=-1))


    def boundary_regularization(
        self,
        coords: Tensor,
        territorial: Tensor,
        sat_threshold: float = 0.8,
    ) -> Tensor:
        """Soft-hinge anti-saturation prior on sigmoid-bounded raw outputs.

        Zero penalty inside [σ=0.2, σ=0.8] (with threshold=0.8); rises quadratically
        only when sigmoid output exits the safe band. Contrast with the previous
        ``(2σ-1)⁴`` form, which was quartic everywhere on (-1, 1) and effectively a
        pull-to-center prior — it suppressed spatial variance even before
        saturation. The hinge form lets the NeRF freely explore the middle band
        while still pushing back against true saturation.

        Unbounded (exp-constrained) columns are not penalised here — that role is
        played by ``physical_prior_loss`` which already pulls log(K_sat), log(k_gw)
        toward literature targets.
        """
        enc = self.coord_enc(self._project_coords(coords))
        x0 = torch.cat([enc, territorial], dim=-1)
        h = torch.nn.functional.silu(self.fc1(x0))
        h = torch.cat([h, x0], dim=-1)
        h = torch.nn.functional.silu(self.fc2(h))
        raw = self.fc_out(h)

        # Colonnes sigmoid-bornées : la liste couvrait 3-31 seulement ; les params
        # 32-36 (K_c, rain_hours, Z2, Z3, vsa_b), sigmoid-bornés eux aussi, n'étaient
        # pas surveillés contre la saturation (revue 2026-07-01).
        sig_cols = (list(range(3, 12)) + list(range(15, 24))
                    + [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36])
        sig = torch.sigmoid(raw[:, sig_cols])
        # |2σ-1| ∈ [0, 1]. Hinge active only when this exceeds sat_threshold.
        excess = torch.clamp(torch.abs(2.0 * sig - 1.0) - sat_threshold, min=0.0)
        return (excess ** 2).mean()

    def physical_prior_loss(self, params: SpatialParams) -> Tensor:
        """Soft L2 penalty pulling the SPATIAL MEAN toward literature targets.

        PRIOR SUR LA MOYENNE, PAS PAR NŒUD (revue 2026-07-01) : l'ancienne forme
        ``((p - c)**2).mean()`` se décompose en (p_bar - c)² + Var(p) — elle
        pénalisait DIRECTEMENT la variance spatiale, au même poids que le biais
        de moyenne. Structurellement anti-NeRF : cause mathématique du collapse
        de k_gw/f_vert/vg_n/frost_alpha (CV ~0.002-0.005). La forme
        ``(p.mean() - c)**2`` ancre la climatologie du champ et laisse la
        différenciation spatiale libre.

        Cibles : ``self._prior_targets`` (résolues par init_from_literature,
        overrides [literature_prior] de la config inclus) — une seule source de
        vérité, plus de contradictions init/prior.
        """
        import math
        device = params.K_sat_1.device
        loss = torch.tensor(0.0, device=device)
        t = getattr(self, "_prior_targets", None) or {}
        def tg(key, fallback):
            return float(t.get(key, fallback))

        # K_sat en log-espace (moyenne du log = médiane géométrique du champ)
        for k, key, fb in [(params.K_sat_1, "K_sat_1", 0.08),
                           (params.K_sat_2, "K_sat_2", 0.04),
                           (params.K_sat_3, "K_sat_3", 0.015)]:
            loss = loss + ((torch.log(k + 1e-8).mean() - math.log(tg(key, fb))) ** 2) * 0.3

        # Porosity (cibles par couche, alignées init)
        for p, key, fb in [(params.porosity_1, "porosity_1", 0.46),
                           (params.porosity_2, "porosity_2", 0.44),
                           (params.porosity_3, "porosity_3", 0.42)]:
            loss = loss + ((p.mean() - tg(key, fb)) ** 2)

        # C_f (aligné init Hock 4.5, plus 3.0)
        loss = loss + ((params.C_f.mean() - tg("C_f", 4.5)) ** 2) * 0.3

        # T_melt
        loss = loss + ((params.T_melt.mean() - tg("T_melt", -0.5)) ** 2) * 0.5

        # frost_alpha
        loss = loss + ((params.frost_alpha.mean() - tg("frost_alpha", 0.5)) ** 2) * 0.3

        # alpha_T
        loss = loss + ((params.alpha_T.mean() - tg("alpha_T", 0.03)) ** 2)

        # vg_n
        loss = loss + ((params.vg_n.mean() - tg("vg_n", 1.5)) ** 2) * 0.3

        # k_gw récession (log-espace)
        loss = loss + ((torch.log(params.k_gw + 1e-8).mean() - math.log(tg("k_gw", 0.02))) ** 2) * 0.3

        # K_c (aligné sur la cible d'init, ex. 0.6 via [literature_prior])
        if hasattr(params, 'K_c'):
            loss = loss + ((params.K_c.mean() - tg("K_c", 1.0)) ** 2) * 0.2

        return loss

    def param_diversity_loss(self, params: SpatialParams, cv_target: float = 0.12) -> Tensor:
        """Anti-collapse : pénalise un coefficient de variation spatial INFÉRIEUR
        à ``cv_target`` pour les paramètres clés.

        Diagnostic 2026-06-12 : sur le bassin open-data le NeRF collapse vers des
        params quasi uniformes (CV 0.006-0.09 vs 0.2-0.47 sur PHYSITEL), incapable
        de reproduire l'hétérogénéité du ruissellement. ``physical_prior_loss``
        aggrave en tirant chaque nœud vers une cible scalaire uniforme.

        Plancher souple (relu) : la perte est nulle dès que CV ≥ cv_target — on ne
        RÉCOMPENSE jamais la variance (pas de dérive vers du bruit / des outliers),
        on combat seulement l'effondrement. La perte Q + les features façonnent OÙ
        va la variance ; ce terme garantit seulement qu'elle existe. K_sat/k_gw
        (log-distribués) sont mesurés en espace log pour ne pas laisser quelques
        nœuds extrêmes satisfaire le plancher à bon compte.
        """
        eps = 1e-8
        loss = torch.tensor(0.0, device=params.K_sat_1.device)
        log_keys = ("K_sat_1", "K_sat_2", "K_sat_3", "k_gw")
        lin_keys = ("f_vert_1", "f_vert_2", "f_vert_3", "K_c")
        n = 0
        # Charnière LINÉAIRE (pas au carré) : gradient constant fort tant que
        # cv < cv_target, nul au-dessus. Le carré s'annulait trop vite pour des
        # params très collapsés (cv~0.006 → perte ~0.0001/param, négligeable).
        for k in log_keys:
            if not hasattr(params, k):
                continue
            v = torch.log(torch.clamp(getattr(params, k), min=eps))
            cv = v.std() / (v.abs().mean() + eps)
            loss = loss + torch.clamp(cv_target - cv, min=0.0)
            n += 1
        for k in lin_keys:
            if not hasattr(params, k):
                continue
            v = getattr(params, k)
            cv = v.std() / (v.abs().mean() + eps)
            loss = loss + torch.clamp(cv_target - cv, min=0.0)
            n += 1
        return loss / max(n, 1)
