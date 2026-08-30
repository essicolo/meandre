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

import os
from dataclasses import fields as _dc_fields
import torch

# Bornes du temps de transfert Muskingum (heures). Défaut historique [4, 48] avec init
# 24 ; MEANDRE_KMUSK="min,max,init" permet de descendre au temps de parcours PHYSIQUE
# (~0.2-0.35 h mesuré par Manning sur troncon.trl). Voir le commentaire dans forward().
_kmb = os.environ.get("MEANDRE_KMUSK", "4,48,24").split(",")
_KMUSK_MIN, _KMUSK_MAX, _KMUSK_INIT = (float(_kmb[0]), float(_kmb[1]), float(_kmb[2]))
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from meandre.spatial.positional_encoding import FourierPositionalEncoding


# Recharge de reference (m/h). Le banc du 2026-08-19 place ce reglage a ~27 % du debit
# porte par la nappe, soit de l'ordre de 150 mm/an sur OUTV -- coherent avec les ordres
# de grandeur publies pour le Quebec meridional (100 a 400 mm/an selon les regions).
# Le debit SEUL prefere une recharge quasi nulle : c'est un arbitrage assume, pas un
# optimum de score (voir la note d'enjeu au registre).
KREC_REF = 2e-5


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

    # RECHARGE PROFONDE, par noeud (2026-08-20). Elle etait un SCALAIRE GLOBAL de la
    # colonne : une seule valeur pour toute une region. C'est bloquant pour ce que le
    # projet doit livrer -- une recharge qui ne varie pas dans l'espace ne peut ni
    # suivre la geologie, ni repondre a l'urbanisation d'un territoire agricole, ni
    # etre confrontee a une carte regionale qui va de 0 a 400 mm/an sur quelques
    # dizaines de kilometres. Sortie du champ spatial, elle recoit les descripteurs
    # territoriaux, dont l'occupation du sol.
    krec: Tensor            # drainage profond L3 -> aquifere (m/h) [1e-7, 1e-4]
    # ── PROPRIETES THERMIQUES DU SOL (gel de Rankinen) ───────────────────────
    # Remarque d'Essi (2026-08-27) : « le gel doit etre physique, en fonction de la
    # temperature et de la diffusion de la chaleur dans le sol ». Ces trois grandeurs
    # etaient des SCALAIRES GLOBAUX identiques sur les 25 656 troncons de la province,
    # et la docstring du clone le disait elle-meme (« KT/CS/CIce uniformes donc le type
    # n'entre pas »). Une argile saturee et un sable sec gelaient donc identiquement,
    # alors que leurs proprietes thermiques different d'un facteur trois a quatre.
    # Elles sont IDENTIFIABLES separement parce qu'elles agissent sur des choses
    # differentes : la conductivite sur la vitesse de descente du front, la capacite
    # sur son inertie, l'amortissement nival sur le couplage a l'air.
    # UNE SEULE sortie, pas deux (correction du 2026-08-27). Rankinen ne depend du sol
    # que par le RAPPORT conductivite / capacite, c'est-a-dire la diffusivite : sa
    # relaxation vaut dt * kt / (ca * (2z)^2) = dt * alpha / (2z)^2. Exposer les deux
    # grandeurs creait donc une REDONDANCE, et le champ l'a exploitee -- apres deux
    # epochs, conductivite et capacite apprises etaient anti-correlees a -0.920 d'un
    # troncon a l'autre, ce qui est thermodynamiquement impossible (les deux croissent
    # avec la teneur en eau). Le modele ne trichait pas : il tirait sur une
    # parametrisation mal posee. Un parametre qui dit ce qu'il fait vaut mieux que deux
    # qui se compensent.
    diff_gel: Tensor        # diffusivite thermique APPARENTE (m2/s) [4e-8, 2.2e-7]
    fs_neige: Tensor        # amortissement par le couvert nival (1/m) [0.5, 6.0]
    # ── Retard de fonte par la canopee (2026-08-28, R56) ──────────────────────
    # Hydrotel coordonne son freshet avec un SEUIL DE FONTE PAR CLASSE d'occupation,
    # cale contre le debit : +3.35 degres sous conifere sur sagu/outv/abit, ce qui
    # n'autorise la fonte que 0.7 % des jours de decembre a mars contre 4.1 % au seuil
    # physique. C'est un verrou, pas une loi de fonte. Il encode un processus REEL
    # (contenu de froid + ombrage de canopee : un manteau sous resineux a +2 degres
    # d'air n'a pas l'energie de fondre, cf. Cryosphere 15:5371 au BEREV), mais comme
    # une constante calibree contre l'hydrogramme, donc sans garantie hors calage.
    # On le remplace ici par un champ APPRIS, rendu identifiable par sa structure
    # plutot que par un ancrage : le seuil de reference est celui du terrain DECOUVERT
    # (T_melt), et chaque strate de couvert ne peut que le RETARDER. D'ou deux offsets
    # strictement non negatifs, empiles :
    #     seuil_decouvert = T_melt
    #     seuil_feuillu   = T_melt + dT_canopee_feu
    #     seuil_conifere  = T_melt + dT_canopee_feu + dT_canopee_conif
    # L'ordre conifere >= feuillu >= decouvert est donc vrai PAR CONSTRUCTION, et c'est
    # ce qui empeche les trois seuils de se compenser librement. Les deux calages
    # d'Hydrotel respectent cet ordre, mais s'accordent mal sur l'amplitude (offsets de
    # 2.95 degres sur la famille sagu contre 0.35 sur la famille gasp) : c'est
    # precisement pourquoi ils entrent comme PRIOR FAIBLE et non comme ancrage.
    dT_canopee_feu: Tensor    # retard de fonte sous feuillu, vs decouvert (C) [0, 3]
    dT_canopee_conif: Tensor  # retard SUPPLEMENTAIRE sous conifere (C) [0, 3]

    N_PARAMS: ClassVar[int] = 42

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
            "K_musk_hours": _KMUSK_INIT, "x_musk": 0.20,
            # ETP scaling — défaut 1.0 (FAO-56 reference comme Hydrotel PM).
            "K_c": 1.0,
            # Sub-daily storm duration — 12h par défaut (vs 6h hardcodé avant).
            # Plus réaliste pour pluies frontales QC (vs orages convectifs courts).
            "rain_hours": 12.0,
            # Soil layer thicknesses — Hydrotel BV3C standard
            "Z2": 0.70,
            "Z3": 1.00,
            "vsa_b": 2.5,
            "krec": KREC_REF,
            # PROPRIETES THERMIQUES : valeurs du C++ (rankinen.cpp), pour que le champ
            # DEMARRE sur le clone fidele. Sans cela il partirait au MILIEU des bornes
            # (1.35, 1.75e6, 3.25), donc sur une autre physique de gel, et l'ecart
            # mesure ne serait plus attribuable au degre de liberte ajoute. Erreur
            # commise puis corrigee le 2026-08-27.
            "diff_gel": 1.6e-7,     # = 0.8 / (1e6 + 4e6), la valeur du C++
            "fs_neige": 2.35,
            # Retards de canopee (R56). Prior FAIBLE et volontairement median : les
            # deux familles de calage d'Hydrotel donnent 2.95/2.95 (sagu, outv, abit)
            # et 0.35/0.34 (gasp, mont, slso) pour le meme processus physique. Un
            # ecart d'un facteur huit sur une constante calee contre le debit dit que
            # ce n'est pas une mesure ; on part au milieu et on laisse la contrainte
            # d'ordre plus les observables faire le travail.
            "dT_canopee_feu": 1.0,
            "dT_canopee_conif": 1.0,
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
        # T_melt: bounded [-2, 3] (aligné sur la transform ; seuil de fonte NeRF)
        raw[i] = inv_bounded(d["T_melt"], -2.0, 3.0); i += 1
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
        raw[i] = inv_bounded(d["K_musk_hours"], _KMUSK_MIN, _KMUSK_MAX); i += 1
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
        # krec: exp(clamp(raw*0.3 + log(KREC_REF)))
        raw[i] = (math.log(d["krec"]) - math.log(KREC_REF)) / 0.3; i += 1
        # proprietes thermiques du gel : bornes de la litterature des sols, valeur de
        # depart = celle du C++ (voir le dictionnaire de cibles).
        raw[i] = inv_bounded(d["diff_gel"], 4e-8, 2.2e-7); i += 1
        raw[i] = inv_bounded(d["fs_neige"], 0.5, 6.0); i += 1
        raw[i] = inv_bounded(d["dT_canopee_feu"], 0.0, 3.0); i += 1
        raw[i] = inv_bounded(d["dT_canopee_conif"], 0.0, 3.0); i += 1

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
                # CONJOINT multi-régions : latent_codes est dimensionné sur le
                # TOTAL des nœuds ; latent_offset (défaut 0) sélectionne la
                # tranche de la région courante.
                off = int(getattr(self, "latent_offset", 0))
                raw = raw + self.latent_codes[off:off + raw.shape[0]]

        return self._apply_constraints(raw)

    def latent_reg(self) -> Tensor:
        """Pénalité de shrinkage L2 des codes latents (partial pooling).

        Tire les z_n vers 0 : chaque bassin ne dévie du feature-mean que si ses
        données le justifient. Zéro si les codes sont désactivés.
        """
        if self.use_latent_codes:
            return self.latent_codes.pow(2).mean()
        return torch.zeros((), device=self.fc_out.weight.device)

    # REPERE FIXE, ET C'EST UN CORRECTIF MAJEUR (2026-08-30).
    # La version precedente calculait le centre ET l'echelle SUR LE LOT PASSE :
    #     lat0 = lat.mean() ; scale = max(|x|.max(), |y|.max())
    # Le champ n'etait donc pas une fonction de la POSITION mais de la position
    # RELATIVE A LA BOITE ENGLOBANTE DU DOMAINE CHARGE. Mesure sur les 17 708 noeuds
    # communs entre un domaine de 6 plateformes et un de 14 : l'echelle passe de 572 a
    # 1111 km et les coordonnees projetees d'un MEME troncon se deplacent de 0.195 en
    # mediane, 0.62 au maximum, sur une plage totale de -1 a 1. Le meme fichier de poids
    # rendait donc 0.4513 de KGE median sur 14 plateformes et 0.7059 sur 6, aux MEMES
    # stations : ce n'etait pas une regression, c'etaient d'autres entrees.
    #
    # Trois consequences, toutes levees par le repere fixe. Une comparaison entre deux
    # ensembles de plateformes redevient valide. Un modele entraine peut etre applique a
    # un nouveau territoire sans que ses parametres se deplacent, ce que le claim de
    # regionalisation exige. Et la carte du champ cesse de dependre de ce qu'on charge.
    #
    # Constantes : centre (-72, 52) et rayon 1200 km, mesures sur l'emprise reelle des
    # quatorze plateformes (lon -80.4 a -55.8, lat 43.3 a 53.1 ; rayon max 1109 km).
    # Elles sont FIXES a dessein -- les recalculer sur les donnees reintroduirait la
    # dependance qu'on vient de retirer.
    _LON0 = -72.0
    _LAT0 = 52.0
    _RAYON_KM = 1200.0

    def _project_coords(self, coords: Tensor) -> Tensor:
        """Projette (lon, lat) en degres vers des coordonnees ISOTROPES, repere FIXE.

        Les degres lon/lat ne sont pas isotropes : a la latitude phi, 1 degre de
        longitude vaut cos(phi)*111 km contre 111 km pour 1 degre de latitude. Traiter
        (lon, lat) comme cartesien distord l'encodage de Fourier. On applique donc une
        projection equirectangulaire vers des kilometres, puis on divise par un rayon
        CONSTANT. Le resultat ne depend que du point, jamais du lot.
        """
        import math
        lon, lat = coords[:, 0], coords[:, 1]
        x = (lon - self._LON0) * math.cos(math.radians(self._LAT0)) * 111.32
        y = (lat - self._LAT0) * 110.574
        return torch.stack([x / self._RAYON_KM, y / self._RAYON_KM], dim=-1)

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

    def set_lake_anchor(self, area_lac_km2, a_ref_km2: float = 20.0, k0: float = 1e-4,
                        alpha: float = 1.0):
        """Ancre k_lake sur la loi d'exutoire : k0 * (a_ref / A)^alpha, jamais au-dessus
        de k0 (la reponse SATURE vers le haut : k x10 et k x100 donnent le meme KGE, seule
        la reduction des grands lacs porte de l'information). area_lac_km2 : (n_nodes,).
        Poser None retire l'ancrage."""
        if area_lac_km2 is None:
            self._lake_k_anchor = None
            return
        import torch as _t
        A = _t.clamp(_t.as_tensor(area_lac_km2, dtype=_t.float32), min=1e-3)
        self._lake_k_anchor = k0 * _t.clamp((a_ref_km2 / A) ** alpha, max=1.0)

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
        # k_lake : log-uniforme centré sur l'ANCRE ; raw=0 → ancre. Par défaut 1e-4
        # (littérature). Avec set_lake_anchor(), l'ancre devient k0*(A_ref/A) : la loi du
        # seuil Q = C*L*h^1.5, égalée à la forme implémentée Q = k*(S/A)^beta*A avec
        # beta=1.5, donne k = C*L/A, et la mesure du 5 août désigne l'exposant 1, soit une
        # largeur d'exutoire fixée par le chenal de sortie et non par l'étendue du lac.
        # Motif : laissée LIBRE, la tête apprend sur 2000-2018 une direction qui ne
        # transfère pas (+0.002 hors échantillon sur OUTV) alors que la même contrainte
        # imposée en inférence rapporte +0.026. On ancre donc, et la tête module autour.
        _anc = getattr(self, "_lake_k_anchor", None)
        log_anc = math.log(1e-4) if _anc is None else torch.log(_anc.to(raw.device))
        log_k = torch.clamp(raw[:, 0] * 0.5 + log_anc, min=math.log(1e-6), max=math.log(1e-2))
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
        # T_melt: [-2, 3] C — élargi 2026-07-25 : borné [-1,1] il ne pouvait PAS
        # atteindre les seuils de fonte calés (+1.6..+2.3°C, valeur mesurée +0.15 KGE
        # sur GASP) maintenant qu'il pilote le seuil du module neige.
        constrained.append(bounded(cols[i], -2.0, 3.0)); i += 1
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
        # K_musk_hours: temps de transfert Muskingum, bornes CONFIGURABLES.
        # MESURE 2026-08-09 (banc de modules, Manning sur la géométrie réelle du trl) :
        # le temps de parcours PHYSIQUE des tronçons vaut ~0.2-0.35 h (longueur médiane
        # 3.6-4.3 km, vitesse ~2 m/s) et 100 % des tronçons sont sous l'ancienne borne
        # basse de 4 h. Le K appris (23.7 h sur le champion gasp) valait donc 60-100×
        # le temps de parcours réel : chaque tronçon se comportait en réservoir d'un
        # jour, atténuant un événement court de 27 % et l'étalant sur 4 jours, effet
        # composé le long de la chaîne topologique. L'optimiseur ne pouvait pas le
        # corriger (borne + perte quasi plate en K, mesurée à 4 % de la perte totale).
        # À K physique le même code reproduit la translation d'Hydrotel (pic 10.00
        # contre 10.61 pour le clone de l'onde cinématique).
        # Stabilité : en mode opérateur un petit K donne c2=0, soit translation pure,
        # numériquement sain. En mode message-passing (n_substeps=2) garder K >= 4.
        constrained.append(bounded(cols[i], _KMUSK_MIN, _KMUSK_MAX)); i += 1
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
        # krec: drainage profond L3 -> aquifere (m/h), log-normal CENTRE sur la
        # reference. raw = 0 rend exactement KREC_REF : c'est ce qui rend inoffensif
        # le remplissage par zeros des anciens points de reprise (le padding de fc_out
        # met poids ET biais a zero). Meme construction que k_gw.
        exponent = torch.clamp(cols[i] * 0.3 + math.log(KREC_REF),
                               min=math.log(1e-7), max=math.log(1e-4))
        constrained.append(torch.exp(exponent)); i += 1
        # PROPRIETES THERMIQUES. Bornes tirees de la litterature des sols : la
        # conductivite d'un sol mineral va de ~0.25 (sec, poreux) a ~2.2 W/m/K (sature,
        # sableux) ; la capacite volumique de ~0.8e6 (sec) a ~3e6 J/m3/K (sature, l'eau
        # portant l'essentiel). L'amortissement nival de Rankinen vaut 2.35 dans le
        # C++ ; on ouvre autour, la densite et la structure du couvert le faisant
        # varier. Centrees sur le defaut, raw=0 rend EXACTEMENT l'ancien comportement,
        # ce qui garde inoffensif le remplissage par zeros d'un ancien point de reprise.
        # DIFFUSIVITE APPARENTE. Valeur du C++ : kt / (cs + cice) = 0.8 / 5e6 =
        # 1.6e-7 m2/s. APPARENTE et non vraie : la capacite au denominateur inclut le
        # terme de glace (4e6), qui porte la chaleur latente de changement de phase --
        # methode classique de la capacite apparente. Elle est donc systematiquement plus
        # basse que les diffusivites de manuel (1e-7 a 1e-6) et ne se compare pas
        # directement a elles.
        #
        # BORNE SUPERIEURE FIXEE PAR LA NUMERIQUE, PAS PAR LA PHYSIQUE. Le schema de
        # Rankinen est EXPLICITE : son taux de relaxation vaut dt*alpha/(2z)^2, soit
        # 8.64e6*alpha au noeud le plus superficiel (5 cm, dt journalier). La stabilite
        # exige ce taux sous 2, donc alpha sous 2.31e-7 ; le clone tourne a 1.38, juste
        # sous la limite. Mon premier essai bornait a 8e-7 en empruntant la plage de la
        # litterature : le profil de temperature divergeait et le gel sortait en NaN
        # (2026-08-27). Ce schema ne peut donc PAS representer les diffusivites reelles
        # les plus elevees -- une limite de la numerique, a lever par un schema implicite
        # si le besoin s'en fait sentir.
        constrained.append(bounded(cols[i], 4e-8, 2.2e-7)); i += 1    # diff_gel
        constrained.append(bounded(cols[i], 0.5, 6.0)); i += 1        # fs_neige
        # Retards de fonte par la canopee (R56). Bornes [0, 3] : non negatives parce
        # qu'un couvert ne peut qu'ombrager et couper l'echange turbulent, jamais
        # accelerer la fonte par rapport au terrain decouvert ; plafonnees a 3 parce
        # que c'est deja l'offset du calage le plus extreme d'Hydrotel (+2.95 sur la
        # famille sagu) et qu'au-dela le manteau ne fondrait plus du tout en avril.
        constrained.append(bounded(cols[i], 0.0, 3.0)); i += 1        # dT_canopee_feu
        constrained.append(bounded(cols[i], 0.0, 3.0)); i += 1        # dT_canopee_conif

        return SpatialParams.from_tensor(torch.stack(constrained, dim=-1))


    # ── Recharge : poser ou geler le champ ────────────────────────────────────
    # krec n'est plus la derniere sortie depuis l'ajout des proprietes thermiques
    # (2026-08-27) : l'index est desormais nomme explicitement plutot que deduit d'une
    # position, ce qui evite qu'un ajout futur ne deplace silencieusement le gel de la
    # recharge vers autre chose.
    _IDX_KREC = [f.name for f in _dc_fields(SpatialParams)].index("krec")

    def set_uniform_krec(self, valeur: float) -> None:
        """Force la recharge a une valeur UNIFORME (m/h) sur tous les noeuds.

        Mode DEGRADE, a n'utiliser que pour les bancs : il annule la variation
        spatiale, qui est justement la raison d'etre du champ. Concretement on met a
        zero les poids de la sortie krec et on porte la valeur dans son biais.
        """
        import math as _m
        i = self._IDX_KREC
        with torch.no_grad():
            self.fc_out.weight[i].zero_()
            self.fc_out.bias[i] = (_m.log(valeur) - _m.log(KREC_REF)) / 0.3

    def freeze_krec(self) -> None:
        """Exclut la recharge de l'apprentissage, sans geler le reste du champ.

        fc_out est UN seul parametre : on ne peut pas y mettre requires_grad par
        ligne. On annule donc le gradient de la ligne krec par un crochet. Motif : la
        recharge est un LIVRABLE, et le debit seul la pousse aux extremes (banc du
        2026-08-19 : a 5e-5 la nappe fournit 69 % du debit et le KGE tombe a 0.589).
        Quand elle est posee pour des raisons PHYSIQUES, elle doit le rester.
        """
        i = self._IDX_KREC

        def _coupe(g):
            g = g.clone()
            g[i] = 0.0
            return g

        self.fc_out.weight.register_hook(_coupe)
        self.fc_out.bias.register_hook(_coupe)

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
        # Retards de canopee : prior FAIBLE (poids 0.1 contre 0.5 pour T_melt), parce
        # que les deux calages d'Hydrotel s'accordent sur l'ORDRE mais pas du tout sur
        # l'amplitude. L'ordre est deja garanti par la construction (offsets bornes
        # positifs, empiles) ; ce terme ne sert qu'a eviter la derive vers les bornes
        # en debut d'entrainement, pas a imposer une valeur.
        loss = loss + ((params.dT_canopee_feu.mean() - tg("dT_canopee_feu", 1.0)) ** 2) * 0.1
        loss = loss + ((params.dT_canopee_conif.mean() - tg("dT_canopee_conif", 1.0)) ** 2) * 0.1

        # frost_alpha
        loss = loss + ((params.frost_alpha.mean() - tg("frost_alpha", 0.5)) ** 2) * 0.3

        # alpha_T
        loss = loss + ((params.alpha_T.mean() - tg("alpha_T", 0.03)) ** 2)

        # vg_n
        loss = loss + ((params.vg_n.mean() - tg("vg_n", 1.5)) ** 2) * 0.3

        # k_gw récession (log-espace)
        loss = loss + ((torch.log(params.k_gw + 1e-8).mean() - math.log(tg("k_gw", 0.02))) ** 2) * 0.3

        # krec, drainage profond de la couche 3 (log-espace, comme k_gw).
        #
        # OPT-IN (`prior_on_krec`), et c'est délibéré : quand krec est imposé par le
        # calage Hydrotel, la sortie du NeRF n'est PAS utilisée (le calage est fusionné
        # par-dessus dans la colonne), donc l'ancrer tirerait un paramètre mort et
        # changerait la perte sans changer la physique. Le drapeau n'est levé que quand
        # krec est réellement libre.
        #
        # POURQUOI IL EN FAUT UN (Essi, 2026-08-22 : « krec devrait être dans le nerf »).
        # krec EST une sortie du champ, mais rien ne l'ancrait : le prior tient k_gw et
        # pas lui. Libre, il n'a donc que le débit pour juge, et le débit seul le pousse
        # à zéro (R11) -- d'où le réflexe de le geler, qui est le mauvais remède : geler
        # une propriété du sous-sol la rend uniforme sur toute une région, interdit de
        # suivre la géologie et de répondre à un changement d'occupation du territoire.
        # Ancrer la MOYENNE en laissant la variation spatiale libre est le même
        # traitement que k_gw, et c'est ce que la forme (p.mean() - c)^2 permet.
        #
        # LA CIBLE. À saturation q3 = krec x z3 x theta ; avec z3 = 2.65 m et theta = 0.40,
        # KREC_REF = 2e-5 m/h donne ~0.51 mm/j, soit ~34 % d'un écoulement de 549 mm/an.
        # L'indice de débit de base des bassins boréaux québécois se situe entre 0.4 et
        # 0.6, donc la cible est du bon ordre, légèrement conservatrice. À comparer au
        # calage Hydrotel, 1.3e-7, qui donne 0.0036 mm/j : un robinet fermé.
        if getattr(self, "prior_on_krec", False) and hasattr(params, "krec"):
            loss = loss + ((torch.log(params.krec + 1e-12).mean()
                            - math.log(tg("krec", KREC_REF))) ** 2) * 0.3

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


    def fit_to_field(self, node_coords, territorial_data, cibles: dict,
                          n_iter: int = 3000, lr: float = 3e-3, log_keys=("K_sat_1", "K_sat_2",
                          "K_sat_3", "k_gw"), verbeux: bool = True):
        """Ajuste le champ par RÉGRESSION sur des valeurs cibles PAR NŒUD.

        Proposition d'Essi (2026-08-13) : « pourquoi ne pas démarrer le NeRF sur le champ
        Hydrotel puis optimiser ? ». C'est le seul départ qui donne à la fois le NIVEAU et
        le CONTRASTE tout en laissant le champ LIBRE ensuite :
          - `init_from_literature` pose un champ constant (dispersion mesurée 0.0017) ;
          - `set_calibrated_soil` COURT-CIRCUITE la sortie du réseau, qui n'apprend alors
            plus rien sur le sol ;
          - le motif de pédotransfert donne le contraste sans le niveau.
        Mesuré le 2026-08-13 sur OUTV : le champ appris reste collé à son initialisation
        (K_sat 0.0373 pour un prior à 0.04) alors que la valeur calibrée vaut 0.317, et en
        30 époques le gradient du débit ne porte la dispersion que de 0.0017 à 0.054 quand
        il en faudrait 0.74. Le point de départ est donc le verrou, pas la capacité.

        cibles : {nom_du_champ: tenseur (n_nodes,)} — les noms sont ceux de SpatialParams.
        log_keys : champs ajustés en log (grandeurs positives couvrant des ordres de
        grandeur) ; les autres en linéaire.
        """
        import torch as _t
        cibles = {k: _t.as_tensor(v, dtype=_t.float32).to(node_coords.device).detach()
                  for k, v in cibles.items()}
        opt = _t.optim.Adam(self.parameters(), lr=lr)
        n0 = None
        for it in range(n_iter):
            opt.zero_grad()
            sp = self(node_coords, territorial_data)
            perte = 0.0
            for k, cible in cibles.items():
                pred = getattr(sp, k)
                if k in log_keys:
                    perte = perte + ((_t.log(pred.clamp(min=1e-9))
                                      - _t.log(cible.clamp(min=1e-9))) ** 2).mean()
                else:
                    ech = cible.abs().mean().clamp(min=1e-6)
                    perte = perte + (((pred - cible) / ech) ** 2).mean()
            perte.backward()
            opt.step()
            if n0 is None:
                n0 = float(perte)
            if verbeux and (it % max(n_iter // 4, 1) == 0 or it == n_iter - 1):
                print(f"    [ajust] iter {it:5d} | perte {float(perte):.5f}")
        if verbeux:
            with _t.no_grad():
                sp = self(node_coords, territorial_data)
            print(f"    [ajust] perte {n0:.5f} -> {float(perte):.5f}")
            for k, cible in cibles.items():
                p = getattr(sp, k).detach()
                cv_p = float(p.std() / p.abs().mean().clamp(min=1e-12))
                cv_c = float(cible.std() / cible.abs().mean().clamp(min=1e-12))
                print(f"    [ajust] {k:14s} médiane {float(p.median()):.4f} "
                      f"(cible {float(cible.median()):.4f}) | dispersion {cv_p:.3f} "
                      f"(cible {cv_c:.3f})")
        return self


# Alias de compatibilite (nettoyage 2026-08-24, convention anglaise pour le code).
# L'identifiant francais reste appelable ; les scripts .runs historiques y tiennent.
SpatialFieldNetwork.ajuster_sur_champ = SpatialFieldNetwork.fit_to_field
