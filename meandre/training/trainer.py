"""Training loop with RunLogger logging.

Le curriculum a quatre phases (encodeur temporel, correcteur residuel, attention de temps
de transfert) a ete retire le 2026-10-23 avec ces modules.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW

from meandre.routing.graph import RiverGraph
from meandre.utils import settings
from meandre.routing.withdrawals import WithdrawalData
from meandre.spatial.territorial import TerritorialFeatures
from meandre.training.loss import CompositeKGELoss, HydroLoss
from meandre.utils.metrics import kge_components, log_nse, mae, nse, nrmse, pbias, rmse
from meandre.utils.state import HydroState

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Hyperparameters of the training loop."""
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    n_epochs: int = 100
    batch_size: int = 1           # typical for physics simulations

    # Soft prior regularization weight on spatial parameters.
    # Penalizes deviation from literature targets (k_gw, krec, K_sat, K_c, etc.).
    # 0 = no regularization (params can drift freely).
    # 0.001-0.01 = soft pull, allows learning from data but prevents extreme drift.
    # Useful to avoid overfitting on dev period (e.g. wet/dry year bias).
    w_prior: float = 0.0

    # Anti-collapse : plancher de variance spatiale sur les params NeRF clés.
    # Combat l'effondrement vers des params uniformes (diagnostic open-data
    # 2026-06-12). Plancher relu → ne récompense pas l'excès de variance.
    w_diversity: float = 0.0
    diversity_cv_target: float = 0.12
    w_latent_reg: float = 1e-3
    # LR dédié élevé pour les codes latents (cold-start auto-décodeur : le NeRF
    # partagé soak le signal, les codes restent à 0 sans LR plus fort).
    latent_lr_mult: float = 50.0
    # LR dédié pour la tête de lac : poids initialisés à zéro, sortie divisée par 2,
    # bornes sur 4 ordres de grandeur — au LR de base elle ne quitte jamais son
    # initialisation (constat sur les 6 régions entraînées, 2026-08-04).
    lake_lr_mult: float = 50.0

    # Boundary regularization on raw network outputs.
    # Penalizes sigmoid saturation (params at clamp bounds) and L2-pulls
    # exp-constrained raw outputs toward center. Complements w_prior
    # (which targets specific literature values in physical units).
    # 0 = off. 0.001-0.01 = soft anti-saturation prior. Only active when
    # spatial encoder is unfrozen.
    w_boundary: float = 0.0

    # LR scheduler shape. Cosine annealing from `lr` to `lr * eta_min_factor`
    # over n_epochs. Set to 1.0 for CONSTANT lr (cosine becomes flat) —
    # useful for short tuning runs where the default 0.01 collapses lr by
    # epoch 2-3 and stalls learning.
    eta_min_factor: float = 0.01

    # Spin-up steps (run model without contributing to loss; state initialisation)
    spinup_steps: int = 730       # 2 years of daily data

    # Truncated BPTT: detach state every N steps during training forward pass.
    # Limits backward-pass depth to tbptt_steps instead of the full sequence.
    # 0 = no truncation (very slow for long sequences).
    tbptt_steps: int = 90         # one season = good balance of speed vs. gradient depth

    # Neige : SWE (mm) au-dessus duquel la couverture est ~saturée, pour mapper
    # SWE simulé → fraction de couverture (SCF = 1-exp(-SWE/ref)) comparée à MODIS
    # snow cover. ~15 mm = la neige couvre le pixel dès une faible accumulation.
    snow_swe_ref: float = 15.0

    # Incertitude de la CLIMATOLOGIE mensuelle GRACE, pour w_tws_clim (R23).
    # 25 mm (un mois) / sqrt(~21 ans d'observations par mois) = 5.4 mm.
    tws_clim_sigma: float = 5.4
    # Borne de Huber sur les z-scores des contraintes auxiliaires. 0 = desactive
    # (comportement fidele aux runs anterieurs). 3.0 recommande : au-dela de trois
    # ecarts-types le gradient cesse de croitre, ce qui empeche une contrainte de
    # prendre le controle de l'optimisation quand son residu est structurellement
    # grand -- cas mesure le 2026-08-26 (GRACE a 966 % du total).
    aux_huber_delta: float = 0.0
    # GRACE compare en FORME (correlation de l'anomalie normalisee) plutot qu'en
    # millimetres absolus. Meme discipline que MODIS ET (R24). A activer quand le
    # diagnostic de stockage montre un rapport d'amplitude eloigne de 1.
    tws_shape_only: bool = False

    # Warm-start spinup: after epoch 0, run only this many steps from the
    # cached spinup state instead of re-running all spinup_steps from zeros.
    # 0 = always run full spinup (safe but slow).
    warm_spinup_steps: int = 90

    # Chunked gradient accumulation: split training sequence into chunks of
    # this many steps, compute loss+backward per chunk, then optimizer.step().
    # Keeps peak GPU memory proportional to chunk_steps, not the full sequence.
    # 0 = no chunking (entire sequence in one pass).
    chunk_steps: int = 365

    # Validation frequency: run val epoch every N training epochs.
    # 1 = every epoch (default); 5 = every 5 epochs.
    val_every: int = 1

    # Early stopping: stop training if val metric hasn't improved for this many
    # epochs. 0 = disabled.
    patience: int = 0

    # Metric used for best-checkpoint selection and early stopping.
    # - validation metric keys ("nse", "kge", "kge_station", "kge_median"):
    #   higher is better, no_improve counted on bare comparisons.
    # - "loss" (train loss): lower is better, requires
    #   ``best_metric_tolerance`` since train_loss is monotone-decreasing.
    #   With tolerance, no_improve counts epochs where the relative
    #   improvement is less than `best_metric_tolerance` (0.5% by default).
    best_metric: str = "nse"

    # Relative improvement threshold for ``best_metric`` updates. Only used
    # when best_metric is loss-like (lower-is-better and monotone). For
    # higher-is-better metrics like KGE that oscillate, set to 0.0.
    best_metric_tolerance: float = 0.005

    # Number of warmup epochs for the LR scheduler.  Set to 0 for warm-start
    # fine-tuning where pre-trained weights don't need a ramp-up from ~zero LR.
    warmup_epochs: int = 5

    # Multiplier for spatial-encoder fc1/fc2 learning rate when fine-tuning
    # with new territorial features (discriminative LR).  None = same LR for all.
    lr_new_features_mult: float | None = None

    # Compile hot sub-modules (VerticalColumn + RoutingLayer) with torch.compile.
    # Fuses per-timestep ops into fewer CUDA kernels — big win on GPU, no-op on CPU.
    # Disabled by default; set True once GPU install is confirmed working.
    compile_modules: bool = False

    # ── Autopilot ──────────────────────────────────────────────────────────
    # When enabled, the trainer automatically adjusts hyperparameters based on
    # validation metrics.  Designed for unsupervised long runs (e.g. all Québec).
    # All thresholds are conservative — they only act on clear signals.

    # Enable autopilot?  When True, forces val_every=1 for fast reaction.
    autopilot: bool = False

    # Grace period: l'autopilot n'intervient PAS pendant les N premières
    # epochs.  Lors d'un fresh start (warm_start=false), la convergence
    # initiale produit des drifts beta/gamma temporaires qui ne sont PAS
    # de l'overfitting — c'est la physique qui cherche ses paramètres.
    # Intervenir trop tôt (réduction du LR, redémarrage) freine la convergence.
    autopilot_grace_epochs: int = 0  # 0 = pas de grace, 10-15 pour fresh start

    # Dérive de β et de γ : seuils au-delà desquels le redémarrage intelligent
    # (plus bas) considère qu'il y a dérive. La pénalité qui augmentait w_residual
    # sur dérive est retirée avec le correcteur résiduel (2026-10-23).
    autopilot_beta_threshold: float = 0.15   # |β - 1| > 0.15 → dérive
    autopilot_gamma_threshold: float = 0.20  # |γ - 1| > 0.20 → dérive

    # ReduceLROnPlateau: if best val metric hasn't improved for this many
    # epochs, multiply LR by autopilot_lr_factor.
    autopilot_lr_patience: int = 8
    autopilot_lr_factor: float = 0.5
    autopilot_lr_min: float = 1e-6

    # Smart restart: if val metric regresses by more than this fraction from
    # best AND beta/gamma drift is detected AND the regression persists for
    # at least ``autopilot_restart_min_no_improve`` epochs, reload best
    # checkpoint and reduce LR.  The min_no_improve guard prevents firing
    # on a single transient oscillation during cold-start, which would
    # otherwise consume the restart budget before the model has had a
    # chance to converge.
    autopilot_restart_regression: float = 0.05  # 5% regression from best
    autopilot_restart_max: int = 3               # max restarts
    autopilot_restart_min_no_improve: int = 3    # consecutive bad epochs required

    # Phase-1 → phase-2 transition (Kendall-Gal recipe).  When the spatial
    # encoder is frozen at startup (``freeze_spatial=true``), the autopilot
    # can flip it to trainable mid-run once the NLL stack has stabilised.
    # Trigger conditions are AND-combined: both ``epoch >= unfreeze_epoch``
    # AND (if set) ``kge_station >= unfreeze_min_kge`` must hold.
    # The transition divides the base LR by ``unfreeze_lr_factor`` (default
    # 20× lower) to mimic the manual two-config recipe and avoid disturbing
    # the converged σ/μ ratio.  One-shot.
    autopilot_unfreeze_spatial_epoch: int | None = None     # e.g. 25
    autopilot_unfreeze_spatial_min_kge: float | None = None # e.g. 0.40
    autopilot_unfreeze_spatial_lr_factor: float = 0.05      # lr → lr × 0.05

@dataclass
class TrainingData:
    """Everything the trainer needs for one simulation run.

    Attributes
    ----------
    forcing:        (n_timesteps, n_nodes, n_forcing)
    q_obs:          (n_timesteps, n_stations) — observed streamflow; NaN = missing
    station_mask:   (n_nodes,) bool — which nodes have observations
    station_idx:    (n_stations,) int — node indices for gauging stations
    graph:          River network structure
    node_coords:    (n_nodes, 2) lon/lat
    territorial:    Static territorial indicators
    withdrawals:    Withdrawal/rejection tensors
    day_of_year:    (n_timesteps,) integer 1-366
    train_slice:    slice — timesteps used for loss (after spinup)
    val_slice:      slice — held-out timesteps for validation
    """

    forcing: Tensor
    q_obs: Tensor
    station_mask: Tensor
    station_idx: Tensor
    graph: RiverGraph
    node_coords: Tensor
    territorial: TerritorialFeatures
    withdrawals: WithdrawalData
    day_of_year: Tensor
    train_slice: slice
    val_slice: slice
    # Multi-objective observations (optional, None when not loaded).
    # Shapes: (n_timesteps, n_nodes) ; NaN = missing (most cells are NaN
    # because MODIS revisit is irregular).
    et_obs: Tensor | None = None    # MODIS MOD16A2 ETR (mm/jour, 8-day agrégé en daily)
    swe_obs: Tensor | None = None   # SWE de MODIS NDSI ou SNODAS (mm)
    tws_obs: Tensor | None = None   # GRACE TWS anomalie (mm), valeur mensuelle au 15, NaN ailleurs
    # POIDS DU TERME MOD16 HORS NEIGE (2026-10-03), aligne sur l'axe du FORCAGE (index 0 =
    # forcing[0], mise en regime comprise), forme (T, n_nodes), sans dimension, de 0 a 1.
    # MOD16 est peu fiable sous la neige ; en mode centre, l'ecart de printemps (MOD16 0,96
    # et 1,39 mm/j en mars et avril, modele 0,16 et 0,51 sur 052805) tirait K_c vers le bas.
    # Typiquement w = exp(-SWE_NEISIM / tau). None = poids uniforme.
    et_weight: Tensor | None = None
    # GRACE est un observable de BASSIN, pas de tronçon : sa moyenne de stockage doit se
    # prendre sur le masque du bassin qu'elle mesure. Tant qu'un entraînement portait sur
    # un seul bassin, la moyenne sur tous les nœuds était ce masque et la question ne se
    # posait pas. Sur le domaine provincial fondu (2026-08-25), elle mélangerait quatorze
    # bassins pour les confronter à une seule série satellitaire. `tws_group` donne le
    # bassin de chaque nœud et `tws_obs` devient (T, n_bassins).
    tws_group: Tensor | None = None   # (n_nodes,) indice de bassin, None = un seul bassin
    # NIVEAUX DE NAPPE MESURES (2026-09-19). Seule observation directe et non circulaire de
    # l'eau souterraine : les cartes de recharge regionales sont calees sur le debit de base
    # et le tableau des lignes rouges les ecarte. Profondeur sous le repere du tubage, en
    # metres, par PUITS et non par noeud, NaN quand la mesure manque. Puits captifs et puits
    # influences deja retires par le chargeur.
    nappe_obs: Tensor | None = None   # (n_timesteps, n_puits)
    nappe_idx: Tensor | None = None   # (n_puits,) indice du noeud portant chaque puits
    # CanSWE : MASSE du manteau mesurée au sol, par SITE et non par nœud (R24). Plusieurs
    # sites peuvent viser le même tronçon et les agréger détruirait la dispersion
    # intra-nœud, qui EST la mesure de l'incertitude de représentativité ponctuelle.
    # Complémentaire de swe_obs (couverture MODIS) : la fraction sature dès qu'il y a un
    # peu de neige et sous-estime sous couvert forestier ; la masse, non.
    swe_mass_obs: Tensor | None = None    # (T, n_sites) mm, NaN hors relevé
    swe_mass_node: Tensor | None = None   # (n_sites,) nœud visé par chaque site

    def __post_init__(self) -> None:
        # CONVENTION D'ALIGNEMENT, payee trois fois en deux jours (R75 bis) : q_obs[0]
        # correspond a forcing[train_slice.start], pas a forcing[0]. Le forcage porte la
        # mise en regime en amont ; les observations commencent au premier jour juge.
        # Un banc qui passe q_obs complet compare des annees decalees et rend un score
        # faux sans aucune erreur d'execution. On avertit ; on n'interdit pas, pour ne
        # casser aucun pilote existant.
        try:
            n_f, n_o = int(self.forcing.shape[0]), int(self.q_obs.shape[0])
            attendu = n_f - int(self.train_slice.start)
            if n_o != attendu:
                print(f"[TrainingData] AVERTISSEMENT alignement : q_obs a {n_o} pas de temps, "
                      f"forcing en a {n_f} et train_slice commence a {self.train_slice.start} ; "
                      f"la convention veut q_obs[0] <-> forcing[train_slice.start], soit "
                      f"{attendu} pas. Score FAUX si ce n'est pas volontaire.")
        except Exception:
            pass


class Trainer:
    """Training loop with RunLogger logging.

    Usage
    -----
    data = TrainingData(...)
    trainer = Trainer(model, loss_fn, train_data=data, config=cfg)
    trainer.fit()
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: HydroLoss | CompositeKGELoss,
        train_data: TrainingData,
        val_data: TrainingData | None = None,
        config: TrainingConfig | None = None,
        run_name: str | None = None,
        run_logger: "RunLogger | None" = None,
        checkpoint_path: str | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        # backward-compat alias (ignored — use run_name instead)
        mlflow_run_name: str | None = None,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.train_data = train_data
        self.val_data = val_data if val_data is not None else train_data
        self._check_auxiliary_targets(self.train_data)
        self.config = config or TrainingConfig()
        self.run_name = run_name or mlflow_run_name or ""
        self.run_logger = run_logger
        self.checkpoint_path = checkpoint_path

        if optimizer is not None:
            self.optimizer = optimizer
        else:
            # Discriminative LR:
            #   - higher rate for fc1/fc2 when new features added (padded)
            #   - higher rate + no weight_decay for fc_out.weight (escape NeRF
            #     collapse: init_from_literature shrinks this layer 100×, so
            #     without a boost it stays stuck near zero → uniform params).
            padded = getattr(model, "_padded_layers", set())
            mult = self.config.lr_new_features_mult
            # MEANDRE_LR_MULT_TETES : multiplicateur commun des tetes a taux accru (sortie du
            # champ, 10 par defaut ; tete de lac, lake_lr_mult, 50 par defaut). Ces
            # multiplicateurs ont ete regles pour UN pas d'Adam par epoque (registre,
            # 2026-09-04) ; avec un pas par bloc, ils sont appliques huit a soixante-cinq
            # fois par epoque. Opt-in, pour mesurer leur part dans l'instabilite du banc.
            _mt = settings.get("trainer.lr_mult_heads")
            fc_out_mult = float(_mt) if _mt is not None else 10.0
            lake_mult = float(_mt) if _mt is not None else self.config.lake_lr_mult

            fc_out_params: list[torch.nn.Parameter] = []
            new_params: list[torch.nn.Parameter] = []
            base_params: list[torch.nn.Parameter] = []
            for name, p in model.named_parameters():
                if name == "spatial_encoder.fc_out.weight":
                    fc_out_params.append(p)
                    continue
                layer = ".".join(name.split(".")[:-1])
                if padded and mult is not None and layer in padded:
                    new_params.append(p)
                else:
                    base_params.append(p)

            groups = [{"params": base_params, "lr": self.config.lr,
                       "weight_decay": self.config.weight_decay}]
            if new_params:
                groups.append({"params": new_params,
                               "lr": self.config.lr * mult,
                               "weight_decay": self.config.weight_decay})
                logger.info(
                    "Discriminative LR: base=%.1e, fc1/fc2=%.1e (%d params)",
                    self.config.lr, self.config.lr * mult,
                    sum(p.numel() for p in new_params),
                )
            if fc_out_params:
                groups.append({"params": fc_out_params,
                               "lr": self.config.lr * fc_out_mult,
                               "weight_decay": 0.0})
                logger.info(
                    "Discriminative LR: fc_out.weight=%.1e (%.0f×), wd=0 — NeRF anti-collapse",
                    self.config.lr * fc_out_mult, fc_out_mult,
                )
            # Tête de lac (k_lake, beta) : oubliée jusqu'ici, seule tête sans
            # traitement particulier. Ses poids partent de ZÉRO exactement, sa sortie
            # est divisée par 2 avant l'exponentielle, et ses bornes couvrent 4 ordres
            # de grandeur (k_lake 1e-6 à 1e-2) : atteindre le haut demanderait une
            # sortie brute de 9 en partant de 0. Diagnostic 2026-08-04 : k_lake vaut
            # 1e-4 et beta 1.5 dans les 6 régions entraînées, soit l'initialisation, avec
            # 1-5 % de dispersion. Or le déficit contre Hydrotel est LACUSTRE (-0.22 de
            # KGE au-dessus de 5 % de nœuds-lacs en amont, contre -0.03 en dessous).
            lake_params = [p for n, p in model.named_parameters() if "fc_lake" in n]
            if lake_params:
                _lk_ids = set(id(p) for p in lake_params)
                base_params[:] = [p for p in base_params if id(p) not in _lk_ids]
                groups.append({"params": lake_params,
                               "lr": self.config.lr * lake_mult,
                               "weight_decay": 0.0})
                logger.info("Discriminative LR: fc_lake=%.1e (%.0f×), wd=0",
                            self.config.lr * lake_mult, lake_mult)
            # Codes latents (effet aléatoire spatial) : LR élevé pour escaper la
            # domination du NeRF partagé au cold-start (auto-décodeur). wd=0 :
            # le shrinkage est déjà géré par w_latent_reg.
            latent_params = [p for n, p in model.named_parameters()
                             if n == "spatial_encoder.latent_codes"]
            if latent_params:
                _lat_ids = set(id(p) for p in latent_params)
                base_params[:] = [p for p in base_params if id(p) not in _lat_ids]
                groups.append({"params": latent_params,
                               "lr": self.config.lr * self.config.latent_lr_mult,
                               "weight_decay": 0.0})
                logger.info("Discriminative LR: latent_codes=%.1e (%.0f×), wd=0",
                            self.config.lr * self.config.latent_lr_mult,
                            self.config.latent_lr_mult)
            # PhenologyModulator (IHI) : GDD seuils ont une échelle naturelle
            # O(100), gradients sigmoïdaux faibles loin du seuil. lr × 100 pour
            # les seuils, lr × 1 pour les amplitudes (K_c_min, K_c_max_factor).
            gdd_threshold_params: list[torch.nn.Parameter] = []
            for name, p in model.named_parameters():
                if "phenology_modulator" in name and ("gdd_emerg" in name or "gdd_mid" in name or "photo_crit" in name):
                    gdd_threshold_params.append(p)
            # Scalaires de la nappe apprise (2026-09-30) : en espace logarithmique, un pas au
            # taux de base ne les bouge pas ; taux x 20, sans decroissance de poids.
            nappe_params = [p for name, p in model.named_parameters() if "nappe_log_kb" in name or "nappe_logit_efrac" in name or "nappe_log_exp1" in name]
            if nappe_params:
                base_params[:] = [p for p in base_params if id(p) not in set(id(g) for g in nappe_params)]
                groups.append({"params": nappe_params, "lr": self.config.lr * 20.0, "weight_decay": 0.0})
                logger.info("Discriminative LR: nappe scalaires=%.1e (20x), wd=0", self.config.lr * 20.0)
            # Parametres appris du profil de sol declare (2026-10-01) : scalaires en espace
            # logarithmique, taux x 50 pour qu'un facteur 3 a 10 soit atteignable en une
            # centaine de pas ; taux nominal pris de MEANDRE_LR_MULT_SOL si pose.
            soil_params = [p for name, p in model.named_parameters() if "soil_learn_" in name]
            if soil_params:
                _ms = float(settings.get("trainer.lr_mult_soil"))
                base_params[:] = [p for p in base_params if id(p) not in set(id(g) for g in soil_params)]
                groups.append({"params": soil_params, "lr": self.config.lr * _ms, "weight_decay": 0.0})
                logger.info("Discriminative LR: profil de sol appris=%.1e (%gx), wd=0", self.config.lr * _ms, _ms)
            # Scalaires des lacs hors reseau (2026-10-02), bornes en douceur : a x50 ils ne
            # bougent que de 10 % en deux epoques ; taux propre, MEANDRE_LR_MULT_LACS.
            # Tete du temps de transfert (2026-10-04) : taux propre, MEANDRE_LR_MULT_ROUTAGE.
            routing_params = [p for name, p in model.named_parameters() if "routing_head" in name]
            if routing_params:
                _mr = float(settings.get("trainer.lr_mult_routing"))
                base_params[:] = [p for p in base_params if id(p) not in set(id(g) for g in routing_params)]
                groups.append({"params": routing_params, "lr": self.config.lr * _mr, "weight_decay": 0.0})
            lakes_params = [p for name, p in model.named_parameters() if "lakes_learn_" in name]

            if lakes_params:
                _ml = float(settings.get("trainer.lr_mult_lakes"))
                base_params[:] = [p for p in base_params if id(p) not in set(id(g) for g in lakes_params)]
                groups.append({"params": lakes_params, "lr": self.config.lr * _ml, "weight_decay": 0.0})
                logger.info("Discriminative LR: lacs hors reseau=%.1e (%gx), wd=0", self.config.lr * _ml, _ml)
            if gdd_threshold_params:
                base_params[:] = [p for p in base_params
                                  if id(p) not in set(id(g) for g in gdd_threshold_params)]
                groups.append({"params": gdd_threshold_params,
                               "lr": self.config.lr * 100.0,
                               "weight_decay": 0.0})
                logger.info(
                    "Discriminative LR: phenology GDD thresholds=%.1e (100×), wd=0 "
                    "— compensate for sigmoid gradient scale O(100)",
                    self.config.lr * 100.0,
                )
            self.optimizer = AdamW(groups)
        # Autopilot: force val_every=1 for fast reaction
        if self.config.autopilot and self.config.val_every > 1:
            logger.info(
                f"[autopilot] val_every forced from {self.config.val_every} to 1 "
                "(autopilot requires per-epoch validation)"
            )
            self.config.val_every = 1

        # Best-metric init depends on direction (lower-is-better for loss-like)
        self._best_metric_lower_is_better = self.config.best_metric in (
            "loss", "val_loss", "nll", "val_nll", "rmse", "nrmse", "mae", "pbias",
        )
        self._best_val_metric = (
            float("inf") if self._best_metric_lower_is_better else -float("inf")
        )

        # Autopilot state
        self._ap_lr_plateau_count: int = 0
        self._ap_restart_count: int = 0
        self._ap_prev_kge_sta: float | None = None

        # Mixed precision (AMP) with bfloat16 — même plage d'exposant que
        # float32 (pas d'overflow dans exp/log de l'hydrologie), mais
        # matmuls plus rapides sur Ampere+ / Ada (Tensor Cores).
        # GradScaler pas nécessaire pour bf16.
        # Activé automatiquement si GPU récent (SM ≥ 8.0) et >= 1000 nœuds.
        self._use_amp = False
        self._amp_dtype = torch.float32
        if torch.cuda.is_available() and next(model.parameters()).is_cuda:
            sm_major, _ = torch.cuda.get_device_capability()
            n_nodes = getattr(model, "n_nodes", 0)
            if sm_major >= 8 and n_nodes >= 1000:
                self._use_amp = True
                self._amp_dtype = torch.bfloat16
                logger.info(
                    f"AMP enabled (bfloat16) — GPU SM {sm_major}.x, "
                    f"{n_nodes} nœuds"
                )
            else:
                logger.info(
                    f"AMP disabled — SM {sm_major}.x, {n_nodes} nœuds "
                    f"(seuils : SM≥8, nœuds≥1000)"
                )
        else:
            logger.info("AMP disabled (pas de GPU)")


        # Maximise intra-op parallelism on CPU
        n_cpu = os.cpu_count() or 1
        torch.set_num_threads(n_cpu)
        try:
            torch.set_num_interop_threads(max(1, n_cpu // 2))
        except RuntimeError:
            pass  # already set or parallel work started

        # torch.compile : désactivé par défaut.  Triton (backend inductor)
        # n'est pas disponible sous Windows.  Le mode "reduce-overhead"
        # (CUDA graphs) est censé contourner le problème mais en pratique
        # PyTorch retombe sur inductor quand le graphe n'est pas capturé
        # intégralement, ce qui crashe au premier forward.
        # Garder compile_modules = false sous Windows.
        if config is not None and config.compile_modules:
            if next(model.parameters()).is_cuda:
                logger.warning(
                    "compile_modules=True but torch.compile requires Triton "
                    "(Linux only).  Skipping — run in eager mode."
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self) -> None:
        """Run training loop for config.n_epochs epochs."""
        from tqdm.auto import tqdm

        self._start_run()

        from meandre.training.scheduler import build_scheduler
        scheduler = build_scheduler(
            self.optimizer, self.config.n_epochs,
            warmup_epochs=self.config.warmup_epochs,
            eta_min_factor=self.config.eta_min_factor,
        )

        pbar = tqdm(range(self.config.n_epochs), desc="Training", unit="epoch")
        last_val_metrics: dict[str, float] = {}
        epochs_without_improvement = 0
        _loss_ema = None          # exponential moving average of train loss
        _rollback_count = 0       # number of LR reductions so far
        _MAX_ROLLBACKS = 3        # stop reducing after this many
        for epoch in pbar:
            self._cur_epoch = epoch   # utilisé par l'offset aléatoire des chunks

            train_loss, train_comps = self._train_epoch()
            self._et_reference_fin_epoque()

            # ── Divergence guard ──────────────────────────────────────
            # Track EMA of train loss; if current loss exceeds 3× EMA,
            # reload best checkpoint and halve LR.
            tl = float(train_loss)
            if _loss_ema is None:
                _loss_ema = tl
            else:
                _loss_ema = 0.8 * _loss_ema + 0.2 * tl
            if tl > 3.0 * _loss_ema and _rollback_count < _MAX_ROLLBACKS and self.checkpoint_path:
                _rollback_count += 1
                self.model.load(self.checkpoint_path)
                for pg in self.optimizer.param_groups:
                    pg["lr"] *= 0.5
                new_lr = self.optimizer.param_groups[0]["lr"]
                msg = (f"  [rollback {_rollback_count}/{_MAX_ROLLBACKS}] "
                       f"train loss spike ({tl:.2f} > 3xEMA {_loss_ema:.2f}) -- "
                       f"reloaded best checkpoint, lr->{new_lr:.2e}")
                print(msg, flush=True)
                logger.warning(msg)
                _loss_ema = None  # reset EMA after rollback
                # Rebuild scheduler with reduced LR
                remaining = self.config.n_epochs - epoch
                scheduler = build_scheduler(
                    self.optimizer, remaining, warmup_epochs=0,
                    eta_min_factor=self.config.eta_min_factor,
                )
                continue

            # Release fragmented CUDA memory between epochs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Validate every val_every epochs (always run on the last epoch)
            val_every = self.config.val_every
            run_val = (val_every <= 1 or epoch % val_every == 0
                       or epoch == self.config.n_epochs - 1)
            if run_val:
                last_val_metrics = self._val_epoch()
                # Water balance diagnostic (every val epoch)
                try:
                    self._water_balance_check(self.train_data)
                except Exception as e:
                    logger.warning(f"Water balance check failed: {e}")
                # Free validation tensors before next training epoch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            scheduler.step()

            val_nse = last_val_metrics.get("nse", float("nan"))
            val_kge = last_val_metrics.get("kge", float("nan"))
            val_rmse = last_val_metrics.get("rmse", float("nan"))
            val_nrmse = last_val_metrics.get("nrmse", float("nan"))
            val_r = last_val_metrics.get("r", float("nan"))
            val_beta = last_val_metrics.get("beta", float("nan"))
            val_gamma = last_val_metrics.get("gamma", float("nan"))
            val_kge_log = last_val_metrics.get("kge_log", float("nan"))
            val_kge_sta = last_val_metrics.get("kge_station", float("nan"))
            val_kge_med = last_val_metrics.get("kge_median", float("nan"))
            val_cov90 = last_val_metrics.get("cov_90", float("nan"))
            val_cov50 = last_val_metrics.get("cov_50", float("nan"))
            pbar.set_postfix(
                loss=f"{float(train_loss):.4f}",
                kge_sta=f"{val_kge_sta:.4f}",
                val_kge=f"{val_kge:.4f}",
                rmse=f"{val_rmse:.2f}",
            )
            current_lr = scheduler.get_last_lr()[0]
            from datetime import datetime
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            epoch_msg = (
                f"[{ts}] Epoch {epoch:4d} | train={train_loss:.4f} "
                f"| val_nse={val_nse:.4f} | val_kge={val_kge:.4f} "
                f"| kge_sta={val_kge_sta:.4f} | kge_med={val_kge_med:.4f}"
                f"| rmse={val_rmse:.2f} | nrmse={val_nrmse:.3f}"
                f" | r={val_r:.3f} | beta={val_beta:.3f} | gamma={val_gamma:.3f}"
                f" | kge_log={val_kge_log:.4f}"
            )
            if not math.isnan(val_cov90):
                epoch_msg += f" | cov90={val_cov90:.3f} | cov50={val_cov50:.3f}"
            val_nll_m = last_val_metrics.get("val_nll", float("nan"))
            if not math.isnan(val_nll_m):
                epoch_msg += f" | nll={val_nll_m:.3f}"
            epoch_msg += f" | lr={current_lr:.2e}"
            logger.info(epoch_msg)
            print(epoch_msg, flush=True)

            # DECOMPOSITION DE LA PERTE (2026-08-22). Sans elle on ne sait pas ce qui
            # tire le modele : un terme multi-objectif mal dose ressemble en tout point
            # a un modele qui diverge. Le pilote quebecois ne passe aucun `run_logger`,
            # donc les composantes n'etaient enregistrees NULLE PART et le seul moyen de
            # juger un poids etait de relancer sans lui. On imprime les termes non nuls
            # avec leur PART du total : c'est la grandeur qui decide, pas la valeur brute.
            if train_comps:
                # PIEGE : les composantes stockees sont les termes BRUTS (L_tws, L_et...),
                # jamais ponderes, alors que le total l'est. Un pourcentage naif compare
                # donc une pomme a une somme de poires et surestime les termes a petit
                # poids. On remultiplie par le poids avant de parler de part.
                _poids = {"tws": getattr(self.loss_fn, "w_tws", 1.0),
                          "tws_clim": getattr(self.loss_fn, "w_tws_clim", 1.0),
                          "et": getattr(self.loss_fn, "w_et", 1.0),
                          "snow": getattr(self.loss_fn, "w_snow", 1.0),
                          "swe_mass": getattr(self.loss_fn, "w_swe_mass", 1.0),
                          "peak": getattr(self.loss_fn, "w_peak", 1.0),
                          # `prior` MANQUAIT a cette table (trouve le 2026-08-28). Il
                          # tombait donc sur le defaut 1.0 et s'imprimait BRUT au milieu
                          # de termes ponderes, sous un en-tete qui annonce "ponderees".
                          # Effet mesure : il affichait 3782 % du total alors que sa part
                          # reelle vaut 182 x 0.005 = 0.91 sur un total de 3.98, soit
                          # 23 %. J'ai bati sur ce chiffre tout un diagnostic de soiree
                          # -- "le prior ecrase le debit d'un facteur cinquante" -- qui
                          # etait faux. Les vrais dominants sont les deux termes GRACE,
                          # eux correctement ponderes. Le commentaire ci-dessus dit
                          # exactement ce qu'il fallait faire ; la table ne le faisait
                          # pas pour ce terme-la.
                          "prior": getattr(self.config, "w_prior", 1.0),
                          "diversity": getattr(self.config, "w_diversity", 1.0)}

                def _poids_de(nom):
                    """Poids d'un terme, cherché par son nom plutôt que dans une table tenue
                    à la main. Celle-ci ne couvrait que huit termes sur la vingtaine
                    existante ; tous les autres, dont le KGE, l'écart quadratique et le
                    biais de volume, tombaient sur le défaut de un et s'imprimaient BRUTS
                    au milieu de termes pondérés, sous un en-tête annonçant le contraire.
                    C'est le piège décrit juste au-dessus pour le prior, qui valait pour
                    presque tous les termes de débit (constaté le 2026-09-20)."""
                    if nom == "nappe" and "nappe_poids_applique" in train_comps:
                        return float(train_comps["nappe_poids_applique"])
                    if nom in _poids:
                        return _poids[nom]
                    for source in (self.loss_fn, self.config):
                        w = getattr(source, f"w_{nom}", None)
                        if w is not None:
                            return float(w)
                    return 1.0

                _nz = {}
                for k, v in train_comps.items():
                    v = float(v)
                    if abs(v) <= 1e-9 or math.isnan(v) or k == "nappe_poids_applique":
                        continue
                    _nom = k.replace("_loss", "")
                    _nz[_nom] = v * _poids_de(_nom)
                # Le poids APPLIQUE aux puits s'imprime en clair : le plafond peut le reduire
                # sous le poids demande, et le produit seul ne permet pas de le voir.
                if "nappe" in _nz and "nappe_poids_applique" in train_comps:
                    _nz["nappe"] = _nz["nappe"]
                    _pa = float(train_comps["nappe_poids_applique"])
                    print(f"            puits : poids demande {self.loss_fn.w_nappe:.2f}, "
                          f"poids applique moyen {_pa:.2f}", flush=True)
                if _nz:
                    _tot = float(train_loss) or 1.0
                    print("            composantes ponderees | " + "  ".join(
                        f"{k}={v:.3g}({100*v/_tot:.0f}%)"
                        for k, v in sorted(_nz.items(), key=lambda kv: -abs(kv[1]))),
                        flush=True)

            if self.run_logger is not None:
                self.run_logger.log_metrics(
                    {"train_loss": float(train_loss)}
                    | {f"train_{k}": float(v) for k, v in train_comps.items()}
                    | ({f"val_{k}": float(v) for k, v in last_val_metrics.items()}
                       if run_val else {}),
                    step=epoch,
                )

            # Save best checkpoint (only when val was actually computed).
            # Two regimes depending on best_metric direction:
            #   - higher-is-better (kge, nse, ...): improvement = strictly larger
            #   - lower-is-better (loss): improvement = relative drop > tolerance
            #     (train_loss is monotone-decreasing, so a strict comparison
            #     would never count no-improve and disable autopilot LR-plateau)
            _bm = self.config.best_metric
            _tol = self.config.best_metric_tolerance
            # Alias : "nll" (pinball moyenne en mode quantile) est rangé sous "val_nll"
            _bm_key = {"nll": "val_nll"}.get(_bm, _bm)
            _default = float("inf") if self._best_metric_lower_is_better else -float("inf")
            if _bm == "loss":
                _cur = float(train_loss)
            elif _bm == "val_loss":
                _cur = last_val_metrics.get("loss", float("inf")) if run_val else float("inf")
            else:
                _cur = last_val_metrics.get(_bm_key, _default) if run_val else _default

            tracked_now = run_val or (_bm == "loss")
            if tracked_now:
                # Use absolute tolerance scaled by |best| so we handle negative
                # metrics correctly (NLL can be negative — a relative-tolerance
                # comparison flips sign and asks for a worse score).
                # MARGE UNIQUEMENT lower-is-better (bug corrigé 2026-08-18) : la doc de
                # best_metric_tolerance dit « For higher-is-better metrics like KGE...
                # set to 0.0 » mais la marge s'appliquait aux deux sens ; à 0.005 elle
                # jetait de vraies améliorations (ex. zn-occ ép. 4 : 0.7587 > best
                # 0.7551 refusé, marge 0.0038) et gonflait no_improve -> coupes LR
                # autopilote prématurées. KGE/NSE : comparaison stricte, comme documenté.
                if self._best_metric_lower_is_better:
                    _margin = _tol * abs(self._best_val_metric)
                    if not math.isfinite(_margin):
                        _margin = 0.0
                else:
                    _margin = 0.0
                if self._best_metric_lower_is_better:
                    is_improvement = _cur < self._best_val_metric - _margin
                else:
                    is_improvement = _cur > self._best_val_metric + _margin

                if is_improvement:
                    self._best_val_metric = _cur
                    epochs_without_improvement = 0
                    if self.checkpoint_path:
                        self.model.save(self.checkpoint_path)
                        print(f"  -> best checkpoint saved ({_bm}={self._best_val_metric:.4f})")
                        logger.info(f"  -> best checkpoint saved ({_bm}={self._best_val_metric:.4f})")
                else:
                    epochs_without_improvement += 1
                    print(f"  [no save] {_bm}={_cur:.4f}, "
                          f"best={self._best_val_metric:.4f}, "
                          f"no_improve={epochs_without_improvement}/{self.config.patience}")
                # Point de reprise de CHAQUE epoque (2026-10-03, MEANDRE_SAUVER_EPOQUES=1) : la
                # validation d'une station sur deux ans saute d'une epoque a l'autre, et le choix
                # du meilleur point y ajoute un bruit que seule l'evaluation de toute la
                # trajectoire permet de mesurer.
                if self.checkpoint_path and settings.get("trainer.save_every_epoch"):
                    self.model.save(self.checkpoint_path.replace(".pt", f"-e{epoch + 1}.pt"))

            # ── Autopilot ──────────────────────────────────────────────
            if self.config.autopilot and run_val:
                self._run_autopilot(
                    epoch, last_val_metrics, epochs_without_improvement,
                )

            # ABANDON PRÉCOCE SUR SEUIL (2026-08-14). Un run de 30 époques coûte 4 à 8 h
            # et son verdict est souvent lisible dès les premières : inutile d'attendre
            # la fin pour constater qu'on plafonne à 0.3 quand la cible est 0.77.
            # MEANDRE_ABANDON="époque:seuil" (ex "5:0.55") : au-delà de cette époque, si
            # la métrique de sélection reste sous le seuil, on arrête.
            _ab = settings.get("trainer.abandon")
            if _ab and last_val_metrics:
                try:
                    _ae, _as = _ab.split(":")
                    _ae, _as = int(_ae), float(_as)
                except ValueError:
                    _ae, _as = None, None
                if _ae is not None and epoch >= _ae:
                    _cur = self._best_val_metric
                    if _cur is not None and float(_cur) < _as:
                        logger.info(
                            f"ABANDON à l'époque {epoch} : meilleure métrique "
                            f"{float(_cur):.4f} sous le seuil {_as} — le verdict est "
                            f"acquis, on n'attend pas la fin.")
                        print(f"[trainer] ABANDON époque {epoch} : {float(_cur):.4f} "
                              f"< seuil {_as}", flush=True)
                        break

            # Early stopping
            if (self.config.patience > 0
                    and epochs_without_improvement >= self.config.patience):
                logger.info(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {self.config.patience} epochs, "
                    f"best {_bm}={self._best_val_metric:.4f})"
                )
                break

        if self.run_logger is not None:
            self.run_logger.end_run()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_spinup(self, data: TrainingData) -> HydroState:
        """Return the spun-up state.

        On the first call (or when warm_spinup_steps == 0), runs the full
        spinup from cold start.  On subsequent calls, reuses the cached
        spinup state and only runs ``warm_spinup_steps`` additional steps
        to let the state adjust to updated model weights — much cheaper
        than replaying the entire spinup every epoch.
        """
        device = data.forcing.device
        spinup_end = min(self.config.spinup_steps, data.train_slice.start)

        if spinup_end == 0:
            return HydroState.zeros(self.model.n_nodes, device=device)

        warm_steps = self.config.warm_spinup_steps
        cached = getattr(self, "_cached_spinup_state", None)

        if cached is not None and warm_steps > 0:
            # Warm spinup: start from cached state, run only the last N days
            warm_start = max(0, spinup_end - warm_steps)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=self._amp_dtype, enabled=self._use_amp):
                _, spun_state = self.model.simulate(
                    forcing=data.forcing[warm_start:spinup_end],
                    initial_state=cached,
                    graph=data.graph,
                    node_coords=data.node_coords,
                    territorial=data.territorial,
                    withdrawals=data.withdrawals.slice(warm_start, spinup_end),
                    day_of_year=data.day_of_year[warm_start:spinup_end],
                )
        else:
            # Full spinup from cold start
            _ref_et = (getattr(self.loss_fn, "et_mode", "level") in ("anomaly", "bassin", "forme")
                       and getattr(self.loss_fn, "w_et", 0.0) > 0 and data.et_obs is not None
                       and getattr(self, "_et_sim_base", None) is None
                       and not settings.get("trainer.et_center_ema"))
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=self._amp_dtype, enabled=self._use_amp):
                _res = self.model.simulate(
                    forcing=data.forcing[:spinup_end],
                    initial_state=HydroState.zeros(self.model.n_nodes, device=device),
                    graph=data.graph,
                    node_coords=data.node_coords,
                    territorial=data.territorial,
                    withdrawals=data.withdrawals.slice(0, spinup_end),
                    day_of_year=data.day_of_year[:spinup_end],
                    return_diagnostics=_ref_et,
                )
            spun_state = _res[1]
            if _ref_et and len(_res) == 3 and settings.get("trainer.et_reference_from_spinup"):
                # Reference initiale de l'ET centree : annees completes de la mise en regime.
                _n_ans = spinup_end // 365
                if _n_ans >= 1:
                    _a0 = spinup_end - 365 * _n_ans
                    _e = _res[2].etr[_a0:spinup_end].to(device)
                    _Wsp = getattr(data, "et_weight", None)
                    if _Wsp is not None:
                        _ww = _Wsp[_a0:spinup_end].to(device, _e.dtype)
                        self._et_sim_base = ((_ww * _e).sum(dim=0) / _ww.sum(dim=0).clamp(min=1e-6)).detach()
                    else:
                        self._et_sim_base = _e.mean(dim=0).detach()
                    logger.info("ET centree : reference initiale sur %d an(s) de mise en regime, moyenne %.3f mm/j", _n_ans, float(self._et_sim_base.mean()))

        # Cache for next epoch
        self._cached_spinup_state = spun_state.detach()
        return spun_state

    def _simulate(
        self, data: TrainingData, time_slice: slice, tbptt_steps: int = 0,
    ) -> tuple[Tensor, HydroState]:
        """Run model over time_slice (after spin-up on the preceding steps)."""
        initial_state = self._run_spinup(data)

        # Main simulation over the requested slice
        Q_sim, final_state = self.model.simulate(
            forcing=data.forcing[time_slice],
            initial_state=initial_state,
            graph=data.graph,
            node_coords=data.node_coords,
            territorial=data.territorial,
            withdrawals=data.withdrawals.slice(time_slice.start, time_slice.stop),
            day_of_year=data.day_of_year[time_slice],
            tbptt_steps=tbptt_steps,
        )
        return Q_sim, final_state

    def _check_auxiliary_targets(self, data: TrainingData) -> None:
        """Un poids de perte strictement positif dont la cible est absente doit CRIER.

        Dette #14 du registre : `w_snow = 0.3` a ete annonce a chaque run du pilote
        quebecois pendant un mois alors que `swe_obs` etait perdu en amont, donc que le
        terme ne s'evaluait jamais. Le pilote imprimait la contrainte, le trainer la
        sautait en silence, et 219 commits sont passes dessus. Meme famille que la
        dette #3 : une entree a repli silencieux fabrique des fantomes.

        On leve, on n'avertit pas : un multi-objectif dont un objectif manque n'est pas
        le modele qu'on croit entrainer, et le score qui en sort n'est comparable a rien.
        """
        manquants = []
        for poids, cible, nom in (
            (getattr(self.loss_fn, "w_snow", 0.0), data.swe_obs, "w_snow / swe_obs (MODIS snow cover)"),
            (getattr(self.loss_fn, "w_tws", 0.0), data.tws_obs, "w_tws / tws_obs (GRACE TWS)"),
            (getattr(self.loss_fn, "w_nappe", 0.0), data.nappe_obs, "w_nappe / nappe_obs (niveaux de nappe)"),
            (getattr(self.loss_fn, "w_et", 0.0), data.et_obs, "w_et / et_obs (MODIS MOD16)"),
            (getattr(self.loss_fn, "w_swe_mass", 0.0), data.swe_mass_obs,
             "w_swe_mass / swe_mass_obs (CanSWE, masse du manteau)"),
        ):
            if poids > 0 and cible is None:
                manquants.append(f"  - {nom} : poids {poids} mais observation absente")
        if manquants:
            raise ValueError(
                "Contrainte auxiliaire demandee sans sa cible :\n" + "\n".join(manquants)
                + "\nMettre le poids a zero, ou fournir l'observation dans TrainingData."
            )

    def _center_et(self, et_sim: Tensor, et_obs: Tensor, data: TrainingData, t0: int | None = None) -> tuple[Tensor, Tensor]:
        """Centrage ET pour ``loss_fn.et_mode == "anomaly"`` (TENDANCE, pas niveau).

        Même discipline que la TWS (corrigée le 2026-08-10) : lignes de base LONGUE
        DURÉE, jamais intra-tronçon (sinon on retire le signal saisonnier qu'on
        prétend imposer). Côté observation : moyenne par nœud de la série MODIS
        entière, calculée une fois. Côté simulation : moyenne mobile exponentielle
        par nœud, détachée (la moyenne long-terme n'est pas connue d'avance).
        Retourne (et_sim_centré, et_obs_centré) ; identité en mode "level".
        """
        if getattr(self.loss_fn, "et_mode", "level") not in ("anomaly", "bassin", "forme"):
            return et_sim, et_obs
        # Poids hors neige du bloc (t0 = indice du premier jour du bloc sur l'axe du forcage).
        _W = getattr(data, "et_weight", None)
        w = None
        if _W is not None and t0 is not None:
            w = _W[t0:t0 + et_sim.shape[0]].to(et_sim.device, et_sim.dtype)
        base_obs = getattr(self, "_et_obs_base", None)
        if base_obs is None:
            eo = data.et_obs
            if _W is not None:
                _s0 = data.train_slice.start
                _wo = _W[_s0:_s0 + eo.shape[0]].to(eo.device, eo.dtype) * (~torch.isnan(eo))
                base_obs = (_wo * torch.nan_to_num(eo, nan=0.0)).sum(dim=0) / _wo.sum(dim=0).clamp(min=1e-6)
            else:
                _wo = (~torch.isnan(eo)).to(eo.dtype)
                cnt = (~torch.isnan(eo)).sum(dim=0).clamp(min=1)
                base_obs = torch.nan_to_num(eo, nan=0.0).sum(dim=0) / cnt  # (n_nodes,)
            self._et_obs_base = base_obs
            _var_o = (_wo * (torch.nan_to_num(eo, nan=0.0) - base_obs) ** 2).sum(dim=0) / _wo.sum(dim=0).clamp(min=1e-6)
            self._et_obs_std = torch.sqrt(_var_o.clamp(min=1e-6))
        # REFERENCE SIMULEE (corrigee le 2026-10-02). L'ancienne partait de la moyenne du
        # PREMIER bloc (45 jours d'hiver, ET ~0,05 mm/j) et la suivait par une moyenne mobile de
        # poids 0,02 par bloc, soit environ six ans de memoire : les anomalies simulees etaient
        # surestimees de l'ecart entre la moyenne annuelle et celle de janvier, et le terme
        # poussait toute l'ET vers le bas (052805, deux epoques : K_c 0,63, ET 1,05 contre 1,44).
        # Desormais : moyenne exacte de l'epoque precedente sur toute la periode d'entrainement,
        # et, a la premiere, moyenne sur les annees completes de la mise en regime.
        # MEANDRE_ET_CENTRAGE_EMA=1 restitue l'ancienne reference.
        _acc = getattr(self, "_et_sim_acc", None)
        _d = et_sim.detach()
        _wd = w if w is not None else torch.ones_like(_d)
        if _acc is None:
            self._et_sim_acc = [(_wd * _d).sum(dim=0), _wd.sum(dim=0), (_wd * _d * _d).sum(dim=0)]
        else:
            _acc[0] = _acc[0] + (_wd * _d).sum(dim=0)
            _acc[1] = _acc[1] + _wd.sum(dim=0)
            _acc[2] = _acc[2] + (_wd * _d * _d).sum(dim=0)
        if settings.get("trainer.et_center_ema") or getattr(self, "_et_sim_base", None) is None:
            cur = _d.mean(dim=0)
            prev = getattr(self, "_et_sim_base", None)
            self._et_sim_base = cur if prev is None else (0.98 * prev + 0.02 * cur)
        if getattr(self.loss_fn, "et_mode", "") == "forme":
            # Chaque cote divise par son propre ecart-type : phase et forme seulement.
            _ss = getattr(self, "_et_sim_std", None)
            if _ss is None:
                raise RuntimeError("et_mode forme : ecart-type simule absent (passe de reference non faite)")
            _zs = (et_sim - self._et_sim_base) / _ss.to(et_sim.device).clamp(min=1e-3)
            _zo = (et_obs - base_obs) / self._et_obs_std.to(et_obs.device).clamp(min=1e-3)
            if w is None:
                return _zs, _zo
            _r = torch.sqrt(w.clamp(min=0.0))
            return _zs * _r, _zo * _r
        if w is None:
            return et_sim - self._et_sim_base, et_obs - base_obs
        # Ecart pondere : chaque cote multiplie par la racine du poids, si bien que l'ecart
        # quadratique de la perte vaut w x ecart^2.
        _r = torch.sqrt(w.clamp(min=0.0))
        return (et_sim - self._et_sim_base) * _r, (et_obs - base_obs) * _r

    def _et_reference_fin_epoque(self) -> None:
        """Pose la reference simulee de l'ET centree sur la moyenne exacte de l'epoque qui
        s'acheve (somme des blocs sur toute la periode d'entrainement), puis remet a zero."""
        _acc = getattr(self, "_et_sim_acc", None)
        if _acc is not None and not settings.get("trainer.et_center_ema"):
            _n = torch.as_tensor(_acc[1])
            if bool((_n > 0).all()):
                self._et_sim_base = _acc[0] / _n
                if len(_acc) > 2:
                    self._et_sim_std = torch.sqrt((_acc[2] / _n - self._et_sim_base ** 2).clamp(min=1e-6))
        self._et_sim_acc = None

    def _et_reference_exacte(self, data: TrainingData, etat: HydroState) -> None:
        """Reference simulee du terme MOD16 centre ou en forme, avant la premiere epoque.

        Moyenne et ecart-type EXACTS de l'ET simulee sur la periode d'entrainement, aux poids
        de depart, par une passe sans gradient par blocs d'un an depuis l'etat de mise en
        regime. Appelee au debut de l'epoque d'entrainement, et non dans la mise en regime :
        le pilote evalue avant d'entrainer, si bien que la premiere mise en regime tourne sur
        les donnees de validation, sans MOD16, et que l'epoque reprend ensuite l'etat en
        cache. La reference n'etait alors jamais posee, et `_center_et` retombait sur la
        moyenne du premier bloc d'hiver (2026-10-06, R311). MEANDRE_ET_REF_MISE_EN_REGIME=1
        restitue la reference de la mise en regime.
        """
        device = data.forcing.device
        _Wsp = getattr(data, "et_weight", None)
        _s, _num, _num2, _den = etat, None, None, None
        _t, _fin = data.train_slice.start, data.train_slice.stop
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=self._amp_dtype, enabled=self._use_amp):
            while _t < _fin:
                _u = min(_t + 365, _fin)
                _r = self.model.simulate(
                    forcing=data.forcing[_t:_u], initial_state=_s, graph=data.graph,
                    node_coords=data.node_coords, territorial=data.territorial,
                    withdrawals=data.withdrawals.slice(_t, _u),
                    day_of_year=data.day_of_year[_t:_u], return_diagnostics=True,
                    poursuivre_etat=True,
                )
                _s = _r[1]
                _e = _r[2].etr.to(device).float()
                _ww = (_Wsp[_t:_u].to(device, _e.dtype) if _Wsp is not None else torch.ones_like(_e))
                _a, _b, _c = (_ww * _e).sum(dim=0), (_ww * _e * _e).sum(dim=0), _ww.sum(dim=0)
                _num = _a if _num is None else _num + _a
                _num2 = _b if _num2 is None else _num2 + _b
                _den = _c if _den is None else _den + _c
                _t = _u
        self._et_sim_base = (_num / _den.clamp(min=1e-6)).detach()
        self._et_sim_std = torch.sqrt((_num2 / _den.clamp(min=1e-6) - self._et_sim_base ** 2).clamp(min=1e-6)).detach()
        print(f"[et] reference initiale exacte sur la periode d'entrainement : moyenne {float(self._et_sim_base.mean()):.3f} mm/j, "
              f"ecart-type {float(self._et_sim_std.mean()):.3f} mm/j", flush=True)

    def _train_epoch(self) -> tuple[Tensor, dict[str, Tensor]]:
        """One training epoch: simulate -> loss -> backward -> step.

        When ``config.chunk_steps > 0``, the training sequence is split into
        checkpointed chunks.  Each chunk's forward activations are freed after
        the forward pass and recomputed during backward — peak memory stays
        proportional to chunk_steps.  The loss (e.g. KGE) is computed on the
        **full** concatenated Q_sim so statistics (r, beta, gamma) are correct.
        """
        self.model.train()
        self.optimizer.zero_grad()

        data = self.train_data
        n_train = data.train_slice.stop - data.train_slice.start
        chunk = self.config.chunk_steps

        if chunk <= 0 or n_train <= chunk:
            # Original single-pass path
            return self._train_epoch_single(data)

        # ── Gradient-accumulation chunked forward ─────────────────────────
        # For chunk-safe losses (MSE, log-MSE, PBIAS): forward+backward per
        # chunk, accumulate gradients, single optimizer.step().
        # Memory ∝ chunk_steps, no checkpointing overhead.
        initial_state = self._run_spinup(data)
        _spinup_a_tourne = min(self.config.spinup_steps, data.train_slice.start) > 0
        if (getattr(self.loss_fn, "et_mode", "level") in ("anomaly", "bassin", "forme")
                and getattr(self.loss_fn, "w_et", 0.0) > 0 and data.et_obs is not None
                and not settings.get("trainer.et_center_ema")
                and not settings.get("trainer.et_reference_from_spinup")
                and (getattr(self, "_et_sim_base", None) is None or getattr(self, "_et_sim_std", None) is None)):
            self._et_reference_exacte(data, initial_state)
            # La passe laisse l'etat interne (manteau, gel, lacs) a la fin de la periode :
            # la mise en regime est rejouee pour que le premier bloc reparte de son etat.
            initial_state = self._run_spinup(data)

        state = initial_state
        t_start = data.train_slice.start
        obs_offset = 0
        total_loss = torch.tensor(0.0, device=data.forcing.device)
        all_components: dict[str, float] = {}
        n_chunks = 0
        import time as _time
        _t_bloc = _time.perf_counter()  # duree du bloc courant, lue par la sonde du recalcul
        if settings.get("probe.terms"):
            # La sonde retropropage chaque terme avec retain_graph ; une fonction compilee qui
            # recycle ses tampons (donated buffers) le refuse des que la taille du bloc change.
            import torch._functorch.config as _fconf
            _fconf.donated_buffer = False
            # Facteurs par troncon, valant un (neutres), hors de l'optimiseur.
            _champs_s = [x for x in settings.get("probe.terms").split(",") if x]
            self._sonde_mult = {k_: torch.ones(self.model.n_nodes, device=data.forcing.device, requires_grad=True)
                                for k_ in _champs_s}
            self._sonde_acc = {}
            self.model.spatial_encoder.multiplicateurs = self._sonde_mult
        sp_tensor: Tensor | None = None  # parametres spatiaux du bloc, pour la tete de quantiles
        # HISTORIQUE DÉTACHÉ des débits aux stations, pour que le KGE et la famille
        # Nash-Sutcliffe portent sur la séquence CONTINUE et non sur un bloc de 45 jours
        # (voir la note dans loss.py). MEANDRE_KGE_CONTINU=0 restitue le comportement
        # historique, pour comparaison.
        _kge_continu = settings.get("trainer.kge_continuous")
        _etat_continu = settings.get("trainer.state_continuous")
        _pas_par_bloc = settings.get("trainer.step_per_block")
        if settings.get("debug.nan_level") == "2" and not torch.is_anomaly_enabled():
            # NIVEAU 2 : mode anomalie pose des la passe AVANT, seul moyen d'obtenir la
            # ligne de code qui cree le tenseur fautif. Le niveau 1 ne nomme que
            # l'operation de la passe arriere (2026-09-08 : PowBackward0), ce qui ne suffit
            # pas a designer laquelle des puissances du modele est en cause. Tres lent :
            # a reserver a un sous-bassin.
            torch.autograd.set_detect_anomaly(True)
            print("[nan-debug] mode anomalie pose sur la passe avant (lent)", flush=True)
        # AMORCAGE DE L'HISTORIQUE (2026-09-04, MEANDRE_HISTORIQUE_AMORCE=1). Avec le
        # KGE continu, le bloc k ne voit que les blocs 0..k-1 : les dix premiers pas de
        # chaque epoque optimisent un KGE d'hiver seul, et avec un pas par bloc ils
        # poussent le modele vers un hiver plus fort et plus plat (mesure : validation
        # 0.83 -> 0.55 apres 65 pas). Ici, une passe SANS gradient en debut d'epoque
        # fournit la serie de toute la periode ; le bloc k voit alors le KGE de toute la
        # periode, lui-meme en direct et le reste detache.
        _amorce = settings.get("trainer.history_priming") and _kge_continu
        _pre_o = _pre_s = None
        _hist_o: list[Tensor] = []
        _hist_s: list[Tensor] = []

        # OFFSET ALÉATOIRE des frontières de chunks, par epoch (revue 2026-07-01) :
        # avec des frontières FIXES, les mêmes 11 jours de burn-in sur 45 (24 % des
        # jours) ne recevaient JAMAIS de gradient, epoch après epoch — tout orage
        # dans ces fenêtres était invisible à l'optimisation. On ALLONGE le premier
        # chunk de `offset` (plutôt que le raccourcir : fenêtre de loss trop courte
        # sinon), ce qui décale toutes les frontières suivantes. Mémoire : premier
        # chunk ≤ 2·chunk−1, dans l'enveloppe déjà tolérée par le merge de reliquat
        # (< chunk+60). Déterministe par epoch (reproductible).
        _g = torch.Generator().manual_seed(int(getattr(self, "_cur_epoch", 0)))
        _first_offset = int(torch.randint(0, chunk, (1,), generator=_g).item())

        if _amorce:
            _col = getattr(self.model, "vertical_column", None)
            _aux0 = getattr(_col, "_aux", None)
            _lac0 = getattr(self.model, "_lake_storage_fin", None)
            with torch.no_grad():
                _Qp, _ = self.model.simulate(
                    forcing=data.forcing[data.train_slice],
                    initial_state=initial_state,
                    graph=data.graph, node_coords=data.node_coords,
                    territorial=data.territorial, withdrawals=data.withdrawals.slice(data.train_slice.start, data.train_slice.stop),
                    day_of_year=data.day_of_year[data.train_slice],
                    poursuivre_etat=(_etat_continu and _spinup_a_tourne),
                )
            _T = data.train_slice.stop - data.train_slice.start
            _pre_s = _Qp[:, data.station_mask].detach()
            _pre_o = data.q_obs[:_T].detach()
            del _Qp
            # La passe a fait avancer l'etat interne jusqu'a la fin de la periode : on
            # remet celui de la mise en regime pour que le premier bloc reparte juste.
            if _col is not None and _aux0 is not None:
                _col._aux = _aux0
            if _lac0 is not None:
                self.model._lake_storage_fin = _lac0

        while t_start < data.train_slice.stop:
            _len = chunk + _first_offset if n_chunks == 0 else chunk
            t_end = min(t_start + _len, data.train_slice.stop)
            # Merge tiny remainder into the previous chunk
            remainder = data.train_slice.stop - t_end
            if 0 < remainder < 60:
                t_end = data.train_slice.stop
            sl = slice(t_start, t_end)
            chunk_len = t_end - t_start
            # PART DU BLOC DANS L'EPOQUE. Les termes ajoutes dans le bloc s'accumulent avec
            # cette part, de sorte que le bilan imprime compare des MOYENNES d'epoque. Sans
            # elle ils s'additionnaient sur les quelque deux cents blocs d'une epoque alors
            # que ceux de la fonction de perte etaient ponderes plus bas : le bilan portait
            # un facteur deux cents entre les deux familles, et se lisait comme une
            # domination de GRACE qui n'existait pas (2026-09-21).
            _part_bloc = chunk_len / n_train

            # ET multi-objectif : on récupère et_sim depuis CE forward (avec
            # gradient) plutôt qu'un 2e forward détaché.
            _need_et = (self.loss_fn.w_et > 0 and data.et_obs is not None)
            _need_tws = (self.loss_fn.w_tws > 0 and data.tws_obs is not None)
            _need_nappe = (getattr(self.loss_fn, "w_nappe", 0.0) > 0
                           and data.nappe_obs is not None and data.nappe_idx is not None)
            # Neige : MODIS snow cover (fraction) cale le taux de FONTE (sp_fonte).
            # On compare une fraction de couverture SIMULÉE = 1-exp(-SWE/SWE_REF)
            # (différentiable, monotone) à snow_frac MODIS. data.swe_obs porte le
            # snow_frac MODIS (0-1), PAS du SWE en mm.
            _need_snow = (self.loss_fn.w_snow > 0 and data.swe_obs is not None)
            # CanSWE : masse mesurée au sol, cible distincte de la couverture MODIS.
            _need_swe_mass = (getattr(self.loss_fn, "w_swe_mass", 0.0) > 0
                              and data.swe_mass_obs is not None)
            _need_diag = _need_et or _need_tws or _need_snow or _need_swe_mass
            with torch.amp.autocast("cuda", dtype=self._amp_dtype, enabled=self._use_amp):
                _sim_out = self.model.simulate(
                    forcing=data.forcing[sl],
                    initial_state=state,
                    graph=data.graph,
                    node_coords=data.node_coords,
                    territorial=data.territorial,
                    withdrawals=data.withdrawals.slice(sl.start, sl.stop),
                    day_of_year=data.day_of_year[sl],
                    tbptt_steps=self.config.tbptt_steps,
                    return_diagnostics=_need_diag,
                    # Les blocs suivants POURSUIVENT la simulation : le manteau neigeux,
                    # le profil de gel, le milieu humide et le stockage des lacs vivent
                    # hors de HydroState et repartaient de zero a chaque bloc, soit tous
                    # les 45 jours. L'entrainement ne voyait donc jamais un hiver
                    # continu (2026-09-03). MEANDRE_ETAT_CONTINU=0 restitue l'ancien
                    # comportement, pour un banc apparie.
                    # ... et aussi au PREMIER bloc quand une mise en regime vient de
                    # tourner : sans cela le manteau spinne etait jete au passage, le
                    # premier hiver de chaque epoque etait faux et le premier bloc
                    # rendait des gradients NaN en janvier (2026-09-04).
                    poursuivre_etat=(_etat_continu and (n_chunks > 0 or _spinup_a_tourne)),
                )
                if _need_diag:
                    Q_chunk, state_out, _diag_chunk = _sim_out
                else:
                    Q_chunk, state_out = _sim_out

                # Burn-in: first chunk after spinup needs longer burn-in to
                # stabilize state; subsequent chunks only need a short warm-up
                # since state is propagated from the previous chunk.
                burnin = min(60, chunk_len // 4) if n_chunks > 0 else min(90, chunk_len // 4)
                q_obs_chunk = data.q_obs[obs_offset + burnin:obs_offset + chunk_len]
                Q_chunk_loss = Q_chunk[burnin:]
                # Parametres spatiaux du bloc, calcules a la demande par la tete de quantiles.
                sp_tensor = None
                # ET sim de ce chunk (avec gradient)
                # GARDE CONTRE UN TERME PRESENT ET INERTE (2026-09-20). Trois fois le
                # meme jour, une piece s'est revelee declaree, affichee et sans effet :
                # les diagnostics detaches sous MEANDRE_DIAG_CPU=1, la table des poids de
                # l'affichage qui ne couvrait que huit termes sur vingt, et un profil de
                # sol vide. La plus couteuse a tourne deux jours en imprimant dans le
                # journal un terme GRACE qui ne produisait aucun gradient. Une grandeur
                # simulee qui n'est pas rattachee au graphe ne peut RIEN contraindre : on
                # le dit une fois par execution, fort, au lieu de le decouvrir en
                # constatant que deux passes different d'un terme donnent le meme chiffre.
                if not getattr(self, "_garde_gradient_faite", False):
                    self._garde_gradient_faite = True
                    _a_verifier = [("evapotranspiration MODIS", _need_et, "etr"),
                                   ("neige", _need_snow, "swe"),
                                   ("niveaux de puits", _need_nappe, "s_gw"),
                                   ("gravimetrie GRACE", _need_tws, "theta3")]
                    # Le terme de pics par rapport a besoin d'un SEUIL venu de
                    # l'observation. Sans lui il est silencieusement nul, et c'est justement
                    # le cas quand on remplace l'ancien terme de pics par le nouveau.
                    if getattr(self.loss_fn, "w_peak_ratio", 0.0) > 0 and                             getattr(self.loss_fn, "peak_threshold", None) is None:
                        print("[garde] TERME INERTE : « pics par rapport » est active mais "
                              "aucun seuil de pointe n'a ete construit. Le terme vaudra zero.",
                              flush=True)
                    for _nom_terme, _actif, _cle in _a_verifier:
                        if not _actif:
                            continue
                        _t = getattr(_diag_chunk, _cle, None)
                        if _t is None or not torch.is_tensor(_t):
                            continue
                        if not _t.requires_grad:
                            print(f"[garde] TERME INERTE : « {_nom_terme} » est active mais "
                                  f"son diagnostic « {_cle} » est DETACHE du graphe. Le terme "
                                  f"gardera sa valeur dans le journal et ne produira aucun "
                                  f"gradient. Verifier MEANDRE_DIAG_CPU et "
                                  f"HydroModel.DIAGNOSTICS_DERIVES.", flush=True)

                et_sim_chunk = et_obs_chunk = None
                if _need_et:
                    et_sim_chunk = _diag_chunk.etr[burnin:]
                    et_obs_chunk = data.et_obs[obs_offset + burnin:obs_offset + chunk_len]
                    et_sim_chunk, et_obs_chunk = self._center_et(et_sim_chunk, et_obs_chunk, data, t0=data.train_slice.start + obs_offset + burnin)

                # Neige : fraction de couverture simulée vs MODIS snow_frac.
                scf_sim_chunk = snow_obs_chunk = None
                if _need_snow:
                    _swe_mm = _diag_chunk.swe[burnin:]
                    scf_sim_chunk = 1.0 - torch.exp(-_swe_mm / self.config.snow_swe_ref)
                    snow_obs_chunk = data.swe_obs[obs_offset + burnin:obs_offset + chunk_len]

                if hasattr(self.loss_fn, "doy_bloc"):
                    self.loss_fn.doy_bloc = data.day_of_year[sl][burnin:]
                loss_chunk, comps = self.loss_fn(
                    q_obs=q_obs_chunk,
                    q_sim=Q_chunk_loss,
                    station_mask=data.station_mask,
                    et_obs=et_obs_chunk,
                    et_sim=et_sim_chunk,
                    swe_obs=snow_obs_chunk,
                    swe_sim=scf_sim_chunk,
                    q_obs_hist=(torch.cat([_pre_o[:obs_offset + burnin], _pre_o[obs_offset + chunk_len:]])
                                if _amorce else
                                (torch.cat(_hist_o) if (_kge_continu and _hist_o) else None)),
                    q_sim_hist=(torch.cat([_pre_s[:obs_offset + burnin], _pre_s[obs_offset + chunk_len:]])
                                if _amorce else
                                (torch.cat(_hist_s) if (_kge_continu and _hist_s) else None)),
                )
                if _kge_continu:
                    _hist_o.append(q_obs_chunk.detach())
                    _hist_s.append(Q_chunk_loss[:, data.station_mask].detach())

                # NIVEAU ANNUEL DE L'ET (2026-10-09, R360). Le terme MOD16 centre ne voit que la
                # forme ; rien n'ancrait le niveau, et le modele evapore 412 mm/an sur les bassins
                # jauges de l'Outaouais contre 550 d'ET implicite (pluie moins debit) et 502 a 540
                # pour MOD16, d'ou un exces de volume de 6 a 17 %. Ici : rapport, par noeud, de
                # l'ET simulee moyenne de l'epoque precedente a la moyenne MOD16 de la periode
                # d'entrainement ; son ecart a 1, en forme absolue (premier ordre au-dela de 5 %),
                # pousse l'ET du bloc vers le niveau observe. MEANDRE_W_ET_NIVEAU (poids, defaut 0).
                _w_niv = float(settings.get("trainer.et_level_weight"))
                L_et_niv = None
                if (_w_niv > 0 and _need_et and getattr(self, "_et_sim_base", None) is not None
                        and getattr(self, "_et_obs_base", None) is not None):
                    _raw_et = _diag_chunk.etr[burnin:]
                    _bo = torch.nan_to_num(self._et_obs_base.to(_raw_et.device), nan=0.0)
                    _ok_n = _bo > 0.05
                    _r_n = (self._et_sim_base.to(_raw_et.device) / _bo.clamp(min=0.05)).detach()
                    _coef = torch.clamp((_r_n - 1.0) / 0.05, -1.0, 1.0)
                    L_et_niv = (_coef * _raw_et.mean(dim=0) / _bo.clamp(min=0.05))[_ok_n].mean()
                    self.loss_fn.w_et_niveau = _w_niv
                    loss_chunk = loss_chunk + _w_niv * L_et_niv
                    all_components["et_niveau"] = all_components.get("et_niveau", 0.0) + float(((_r_n - 1.0).abs())[_ok_n].mean()) * _part_bloc

                # ── NIVEAUX DE NAPPE MESURES : contrainte de FORME ──────────
                # Anomalies REDUITES, centrees et normalisees DANS LE BLOC, donc compatible
                # avec l'accumulation par blocs sans calendrier a transporter. La serie est
                # prise au pas journalier plutot qu'au mois : le lissage mensuel servait au
                # diagnostic, la perte gagne a voir tous les points.
                if _need_nappe:
                    from meandre.training.loss import nappe_anomaly_loss
                    # La grandeur simulee est la PROFONDEUR de la surface libre quand la
                    # nappe libre est active, sinon le STOCK souterrain. Le terme travaille
                    # sur des anomalies reduites, donc l'echelle est eliminee et les deux
                    # conviennent ; il faut seulement le meme sens que le niveau mesure, qui
                    # est une profondeur sous le repere du tubage. Un stock qui monte est
                    # une profondeur qui diminue, d'ou le signe. Sans cela le terme ne
                    # s'appliquerait qu'aux configurations portant la nappe libre, et une
                    # ronde appariee cesserait de l'etre.
                    _pn = getattr(_diag_chunk, "profondeur_nappe", None)
                    _zn = (_pn[burnin:] if _pn is not None else -_diag_chunk.s_gw[burnin:])
                    # L'indice des puits vit sur la carte et le diagnostic peut vivre sur
                    # le processeur : on aligne AVANT d'indexer, un tenseur d'indices
                    # devant etre sur l'appareil du tenseur indexe.
                    _zn = _zn[:, data.nappe_idx.to(_zn.device)]
                    # On calcule LA OU VIT LE DIAGNOSTIC, sans le rapatrier sur la carte :
                    # sous MEANDRE_DIAG_CPU=1 il est sur le processeur, et le ramener a
                    # sature les 8 Go de la carte. Le gradient traverse les appareils.
                    _on = data.nappe_obs[obs_offset + burnin:obs_offset + chunk_len]
                    if _on.device != _zn.device:
                        _on = _on.to(_zn.device)
                    _mn = ~torch.isnan(_on)
                    L_nappe = nappe_anomaly_loss(_zn, torch.nan_to_num(_on), masque=_mn)
                    # PART PLAFONNEE (2026-09-23). Le terme vaut 2(1 - r) en moyenne sur les
                    # puits : la ou la nappe est mal simulee, il prend jusqu'a la moitie de la
                    # perte a poids egal, 53 % au Saint-Laurent nord-ouest et 49 % en Gaspesie
                    # contre 15 % en Outaouais, et tire le volume du debit, beta tombant de
                    # 1,05 a 0,90 en huit epoques. On borne sa part aux termes de debit du
                    # bloc par un facteur DETACHE : la direction du gradient est conservee,
                    # seule son amplitude est limitee.
                    # La reference est le reste de la perte du bloc a cet instant : les
                    # termes de debit, plus l'evapotranspiration et la neige que la fonction
                    # de perte a deja ajoutes. La gravimetrie et la masse du manteau viennent
                    # apres et n'en font pas partie.
                    _part_max = float(settings.get("trainer.groundwater_max_share"))
                    _poids_n = self.loss_fn.w_nappe
                    _ref = float(loss_chunk.detach())
                    if _part_max > 0 and _ref > 0 and float(L_nappe.detach()) > 0:
                        _poids_n = min(_poids_n, _part_max * _ref / float(L_nappe.detach()))
                    loss_chunk = loss_chunk + _poids_n * L_nappe
                    all_components["nappe_loss"] = (
                        all_components.get("nappe_loss", 0.0) + float(L_nappe.detach()) * _part_bloc)
                    # Le bilan multiplie chaque terme par son poids NOMINAL. Pour les puits le
                    # poids applique peut etre plus petit, a cause du plafond : on garde le
                    # poids reellement applique, moyenne sur l'epoque, pour que le bilan
                    # montre ce que l'optimiseur a vu et non ce que la configuration demandait.
                    all_components["nappe_poids_applique"] = (
                        all_components.get("nappe_poids_applique", 0.0) + _poids_n * _part_bloc)

                # ── GRACE TWS : stockage total basin-moyen (avec gradient) ──
                # storage = Σθ_i·z_i·1000 + SWE + S_gw + canopy + wetland (mm).
                # Comparé à GRACE aux mois valides, centré intra-chunk (chunk-safe).
                if _need_tws:
                    from meandre.training.loss import tws_anomaly_loss
                    _sp = self.model.spatial_encoder(
                        data.node_coords, data.territorial.to_tensor()
                    )
                    # z1 EST UNE PROPRIETE DE LA COLONNE (0.15 m), pas 0.30 : la valeur
                    # codee en dur ici doublait la respiration saisonnière de la couche
                    # de surface (8 mm). Le decalage constant s'annulait au centrage,
                    # l'amplitude non. Corrige le 2026-08-21.
                    _z1 = getattr(self.model.vertical_column, "z1", 0.15)
                    # MEANDRE_DIAG_CPU=1 accumule les diagnostics sur le PROCESSEUR pour
                    # tenir dans la memoire de la carte. Les parametres du champ, eux,
                    # restent sur la carte : sans alignement la multiplication leve une
                    # exception et tout entrainement devient impossible sous ce reglage
                    # (mesure 2026-09-19). Un bloc pese quelques megaoctets, le transfert
                    # est negligeable.
                    _dev = _sp.Z2.device
                    _al = lambda x: x if (x is None or x.device == _dev) else x.to(_dev)
                    _soil_mm = ((_al(_diag_chunk.theta1) * _z1
                                 + _al(_diag_chunk.theta2) * _sp.Z2
                                 + _al(_diag_chunk.theta3) * _sp.Z3) * 1000.0)
                    # `_diag_chunk.wetland` est le champ MORT (recopie tel quel d'un pas
                    # a l'autre en mode hydrotel) : constant, donc invisible apres
                    # centrage. Le vrai stock est `wet_vol`, expose le 2026-08-20.
                    _wet = _al(getattr(_diag_chunk, "wet_vol", None))
                    if _wet is None:
                        _wet = torch.zeros_like(_soil_mm)
                    _stor = (_soil_mm + _al(_diag_chunk.swe) + _al(_diag_chunk.s_gw)
                             + _al(_diag_chunk.canopy) + _wet)  # (T, n_nodes) mm
                    _grp = getattr(data, "tws_group", None)
                    if _grp is None:
                        _stor_basin = _stor.mean(dim=1)[burnin:]  # (T-burnin,) moy-bassin
                    else:
                        # MOYENNE PAR BASSIN. GRACE mesure une colonne d'eau sur un
                        # bassin ; sur le domaine provincial, moyenner les 25 656
                        # tronçons confronterait la Gaspésie et l'Abitibi mêlées à une
                        # seule série. Un index_add par bassin coûte un balayage.
                        _ng = int(data.tws_obs.shape[1])
                        _cnt = getattr(self, "_tws_group_count", None)
                        if _cnt is None:
                            _cnt = torch.zeros(_ng, device=_stor.device).index_add_(
                                0, _grp, torch.ones_like(_grp, dtype=_stor.dtype))
                            self._tws_group_count = _cnt.clamp(min=1.0)
                            _cnt = self._tws_group_count
                        _sum = torch.zeros(_stor.shape[0], _ng, device=_stor.device,
                                           dtype=_stor.dtype).index_add_(1, _grp, _stor)
                        _stor_basin = (_sum / _cnt)[burnin:]     # (T-burnin, n_bassins)
                    _tws_chunk = data.tws_obs[obs_offset + burnin:obs_offset + chunk_len]
                    _vt = ~torch.isnan(_tws_chunk)
                    if int(_vt.sum()) >= 2:  # ≥2 mois pour centrer
                        _s = _stor_basin[_vt]
                        _g = _tws_chunk[_vt]
                        # σ = incertitude GRACE typique (~25 mm) → z-score, L_tws~O(1)
                        # (sinon mm² ~800 écrase Q). "fit à l'incertitude GRACE près".
                        # LIGNE DE BASE LONGUE DURÉE (corrigée le 2026-08-10). Les deux
                        # lignes de base étaient calculées DANS le tronçon de séquence
                        # (`_s.mean()`, `_g.mean()`), alors que loss.tws_anomaly_loss
                        # documente une référence longue durée : on retirait donc une
                        # partie du signal saisonnier que la contrainte prétend imposer.
                        # Côté observation : moyenne de la série ENTIÈRE, calculée une
                        # fois. Côté simulation : moyenne mobile exponentielle détachée,
                        # puisque la valeur longue durée n'est pas connue d'avance.
                        # Indice de bassin de chaque valeur retenue : les lignes de base
                        # se prennent PAR BASSIN, sinon on centrerait la Gaspésie sur la
                        # moyenne provinciale et on inventerait un biais là où il n'y en
                        # a pas. Sans groupe, `_gi` reste None et rien ne change.
                        _gi = _vt.nonzero(as_tuple=True)[1] if _vt.ndim == 2 else None
                        _gb = getattr(self, "_tws_obs_base", None)
                        if _gb is None:
                            _a = data.tws_obs
                            if _gi is None:
                                _ao = _a[~torch.isnan(_a)]
                                _gb = _ao.mean() if _ao.numel() else torch.zeros((), device=_s.device)
                            else:
                                _ok = ~torch.isnan(_a)
                                _n = _ok.sum(dim=0).clamp(min=1)
                                _gb = torch.where(_ok, _a, torch.zeros_like(_a)).sum(dim=0) / _n
                            self._tws_obs_base = _gb
                        _gbv = _gb if _gi is None else _gb[_gi]
                        _pb = getattr(self, "_tws_sim_base", None)
                        if _gi is None:
                            _sb = _s.mean().detach()
                            self._tws_sim_base = _sb if _pb is None else (0.98 * _pb + 0.02 * _sb)
                        else:
                            _ng2 = int(data.tws_obs.shape[1])
                            _cs = torch.zeros(_ng2, device=_s.device).index_add_(0, _gi, _s.detach())
                            _cn = torch.zeros(_ng2, device=_s.device).index_add_(
                                0, _gi, torch.ones_like(_s))
                            _vu = _cn > 0        # bassins PRÉSENTS dans ce tronçon
                            _sb = torch.where(_vu, _cs / _cn.clamp(min=1),
                                              torch.zeros_like(_cs))
                            # Un bassin absent du tronçon courant doit GARDER sa ligne de
                            # base. Le tirer vers zéro parce qu'il n'a pas de mois GRACE
                            # ici fabriquait un écart de plusieurs centaines de mm, et la
                            # perte GRACE mangeait 3300 % du total au premier epoch du
                            # domaine fondu (2026-08-25). Avec une seule région le cas
                            # n'existait pas : le bassin était toujours présent.
                            if _pb is None:
                                _pb = _sb.clone()
                                _pb[~_vu] = _s.detach().mean()
                            self._tws_sim_base = torch.where(
                                _vu, 0.98 * _pb + 0.02 * _sb, _pb)
                        _sbv = (self._tws_sim_base if _gi is None
                                else self._tws_sim_base[_gi])
                        # ── GRACE EN FORME PLUTOT QU'EN NIVEAU (opt-in) ──────────
                        # Le projet a deja tranche cette question pour MODIS : la donnee
                        # sert pour la FORME, jamais pour le NIVEAU (R24, et_mode
                        # "anomaly"). GRACE est pourtant compare en millimetres absolus,
                        # alors que rien ne garantit que le stockage simule et le
                        # stockage satellitaire aient la meme AMPLITUDE : la colonne
                        # respire sur trois metres de sol plus la neige et la nappe,
                        # GRACE integre sur une empreinte de plusieurs centaines de
                        # kilometres. Quand les amplitudes different, la contrainte est
                        # INSATISFAISABLE et pousse indefiniment -- mesure le 2026-08-26,
                        # ou meme bornee elle a fait tomber le modele de 0.52 a 0.34.
                        # En mode forme, chaque cote est divise par son propre
                        # ecart-type (detache cote simulation, sinon le modele reduirait
                        # simplement son amplitude pour tricher) : ne restent que la
                        # phase et la forme, ce que GRACE mesure le plus surement.
                        if bool(getattr(self.config, "tws_shape_only", False)):
                            _rs = _s - _sbv
                            _ro = _g - _gbv
                            # ECARTS-TYPES DE LONGUE DUREE (2026-10-10). Pris sur les seules
                            # valeurs mensuelles du bloc, ils valaient un ou trois mois : le
                            # terme etait eteint a 30 jours (0,0004) et explosait a 90 jours
                            # (83,9, 98 % de la perte). Observation : toute la serie, une fois ;
                            # simulation : moyenne mobile detachee, comme la ligne de base.
                            # MEANDRE_TWS_SD_BLOC=1 restitue l'ecart-type du bloc.
                            if settings.get("debug.tws_sd_block"):
                                _ss = _rs.detach().std().clamp(min=1e-3)
                                _so = _ro.std().clamp(min=1e-3)
                            else:
                                if getattr(self, "_tws_obs_sd", None) is None:
                                    # anomalie par bassin quand il y a des groupes, pour ne pas
                                    # compter l'ecart de niveau entre bassins comme une amplitude
                                    _a2 = data.tws_obs - (self._tws_obs_base if data.tws_obs.ndim == 2 and _gi is not None else 0.0)
                                    _ao2 = _a2[~torch.isnan(_a2)]
                                    self._tws_obs_sd = (_ao2.std() if _ao2.numel() > 2 else torch.ones((), device=_s.device)).clamp(min=1e-3)
                                _so = self._tws_obs_sd
                                _v2 = _rs.detach().pow(2).mean()
                                _pv = getattr(self, "_tws_sim_var", None)
                                self._tws_sim_var = _v2 if _pv is None else (0.98 * _pv + 0.02 * _v2)
                                _ss = self._tws_sim_var.sqrt().clamp(min=1e-3)
                            L_tws = ((_rs / _ss) - (_ro / _so)).pow(2).mean()
                            loss_chunk = loss_chunk + self.loss_fn.w_tws * L_tws
                            all_components["tws_loss"] = (
                                all_components.get("tws_loss", 0.0) + float(L_tws.detach()) * _part_bloc)
                            _skip_niveau = True
                        else:
                            _skip_niveau = False
                        _d0 = float(getattr(self.config, "aux_huber_delta", 0.0))
                        if _skip_niveau:
                            pass
                        elif _d0 > 0:
                            _zt = ((_s - _sbv) - (_g - _gbv)) / 25.0
                            _at = _zt.abs()
                            L_tws = torch.where(_at <= _d0, 0.5 * _zt.pow(2),
                                                _d0 * (_at - 0.5 * _d0)).mean()
                        else:
                            L_tws = tws_anomaly_loss(_s, _g, _sbv, _gbv, sigma=25.0)
                        if not _skip_niveau:
                            loss_chunk = loss_chunk + self.loss_fn.w_tws * L_tws
                            all_components["tws_loss"] = (
                                all_components.get("tws_loss", 0.0) + float(L_tws.detach()) * _part_bloc)

                        # ── TERME CLIMATOLOGIQUE (2026-08-21, R23) ──────────────
                        # Le terme ci-dessus compare des mois INDIVIDUELS a sigma=25 mm,
                        # l'incertitude d'UNE estimation GRACE. Mesure sur le champion
                        # OUTV : 65 % du residu est un biais saisonnier SYSTEMATIQUE
                        # (+44 mm en mars, -47 en mai), mais la perte le lit comme
                        # 1.05 ecart-type et se declare satisfaite. Or un biais moyenne
                        # sur ~21 ans ne se juge pas a l'incertitude d'un mois : la
                        # bonne echelle est 25/sqrt(21) = 5.4 mm, contre laquelle le
                        # meme residu vaut 4.8 ecarts-types (9.2 en mai).
                        #
                        # On ajoute donc un terme sur le BIAIS PAR MOIS CALENDAIRE, sans
                        # retirer le terme mensuel (qui porte l'interannuel, un vrai
                        # signal qu'on ne veut pas ecraser). Astuce du passage direct :
                        # la VALEUR penalisee est le biais accumule (moyenne mobile
                        # detachee par mois), le GRADIENT passe par le residu courant.
                        # Chaque mois est donc pousse proportionnellement au biais
                        # systematique de son mois calendaire, et le terme reste
                        # calculable tronçon par tronçon.
                        # En mode FORME, le terme climatologique est neutralise : il
                        # penalise un biais mensuel EN MILLIMETRES, donc il
                        # reintroduirait exactement le niveau qu'on vient de retirer, et
                        # avec la tolerance la plus serree des deux. L'information
                        # saisonniere n'est pas perdue pour autant : le terme de forme
                        # compare les anomalies normalisees mois par mois.
                        _w_clim = (0.0 if bool(getattr(self.config, "tws_shape_only", False))
                                   else getattr(self.loss_fn, "w_tws_clim", 0.0))
                        if _w_clim > 0:
                            _bnd = getattr(self, "_doy_month_bounds", None)
                            if _bnd is None:
                                _bnd = torch.tensor(
                                    [31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335],
                                    device=_s.device, dtype=torch.long)
                                self._doy_month_bounds = _bnd
                            _d0 = data.day_of_year[sl][burnin:]
                            if _gi is not None:      # (T,) -> (T, n_bassins) avant masque
                                _d0 = _d0[:, None].expand_as(_vt)
                            _doy = _d0[_vt].long()
                            _mth = torch.bucketize(_doy, _bnd)          # 0..11
                            _res = (_s - _sbv) - (_g - _gbv)
                            # Le biais saisonnier s'accumule PAR MOIS ET PAR BASSIN : un
                            # excès de mars en Gaspésie et un déficit de mars en Abitibi
                            # se compenseraient dans une case unique, et le terme qui
                            # existe justement pour voir le biais systématique (R23) ne
                            # verrait plus rien.
                            _bias = getattr(self, "_tws_month_bias", None)
                            _shape = (12,) if _gi is None else (12, int(data.tws_obs.shape[1]))
                            if _bias is None or tuple(_bias.shape) != _shape:
                                _bias = torch.zeros(_shape, device=_s.device)
                            _cur = _bias.clone()
                            if _gi is None:
                                for _m in torch.unique(_mth):
                                    _cur[_m] = _res[_mth == _m].mean().detach()
                            else:
                                _flat = _mth * _shape[1] + _gi
                                _sm = torch.zeros(_shape[0] * _shape[1], device=_s.device)
                                _sm.index_add_(0, _flat, _res.detach())
                                _cn2 = torch.zeros_like(_sm).index_add_(
                                    0, _flat, torch.ones_like(_res))
                                _seen = _cn2 > 0
                                _cur = _cur.reshape(-1)
                                _cur[_seen] = (_sm[_seen] / _cn2[_seen])
                                _cur = _cur.reshape(_shape)
                            self._tws_month_bias = 0.9 * _bias + 0.1 * _cur
                            # passage direct : valeur = biais accumule, gradient = residu
                            _acc = (self._tws_month_bias[_mth] if _gi is None
                                    else self._tws_month_bias[_mth, _gi])
                            _st = _res - _res.detach() + _acc
                            # TOLERANCE PAR CASE. 5.4 mm vaut 25/sqrt(21), l'incertitude
                            # d'une climatologie sur vingt et un ans. En accumulant par
                            # mois ET PAR BASSIN (2026-08-25), chaque case ne voit plus
                            # qu'une fraction de ces echantillons : garder 5.4 revient a
                            # juger une moyenne bruitee a la tolerance d'une moyenne
                            # longue. La tolerance suit desormais le compte reel.
                            _sg = float(getattr(self.config, "tws_clim_sigma", 5.4))
                            if _gi is not None:
                                _nc = getattr(self, "_tws_cell_n", None)
                                if _nc is None:
                                    _nc = torch.zeros(_shape[0] * _shape[1], device=_s.device)
                                    self._tws_cell_n = _nc
                                self._tws_cell_n = self._tws_cell_n + _cn2
                                _n_eff = self._tws_cell_n.reshape(_shape)[_mth, _gi].clamp(min=1.0)
                                _sg_t = 25.0 / _n_eff.sqrt()
                            else:
                                _sg_t = torch.full_like(_st, _sg)
                            # HUBER SUR LE Z-SCORE (opt-in, `aux_huber_delta`). Sans
                            # borne, un ecart de 240 mm contre 5.4 de tolerance donne un
                            # terme de plusieurs milliers et un gradient proportionnel :
                            # la contrainte cesse d'etre un garde-fou et devient
                            # l'objectif. Mesure le 2026-08-26 sur la province, ou la
                            # perte d'entrainement est passee de 4.4 a 48.9 pendant que
                            # le debit se degradait. Au-dela de delta, le gradient reste
                            # CONSTANT au lieu de croitre.
                            _z = _st / _sg_t
                            _d = float(getattr(self.config, "aux_huber_delta", 0.0))
                            if _d > 0:
                                _a = _z.abs()
                                L_tws_clim = torch.where(_a <= _d, 0.5 * _z.pow(2),
                                                         _d * (_a - 0.5 * _d)).mean()
                            else:
                                L_tws_clim = _z.pow(2).mean()
                            loss_chunk = loss_chunk + _w_clim * L_tws_clim
                            all_components["tws_clim_loss"] = (
                                all_components.get("tws_clim_loss", 0.0)
                                + float(L_tws_clim.detach()) * _part_bloc)

                # ── CanSWE : MASSE du manteau mesurée au sol (2026-08-21, R24) ──
                # La cible existante (w_snow) est la FRACTION DE COUVERTURE MODIS, qui
                # sature dès qu'il y a un peu de neige (SCF = 1-exp(-SWE/15)) et ne dit
                # donc presque rien de la quantité d'eau stockée -- justement ce qui
                # manque au modèle. Pire, MODIS mesure une réflectance et sous-estime la neige sous
                # couvert forestier, si bien que le terme de couverture demandait au
                # modèle de fondre PLUS TÔT, contre GRACE. Les relevés CanSWE sont des
                # mesures de masse au sol, insensibles au couvert.
                #
                # Échelle : on divise par SWE_SCALE (100 mm, l'ordre de grandeur d'un
                # manteau québécois) plutôt que par un sigma d'incertitude. La
                # représentativité ponctuelle d'un relevé face à un tronçon n'a pas
                # d'écart-type connu ; la borner par les filtres de distance et
                # d'altitude (build_swe_targets) est plus honnête que d'inventer un
                # sigma qui donnerait à la contrainte une autorité qu'elle n'a pas.
                if (getattr(self.loss_fn, "w_swe_mass", 0.0) > 0
                        and data.swe_mass_obs is not None):
                    _sm_obs = data.swe_mass_obs[obs_offset + burnin:obs_offset + chunk_len]
                    _vm = ~torch.isnan(_sm_obs)
                    if bool(_vm.any()):
                        _sw = _diag_chunk.swe[burnin:]
                        # Meme alignement que pour les puits : l'indice doit vivre sur
                        # l'appareil du tenseur indexe, et l'observe sur celui du simule.
                        _sm_sim = _sw[:, data.swe_mass_node.to(_sw.device)]
                        if _sm_obs.device != _sm_sim.device:
                            _sm_obs, _vm = _sm_obs.to(_sm_sim.device), _vm.to(_sm_sim.device)
                        _SWE_SCALE = 100.0
                        L_swe_mass = (((_sm_sim[_vm] - _sm_obs[_vm]) / _SWE_SCALE) ** 2).mean()
                        loss_chunk = loss_chunk + self.loss_fn.w_swe_mass * L_swe_mass
                        all_components["swe_mass_loss"] = (
                            all_components.get("swe_mass_loss", 0.0)
                            + float(L_swe_mass.detach()) * _part_bloc)

                # ── Régression quantile (Phase 2 v2) : q_τ = μ + δ_τ ─────
                # δ_τ via quantile_head (avec gradient), μ détaché (la tête
                # quantile n'entraîne PAS le backbone — médiane = μ par construction).
                if (self.loss_fn.w_quantile > 0
                        and getattr(self.model, "use_quantile_head", False)
                        and hasattr(self.model, "quantile_head")):
                    from meandre.training.loss import quantile_loss as _qloss
                    if sp_tensor is None:
                        sp = self.model.spatial_encoder(
                            data.node_coords, data.territorial.to_tensor()
                        )
                        sp_tensor = sp.to_tensor()
                    _Q_det = Q_chunk_loss.detach()
                    _offsets = self.model.quantile_head(sp_tensor, _Q_det)  # (T-burnin, N, K)
                    _q_pred = _Q_det.unsqueeze(-1) + _offsets                # (T-burnin, N, K)
                    _q_pred_st = _q_pred[:, data.station_mask, :]            # (T-burnin, n_st, K)
                    _valid = ~torch.isnan(q_obs_chunk) & ~torch.isnan(_q_pred_st[..., 0])
                    if int(_valid.sum()) > 0:
                        _y = q_obs_chunk[_valid]                             # (M,)
                        _q = _q_pred_st[_valid]                              # (M, K)
                        _taus = torch.tensor(
                            self.model.quantile_head.taus,
                            device=_q.device, dtype=_q.dtype,
                        )
                        L_q = _qloss(_y, _q, _taus)
                        loss_chunk = loss_chunk + self.loss_fn.w_quantile * L_q
                        all_components["quantile_loss"] = (
                            all_components.get("quantile_loss", 0.0) + float(L_q.detach()) * _part_bloc)

            # BLOC SANS DONNEE (2026-09-04). Si la perte de donnees n'a pas de gradient
            # (aucune station gardee dans ce bloc), le prior ajoute ci-dessous en aurait
            # un, et le pas par bloc se ferait sur le prior SEUL : mesure sur le banc de
            # sous-bassin, 23 pas sur 24 par epoque tiraient le champ vers la litterature
            # sans qu'aucune observation ne pese. Un tel bloc ne doit pas faire de pas.
            _sans_donnees = not loss_chunk.requires_grad
            # Soft physical prior regularization on spatial params (k_gw, krec, K_sat, K_c, ...).
            # Pulls toward literature targets — prevents drift toward overfit values.
            if self.config.w_prior > 0 or self.config.w_diversity > 0:
                params_t = self.model.spatial_encoder(
                    data.node_coords, data.territorial.to_tensor()
                )
                if self.config.w_prior > 0:
                    prior_loss = self.model.spatial_encoder.physical_prior_loss(params_t)
                    loss_chunk = loss_chunk + self.config.w_prior * prior_loss
                    all_components["prior"] = all_components.get("prior", 0.0) + float(prior_loss.detach()) * _part_bloc
                if self.config.w_diversity > 0:
                    div_loss = self.model.spatial_encoder.param_diversity_loss(
                        params_t, cv_target=self.config.diversity_cv_target)
                    loss_chunk = loss_chunk + self.config.w_diversity * div_loss
                    all_components["diversity"] = all_components.get("diversity", 0.0) + float(div_loss.detach()) * _part_bloc

            if self.config.w_latent_reg > 0 and getattr(self.model.spatial_encoder, "use_latent_codes", False):
                latent_loss = self.model.spatial_encoder.latent_reg()
                loss_chunk = loss_chunk + self.config.w_latent_reg * latent_loss
                all_components["latent_reg"] = all_components.get("latent_reg", 0.0) + float(latent_loss.detach()) * _part_bloc

            if self.config.w_boundary > 0:
                boundary_loss = self.model.spatial_encoder.boundary_regularization(
                    data.node_coords, data.territorial.to_tensor()
                )
                loss_chunk = loss_chunk + self.config.w_boundary * boundary_loss
                all_components["boundary"] = all_components.get("boundary", 0.0) + float(boundary_loss.detach()) * _part_bloc

            # Scale by chunk fraction so total gradient ≈ full-series gradient
            weight = _part_bloc
            if settings.get("debug.grad") and n_chunks < 3:
                # QUI POUSSE LE PLUS FORT (2026-09-05). Norme du gradient de chaque terme
                # pondere, mesuree sur les trois premiers blocs, pour savoir quel terme
                # domine la descente : la flotte du 4 septembre a diverge dans les regions
                # ou GRACE est active, tous les termes montant ensemble.
                _params = [p_ for p_ in self.model.parameters() if p_.requires_grad]
                _termes = {k_.replace("_loss", ""): v_ for k_, v_ in comps.items()
                           if torch.is_tensor(v_) and v_.requires_grad}
                for _nom, _var in (("prior", locals().get("prior_loss")),
                                   ("tws", locals().get("L_tws")),
                                   ("tws_clim", locals().get("L_tws_clim"))):
                    if torch.is_tensor(_var) and _var.requires_grad:
                        _termes[_nom] = _var
                _lignes = []
                for _nom, _v in _termes.items():
                    _w = getattr(self.loss_fn, f"w_{_nom}", None)
                    if _w is None:
                        _w = getattr(self.config, f"w_{_nom}", 1.0)
                    _g = torch.autograd.grad(_v, _params, retain_graph=True, allow_unused=True)
                    _n2 = sum(float((g_ ** 2).sum()) for g_ in _g if g_ is not None)
                    _lignes.append(f"{_nom}={float(_v):.3g} x{float(_w):g} -> grad {float(_w) * _n2 ** 0.5:.3g}")
                print(f"[grad-debug] bloc {n_chunks} (t={t_start}) : " + " | ".join(_lignes), flush=True)
                if n_chunks == 2 and settings.get("debug.grad_stop"):
                    raise SystemExit("grad-debug termine apres trois blocs")
            if getattr(self, "_sonde_mult", None) is not None and loss_chunk.requires_grad:
                # SONDE PAR TERME (2026-10-06, MEANDRE_SONDE_TERMES) : gradient de chaque terme
                # pondere de la perte par rapport a un facteur par troncon pose sur les champs
                # demandes, cumule sur l'epoque. Negatif : augmenter le champ fait baisser le
                # terme, l'optimiseur le poussera vers le haut.
                # La perte rend aussi son total parmi ses composantes : il est ecarte ici, le
                # total du bloc etant mesure a part.
                _termes = {k_.replace("_loss", ""): v_ for k_, v_ in comps.items()
                           if torch.is_tensor(v_) and v_.requires_grad and k_ not in ("total", "total_loss")}
                for _nom, _var in (("prior", locals().get("prior_loss")),
                                   ("tws", locals().get("L_tws")),
                                   ("tws_clim", locals().get("L_tws_clim")),
                                   ("nappe", locals().get("L_nappe")),
                                   ("et_niveau", locals().get("L_et_niv"))):
                    if torch.is_tensor(_var) and _var.requires_grad:
                        _termes[_nom] = _var
                _noms_m = list(self._sonde_mult)
                _facteurs = [self._sonde_mult[k_] for k_ in _noms_m]
                _seuls = [x for x in settings.get("probe.terms_only").split(",") if x]
                if _seuls:
                    _termes = {k_: v_ for k_, v_ in _termes.items() if k_ in _seuls}
                for _nom, _v in list(_termes.items()) + [("total", loss_chunk)]:
                    if _nom == "total":
                        _w = 1.0
                    else:
                        _w = getattr(self.loss_fn, f"w_{_nom}", None)
                        if _w is None:
                            _w = getattr(self.config, f"w_{_nom}", 1.0)
                    if float(_w) == 0.0:
                        continue
                    _g = torch.autograd.grad(_v * float(_w) * weight, _facteurs, retain_graph=True, allow_unused=True)
                    _acc = self._sonde_acc.setdefault(_nom, {k_: torch.zeros_like(f_) for k_, f_ in zip(_noms_m, _facteurs)})
                    for k_, g_ in zip(_noms_m, _g):
                        if g_ is not None:
                            _acc[k_] += torch.nan_to_num(g_.detach())
            if not torch.isnan(loss_chunk) and loss_chunk.requires_grad:
                # FILET PAR BLOC (2026-09-04). Le pas d'optimisation est UNIQUE par
                # epoque, sur les gradients accumules de tous les blocs. Un seul bloc
                # dont la retropropagation rend inf ou NaN empoisonne donc la somme, et
                # le nettoyage d'avant le pas mettait a ZERO le gradient de tout le champ
                # spatial : il n'apprenait rien de l'epoque, silencieusement (validation
                # identique a la quatrieme decimale d'une epoque a l'autre). On prend une
                # copie des gradients avant le bloc ; si le bloc les rend non finis, on
                # la restaure et on compte le bloc comme jete, au lieu de perdre l'epoque.
                _instantane = {p_: (p_.grad.detach().clone() if p_.grad is not None else None)
                               for p_ in self.model.parameters() if p_.requires_grad}
                if settings.get("debug.nan_level") == "2":
                    # NIVEAU 2 SEULEMENT (2026-09-09). Ce bloc interrompt la
                    # retropropagation des qu'un NaN parait, si bien que les gradients ne
                    # deviennent jamais non finis et que l'autopsie PAR TERME, qui vit plus
                    # bas et se declenche sur ce critere, ne s'executait jamais. Les deux
                    # niveaux sont donc separes : 1 pour l'autopsie par terme a vitesse
                    # normale, 2 pour la trace d'operation, lente et exclusive.
                    # finis, le gradient non ; le mode anomalie de torch nomme l'operation
                    # dont la retropropagation produit le premier NaN, avec sa pile.
                    try:
                        with torch.autograd.detect_anomaly(check_nan=True):
                            (loss_chunk * weight).backward(retain_graph=True)
                    except RuntimeError as _e:
                        import traceback as _tb
                        print("[nan-debug] operation fautive :\n" + "".join(_tb.format_exc())[-3000:],
                              flush=True)
                        _noms = [n_ for n_, p_ in self.model.named_parameters()
                                 if p_.grad is not None and not torch.isfinite(p_.grad).all()]
                        print(f"[nan-debug] parametres a gradient non fini ({len(_noms)}) : {_noms[:15]}",
                              flush=True)
                else:
                    (loss_chunk * weight).backward()
                if settings.get("probe.conflict"):
                    # SONDE DES CONFLITS (2026-10-14) : gradient de la perte de chaque bloc par
                    # rapport a la sortie brute du champ (troncons x sorties), sans pas
                    # d'optimisation ; arret apres MEANDRE_SONDE_CONFLIT_BLOCS blocs. Les bassins
                    # etant independants, la part de chaque bassin s'en deduit exactement.
                    _se = self.model.spatial_encoder
                    _gs = [t_.grad for t_ in getattr(_se, "brut_capture", []) if t_.grad is not None]
                    _se.brut_capture = []
                    _g = torch.stack(_gs).sum(0) if _gs else None
                    _liste = getattr(self, "_sonde_conflit", [])
                    if _g is not None:
                        _liste.append(_g.detach().float().cpu().numpy())
                    self._sonde_conflit = _liste
                    _nrm = [f"{float(t_.norm()):.2e}" for t_ in _gs]
                    print(f"[sonde-conflit] bloc {n_chunks} (t={t_start}) : perte {float(loss_chunk):.4f} | {len(_gs)} appels du champ, normes {_nrm}", flush=True)
                    if len(_liste) >= int(settings.get("probe.conflict_blocks")):
                        import numpy as _np_c
                        _np_c.save(settings.get("probe.conflict"), _np_c.stack(_liste))
                        print(f"[sonde-conflit] {len(_liste)} blocs ecrits dans {settings.get('probe.conflict')}", flush=True)
                        os._exit(0)
                    self.optimizer.zero_grad()
                if settings.get("probe.segment") and n_chunks >= 1:
                    # SONDE DU RECALCUL PAR SEGMENTS : gradient et memoire du deuxieme bloc (le
                    # premier porte la mise en regime), puis arret. Deux executions, avec et sans
                    # MEANDRE_SEGMENT_GRADIENT, doivent rendre la meme empreinte du gradient.
                    _gs = [p_.grad.detach().double().flatten() for p_ in self.model.parameters() if p_.grad is not None]
                    _gv = torch.cat(_gs) if _gs else torch.zeros(1, dtype=torch.float64)
                    _pic = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
                    print(f"[sonde-segment] bloc de {chunk_len} jours, segments {settings.get('compute.gradient_segments')} | perte {float(loss_chunk):.8f} | norme du gradient {float(_gv.norm()):.8e} | somme {float(_gv.sum()):.8e} | pic memoire {_pic:.2f} Go | duree du bloc {_time.perf_counter() - _t_bloc:.1f} s", flush=True)
                    raise SystemExit(0)
                _pourri = any(p_.grad is not None and not torch.isfinite(p_.grad).all()
                              for p_ in _instantane)
                if _pourri and settings.get("debug.nan_level") == "1":
                    # AVANT restauration : quels parametres, NaN ou inf, et quel TERME de
                    # la perte porte le gradient non fini (le detecteur d'anomalie ne voit
                    # que les NaN ; un inf de debordement passe).
                    _touches = []
                    for n_, p_ in self.model.named_parameters():
                        if p_.grad is not None and not torch.isfinite(p_.grad).all():
                            _g = p_.grad
                            _touches.append(f"{n_} nan={int(torch.isnan(_g).sum())} inf={int(torch.isinf(_g).sum())}"
                                            f" maxfini={float(_g[torch.isfinite(_g)].abs().max()) if torch.isfinite(_g).any() else float('nan'):.3g}")
                    print(f"[nan-debug] bloc {n_chunks} (t={t_start}) perte={float(loss_chunk):.4g} | "
                          f"{len(_touches)} parametre(s) touche(s) : " + " ; ".join(_touches[:10]), flush=True)
                    _params = [p_ for p_ in self.model.parameters() if p_.requires_grad]
                    _termes = {k_.replace("_loss", ""): v_ for k_, v_ in comps.items()
                               if torch.is_tensor(v_) and v_.requires_grad}
                    for _nom, _var in (("prior", locals().get("prior_loss")), ("tws", locals().get("L_tws")),
                                       ("tws_clim", locals().get("L_tws_clim"))):
                        if torch.is_tensor(_var) and _var.requires_grad:
                            _termes[_nom] = _var
                    for _nom, _v in _termes.items():
                        try:
                            _g = torch.autograd.grad(_v, _params, retain_graph=True, allow_unused=True)
                            _fini = all(torch.isfinite(g_).all() for g_ in _g if g_ is not None)
                            _n2 = sum(float((g_[torch.isfinite(g_)] ** 2).sum()) for g_ in _g if g_ is not None)
                            print(f"[nan-debug]   terme {_nom:10s} valeur {float(_v):.4g} gradient fini={_fini} "
                                  f"norme finie {_n2 ** 0.5:.3g}", flush=True)
                        except RuntimeError as _e:
                            print(f"[nan-debug]   terme {_nom:10s} : {str(_e)[:120]}", flush=True)
                if _pourri:
                    for p_, g_ in _instantane.items():
                        p_.grad = g_
                    _n_jetes = getattr(self, "_n_blocs_jetes", 0) + 1
                    self._n_blocs_jetes = _n_jetes
                    if _n_jetes <= 3 or _n_jetes % 20 == 0:
                        print(f"[train] bloc {n_chunks} (t={t_start}) : gradient non fini, "
                              f"bloc JETE, gradients restaures ({_n_jetes} bloc(s) jete(s) "
                              f"jusqu'ici)", flush=True)
                    if settings.get("debug.nan_level") == "1":
                        # Autopsie du bloc (2026-09-05) : quels parametres portent le
                        # gradient non fini, et la perte elle-meme etait-elle finie ?
                        _noms = [n_ for n_, p_ in self.model.named_parameters()
                                 if p_.grad is not None and not torch.isfinite(p_.grad).all()]
                        print(f"[nan-debug] perte={float(loss_chunk):.4g} finie="
                              f"{bool(torch.isfinite(loss_chunk))} | q_sim non finis="
                              f"{int((~torch.isfinite(Q_chunk_loss)).sum())} sur {Q_chunk_loss.numel()}"
                              f" | parametres touches ({len(_noms)}) : {_noms[:12]}", flush=True)
                del _instantane
                # UN PAS PAR BLOC (2026-09-04, MEANDRE_PAS_PAR_BLOC=1). Le trainer ne
                # faisait qu'UN pas d'optimisation par epoque, sur les gradients
                # accumules de tous les blocs (plein lot), avec un rechauffement du taux
                # sur cinq epoques : trente epoques valaient trente pas d'Adam dont les
                # cinq premiers presque nuls, et le point de reprise apres une epoque
                # etait IDENTIQUE a l'initialisation. Un champ spatial n'apprend pas en
                # trente pas. Ici, un pas apres chaque bloc (la troncature du gradient
                # habituelle) : soixante-cinq pas par epoque au lieu d'un.
                if _pas_par_bloc and _sans_donnees:
                    # Gradient du prior seul : jete, pas de pas (voir plus haut).
                    self.optimizer.zero_grad()
                    _n_vides = getattr(self, "_n_blocs_sans_donnees", 0) + 1
                    self._n_blocs_sans_donnees = _n_vides
                    if _n_vides <= 2:
                        print(f"[train] bloc {n_chunks} (t={t_start}) : aucune donnee, pas "
                              f"d'optimisation SAUTE ({_n_vides} bloc(s) jusqu'ici)", flush=True)
                elif _pas_par_bloc and not _pourri:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            elif not loss_chunk.requires_grad and loss_chunk.item() == 0.0:
                n_no_grad = getattr(self, "_n_no_grad_chunks", 0) + 1
                self._n_no_grad_chunks = n_no_grad
                if n_no_grad == 1:
                    logger.warning(
                        "Loss chunk has no gradient (no valid observations). "
                        "Check that observation dates overlap with the training period."
                    )

            total_loss = total_loss + loss_chunk.detach() * weight
            for k, v in comps.items():
                all_components[k] = all_components.get(k, 0.0) + float(v.detach()) * weight
            n_chunks += 1
            _t_bloc = _time.perf_counter()

            # Detach state for next chunk (TBPTT across chunks)
            state = state_out.detach()
            obs_offset += chunk_len
            t_start = t_end

            # Free chunk memory
            del Q_chunk, loss_chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Cache end-of-train state so _val_epoch can start from
        # the realistic state (after train period forward pass) instead of
        # re-simulating from spinup.  Saves ~50 min/epoch on GPU.
        self._cached_train_end_state = state.detach()

        # Warn if all chunks had no gradient (no obs overlap)
        n_no_grad = getattr(self, "_n_no_grad_chunks", 0)
        if n_no_grad == n_chunks and n_chunks > 0:
            logger.warning(
                "ALL %d chunks in this epoch had zero loss with no gradient. "
                "The model will NOT learn. Verify that observations cover the "
                "training period.",
                n_chunks,
            )
        self._n_no_grad_chunks = 0  # reset for next epoch

        # Clip and step (once, on accumulated gradients)
        # First, surgically replace NaN/Inf gradients with zeros so valid
        # gradients from other parameters still contribute to the update.
        nan_params = []
        for name, p in self.model.named_parameters():
            if p.grad is not None and (p.grad.isnan().any() or p.grad.isinf().any()):
                n_nan = p.grad.isnan().sum().item()
                n_inf = p.grad.isinf().sum().item()
                nan_params.append(f"{name}({n_nan}nan,{n_inf}inf)")
                p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
        if nan_params:
            logger.warning(
                f"NaN/Inf gradients in {len(nan_params)} params — "
                f"zeroed: {nan_params[:5]}"
            )
        if not _pas_par_bloc:
            total_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )
            self.optimizer.step()

        if getattr(self, "_sonde_mult", None) is not None:
            import numpy as _np_s
            _out = settings.get("probe.terms_output")
            _np_s.savez_compressed(_out, **{f"{t_}__{k_}": g_.cpu().numpy() for t_, d_ in self._sonde_acc.items()
                                         for k_, g_ in d_.items()})
            print(f"[sonde] gradient par terme, somme sur le domaine (negatif : le champ sera pousse vers le haut) -> {_out}", flush=True)
            for t_, d_ in sorted(self._sonde_acc.items()):
                print(f"[sonde]   {t_:14s} " + "  ".join(f"{k_} {float(g_.sum()):+.3e}" for k_, g_ in d_.items()), flush=True)
            self.model.spatial_encoder.multiplicateurs = None
            self._sonde_mult = None
        comp_tensors = {k: torch.tensor(v) for k, v in all_components.items()}
        return total_loss, comp_tensors

    def _train_epoch_single(self, data: TrainingData) -> tuple[Tensor, dict[str, Tensor]]:
        """Original single-pass training (no chunking)."""
        with torch.amp.autocast("cuda", dtype=self._amp_dtype, enabled=self._use_amp):
            Q_sim, final_state = self._simulate(
                data, data.train_slice, tbptt_steps=self.config.tbptt_steps,
            )
            # Cache for fast val_epoch (see chunked path).
            self._cached_train_end_state = final_state.detach()

            n_train = data.train_slice.stop - data.train_slice.start
            q_obs_train = data.q_obs[:n_train]

            # ── Multi-objective ETR wiring (MOD16A2) ─────────────────
            # When data.et_obs is populated (MODIS loader ran) and w_et > 0,
            # we need et_sim from the model diagnostics. We re-run simulate with
            # return_diagnostics=True only if the extra loss weight justifies it.
            et_sim = None
            need_et = self.loss_fn.w_et > 0 and data.et_obs is not None
            if need_et:
                # Re-run the train forward with diagnostics. Uses the cached
                # spinup state so only the train period is re-simulated.
                cached_state_for_et = getattr(self, "_cached_train_end_state", None)
                # Use the spinup state (beginning of train) if not cached yet
                if not hasattr(self, "_cached_spinup_state_for_et"):
                    self._cached_spinup_state_for_et = getattr(
                        self, "_cached_spinup_state", None
                    )
                with torch.no_grad():
                    _initial = (self._cached_spinup_state_for_et
                                if self._cached_spinup_state_for_et is not None
                                else None)
                    if _initial is not None:
                        _result = self.model.simulate(
                            forcing=data.forcing[data.train_slice],
                            initial_state=_initial,
                            graph=data.graph,
                            node_coords=data.node_coords,
                            territorial=data.territorial,
                            withdrawals=data.withdrawals.slice(data.train_slice.start, data.train_slice.stop),
                            day_of_year=data.day_of_year[data.train_slice],
                            return_diagnostics=True,
                            tbptt_steps=self.config.tbptt_steps,
                        )
                        if isinstance(_result, tuple) and len(_result) == 3:
                            _, _, _diag = _result
                            et_sim = _diag.etr  # (T, n_nodes) mm/day

            _et_obs_train = data.et_obs[:n_train] if data.et_obs is not None else None
            if et_sim is not None and _et_obs_train is not None:
                et_sim, _et_obs_train = self._center_et(et_sim, _et_obs_train, data, t0=data.train_slice.start)

            loss, components = self.loss_fn(
                q_obs=q_obs_train,
                q_sim=Q_sim,
                station_mask=data.station_mask,
                et_obs=_et_obs_train,
                et_sim=et_sim,
            )

        # Soft physical prior regularization
        if self.config.w_prior > 0 or self.config.w_diversity > 0:
            params_t = self.model.spatial_encoder(
                data.node_coords, data.territorial.to_tensor()
            )
            if self.config.w_prior > 0:
                prior_loss = self.model.spatial_encoder.physical_prior_loss(params_t)
                loss = loss + self.config.w_prior * prior_loss
                components["prior"] = prior_loss.detach()
            if self.config.w_diversity > 0:
                div_loss = self.model.spatial_encoder.param_diversity_loss(
                    params_t, cv_target=self.config.diversity_cv_target)
                loss = loss + self.config.w_diversity * div_loss
                components["diversity"] = div_loss.detach()

        if self.config.w_boundary > 0:
            boundary_loss = self.model.spatial_encoder.boundary_regularization(
                data.node_coords, data.territorial.to_tensor()
            )
            loss = loss + self.config.w_boundary * boundary_loss
            components["boundary"] = boundary_loss.detach()

        if torch.isnan(loss):
            logger.warning(
                "NaN loss detected — skipping backward. "
                f"Components: { {k: float(v) for k, v in components.items()} }"
            )
            self.optimizer.zero_grad()
            return loss.detach(), {k: v.detach() for k, v in components.items()}

        loss.backward()

        # Surgical NaN cleanup: zero only affected params, keep valid gradients
        nan_params = []
        for name, p in self.model.named_parameters():
            if p.grad is not None and (p.grad.isnan().any() or p.grad.isinf().any()):
                n_nan = p.grad.isnan().sum().item()
                n_inf = p.grad.isinf().sum().item()
                nan_params.append(f"{name}({n_nan}nan,{n_inf}inf)")
                p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
        if nan_params:
            logger.warning(
                f"NaN/Inf gradients in {len(nan_params)} params — "
                f"zeroed: {nan_params[:5]}"
            )

        total_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.grad_clip
        )
        self.optimizer.step()

        return loss.detach(), {k: v.detach() for k, v in components.items()}

    def _val_epoch(self) -> dict[str, float]:
        """Validation, plus la PERTE DE DEBIT sur la periode de validation (cle « loss »).

        Le choix du meilleur point de reprise par `best_metric = "val_loss"` lisait une cle
        « loss » que la validation ne remplissait jamais (2026-10-01) : seul le KGE de
        validation pouvait choisir, et le terme d'etiage, present dans la perte
        d'entrainement, ne jouait aucun role dans le modele retenu. On calcule ici les
        termes de debit de la perte, avec ses poids, sur la serie de validation entiere.
        """
        out = self._val_epoch_metriques()
        q = getattr(self, "_val_series", None)
        if q is not None and self.loss_fn is not None:
            out["loss"] = self._perte_debit_validation(*q)
        return out

    def _perte_debit_validation(self, q_obs_val: Tensor, q_sim_val: Tensor) -> float:
        """Termes de debit de la perte, memes poids, par station puis moyennes ponderees."""
        from meandre.training.loss import (differentiable_kge_loss, differentiable_pbias_loss,
                                           differentiable_log_mse_loss, differentiable_etiage_loss,
                                           differentiable_fdc_bas_loss)
        lf = self.loss_fn
        termes = []
        for s in range(q_sim_val.shape[1]):
            v = ~torch.isnan(q_obs_val[:, s]) & ~torch.isnan(q_sim_val[:, s])
            if int(v.sum()) < 30:
                continue
            qo, qs = q_obs_val[v, s].float(), q_sim_val[v, s].float()
            t = getattr(lf, "w_kge", 0.0) * differentiable_kge_loss(qo, qs)
            t = t + getattr(lf, "w_pbias", 0.0) * differentiable_pbias_loss(qo, qs)
            if getattr(lf, "w_log_mse", 0.0) > 0:
                t = t + lf.w_log_mse * differentiable_log_mse_loss(qo, qs)
            if getattr(lf, "w_etiage", 0.0) > 0:
                t = t + lf.w_etiage * differentiable_etiage_loss(qo, qs)
            if getattr(lf, "w_fdc_bas", 0.0) > 0:
                t = t + lf.w_fdc_bas * differentiable_fdc_bas_loss(qo, qs)
            termes.append(float(t))
        return float(sum(termes) / len(termes)) if termes else float("inf")

    def _val_epoch_metriques(self) -> dict[str, float]:
        _etat_continu_val = settings.get("trainer.state_continuous")
        """Validation: simulate val period, compute evaluation metrics.

        Reports both pooled metrics (all stations flattened) and per-station
        weighted KGE/NSE consistent with the training loss function.

        Protocol (corrected): cold-start spinup, then **continuous** forward
        pass from end-of-spinup all the way through to the end of val.  The
        previous protocol jumped from end-of-spinup directly to val_slice,
        skipping any intermediate years — that produces a phantom KGE
        because the model state at start of val never sees the actual
        history (e.g. 18 years of train period).  The val metric reported
        here matches what an independent eval script (`eval_test.py`)
        would compute on the same checkpoint.

        The training cache (`_cached_spinup_state`) is preserved so the
        next train epoch still benefits from warm spinup.
        """
        self.model.eval()
        data = self.val_data

        saved_cache = getattr(self, "_cached_spinup_state", None)
        self._cached_spinup_state = None
        try:
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=self._amp_dtype, enabled=self._use_amp):
                # Fast path: reuse end-of-train state cached by _train_epoch.
                # This is the state the model would have if simulated continuously
                # from cold start through the train period — exactly what we need
                # at the start of val.  Saves a full 22-year forward pass.
                # Slight inconsistency: the cached state was produced under the
                # weights BEFORE this epoch's optimizer.step().  For early epochs
                # with large weight updates this could lag; for later epochs the
                # difference is negligible.
                cached_state = getattr(self, "_cached_train_end_state", None)

                if cached_state is not None:
                    # VALIDATION SANS NEIGE (2026-09-04). L'etat mis en cache est celui
                    # de la fin de l'entrainement, la veille du debut de la validation ;
                    # mais simulate() REFABRIQUAIT l'etat riche de la colonne, donc la
                    # validation demarrait sans manteau et ratait la crue : sur un
                    # sous-bassin gaspesien le trainer notait 0.399 (beta 0.62) un modele
                    # qui vaut 0.829 en simulation continue. Ce chiffre faux choisissait le
                    # meilleur point de reprise et pilotait l'autopilote. Meme correctif
                    # que pour les blocs (R64) : l'etat riche traverse la frontiere.
                    Q_sim, _ = self.model.simulate(
                        forcing=data.forcing[data.val_slice],
                        initial_state=cached_state,
                        graph=data.graph,
                        node_coords=data.node_coords,
                        territorial=data.territorial,
                        withdrawals=data.withdrawals.slice(data.val_slice.start, data.val_slice.stop),
                        day_of_year=data.day_of_year[data.val_slice],
                        tbptt_steps=self.config.tbptt_steps,
                        poursuivre_etat=_etat_continu_val,
                    )
                else:
                    # Slow path: no cached state (e.g. before first train epoch).
                    # Cold spinup + continuous forward through train + val.
                    initial_state = self._run_spinup(data)
                    spinup_end = min(self.config.spinup_steps, data.val_slice.start)
                    full_slice = slice(spinup_end, data.val_slice.stop)
                    # La mise en regime vient de laisser l'etat riche de la colonne au
                    # jour spinup_end : la simulation qui suit doit le CONSERVER.
                    Q_full, _ = self.model.simulate(
                        forcing=data.forcing[full_slice],
                        initial_state=initial_state,
                        poursuivre_etat=(_etat_continu_val and spinup_end > 0),
                        graph=data.graph,
                        node_coords=data.node_coords,
                        territorial=data.territorial,
                        withdrawals=data.withdrawals.slice(full_slice.start, full_slice.stop),
                        day_of_year=data.day_of_year[full_slice],
                        tbptt_steps=self.config.tbptt_steps,
                    )
                    val_offset = data.val_slice.start - spinup_end
                    Q_sim = Q_full[val_offset:]
                    del Q_full
        finally:
            self._cached_spinup_state = saved_cache

        # q_obs is pre-sliced to start at the beginning of the val period.
        n_val = data.val_slice.stop - data.val_slice.start
        q_obs_val = data.q_obs[:n_val]  # (T_val, n_stations)

        q_sim_at_stations = Q_sim[:, data.station_mask]
        # Series gardees pour la perte de debit de validation (_val_epoch).
        self._val_series = (q_obs_val.detach(), q_sim_at_stations.detach())

        # ── Per-station KGE (consistent with per_station loss) ────────
        from meandre.utils.metrics import kge as _kge_fn
        n_stations = q_sim_at_stations.shape[1]
        kge_per = []
        nse_per = []
        for s in range(n_stations):
            q_o_s = q_obs_val[:, s]
            q_s_s = q_sim_at_stations[:, s]
            valid_s = ~torch.isnan(q_o_s) & ~torch.isnan(q_s_s)
            if valid_s.sum() < 30:
                continue
            kge_per.append(float(_kge_fn(q_o_s[valid_s], q_s_s[valid_s])))
            nse_per.append(float(nse(q_o_s[valid_s], q_s_s[valid_s])))

        if kge_per:
            # Weighted by station_weights if available (same as loss function)
            if (self.loss_fn is not None
                    and hasattr(self.loss_fn, 'station_weights')
                    and self.loss_fn.station_weights is not None
                    and len(self.loss_fn.station_weights) == n_stations):
                sw = self.loss_fn.station_weights[:len(kge_per)]
                w = sw.cpu().numpy()
                w = w / w.sum()
            else:
                w = torch.full((len(kge_per),), 1.0 / len(kge_per)).numpy()
            kge_vals = torch.tensor(kge_per)
            nse_vals = torch.tensor(nse_per)
            kge_weighted = float((kge_vals * torch.from_numpy(w)).sum())
            nse_weighted = float((nse_vals * torch.from_numpy(w)).sum())
            kge_median = float(kge_vals.median())
        else:
            kge_weighted = float("nan")
            nse_weighted = float("nan")
            kge_median = float("nan")

        # ── Pooled metrics (all stations flattened) ───────────────────
        q_o = q_obs_val.reshape(-1)
        q_s = q_sim_at_stations.reshape(-1)
        valid = ~torch.isnan(q_o) & ~torch.isnan(q_s)
        q_o, q_s = q_o[valid], q_s[valid]

        if q_o.numel() == 0:
            return {"nse": float("nan"), "kge": float("nan"), "pbias": float("nan"),
                    "rmse": float("nan"), "nrmse": float("nan"), "mae": float("nan"),
                    "r": float("nan"), "beta": float("nan"), "gamma": float("nan"),
                    "r_log": float("nan"), "beta_log": float("nan"), "gamma_log": float("nan"),
                    "kge_log": float("nan"),
                    "kge_station": kge_weighted, "nse_station": nse_weighted,
                    "kge_median": kge_median}

        kge_info = kge_components(q_o, q_s)

        # ── Probabilistic calibration metrics ──────────────────────
        # Well-calibrated model: cov_90 ≈ 0.90, cov_50 ≈ 0.50.
        calibration = {}
        # ── Calibration mode quantile (Phase 2 v2) ──────────────────────
        # cov_50, cov_90 par appartenance dans les intervalles inter-quantiles
        # en m³/s ; val_nll remplacé par la pinball moyenne ; CRPS ≈ 2·pinball.
        if (self.loss_fn.w_quantile > 0
                and getattr(self.model, "use_quantile_head", False)
                and hasattr(self.model, "quantile_head")
                and Q_sim is not None):
            from meandre.training.loss import quantile_loss as _qloss, crps_from_quantiles as _crps
            with torch.no_grad():
                sp_q = self.model.spatial_encoder(
                    data.node_coords, data.territorial.to_tensor()
                )
                # La tete traite chaque troncon independamment : on ne l'evalue qu'aux
                # stations. Sur tout le Quebec (27 834 troncons, trois ans), l'enveloppe
                # complete demandait plus de 7 Go et saturait la carte (2026-10-18).
                _m = data.station_mask
                Q_st = Q_sim.detach()[:, _m]
                offsets_st = self.model.quantile_head(sp_q.to_tensor()[_m], Q_st)
            q_pred_st = Q_st.unsqueeze(-1) + offsets_st           # (T, n_st, K)
            taus = list(self.model.quantile_head.taus)
            valid_q = ~torch.isnan(q_obs_val) & ~torch.isnan(q_pred_st[..., 0])
            # Couvertures par intervalles inter-quantiles (en m³/s, sans hypothèse)
            for level, lo_tau, hi_tau in [(50, 0.25, 0.75), (90, 0.05, 0.95)]:
                if lo_tau in taus and hi_tau in taus:
                    i_lo = taus.index(lo_tau); i_hi = taus.index(hi_tau)
                    lo = q_pred_st[..., i_lo]; hi = q_pred_st[..., i_hi]
                    in_int = (q_obs_val >= lo) & (q_obs_val <= hi)
                    cov = (in_int & valid_q).sum().float() / valid_q.sum().clamp(min=1).float()
                    calibration[f"cov_{level}"] = float(cov)
            # Pinball moyenne + CRPS (en m³/s)
            y_v = q_obs_val[valid_q]; qp_v = q_pred_st[valid_q]
            taus_t = torch.tensor(taus, device=qp_v.device, dtype=qp_v.dtype)
            calibration["val_nll"] = float(_qloss(y_v, qp_v, taus_t))  # = best_metric
            calibration["val_crps"] = float(_crps(y_v, qp_v, taus_t))

        return {
            "nse": float(nse(q_o, q_s)),
            "kge": float(kge_info["kge"]),
            "pbias": float(pbias(q_o, q_s)),
            "rmse": float(rmse(q_o, q_s)),
            "nrmse": float(nrmse(q_o, q_s)),
            "mae": float(mae(q_o, q_s)),
            "log_nse": float(log_nse(q_o, q_s)),
            "r": float(kge_info["r"]),
            "beta": float(kge_info["beta"]),
            "gamma": float(kge_info["gamma"]),
            "r_log": float(kge_info["r_log"]),
            "beta_log": float(kge_info["beta_log"]),
            "gamma_log": float(kge_info["gamma_log"]),
            "kge_log": float(kge_info["kge_log"]),
            "kge_station": kge_weighted,
            "nse_station": nse_weighted,
            "kge_median": kge_median,
            **calibration,
        }

    def _water_balance_check(self, data: TrainingData, period: str = "train") -> None:
        """Log basin-average water balance for diagnostics.

        Runs a 365-day diagnostic simulation and prints:
        P, ETR, lateral, recharge, baseflow, deltaStorage, residual (mm/yr).
        """
        self.model.eval()
        sl = data.train_slice if period == "train" else data.val_slice
        # Use LAST 365 days of the period to show steady-state behavior
        n_days = min(365, sl.stop - sl.start)
        diag_sl = slice(sl.stop - n_days, sl.stop)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=self._amp_dtype, enabled=self._use_amp):
            initial_state = self._run_spinup(data)
            result = self.model.simulate(
                forcing=data.forcing[diag_sl],
                initial_state=initial_state,
                graph=data.graph,
                node_coords=data.node_coords,
                territorial=data.territorial,
                withdrawals=data.withdrawals.slice(diag_sl.start, diag_sl.stop),
                day_of_year=data.day_of_year[diag_sl],
                return_diagnostics=True,
            )
            Q_sim, final_state, diag = result

        # Basin-average fluxes (mm/day → mm/yr)
        scale = 365.0 / n_days
        n = data.forcing.shape[1]  # n_nodes
        P_total = data.forcing[diag_sl, :, 0].sum(dim=0).mean().item() * scale
        etr = diag.etr.sum(dim=0).mean().item() * scale
        lateral = diag.lateral_mm.sum(dim=0).mean().item() * scale
        recharge = diag.recharge.sum(dim=0).mean().item() * scale
        baseflow = diag.q_baseflow.sum(dim=0).mean().item() * scale

        # Storage change in mm: soil theta (m³/m³) × layer depth → mm,
        # plus SWE, canopy, wetland, S_gw (already mm).
        # Exclude t_soil and T_water (temperatures, not storage).
        def _storage_mm(st):
            soil_mm = (st.theta1 * 300 + st.theta2 * 700 + st.theta3 * 1000)  # mm
            return (soil_mm + st.swe + st.canopy_storage
                    + st.wetland_storage + st.S_gw).mean().item()
        delta_s = (_storage_mm(final_state) - _storage_mm(initial_state)) * scale

        residual = P_total - etr - lateral - delta_s
        logger.info(
            f"  Water balance ({n_days}d, basin avg, mm/yr): "
            f"P={P_total:.0f} ETR={etr:.0f} lateral={lateral:.0f} "
            f"recharge={recharge:.0f} baseflow={baseflow:.0f} "
            f"dS={delta_s:.0f} residual={residual:.0f}"
        )

        # Free diagnostic memory
        del diag, Q_sim
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _run_autopilot(
        self,
        epoch: int,
        val_metrics: dict[str, float],
        epochs_without_improvement: int,
    ) -> None:
        """Autopilot: automatic hyperparameter adjustments based on val metrics.

        Called after each validation epoch when autopilot is enabled.
        Actions are logged and conservative — they only intervene on clear
        signals (beta/gamma drift, LR plateau, regression from best).
        """
        cfg = self.config
        beta = val_metrics.get("beta", 1.0)
        gamma = val_metrics.get("gamma", 1.0)
        kge_sta = val_metrics.get("kge_station", 0.0)

        # ── Grace period ────────────────────────────────────────────
        if epoch < cfg.autopilot_grace_epochs:
            logger.debug(
                f"  [autopilot] epoch={epoch} < grace={cfg.autopilot_grace_epochs}"
                f" — skipping (beta={beta:.3f} gamma={gamma:.3f} kge_sta={kge_sta:.4f})"
            )
            return

        # ── 1. Dérive de β / γ (sert au redémarrage intelligent) ─────
        beta_drift = abs(beta - 1.0)
        gamma_drift = abs(gamma - 1.0)
        actions: list[str] = []

        # ── 2. ReduceLROnPlateau ─────────────────────────────────────
        # Bug fix: ne réduire qu'UNE FOIS par plateau de lr_patience epochs.
        # Après une réduction, on attend lr_patience epochs SUPPLÉMENTAIRES
        # avant de réduire à nouveau.  Sans ça, LR est divisé à chaque epoch
        # dès que no_improve dépasse le seuil.
        if not hasattr(self, "_ap_last_lr_reduce_epoch"):
            self._ap_last_lr_reduce_epoch = -cfg.autopilot_lr_patience - 1

        epochs_since_last_reduce = epoch - self._ap_last_lr_reduce_epoch
        if (epochs_without_improvement >= cfg.autopilot_lr_patience
                and epochs_since_last_reduce >= cfg.autopilot_lr_patience):
            self._ap_lr_plateau_count += 1
            self._ap_last_lr_reduce_epoch = epoch
            for pg in self.optimizer.param_groups:
                new_lr = max(pg["lr"] * cfg.autopilot_lr_factor, cfg.autopilot_lr_min)
                pg["lr"] = new_lr
            actions.append(
                f"LR plateau ({epochs_without_improvement} epochs no improve) "
                f"→ LR ×{cfg.autopilot_lr_factor} → {self.optimizer.param_groups[0]['lr']:.2e}"
            )

        # ── 3. Smart restart on regression + drift ───────────────────
        # Note: regression is meaningful only for higher-is-better metrics.
        # When best_metric is loss-like (lower-is-better) the formula is
        # inverted; we just disable smart restart in that case and rely on
        # LR plateau detection above.
        if self._best_metric_lower_is_better:
            regression = -1.0  # never triggers
        else:
            # Comparer la MÊME métrique que celle suivie par best_metric (poolé
            # par défaut). Utiliser kge_station ici était un bug : elle vit sur
            # une autre échelle (par-station ~0.57 vs best poolé ~0.77), ce qui
            # rapportait une « régression » spurious ~26 % à chaque epoch et
            # déclenchait de faux restarts dès que gamma dérivait aussi.
            bm_key = "val_nll" if cfg.best_metric == "nll" else cfg.best_metric
            kge_current = val_metrics.get(bm_key, kge_sta)
            regression = (self._best_val_metric - kge_current) / (abs(self._best_val_metric) + 1e-8)
        drift_detected = (beta_drift > cfg.autopilot_beta_threshold
                          or gamma_drift > cfg.autopilot_gamma_threshold)

        if (regression > cfg.autopilot_restart_regression
                and drift_detected
                and epochs_without_improvement >= cfg.autopilot_restart_min_no_improve
                and self._ap_restart_count < cfg.autopilot_restart_max
                and self.checkpoint_path):
            self._ap_restart_count += 1
            self.model.load(self.checkpoint_path)
            for pg in self.optimizer.param_groups:
                pg["lr"] = max(pg["lr"] * 0.5, cfg.autopilot_lr_min)
            # Reset cached spinup state (model weights changed)
            self._cached_spinup_state = None
            actions.append(
                f"RESTART {self._ap_restart_count}/{cfg.autopilot_restart_max} "
                f"regression={regression:.1%} + drift → reload best, "
                f"LR → {self.optimizer.param_groups[0]['lr']:.2e}"
            )

        # ── 4. Phase 1 → phase 2 : auto-unfreeze spatial encoder ──────
        # One-shot. After the NLL stack converged on a frozen NeRF (phase 1),
        # unfreeze and reduce LR (phase 2 fine-tune). Replaces the manual
        # two-config Kendall-Gal recipe.
        if (cfg.autopilot_unfreeze_spatial_epoch is not None
                and not getattr(self, "_ap_unfroze_spatial", False)
                and epoch >= cfg.autopilot_unfreeze_spatial_epoch):
            min_kge = cfg.autopilot_unfreeze_spatial_min_kge
            if min_kge is None or kge_sta >= min_kge:
                spatial = self.model.spatial_encoder
                n_unfrozen = 0
                for p in spatial.parameters():
                    if not p.requires_grad:
                        p.requires_grad = True
                        n_unfrozen += p.numel()
                # Scale LR for phase 2 fine-tuning
                lr_factor = cfg.autopilot_unfreeze_spatial_lr_factor
                for pg in self.optimizer.param_groups:
                    pg["lr"] = max(pg["lr"] * lr_factor, cfg.autopilot_lr_min)
                # Reset plateau counters so phase 2 gets fresh patience
                self._ap_last_lr_reduce_epoch = epoch
                self._ap_unfroze_spatial = True
                actions.append(
                    f"PHASE 2: unfreeze spatial encoder ({n_unfrozen} params) "
                    f"at epoch={epoch} (kge_sta={kge_sta:.4f}"
                    + (f" ≥ {min_kge}" if min_kge is not None else "")
                    + f"), LR × {lr_factor} → {self.optimizer.param_groups[0]['lr']:.2e}"
                )

        # ── Log autopilot actions ─────────────────────────────────────
        if actions:
            for a in actions:
                msg = f"  [autopilot] {a}"
                print(msg, flush=True)
                logger.info(msg)
        else:
            # Brief status when no action taken
            logger.debug(
                f"  [autopilot] epoch={epoch} beta={beta:.3f} gamma={gamma:.3f} "
                f"kge_sta={kge_sta:.4f} — no action"
            )

        # Track previous kge_sta for trend detection
        self._ap_prev_kge_sta = kge_sta

    def _start_run(self) -> None:
        if self.run_logger is None:
            return
        self.run_logger.start_run(self.run_name)
        self.run_logger.log_params({
            k: v for k, v in vars(self.config).items()
            if not k.startswith("_")
        })
