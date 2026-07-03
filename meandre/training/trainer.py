"""Training loop with RunLogger logging and curriculum support.

Implements the four-phase curriculum from README section 6.6:
    Phase 1: Pure physics + NeRF (temporal modules disabled)
    Phase 2: Enable temporal context encoder
    Phase 3: Enable state residual corrector
    Phase 4: Enable travel-time attention in routing
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
from meandre.routing.withdrawals import WithdrawalData
from meandre.spatial.territorial import TerritorialFeatures
from meandre.training.loss import CompositeKGELoss, HydroLoss
from meandre.utils.metrics import kge_components, log_nse, mae, nse, nrmse, pbias, rmse
from meandre.utils.state import HydroState

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Hyperparameters and curriculum schedule."""
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

    # Boundary regularization on raw network outputs.
    # Penalizes sigmoid saturation (params at clamp bounds) and L2-pulls
    # exp-constrained raw outputs toward center. Complements w_prior
    # (which targets specific literature values in physical units).
    # 0 = off. 0.001-0.01 = soft anti-saturation prior. Only active when
    # spatial encoder is unfrozen.
    w_boundary: float = 0.0

    # Anchor on the noise head's log_sigma_a (and optionally log_sigma_b).
    # Counters the Gaussian NLL degeneracy where σ inflates to fit any μ.
    # Apply to noise_head (Q), noise_head_et, noise_head_swe symmetrically.
    # 0 = no regularization. Typical w_sigma_anchor ≈ 0.1-1.0.
    w_sigma_anchor: float = 0.0
    sigma_anchor_target_a: float = -3.0   # log σ baseline (5% when b=1 at Q=100)
    sigma_anchor_target_b: float | None = None   # None = let b float free
    # Pénalité variance inter-nœuds des coefficients (a, b) du SpatialNoiseHead.
    # Mode "hybride" : tire la moyenne ET serre la dispersion spatiale, pour
    # éviter qu'une fraction de nœuds gonfle σ massivement pendant que la
    # moyenne reste correcte. 0 = ancien comportement.
    sigma_anchor_var_weight: float = 0.0

    # Concrete Dropout KL regularisation weight for the temporal encoder.
    # Penalises dropout rate deviation from the prior. 0 = disabled (standard
    # training). Typical: 0.01-0.1 for moderate epistemic uncertainty.
    w_concrete_kl: float = 0.0

    # LR scheduler shape. Cosine annealing from `lr` to `lr * eta_min_factor`
    # over n_epochs. Set to 1.0 for CONSTANT lr (cosine becomes flat) —
    # useful for short tuning runs where the default 0.01 collapses lr by
    # epoch 2-3 and stalls learning.
    eta_min_factor: float = 0.01

    # Curriculum: epoch at which each module is enabled
    enable_temporal_context_epoch: int = 10
    enable_residual_corrector_epoch: int = 30
    enable_travel_time_attn_epoch: int = 50

    # Residual corrector warm-up: over this many epochs after activation,
    # the effective gate multiplier ramps linearly from 0.0 to 1.0.  This
    # avoids the large loss spike that happens when gate_logit and random
    # GRU weights are activated at once.  Set to 0 to disable.
    residual_warmup_epochs: int = 5

    # TTA warm-up: over this many epochs after TTA activation, the routing
    # aggregation blends linearly from simple sum (Σ Q_actuels, factor=0)
    # to full TTA attention (factor=1).  This avoids a sudden semantic
    # change in routing when warm-starting from a checkpoint trained
    # without TTA.  Set to 0 to use pure TTA immediately.
    tta_warmup_epochs: int = 10

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
    # Intervenir trop tôt (w_residual++, LR reduction) freine la convergence.
    autopilot_grace_epochs: int = 0  # 0 = pas de grace, 10-15 pour fresh start

    # Beta drift: if pooled β deviates from 1.0 by more than this, increase
    # w_residual by autopilot_beta_penalty per epoch until β returns.
    # Catches residual-corrector overfitting (β = volume bias in KGE).
    autopilot_beta_threshold: float = 0.15   # |β - 1| > 0.15 → act
    autopilot_beta_penalty: float = 0.005    # increment w_residual per epoch

    # Gamma drift: if pooled γ deviates from 1.0 by more than this, increase
    # w_residual.  Catches timing/variability overfitting.
    autopilot_gamma_threshold: float = 0.20  # |γ - 1| > 0.20 → act
    autopilot_gamma_penalty: float = 0.003   # increment w_residual per epoch

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

    # ── NLL autopilot (Kendall & Gal phase 2) ──────────────────────
    # Automatically adjust w_nll based on KGE degradation from the
    # deterministic baseline.  Prevents the NLL from destroying the
    # mean prediction while still allowing gradual uncertainty learning.
    autopilot_nll: bool = False
    autopilot_nll_initial_kge: float | None = None  # set automatically if None
    autopilot_nll_max_regression: float = 0.05  # max tolerable KGE drop (5%)
    autopilot_nll_ramp_rate: float = 1.5        # multiply w_nll by this on ramp
    autopilot_nll_max: float = 0.5             # cap for w_nll
    autopilot_nll_min: float = 0.001          # floor for w_nll

    # Metric-driven curriculum: instead of fixed epoch numbers, activate
    # modules when kge_station reaches a threshold.  Overrides the fixed
    # epoch settings when autopilot is enabled and these are > 0.
    autopilot_activate_residual_at_kge: float | None = None  # e.g. 0.55
    autopilot_activate_tta_at_kge: float | None = None       # e.g. 0.65

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

    # ── Kendall-Gal auto phase 1 → phase 2 transition ──────────────
    # When enabled, the trainer starts in phase 1 (deterministic, backbone
    # trainable, w_nll=0) and automatically switches to phase 2 (probabilistic,
    # backbone frozen, w_nll ramped) when kge_station crosses a threshold OR
    # the kge_station plateau patience is exhausted.  Reproduces the two-config
    # warm-start recipe (deterministic.toml → probabilistic.toml) in a single
    # run.  One-shot transition.
    kendall_gal_auto: bool = False
    kga_phase1_kge_threshold: float = 0.85       # transition if kge_sta >= this
    kga_phase1_plateau_patience: int = 15        # or N epochs without kge_sta improvement
    kga_phase1_min_epochs: int = 5               # safeguard: never transition before this epoch
    kga_phase2_freeze_spatial: bool = True
    kga_phase2_freeze_temporal: bool = True       # keeps ConcreteDropout trainable
    kga_phase2_freeze_backbone: bool = True       # vertical column + routing
    kga_phase2_best_metric: str = "nll"
    kga_phase2_lr: float | None = None            # None = keep current LR
    kga_phase2_reset_no_improve: bool = True      # reset early-stopping counter
    # Loss-weight overrides applied at transition. Any key matching a HydroLoss
    # attribute (w_nll, w_kge, w_mse, w_log_mse, w_pbias, w_nrmse, w_nse,
    # w_log_nse, w_physics, w_residual, w_nll_et, w_nll_swe) will be setattr'd.
    # Example: {"w_nll": 0.1, "w_kge": 1.0, "w_mse": 0.0}
    kga_phase2_loss_weights: dict | None = None


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
    # because MODIS revisit is irregular). Used by gaussian_nll_loss
    # against the corresponding sim fields when w_nll_et / w_nll_swe > 0.
    et_obs: Tensor | None = None    # MODIS MOD16A2 ETR (mm/jour, 8-day agrégé en daily)
    swe_obs: Tensor | None = None   # SWE de MODIS NDSI ou SNODAS (mm)
    tws_obs: Tensor | None = None   # GRACE TWS anomalie (mm), valeur mensuelle au 15, NaN ailleurs
    # Indices hydrométéorologiques précalculés pour ContextualQuantileHead.
    # Forme : (T, n_st, 5) — GDD, API, SPI, FN, SWE_proxy normalisés z-score.
    indices_ihi: Tensor | None = None


class Trainer:
    """Training loop with curriculum scheduling and RunLogger logging.

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
                               "lr": self.config.lr * 10.0,
                               "weight_decay": 0.0})
                logger.info(
                    "Discriminative LR: fc_out.weight=%.1e (10×), wd=0 — NeRF anti-collapse",
                    self.config.lr * 10.0,
                )
            # Noise head learns at 10× base LR — sigma needs to adapt fast
            # while spatial_encoder learns slowly from the combined KGE+NLL signal.
            noise_params: list[torch.nn.Parameter] = []
            for name, p in model.named_parameters():
                if "noise_head" in name:
                    noise_params.append(p)
            if noise_params:
                # Remove noise_params from base_params to avoid double-counting
                base_params_set = set(id(p) for p in base_params)
                base_params[:] = [p for p in base_params if id(p) not in set(id(p) for p in noise_params)]
                groups.append({"params": noise_params,
                               "lr": self.config.lr * 10.0,
                               "weight_decay": 0.0})
                logger.info(
                    "Discriminative LR: noise_head=%.1e (10×), wd=0 — fast sigma adaptation",
                    self.config.lr * 10.0,
                )
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
                if "phenology_modulator" in name and ("gdd_emerg" in name or "gdd_mid" in name):
                    gdd_threshold_params.append(p)
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
            "val_flatness", "flatness",
        )
        self._best_val_metric = (
            float("inf") if self._best_metric_lower_is_better else -float("inf")
        )

        # Autopilot state
        self._ap_w_residual_orig: float | None = None  # original w_residual
        self._ap_lr_plateau_count: int = 0
        self._ap_restart_count: int = 0
        self._ap_prev_kge_sta: float | None = None

        # Kendall-Gal auto state (None = feature disabled)
        self._kga_phase: int | None = 1 if self.config.kendall_gal_auto else None
        self._kga_best_kge: float = -float("inf")
        self._kga_plateau_counter: int = 0
        self._kga_phase1_start_epoch: int = 0

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
            self._apply_curriculum(epoch)
            self._cur_epoch = epoch   # utilisé par l'offset aléatoire des chunks

            train_loss, train_comps = self._train_epoch()

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
            val_flat = last_val_metrics.get("val_flatness", float("nan"))
            if not math.isnan(val_nll_m):
                epoch_msg += f" | nll={val_nll_m:.3f}"
            if not math.isnan(val_flat):
                # val_flatness ~ d^2 ; d s'obtient par sqrt (ASCII pour Windows cp1252)
                epoch_msg += f" | flat_d2={val_flat:.4f} (d={val_flat**0.5:.3f})"
            epoch_msg += f" | lr={current_lr:.2e}"
            logger.info(epoch_msg)
            print(epoch_msg, flush=True)

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
            # Alias: "nll"/"flatness" in calibration dict are keyed as "val_*"
            _bm_key = {"nll": "val_nll", "flatness": "val_flatness"}.get(_bm, _bm)
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
                _margin = _tol * abs(self._best_val_metric)
                if not math.isfinite(_margin):
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

            # ── Kendall-Gal phase 1→2 transition (one-shot) ────────────
            # Checked BEFORE regular autopilot so we don't trigger LR plateau
            # actions on a metric we're about to swap away from.
            if self.config.kendall_gal_auto and run_val and self._kga_phase == 1:
                transitioned = self._maybe_transition_kga(epoch, last_val_metrics)
                if transitioned and self.config.kga_phase2_reset_no_improve:
                    epochs_without_improvement = 0

            # ── Autopilot ──────────────────────────────────────────────
            if self.config.autopilot and run_val:
                self._run_autopilot(
                    epoch, last_val_metrics, epochs_without_improvement,
                )

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

    def _run_spinup(self, data: TrainingData) -> tuple[HydroState, Tensor | None]:
        """Return spun-up (state, h_context).

        On the first call (or when warm_spinup_steps == 0), runs the full
        spinup from cold start.  On subsequent calls, reuses the cached
        spinup state and only runs ``warm_spinup_steps`` additional steps
        to let the state adjust to updated model weights — much cheaper
        than replaying the entire spinup every epoch.
        """
        device = data.forcing.device
        spinup_end = min(self.config.spinup_steps, data.train_slice.start)

        if spinup_end == 0:
            return HydroState.zeros(self.model.n_nodes, device=device), None

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
                    withdrawals=data.withdrawals,
                    day_of_year=data.day_of_year[warm_start:spinup_end],
                )
        else:
            # Full spinup from cold start
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=self._amp_dtype, enabled=self._use_amp):
                _, spun_state = self.model.simulate(
                    forcing=data.forcing[:spinup_end],
                    initial_state=HydroState.zeros(self.model.n_nodes, device=device),
                    graph=data.graph,
                    node_coords=data.node_coords,
                    territorial=data.territorial,
                    withdrawals=data.withdrawals,
                    day_of_year=data.day_of_year[:spinup_end],
                )

        # Cache for next epoch
        self._cached_spinup_state = spun_state.detach()

        h_ctx = self.model._last_h_context
        return spun_state, h_ctx

    def _simulate(
        self, data: TrainingData, time_slice: slice, tbptt_steps: int = 0,
    ) -> tuple[Tensor, HydroState]:
        """Run model over time_slice (after spin-up on the preceding steps)."""
        initial_state, h_ctx = self._run_spinup(data)

        # Main simulation over the requested slice
        Q_sim, final_state = self.model.simulate(
            forcing=data.forcing[time_slice],
            initial_state=initial_state,
            graph=data.graph,
            node_coords=data.node_coords,
            territorial=data.territorial,
            withdrawals=data.withdrawals,
            day_of_year=data.day_of_year[time_slice],
            h_context=h_ctx,
            tbptt_steps=tbptt_steps,
        )
        return Q_sim, final_state

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
        initial_state, h_ctx = self._run_spinup(data)

        state = initial_state
        t_start = data.train_slice.start
        obs_offset = 0
        total_loss = torch.tensor(0.0, device=data.forcing.device)
        all_components: dict[str, float] = {}
        n_chunks = 0
        sp_tensor: Tensor | None = None  # cached spatial params for SpatialNoiseHead

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

        while t_start < data.train_slice.stop:
            _len = chunk + _first_offset if n_chunks == 0 else chunk
            t_end = min(t_start + _len, data.train_slice.stop)
            # Merge tiny remainder into the previous chunk
            remainder = data.train_slice.stop - t_end
            if 0 < remainder < 60:
                t_end = data.train_slice.stop
            sl = slice(t_start, t_end)
            chunk_len = t_end - t_start

            # ET multi-objectif : on récupère et_sim depuis CE forward (avec
            # gradient) plutôt qu'un 2e forward détaché — sinon le terme ET
            # n'entraîne que σ_ET, pas le backbone (μ).
            _need_et = ((self.loss_fn.w_nll_et > 0 or self.loss_fn.w_et > 0)
                        and data.et_obs is not None)
            _need_tws = (self.loss_fn.w_tws > 0 and data.tws_obs is not None)
            # Neige : MODIS snow cover (fraction) cale le taux de FONTE (sp_fonte).
            # On compare une fraction de couverture SIMULÉE = 1-exp(-SWE/SWE_REF)
            # (différentiable, monotone) à snow_frac MODIS. data.swe_obs porte le
            # snow_frac MODIS (0-1), PAS du SWE en mm.
            _need_snow = (self.loss_fn.w_snow > 0 and data.swe_obs is not None)
            _need_diag = _need_et or _need_tws or _need_snow
            with torch.amp.autocast("cuda", dtype=self._amp_dtype, enabled=self._use_amp):
                _sim_out = self.model.simulate(
                    forcing=data.forcing[sl],
                    initial_state=state,
                    graph=data.graph,
                    node_coords=data.node_coords,
                    territorial=data.territorial,
                    withdrawals=data.withdrawals,
                    day_of_year=data.day_of_year[sl],
                    h_context=h_ctx,
                    tbptt_steps=self.config.tbptt_steps,
                    return_diagnostics=_need_diag,
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
                # Detach Q_sim before noise_head : σ adapts to current
                # prediction magnitude (heteroscedastic) but gradient doesn't
                # back-propagate through Q_sim via σ. Prevents the NLL
                # degeneracy where model inflates μ_Q to gonfler σ and reduce
                # ((q_obs - q_sim) / σ)² (Kendall & Gal 2017 §2.3).
                log_sigma_chunk = None
                # sp_tensor partagé entre noise_head et quantile_head pour
                # éviter de recalculer le NeRF spatial. None = pas encore
                # calculé (lazy ; recalculé si une tête le requiert).
                sp_tensor = None
                if self.loss_fn.w_nll > 0:
                    Q_det = Q_chunk_loss.detach()
                    from meandre.utils.noise_head import SpatialNoiseHead
                    if isinstance(self.model.noise_head, SpatialNoiseHead):
                        # Per-node noise head conditions on spatial params so
                        # different catchments get different uncertainty profiles.
                        sp = self.model.spatial_encoder(
                            data.node_coords, data.territorial.to_tensor()
                        )
                        sp_tensor = sp.to_tensor()
                        log_sigma_chunk = self.model.noise_head(sp_tensor, Q_det)
                    else:
                        log_sigma_chunk = self.model.noise_head(Q_det)
                # ET sim de ce chunk (avec gradient) + σ_ET (détaché, comme Q)
                et_sim_chunk = et_obs_chunk = log_sigma_et_chunk = None
                if _need_et:
                    et_sim_chunk = _diag_chunk.etr[burnin:]
                    et_obs_chunk = data.et_obs[obs_offset + burnin:obs_offset + chunk_len]
                    if self.loss_fn.w_nll_et > 0 and hasattr(self.model, "noise_head_et"):
                        log_sigma_et_chunk = self.model.noise_head_et(et_sim_chunk.detach())

                # Neige : fraction de couverture simulée vs MODIS snow_frac.
                scf_sim_chunk = snow_obs_chunk = None
                if _need_snow:
                    _swe_mm = _diag_chunk.swe[burnin:]
                    scf_sim_chunk = 1.0 - torch.exp(-_swe_mm / self.config.snow_swe_ref)
                    snow_obs_chunk = data.swe_obs[obs_offset + burnin:obs_offset + chunk_len]

                loss_chunk, comps = self.loss_fn(
                    q_obs=q_obs_chunk,
                    q_sim=Q_chunk_loss,
                    station_mask=data.station_mask,
                    log_sigma_sim=log_sigma_chunk,
                    et_obs=et_obs_chunk,
                    et_sim=et_sim_chunk,
                    log_sigma_et_sim=log_sigma_et_chunk,
                    swe_obs=snow_obs_chunk,
                    swe_sim=scf_sim_chunk,
                    residual_gate_logits=(
                        self.model.residual_corrector.gate_logit
                        if self.model.use_residual and self.model.residual_corrector is not None
                        else None
                    ),
                    log_df=getattr(self.model.noise_head, "log_df", None),
                )

                # ── GRACE TWS : stockage total basin-moyen (avec gradient) ──
                # storage = Σθ_i·z_i·1000 + SWE + S_gw + canopy + wetland (mm).
                # Comparé à GRACE aux mois valides, centré intra-chunk (chunk-safe).
                if _need_tws:
                    from meandre.training.loss import tws_anomaly_loss
                    _sp = self.model.spatial_encoder(
                        data.node_coords, data.territorial.to_tensor()
                    )
                    _soil_mm = ((_diag_chunk.theta1 * 0.30
                                 + _diag_chunk.theta2 * _sp.Z2
                                 + _diag_chunk.theta3 * _sp.Z3) * 1000.0)
                    _stor = (_soil_mm + _diag_chunk.swe + _diag_chunk.s_gw
                             + _diag_chunk.canopy + _diag_chunk.wetland)  # (T, n_nodes) mm
                    _stor_basin = _stor.mean(dim=1)[burnin:]  # (T-burnin,) moy-bassin
                    _tws_chunk = data.tws_obs[obs_offset + burnin:obs_offset + chunk_len]
                    _vt = ~torch.isnan(_tws_chunk)
                    if int(_vt.sum()) >= 2:  # ≥2 mois pour centrer
                        _s = _stor_basin[_vt]
                        _g = _tws_chunk[_vt]
                        # σ = incertitude GRACE typique (~25 mm) → z-score, L_tws~O(1)
                        # (sinon mm² ~800 écrase Q). "fit à l'incertitude GRACE près".
                        L_tws = tws_anomaly_loss(_s, _g, _s.mean().detach(), _g.mean(), sigma=25.0)
                        loss_chunk = loss_chunk + self.loss_fn.w_tws * L_tws
                        all_components["tws_loss"] = (
                            all_components.get("tws_loss", 0.0) + float(L_tws.detach()))

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
                            all_components.get("quantile_loss", 0.0) + float(L_q.detach()))

                # ── ContextualQuantileHead (IHI, Phase A) — pinball loss
                # avec features riches : sp + Q_sim + log Q_sim + indices IHI + DOY.
                # Médiane libre (pas ancrée à Q_sim).
                if (self.loss_fn.w_quantile > 0
                        and getattr(self.model, "use_contextual_quantile_head", False)
                        and hasattr(self.model, "contextual_quantile_head")
                        and data.indices_ihi is not None):
                    if sp_tensor is None:
                        sp = self.model.spatial_encoder(
                            data.node_coords, data.territorial.to_tensor()
                        )
                        sp_tensor = sp.to_tensor()
                    _Q_det = Q_chunk_loss.detach()                                  # (T_chunk, N)
                    _sp_st = sp_tensor[data.station_mask]                           # (n_st, F_sp)
                    _Q_st = _Q_det[:, data.station_mask]                            # (T, n_st)
                    _y_obs = q_obs_chunk                                            # (T, n_st)
                    T_chunk_eff = _Q_st.shape[0]
                    # Slice des indices au chunk (déjà aux stations) + DOY sin/cos
                    _idx_chunk = data.indices_ihi[obs_offset + burnin:obs_offset + chunk_len]
                    # Vérifier dimension
                    if _idx_chunk.shape[0] != T_chunk_eff:
                        _idx_chunk = _idx_chunk[:T_chunk_eff]
                    # DOY sin/cos (T_chunk,)
                    import math
                    _doy_chunk = data.day_of_year[sl.start + burnin:sl.stop][:T_chunk_eff].float()
                    _doy_rad = 2 * math.pi * _doy_chunk / 366.0
                    _doy_sc = torch.stack([_doy_rad.sin(), _doy_rad.cos()], dim=-1)  # (T, 2)
                    # Build features (T, n_st, F_total)
                    _sp_exp = _sp_st.unsqueeze(0).expand(T_chunk_eff, -1, -1)        # (T, n_st, F_sp)
                    _Q_feat = _Q_st.unsqueeze(-1)                                    # (T, n_st, 1)
                    _logQ_feat = torch.log(_Q_st.clamp(min=0) + 1.0).unsqueeze(-1)   # (T, n_st, 1)
                    _doy_feat = _doy_sc.unsqueeze(1).expand(-1, _sp_st.shape[0], -1) # (T, n_st, 2)
                    _features = torch.cat(
                        [_sp_exp, _Q_feat, _logQ_feat, _idx_chunk, _doy_feat], dim=-1,
                    )                                                                # (T, n_st, F_total)
                    # Filtrer valides
                    _v = ~torch.isnan(_y_obs) & ~torch.isnan(_Q_st)
                    if int(_v.sum()) > 0:
                        _y_flat = _y_obs[_v]
                        _x_flat = _features[_v]
                        _Q_flat = _Q_st[_v]
                        L_cqh = self.model.contextual_quantile_head.pinball(
                            _y_flat, _x_flat, _Q_flat,
                        )
                        loss_chunk = loss_chunk + self.loss_fn.w_quantile * L_cqh
                        all_components["cqh_pinball"] = (
                            all_components.get("cqh_pinball", 0.0) + float(L_cqh.detach()))

                # ── Mixture Density Network (option 2b) — NLL non-paramétrique ──
                # p(y | x) = Σ_k π_k · N(y | μ_k, σ_k²) ; loss = -log_prob.
                # Q_sim détaché : la tête n'entraîne PAS le backbone, seulement le
                # mapping features → densité (Phase 2 v3 style, comme quantile head).
                if (self.loss_fn.w_mixture > 0
                        and getattr(self.model, "use_mixture_head", False)
                        and hasattr(self.model, "mixture_head")):
                    if sp_tensor is None:
                        sp = self.model.spatial_encoder(
                            data.node_coords, data.territorial.to_tensor()
                        )
                        sp_tensor = sp.to_tensor()
                    _Q_mdn = Q_chunk_loss.detach()                                  # (T-burnin, n_nodes)
                    _sp_st = sp_tensor[data.station_mask]                           # (n_st, n_features)
                    _Q_mdn_st = _Q_mdn[:, data.station_mask]                        # (T, n_st)
                    _y_mdn = q_obs_chunk
                    _v = ~torch.isnan(_y_mdn) & ~torch.isnan(_Q_mdn_st)
                    if int(_v.sum()) > 0:
                        # Broadcast features (statique) sur T
                        T_chunk = _Q_mdn_st.shape[0]
                        sp_exp = _sp_st.unsqueeze(0).expand(T_chunk, -1, -1)         # (T, n_st, F)
                        y_flat = _y_mdn[_v]
                        sp_flat = sp_exp[_v]
                        q_flat = _Q_mdn_st[_v]
                        L_mdn = -self.model.mixture_head.log_prob(y_flat, sp_flat, q_flat).mean()
                        loss_chunk = loss_chunk + self.loss_fn.w_mixture * L_mdn
                        all_components["mixture_nll"] = (
                            all_components.get("mixture_nll", 0.0) + float(L_mdn.detach()))


            # Soft physical prior regularization on spatial params (k_gw, krec, K_sat, K_c, ...).
            # Pulls toward literature targets — prevents drift toward overfit values.
            if self.config.w_prior > 0 or self.config.w_diversity > 0:
                params_t = self.model.spatial_encoder(
                    data.node_coords, data.territorial.to_tensor()
                )
                if self.config.w_prior > 0:
                    prior_loss = self.model.spatial_encoder.physical_prior_loss(params_t)
                    loss_chunk = loss_chunk + self.config.w_prior * prior_loss
                    all_components["prior"] = all_components.get("prior", 0.0) + float(prior_loss.detach())
                if self.config.w_diversity > 0:
                    div_loss = self.model.spatial_encoder.param_diversity_loss(
                        params_t, cv_target=self.config.diversity_cv_target)
                    loss_chunk = loss_chunk + self.config.w_diversity * div_loss
                    all_components["diversity"] = all_components.get("diversity", 0.0) + float(div_loss.detach())

            if self.config.w_latent_reg > 0 and getattr(self.model.spatial_encoder, "use_latent_codes", False):
                latent_loss = self.model.spatial_encoder.latent_reg()
                loss_chunk = loss_chunk + self.config.w_latent_reg * latent_loss
                all_components["latent_reg"] = all_components.get("latent_reg", 0.0) + float(latent_loss.detach())

            if self.config.w_boundary > 0:
                boundary_loss = self.model.spatial_encoder.boundary_regularization(
                    data.node_coords, data.territorial.to_tensor()
                )
                loss_chunk = loss_chunk + self.config.w_boundary * boundary_loss
                all_components["boundary"] = all_components.get("boundary", 0.0) + float(boundary_loss.detach())

            # Noise head σ anchor — counters NLL degeneracy (σ inflates to
            # mask a bad μ). Applied symmetrically to Q / ET / SWE heads.
            if self.config.w_sigma_anchor > 0 and self.loss_fn.w_nll > 0:
                from meandre.utils.noise_head import SpatialNoiseHead
                if isinstance(self.model.noise_head, SpatialNoiseHead):
                    if sp_tensor is None:
                        sp = self.model.spatial_encoder(
                            data.node_coords, data.territorial.to_tensor()
                        )
                        sp_tensor = sp.to_tensor()
                    anchor = self.model.noise_head.anchor_loss(
                        sp_tensor,
                        self.config.sigma_anchor_target_a,
                        self.config.sigma_anchor_target_b,
                        var_weight=self.config.sigma_anchor_var_weight,
                    )
                else:
                    anchor = self.model.noise_head.anchor_loss(
                        self.config.sigma_anchor_target_a,
                        self.config.sigma_anchor_target_b,
                    )
                if self.loss_fn.w_nll_et > 0:
                    anchor = anchor + self.model.noise_head_et.anchor_loss(
                        self.config.sigma_anchor_target_a,
                        self.config.sigma_anchor_target_b,
                    )
                if self.loss_fn.w_nll_swe > 0:
                    anchor = anchor + self.model.noise_head_swe.anchor_loss(
                        self.config.sigma_anchor_target_a,
                        self.config.sigma_anchor_target_b,
                    )
                loss_chunk = loss_chunk + self.config.w_sigma_anchor * anchor
                all_components["sigma_anchor"] = (
                    all_components.get("sigma_anchor", 0.0) + float(anchor.detach())
                )

            # Concrete Dropout KL regularisation (epistemic uncertainty).
            # Only active when concrete_dropout is enabled in the temporal encoder.
            if self.config.w_concrete_kl > 0 and self.model.temporal_encoder is not None:
                concrete_kl = self.model.temporal_encoder.concrete_kl()
                loss_chunk = loss_chunk + self.config.w_concrete_kl * concrete_kl
                all_components["concrete_kl"] = (
                    all_components.get("concrete_kl", 0.0) + float(concrete_kl.detach())
                )

            # Scale by chunk fraction so total gradient ≈ full-series gradient
            weight = chunk_len / n_train
            if not torch.isnan(loss_chunk) and loss_chunk.requires_grad:
                (loss_chunk * weight).backward()
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

            # Detach state for next chunk (TBPTT across chunks)
            state = state_out.detach()
            h_ctx = (
                self.model._last_h_context.detach()
                if self.model._last_h_context is not None else None
            )
            obs_offset += chunk_len
            t_start = t_end

            # Free chunk memory
            del Q_chunk, loss_chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Cache end-of-train state + h_context so _val_epoch can start from
        # the realistic state (after train period forward pass) instead of
        # re-simulating from spinup.  Saves ~50 min/epoch on GPU.
        self._cached_train_end_state = state.detach()
        self._cached_train_end_h_ctx = h_ctx.detach() if h_ctx is not None else None

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
        total_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.grad_clip
        )
        self.optimizer.step()

        comp_tensors = {k: torch.tensor(v) for k, v in all_components.items()}
        return total_loss, comp_tensors

    def _checkpointed_chunk(
        self,
        data: TrainingData,
        sl: slice,
        state: HydroState,
        h_ctx: Tensor | None,
    ) -> tuple[Tensor, HydroState]:
        """Simulate one chunk with gradient checkpointing.

        The state must flow between chunks (it carries soil moisture, SWE, etc.),
        but we detach it at chunk boundaries so the backward graph of each chunk
        is independent.  This is equivalent to TBPTT with segment length =
        chunk_steps — the gradient for each chunk is exact, but cross-chunk
        temporal gradients are truncated.

        The output state is captured as a side-effect during the forward pass
        (detached, so it doesn't affect the grad graph) to avoid running the
        simulation twice.
        """
        from torch.utils.checkpoint import checkpoint

        # Pack state into a single tensor so checkpoint can track it
        state_tensor = state.to_tensor()  # (n_nodes, n_state_vars)

        # Mutable container to capture the output state during forward
        captured_state: list[HydroState] = []

        def _run_chunk(state_t: Tensor) -> Tensor:
            """Inner function wrapped by checkpoint — must return tensors only."""
            s = HydroState.from_tensor(state_t)
            with torch.amp.autocast("cuda", dtype=self._amp_dtype, enabled=self._use_amp):
                Q_chunk, state_out = self.model.simulate(
                    forcing=data.forcing[sl],
                    initial_state=s,
                    graph=data.graph,
                    node_coords=data.node_coords,
                    territorial=data.territorial,
                    withdrawals=data.withdrawals,
                    day_of_year=data.day_of_year[sl],
                    h_context=h_ctx,
                    tbptt_steps=self.config.tbptt_steps,
                )
            # Capture detached state as side-effect (not part of grad graph).
            # checkpoint calls _run_chunk twice (forward + recompute during
            # backward), so always overwrite with the latest.
            captured_state.clear()
            captured_state.append(state_out.detach())
            return Q_chunk

        # use_reentrant=False is the modern, safer API
        Q_chunk = checkpoint(_run_chunk, state_tensor, use_reentrant=False)

        return Q_chunk, captured_state[0]

    def _train_epoch_single(self, data: TrainingData) -> tuple[Tensor, dict[str, Tensor]]:
        """Original single-pass training (no chunking)."""
        with torch.amp.autocast("cuda", dtype=self._amp_dtype, enabled=self._use_amp):
            Q_sim, final_state = self._simulate(
                data, data.train_slice, tbptt_steps=self.config.tbptt_steps,
            )
            # Cache for fast val_epoch (see chunked path).
            self._cached_train_end_state = final_state.detach()
            self._cached_train_end_h_ctx = (
                self.model._last_h_context.detach()
                if self.model._last_h_context is not None else None
            )

            n_train = data.train_slice.stop - data.train_slice.start
            q_obs_train = data.q_obs[:n_train]

            # Detach Q_sim before noise_head (cf. chunked path above).
            log_sigma_sim = None
            if self.loss_fn.w_nll > 0:
                Q_det = Q_sim.detach()
                from meandre.utils.noise_head import SpatialNoiseHead
                if isinstance(self.model.noise_head, SpatialNoiseHead):
                    sp = self.model.spatial_encoder(
                        data.node_coords, data.territorial.to_tensor()
                    )
                    log_sigma_sim = self.model.noise_head(sp.to_tensor(), Q_det)
                else:
                    log_sigma_sim = self.model.noise_head(Q_det)
            # ── Multi-objective ETR wiring (MOD16A2) ─────────────────
            # When data.et_obs is populated (MODIS loader ran) and w_nll_et > 0,
            # we need et_sim from the model diagnostics. We re-run simulate with
            # return_diagnostics=True only if the extra loss weight justifies it.
            et_sim = None
            log_sigma_et_sim = None
            need_et = (self.loss_fn.w_nll_et > 0 or self.loss_fn.w_et > 0) \
                      and data.et_obs is not None
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
                            withdrawals=data.withdrawals,
                            day_of_year=data.day_of_year[data.train_slice],
                            return_diagnostics=True,
                            tbptt_steps=self.config.tbptt_steps,
                        )
                        if isinstance(_result, tuple) and len(_result) == 3:
                            _, _, _diag = _result
                            et_sim = _diag.etr  # (T, n_nodes) mm/day
                if et_sim is not None and hasattr(self.model, "noise_head_et"):
                    log_sigma_et_sim = self.model.noise_head_et(et_sim.detach())

            loss, components = self.loss_fn(
                q_obs=q_obs_train,
                q_sim=Q_sim,
                station_mask=data.station_mask,
                log_sigma_sim=log_sigma_sim,
                et_obs=data.et_obs[:n_train] if data.et_obs is not None else None,
                et_sim=et_sim,
                log_sigma_et_sim=log_sigma_et_sim,
                residual_gate_logits=(
                    self.model.residual_corrector.gate_logit
                    if self.model.use_residual and self.model.residual_corrector is not None
                    else None
                ),
                log_df=getattr(self.model.noise_head, "log_df", None),
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

        # Noise head σ anchor (single-pass path) — same logic as chunked.
        if self.config.w_sigma_anchor > 0 and self.loss_fn.w_nll > 0:
            from meandre.utils.noise_head import SpatialNoiseHead
            if isinstance(self.model.noise_head, SpatialNoiseHead):
                sp = self.model.spatial_encoder(
                    data.node_coords, data.territorial.to_tensor()
                )
                anchor = self.model.noise_head.anchor_loss(
                    sp.to_tensor(),
                    self.config.sigma_anchor_target_a,
                    self.config.sigma_anchor_target_b,
                    var_weight=self.config.sigma_anchor_var_weight,
                )
            else:
                anchor = self.model.noise_head.anchor_loss(
                    self.config.sigma_anchor_target_a,
                    self.config.sigma_anchor_target_b,
                )
            if self.loss_fn.w_nll_et > 0:
                anchor = anchor + self.model.noise_head_et.anchor_loss(
                    self.config.sigma_anchor_target_a,
                    self.config.sigma_anchor_target_b,
                )
            if self.loss_fn.w_nll_swe > 0:
                anchor = anchor + self.model.noise_head_swe.anchor_loss(
                    self.config.sigma_anchor_target_a,
                    self.config.sigma_anchor_target_b,
                )
            loss = loss + self.config.w_sigma_anchor * anchor
            components["sigma_anchor"] = anchor.detach()

        # Concrete Dropout KL (single-pass path)
        if self.config.w_concrete_kl > 0 and self.model.temporal_encoder is not None:
            concrete_kl = self.model.temporal_encoder.concrete_kl()
            loss = loss + self.config.w_concrete_kl * concrete_kl
            components["concrete_kl"] = concrete_kl.detach()

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
                cached_h_ctx = getattr(self, "_cached_train_end_h_ctx", None)

                if cached_state is not None:
                    Q_sim, _ = self.model.simulate(
                        forcing=data.forcing[data.val_slice],
                        initial_state=cached_state,
                        graph=data.graph,
                        node_coords=data.node_coords,
                        territorial=data.territorial,
                        withdrawals=data.withdrawals,
                        day_of_year=data.day_of_year[data.val_slice],
                        h_context=cached_h_ctx,
                        tbptt_steps=self.config.tbptt_steps,
                    )
                else:
                    # Slow path: no cached state (e.g. before first train epoch).
                    # Cold spinup + continuous forward through train + val.
                    initial_state, h_ctx = self._run_spinup(data)
                    spinup_end = min(self.config.spinup_steps, data.val_slice.start)
                    full_slice = slice(spinup_end, data.val_slice.stop)
                    Q_full, _ = self.model.simulate(
                        forcing=data.forcing[full_slice],
                        initial_state=initial_state,
                        graph=data.graph,
                        node_coords=data.node_coords,
                        territorial=data.territorial,
                        withdrawals=data.withdrawals,
                        day_of_year=data.day_of_year[full_slice],
                        h_context=h_ctx,
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
        # Coverage: fraction of obs within [Q_sim ± z*sigma].
        # Well-calibrated model: cov_90 ≈ 0.90, cov_50 ≈ 0.50.
        calibration = {}
        if (self.loss_fn.w_nll > 0
                and hasattr(self.model, 'noise_head')
                and Q_sim is not None):
            from meandre.utils.noise_head import SpatialNoiseHead
            with torch.no_grad():
                Q_det = Q_sim.detach()
                if isinstance(self.model.noise_head, SpatialNoiseHead):
                    sp = self.model.spatial_encoder(
                        data.node_coords, data.territorial.to_tensor()
                    )
                    log_sigma = self.model.noise_head(sp.to_tensor(), Q_det)
                else:
                    log_sigma = self.model.noise_head(Q_det)
            sigma = log_sigma[:, data.station_mask].exp()  # (T, n_stations)
            q_obs_v = q_obs_val  # (T, n_stations)
            valid_cov = ~torch.isnan(q_obs_v) & ~torch.isnan(Q_sim[:, data.station_mask])
            # σ vit dans l'espace de la NLL (Box-Cox si nll_lambda != 1). La
            # couverture doit être mesurée dans CE MÊME espace, sinon un σ
            # Box-Cox appliqué comme intervalle ± linéaire en m³/s est
            # dimensionnellement incohérent (la couverture s'effondre quand le
            # σ Box-Cox se resserre légitimement).
            from meandre.training.loss import box_cox
            _lam_cov = getattr(self.loss_fn, "nll_lambda", 1.0)
            if _lam_cov != 1.0:
                q_obs_cov = box_cox(q_obs_v, _lam_cov)
                q_sim_cov = box_cox(q_sim_at_stations, _lam_cov)
            else:
                q_obs_cov = q_obs_v
                q_sim_cov = q_sim_at_stations
            for level, z in [(50, 0.674), (90, 1.645)]:
                lo = q_sim_cov - z * sigma
                hi = q_sim_cov + z * sigma
                in_interval = (q_obs_cov >= lo) & (q_obs_cov <= hi)
                cov = (in_interval & valid_cov).sum().float() / valid_cov.sum().float()
                calibration[f"cov_{level}"] = float(cov)
            # Mean sigma statistics
            calibration["sigma_mean"] = float(sigma[valid_cov].mean())
            calibration["sigma_median"] = float(sigma[valid_cov].median())
            # NLL et flatness sur le set de validation, dans l'espace
            # configuré par loss_fn.nll_lambda (Box-Cox/normal/log-normal).
            from meandre.training.loss import gaussian_nll_loss, flatness_loss, student_t_nll_loss
            qo_p = q_obs_v[valid_cov]
            qs_p = q_sim_at_stations[valid_cov]
            ls_p = log_sigma[:, data.station_mask][valid_cov]
            _lam = getattr(self.loss_fn, "nll_lambda", 1.0)
            _dist = getattr(self.loss_fn, "nll_distribution", "normal")
            if _dist == "student-t" and hasattr(self.model.noise_head, "log_df"):
                calibration["val_nll"] = float(
                    student_t_nll_loss(qo_p, qs_p, ls_p, self.model.noise_head.log_df, lam=_lam)
                )
            else:
                calibration["val_nll"] = float(
                    gaussian_nll_loss(qo_p, qs_p, ls_p, lam=_lam)
                )
            calibration["val_flatness"] = float(
                flatness_loss(
                    qo_p, qs_p, ls_p,
                    lam=_lam,
                    n_bins=getattr(self.loss_fn, "flatness_n_bins", 21),
                    bandwidth=getattr(self.loss_fn, "flatness_bandwidth", 0.02),
                )
            )

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
                offsets_v = self.model.quantile_head(sp_q.to_tensor(), Q_sim.detach())
            q_pred_v = Q_sim.detach().unsqueeze(-1) + offsets_v  # (T, N, K)
            q_pred_st = q_pred_v[:, data.station_mask, :]         # (T, n_st, K)
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
            initial_state, h_ctx = self._run_spinup(data)
            result = self.model.simulate(
                forcing=data.forcing[diag_sl],
                initial_state=initial_state,
                graph=data.graph,
                node_coords=data.node_coords,
                territorial=data.territorial,
                withdrawals=data.withdrawals,
                day_of_year=data.day_of_year[diag_sl],
                h_context=h_ctx,
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

    def _apply_curriculum(self, epoch: int) -> None:
        """Enable/disable modules according to curriculum phase schedule.

        Both the forward pass AND gradients are controlled so that disabled
        modules neither consume compute time nor receive gradient updates.
        """
        cfg = self.config

        if self.model.temporal_encoder is not None:
            enabled = epoch >= cfg.enable_temporal_context_epoch
            self.model.use_temporal = enabled           # skip forward pass when off
            for p in self.model.temporal_encoder.parameters():
                p.requires_grad_(enabled)

        if self.model.residual_corrector is not None:
            enabled = epoch >= cfg.enable_residual_corrector_epoch
            self.model.use_residual = enabled           # skip forward pass when off
            for p in self.model.residual_corrector.parameters():
                p.requires_grad_(enabled)
            # Linear warm-up of the effective gate multiplier (0 -> 1) over
            # the first `residual_warmup_epochs` epochs after activation.
            if enabled and cfg.residual_warmup_epochs > 0:
                k = epoch - cfg.enable_residual_corrector_epoch
                factor = min(1.0, max(0.0, (k + 1) / cfg.residual_warmup_epochs))
            else:
                factor = 1.0 if enabled else 0.0
            self.model.residual_corrector.warmup_factor.fill_(factor)

        if hasattr(self.model.routing, "tta"):
            enabled = epoch >= cfg.enable_travel_time_attn_epoch
            self.model.routing.use_tta = enabled
            for p in self.model.routing.tta.parameters():
                p.requires_grad_(enabled)
            # Linear warm-up of TTA blending factor (0 -> 1) over
            # `tta_warmup_epochs` epochs after activation.
            # factor=0 → pure simple sum (Σ Q_actuels), factor=1 → pure TTA.
            if enabled and cfg.tta_warmup_epochs > 0:
                k = epoch - cfg.enable_travel_time_attn_epoch
                tta_factor = min(1.0, max(0.0, (k + 1) / cfg.tta_warmup_epochs))
            else:
                tta_factor = 1.0 if enabled else 0.0
            if hasattr(self.model.routing, "tta_warmup_factor"):
                self.model.routing.tta_warmup_factor.fill_(tta_factor)

    def _apply_phase2_config(self, epoch: int, kge_at_transition: float) -> None:
        """Apply Kendall-Gal phase 2 configuration (one-shot).

        Freezes the requested modules, swaps loss weights and best_metric,
        and resets the early-stopping counter / best metric tracking.
        """
        cfg = self.config

        # ── Freeze modules ──────────────────────────────────────────
        n_frozen_total = 0
        if cfg.kga_phase2_freeze_spatial and hasattr(self.model, "spatial_encoder"):
            for p in self.model.spatial_encoder.parameters():
                p.requires_grad = False
            n_frozen_total += sum(p.numel() for p in self.model.spatial_encoder.parameters())

        if cfg.kga_phase2_freeze_temporal and hasattr(self.model, "temporal_encoder"):
            # Keep ConcreteDropout layers trainable (epistemic uncertainty)
            for name, p in self.model.temporal_encoder.named_parameters():
                if "drop." not in name:
                    p.requires_grad = False
                    n_frozen_total += p.numel()

        if cfg.kga_phase2_freeze_backbone:
            if hasattr(self.model, "vertical_column"):
                for p in self.model.vertical_column.parameters():
                    p.requires_grad = False
                    n_frozen_total += p.numel()
            if hasattr(self.model, "routing"):
                for p in self.model.routing.parameters():
                    p.requires_grad = False
                    n_frozen_total += p.numel()

        # ── Swap loss weights ───────────────────────────────────────
        applied_weights = {}
        if cfg.kga_phase2_loss_weights:
            for key, val in cfg.kga_phase2_loss_weights.items():
                if hasattr(self.loss_fn, key):
                    setattr(self.loss_fn, key, float(val))
                    applied_weights[key] = float(val)
                else:
                    logger.warning(
                        f"  [kga] phase2 loss weight '{key}' not found on loss_fn, skipped"
                    )

        # ── Swap best_metric tracking ───────────────────────────────
        old_best_metric = cfg.best_metric
        cfg.best_metric = cfg.kga_phase2_best_metric
        self._best_metric_lower_is_better = cfg.best_metric in (
            "loss", "val_loss", "nll", "val_nll", "rmse", "nrmse", "mae", "pbias",
        )
        self._best_val_metric = (
            float("inf") if self._best_metric_lower_is_better else -float("inf")
        )

        # ── Optionally adjust LR ────────────────────────────────────
        if cfg.kga_phase2_lr is not None:
            for pg in self.optimizer.param_groups:
                pg["lr"] = float(cfg.kga_phase2_lr)

        # ── Flip phase flag ─────────────────────────────────────────
        self._kga_phase = 2

        msg = (
            f"  [kga] PHASE 1→2 transition at epoch {epoch} "
            f"(kge_sta={kge_at_transition:.4f}) | "
            f"frozen={n_frozen_total:,} params | "
            f"best_metric: {old_best_metric}→{cfg.best_metric} | "
            f"loss_weights: {applied_weights}"
        )
        print(msg)
        logger.info(msg)

    def _maybe_transition_kga(
        self, epoch: int, val_metrics: dict[str, float],
    ) -> bool:
        """Check phase 1→2 transition condition. Returns True if reset needed.

        Uses the SAME metric as ``best_metric`` (set in the config) to evaluate
        the threshold — so the user sees a consistent number in the log and in
        the threshold comparison. For example: if best_metric='kge' (pooled),
        the threshold is compared against val_metrics['kge'] (pooled), not
        kge_station (per-station weighted).

        Updates plateau counter and triggers _apply_phase2_config when:
          - epoch - phase1_start >= kga_phase1_min_epochs (safeguard), AND
          - (best_metric_value >= kga_phase1_kge_threshold) OR
            (plateau_counter >= kga_phase1_plateau_patience)
        """
        if self._kga_phase != 1:
            return False
        cfg = self.config

        # Read the SAME metric the user is tracking via best_metric.
        # Falls back to kge_station for backward compatibility.
        bm = cfg.best_metric
        bm_key = "val_nll" if bm == "nll" else bm
        kge_val = val_metrics.get(bm_key, val_metrics.get("kge_station", float("nan")))
        if not math.isfinite(kge_val):
            return False

        # Update plateau counter (improvement = strict gain on the tracked metric)
        if kge_val > self._kga_best_kge:
            self._kga_best_kge = kge_val
            self._kga_plateau_counter = 0
        else:
            self._kga_plateau_counter += 1

        epochs_in_phase = epoch - self._kga_phase1_start_epoch
        if epochs_in_phase < cfg.kga_phase1_min_epochs:
            return False

        should_transition = (
            kge_val >= cfg.kga_phase1_kge_threshold
            or self._kga_plateau_counter >= cfg.kga_phase1_plateau_patience
        )
        if should_transition:
            self._apply_phase2_config(epoch, kge_val)
            return True
        return False

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

        # ── 1. Beta / Gamma drift → increase w_residual ─────────────
        # Original w_residual is captured on first autopilot call
        if self._ap_w_residual_orig is None:
            self._ap_w_residual_orig = float(self.loss_fn.w_residual)

        beta_drift = abs(beta - 1.0)
        gamma_drift = abs(gamma - 1.0)
        actions: list[str] = []

        # Clamp max : w_residual ne peut pas dépasser 5× sa valeur originale
        # pour éviter d'écraser complètement le residual corrector.
        _w_res_max = (self._ap_w_residual_orig or 0.01) * 5.0

        # Skip drift→w_residual handler when the residual corrector is
        # disabled — incrementing w_residual has no effect (no gradient
        # flows through a disabled module) and just pollutes the log.
        residual_active = (
            epoch >= cfg.enable_residual_corrector_epoch
            and hasattr(self.model, "residual_corrector")
            and getattr(self.model, "residual_corrector", None) is not None
        )

        if not residual_active:
            beta_drift_logged = beta_drift > cfg.autopilot_beta_threshold
            gamma_drift_logged = gamma_drift > cfg.autopilot_gamma_threshold
            if beta_drift_logged or gamma_drift_logged:
                logger.debug(
                    f"  [autopilot] β/γ drift detected (β-drift={beta_drift:.3f}, "
                    f"γ-drift={gamma_drift:.3f}) but residual corrector inactive "
                    f"(enable_residual_corrector_epoch={cfg.enable_residual_corrector_epoch}) "
                    f"— skipping w_residual update"
                )
        else:
            if beta_drift > cfg.autopilot_beta_threshold:
                self.loss_fn.w_residual = min(
                    self.loss_fn.w_residual + cfg.autopilot_beta_penalty, _w_res_max
                )
                actions.append(
                    f"beta drift |beta-1|={beta_drift:.3f} > {cfg.autopilot_beta_threshold} "
                    f"→ w_residual += {cfg.autopilot_beta_penalty} → {self.loss_fn.w_residual:.4f}"
                )

            if gamma_drift > cfg.autopilot_gamma_threshold:
                self.loss_fn.w_residual = min(
                    self.loss_fn.w_residual + cfg.autopilot_gamma_penalty, _w_res_max
                )
                actions.append(
                    f"gamma drift |gamma-1|={gamma_drift:.3f} > {cfg.autopilot_gamma_threshold} "
                    f"→ w_residual += {cfg.autopilot_gamma_penalty} → {self.loss_fn.w_residual:.4f}"
                )

            # Reset w_residual toward original when drift subsides
            if (beta_drift < cfg.autopilot_beta_threshold * 0.5
                    and gamma_drift < cfg.autopilot_gamma_threshold * 0.5
                    and self.loss_fn.w_residual > self._ap_w_residual_orig * 1.1):
                self.loss_fn.w_residual = max(
                    self._ap_w_residual_orig,
                    self.loss_fn.w_residual * 0.9,
                )
                actions.append(
                    f"beta/gamma recovered → w_residual decay → {self.loss_fn.w_residual:.4f}"
                )

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
            # Reset w_residual to original after restart
            if self._ap_w_residual_orig is not None:
                self.loss_fn.w_residual = self._ap_w_residual_orig
            # Reset cached spinup state (model weights changed)
            self._cached_spinup_state = None
            actions.append(
                f"RESTART {self._ap_restart_count}/{cfg.autopilot_restart_max} "
                f"regression={regression:.1%} + drift → reload best, "
                f"LR → {self.optimizer.param_groups[0]['lr']:.2e}, "
                f"w_residual → {self.loss_fn.w_residual:.4f}"
            )

        # ── 4. Metric-driven curriculum ───────────────────────────────
        # Activate modules when kge_station crosses a threshold (instead of
        # fixed epoch numbers).  Only override if the threshold is set.
        if cfg.autopilot_activate_residual_at_kge is not None:
            threshold = cfg.autopilot_activate_residual_at_kge
            if (kge_sta >= threshold
                    and not self.model.use_residual
                    and self.model.residual_corrector is not None):
                # Override the fixed epoch — activate now
                cfg.enable_residual_corrector_epoch = epoch
                actions.append(
                    f"Metric curriculum: kge_sta={kge_sta:.4f} ≥ {threshold} "
                    f"→ activate residual corrector"
                )

        if cfg.autopilot_activate_tta_at_kge is not None:
            threshold = cfg.autopilot_activate_tta_at_kge
            if (kge_sta >= threshold
                    and hasattr(self.model.routing, 'use_tta')
                    and not self.model.routing.use_tta):
                cfg.enable_travel_time_attn_epoch = epoch
                actions.append(
                    f"Metric curriculum: kge_sta={kge_sta:.4f} ≥ {threshold} "
                    f"→ activate TTA"
                )

        # ── 5. Phase 1 → phase 2 : auto-unfreeze spatial encoder ──────
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

        # ── 6. NLL autopilot (Kendall & Gal phase 2) ────────────────
        # Adjust w_nll based on KGE regression from deterministic baseline.
        # If val_kge drops too much, reduce w_nll to preserve mean prediction.
        # If val_kge is stable, gradually ramp w_nll to learn more uncertainty.
        if cfg.autopilot_nll and self.loss_fn.w_nll > 0:
            if not hasattr(self, "_ap_nll_initial_kge"):
                # Capture baseline KGE on first autopilot call
                baseline = cfg.autopilot_nll_initial_kge
                if baseline is None:
                    baseline = val_metrics.get("kge", 0.0)
                self._ap_nll_initial_kge = baseline
                self._ap_nll_original = float(self.loss_fn.w_nll)

            baseline = self._ap_nll_initial_kge
            current_kge = val_metrics.get("kge", 0.0)
            regression = (baseline - current_kge) / (abs(baseline) + 1e-8)
            w_nll = self.loss_fn.w_nll

            if regression > cfg.autopilot_nll_max_regression:
                # KGE dropped too much — halve w_nll to protect mean prediction
                new_w = max(w_nll * 0.5, cfg.autopilot_nll_min)
                self.loss_fn.w_nll = new_w
                actions.append(
                    f"KGE regression {regression:.1%} > {cfg.autopilot_nll_max_regression:.0%} "
                    f"(kge={current_kge:.4f} vs baseline={baseline:.4f}) "
                    f"→ w_nll ×0.5 → {new_w:.4f}"
                )
            elif regression < 0.01 and w_nll < self._ap_nll_original:
                # KGE stable and w_nll was previously reduced — ramp back up
                new_w = min(w_nll * cfg.autopilot_nll_ramp_rate, self._ap_nll_original)
                self.loss_fn.w_nll = new_w
                actions.append(
                    f"KGE stable (kge={current_kge:.4f}) → w_nll ramp ×{cfg.autopilot_nll_ramp_rate} → {new_w:.4f}"
                )
            elif regression < 0.01 and w_nll < cfg.autopilot_nll_max:
                # KGE stable — cautiously increase w_nll to learn more uncertainty
                new_w = min(w_nll * cfg.autopilot_nll_ramp_rate, cfg.autopilot_nll_max)
                self.loss_fn.w_nll = new_w
                actions.append(
                    f"KGE stable (kge={current_kge:.4f}) → w_nll ramp ×{cfg.autopilot_nll_ramp_rate} → {new_w:.4f}"
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
