"""Training loop with RunLogger logging and curriculum support.

Implements the four-phase curriculum from README section 6.6:
    Phase 1: Pure physics + NeRF (temporal modules disabled)
    Phase 2: Enable temporal context encoder
    Phase 3: Enable state residual corrector
    Phase 4: Enable travel-time attention in routing
"""

from __future__ import annotations

import logging
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

    # Anchor on the noise head's log_sigma_a (and optionally log_sigma_b).
    # Counters the Gaussian NLL degeneracy where σ inflates to fit any μ.
    # Apply to noise_head (Q), noise_head_et, noise_head_swe symmetrically.
    # 0 = no regularization. Typical w_sigma_anchor ≈ 0.1-1.0.
    w_sigma_anchor: float = 0.0
    sigma_anchor_target_a: float = -3.0   # log σ baseline (5% when b=1 at Q=100)
    sigma_anchor_target_b: float | None = None   # None = let b float free

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
    # best AND beta/gamma drift is detected, reload best checkpoint and
    # reduce LR.  More targeted than the existing divergence guard (which
    # only checks train loss spikes).
    autopilot_restart_regression: float = 0.05  # 5% regression from best
    autopilot_restart_max: int = 3               # max restarts

    # Metric-driven curriculum: instead of fixed epoch numbers, activate
    # modules when kge_station reaches a threshold.  Overrides the fixed
    # epoch settings when autopilot is enabled and these are > 0.
    autopilot_activate_residual_at_kge: float | None = None  # e.g. 0.55
    autopilot_activate_tta_at_kge: float | None = None       # e.g. 0.65


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
            self.optimizer = AdamW(groups)
        # Autopilot: force val_every=1 for fast reaction
        if self.config.autopilot and self.config.val_every > 1:
            logger.info(
                f"[autopilot] val_every forced from {self.config.val_every} to 1 "
                "(autopilot requires per-epoch validation)"
            )
            self.config.val_every = 1

        # Best-metric init depends on direction (lower-is-better for loss-like)
        self._best_metric_lower_is_better = self.config.best_metric in ("loss", "val_loss")
        self._best_val_metric = (
            float("inf") if self._best_metric_lower_is_better else -float("inf")
        )

        # Autopilot state
        self._ap_w_residual_orig: float | None = None  # original w_residual
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
        )

        pbar = tqdm(range(self.config.n_epochs), desc="Training", unit="epoch")
        last_val_metrics: dict[str, float] = {}
        epochs_without_improvement = 0
        _loss_ema = None          # exponential moving average of train loss
        _rollback_count = 0       # number of LR reductions so far
        _MAX_ROLLBACKS = 3        # stop reducing after this many
        for epoch in pbar:
            self._apply_curriculum(epoch)

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
                scheduler = build_scheduler(self.optimizer, remaining, warmup_epochs=0)
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
                f" | lr={current_lr:.2e}"
            )
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
            if _bm == "loss":
                _cur = float(train_loss)
            elif _bm == "val_loss":
                _cur = last_val_metrics.get("loss", float("inf")) if run_val else float("inf")
            else:
                _cur = last_val_metrics.get(_bm, -float("inf")) if run_val else -float("inf")

            tracked_now = run_val or (_bm == "loss")
            if tracked_now:
                if self._best_metric_lower_is_better:
                    is_improvement = _cur < self._best_val_metric * (1 - _tol)
                else:
                    is_improvement = _cur > self._best_val_metric * (1 + _tol)

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

        while t_start < data.train_slice.stop:
            t_end = min(t_start + chunk, data.train_slice.stop)
            # Merge tiny remainder into the previous chunk
            remainder = data.train_slice.stop - t_end
            if 0 < remainder < 60:
                t_end = data.train_slice.stop
            sl = slice(t_start, t_end)
            chunk_len = t_end - t_start

            with torch.amp.autocast("cuda", dtype=self._amp_dtype, enabled=self._use_amp):
                Q_chunk, state_out = self.model.simulate(
                    forcing=data.forcing[sl],
                    initial_state=state,
                    graph=data.graph,
                    node_coords=data.node_coords,
                    territorial=data.territorial,
                    withdrawals=data.withdrawals,
                    day_of_year=data.day_of_year[sl],
                    h_context=h_ctx,
                    tbptt_steps=self.config.tbptt_steps,
                )

                # Burn-in: first chunk after spinup needs longer burn-in to
                # stabilize state; subsequent chunks only need a short warm-up
                # since state is propagated from the previous chunk.
                burnin = min(60, chunk_len // 4) if n_chunks > 0 else min(90, chunk_len // 4)
                q_obs_chunk = data.q_obs[obs_offset + burnin:obs_offset + chunk_len]
                Q_chunk_loss = Q_chunk[burnin:]
                log_sigma_chunk = (
                    self.model.noise_head(Q_chunk_loss)
                    if self.loss_fn.w_nll > 0 else None
                )
                loss_chunk, comps = self.loss_fn(
                    q_obs=q_obs_chunk,
                    q_sim=Q_chunk_loss,
                    station_mask=data.station_mask,
                    log_sigma_sim=log_sigma_chunk,
                    residual_gate_logits=(
                        self.model.residual_corrector.gate_logit
                        if self.model.use_residual and self.model.residual_corrector is not None
                        else None
                    ),
                )

            # Soft physical prior regularization on spatial params (k_gw, krec, K_sat, K_c, ...).
            # Pulls toward literature targets — prevents drift toward overfit values.
            if self.config.w_prior > 0:
                params_t = self.model.spatial_encoder(
                    data.node_coords, data.territorial.to_tensor()
                )
                prior_loss = self.model.spatial_encoder.physical_prior_loss(params_t)
                loss_chunk = loss_chunk + self.config.w_prior * prior_loss
                all_components["prior"] = all_components.get("prior", 0.0) + float(prior_loss.detach())

            # Noise head σ anchor — counters NLL degeneracy (σ inflates to
            # mask a bad μ). Applied symmetrically to Q / ET / SWE heads.
            if self.config.w_sigma_anchor > 0 and self.loss_fn.w_nll > 0:
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
        total_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.grad_clip
        )
        if torch.isnan(total_norm) or torch.isinf(total_norm):
            logger.warning(
                f"NaN/Inf gradient norm ({float(total_norm):.2f}) — "
                "zeroing gradients to protect weights."
            )
            self.optimizer.zero_grad()
        else:
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

            log_sigma_sim = (
                self.model.noise_head(Q_sim) if self.loss_fn.w_nll > 0 else None
            )
            # TODO multi-obj wiring (NLL ET / SWE) :
            # Quand data.et_obs ou data.swe_obs sont fournis et w_nll_et/_swe > 0,
            # appeler self.model.simulate(..., return_diagnostics=True) au lieu de
            # _simulate() ; extraire et_sim=diag.etr et swe_sim=diag.swe ; appliquer
            # self.model.noise_head_et / _swe pour les log_sigma respectifs ;
            # passer à loss_fn. La même logique s'applique au chemin chunké
            # ci-dessus. Bloqué par l'absence de loader MODIS (cf.
            # meandre/data/open_data.py download_modis_et / download_modis_swe).
            loss, components = self.loss_fn(
                q_obs=q_obs_train,
                q_sim=Q_sim,
                station_mask=data.station_mask,
                log_sigma_sim=log_sigma_sim,
                residual_gate_logits=(
                    self.model.residual_corrector.gate_logit
                    if self.model.use_residual and self.model.residual_corrector is not None
                    else None
                ),
            )

        # Soft physical prior regularization
        if self.config.w_prior > 0:
            params_t = self.model.spatial_encoder(
                data.node_coords, data.territorial.to_tensor()
            )
            prior_loss = self.model.spatial_encoder.physical_prior_loss(params_t)
            loss = loss + self.config.w_prior * prior_loss
            components["prior"] = prior_loss.detach()

        # Noise head σ anchor (single-pass path) — same logic as chunked.
        if self.config.w_sigma_anchor > 0 and self.loss_fn.w_nll > 0:
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

        if torch.isnan(loss):
            logger.warning(
                "NaN loss detected — skipping backward. "
                f"Components: { {k: float(v) for k, v in components.items()} }"
            )
            self.optimizer.zero_grad()
            return loss.detach(), {k: v.detach() for k, v in components.items()}

        loss.backward()

        total_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.grad_clip
        )
        if torch.isnan(total_norm) or torch.isinf(total_norm):
            logger.warning(
                f"NaN/Inf gradient norm ({float(total_norm):.2f}) — "
                "zeroing gradients to protect weights."
            )
            self.optimizer.zero_grad()
        else:
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
            regression = (self._best_val_metric - kge_sta) / (abs(self._best_val_metric) + 1e-8)
        drift_detected = (beta_drift > cfg.autopilot_beta_threshold
                          or gamma_drift > cfg.autopilot_gamma_threshold)

        if (regression > cfg.autopilot_restart_regression
                and drift_detected
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
