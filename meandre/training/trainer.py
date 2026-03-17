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

    # Curriculum: epoch at which each module is enabled
    enable_temporal_context_epoch: int = 10
    enable_residual_corrector_epoch: int = 30
    enable_travel_time_attn_epoch: int = 50

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

    # Early stopping: stop training if val_nse hasn't improved for this many
    # epochs. 0 = disabled.
    patience: int = 0

    # Compile hot sub-modules (VerticalColumn + RoutingLayer) with torch.compile.
    # Fuses per-timestep ops into fewer CUDA kernels — big win on GPU, no-op on CPU.
    # Disabled by default; set True once GPU install is confirmed working.
    compile_modules: bool = False


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

        self.optimizer = optimizer or AdamW(
            model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        self._best_val_nse = -float("inf")

        # Mixed precision (AMP) with bfloat16 — same exponent range as float32
        # (no overflow in exp/log physics ops), but faster matmuls on Ampere+.
        # GradScaler not needed for bfloat16.
        self._use_amp = next(model.parameters()).is_cuda
        self._amp_dtype = torch.bfloat16
        if self._use_amp:
            logger.info("AMP enabled (bfloat16 autocast)")

        # Spinup warm-start caches: separate per data object to avoid cross-contamination
        # when train and val have different temporal positions.
        self._spinup_caches: dict[int, tuple[HydroState, Tensor | None]] = {}

        # Maximise intra-op parallelism on CPU
        n_cpu = os.cpu_count() or 1
        torch.set_num_threads(n_cpu)
        try:
            torch.set_num_interop_threads(max(1, n_cpu // 2))
        except RuntimeError:
            pass  # already set or parallel work started

        # Optionally fuse per-timestep ops via torch.compile.
        # Requires Triton (inductor backend), which is Linux-only.
        # On Windows the compile attempt raises BackendCompilerFailed — we
        # catch it and fall back to eager execution with a warning.
        if config is not None and config.compile_modules:
            if next(model.parameters()).is_cuda:
                try:
                    import triton  # noqa: F401 — just test availability
                    model.vertical_column = torch.compile(model.vertical_column)
                    model.routing = torch.compile(model.routing)
                    logger.info("torch.compile enabled for vertical_column + routing.")
                except (ImportError, Exception):
                    logger.warning(
                        "torch.compile requested but Triton is unavailable "
                        "(Windows?). Falling back to eager execution."
                    )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self) -> None:
        """Run training loop for config.n_epochs epochs."""
        from tqdm.auto import tqdm

        self._start_run()

        from meandre.training.scheduler import build_scheduler
        scheduler = build_scheduler(self.optimizer, self.config.n_epochs)

        pbar = tqdm(range(self.config.n_epochs), desc="Training", unit="epoch")
        last_val_metrics: dict[str, float] = {}
        epochs_without_improvement = 0
        for epoch in pbar:
            self._apply_curriculum(epoch)

            train_loss, train_comps = self._train_epoch()

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
            pbar.set_postfix(
                loss=f"{float(train_loss):.4f}",
                val_nse=f"{val_nse:.4f}",
                val_kge=f"{val_kge:.4f}",
                rmse=f"{val_rmse:.2f}",
            )
            current_lr = scheduler.get_last_lr()[0]
            logger.info(
                f"Epoch {epoch:4d} | train={train_loss:.4f} "
                f"| val_nse={val_nse:.4f} | val_kge={val_kge:.4f} "
                f"| rmse={val_rmse:.2f} | nrmse={val_nrmse:.3f}"
                f" | r={val_r:.3f} | beta={val_beta:.3f} | gamma={val_gamma:.3f}"
                f" | kge_log={val_kge_log:.4f}"
                f" | lr={current_lr:.2e}"
            )

            if self.run_logger is not None:
                self.run_logger.log_metrics(
                    {"train_loss": float(train_loss)}
                    | {f"train_{k}": float(v) for k, v in train_comps.items()}
                    | ({f"val_{k}": float(v) for k, v in last_val_metrics.items()}
                       if run_val else {}),
                    step=epoch,
                )

            # Save best checkpoint (only when val was actually computed)
            if run_val and last_val_metrics.get("nse", -999) > self._best_val_nse:
                self._best_val_nse = last_val_metrics["nse"]
                epochs_without_improvement = 0
                if self.checkpoint_path:
                    self.model.save(self.checkpoint_path)
                    print(f"  -> best checkpoint saved (NSE={self._best_val_nse:.4f})")
                    logger.info(f"  -> best checkpoint saved (NSE={self._best_val_nse:.4f})")
            elif run_val:
                epochs_without_improvement += 1
                print(f"  [no save] val_nse={last_val_metrics.get('nse', 'MISSING')}, "
                      f"best={self._best_val_nse}, no_improve={epochs_without_improvement}/{self.config.patience}")

            # Early stopping
            if (self.config.patience > 0
                    and epochs_without_improvement >= self.config.patience):
                logger.info(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {self.config.patience} epochs, "
                    f"best NSE={self._best_val_nse:.4f})"
                )
                break

        if self.run_logger is not None:
            self.run_logger.end_run()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_spinup(self, data: TrainingData) -> tuple[HydroState, Tensor | None]:
        """Return spun-up (state, h_context), using warm-start cache when available.

        Full spinup: run spinup_steps from zeros (epoch 0 or cache disabled).
        Warm spinup: run warm_spinup_steps from last epoch's cached spinup state.

        Caches are keyed by data object id to prevent cross-contamination when
        train and val data have different temporal positions.
        """
        cfg = self.config
        device = data.forcing.device
        spinup_end = min(cfg.spinup_steps, data.train_slice.start)

        if spinup_end == 0:
            return HydroState.zeros(self.model.n_nodes, device=device), None

        data_id = id(data)
        warm = cfg.warm_spinup_steps
        cached = self._spinup_caches.get(data_id)
        if warm > 0 and cached is not None:
            # Mini-spinup from cached state: only re-run the last `warm` steps
            cached_state, cached_h = cached
            warm_start = max(0, spinup_end - warm)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=self._amp_dtype, enabled=self._use_amp):
                _, spun_state = self.model.simulate(
                    forcing=data.forcing[warm_start:spinup_end],
                    initial_state=cached_state,
                    graph=data.graph,
                    node_coords=data.node_coords,
                    territorial=data.territorial,
                    withdrawals=data.withdrawals,
                    day_of_year=data.day_of_year[warm_start:spinup_end],
                    h_context=cached_h,
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

        h_ctx = self.model._last_h_context
        # Cache per data object for next epoch
        self._spinup_caches[data_id] = (
            spun_state.detach(),
            h_ctx.detach() if h_ctx is not None else None,
        )
        return spun_state, h_ctx

    def _simulate(
        self, data: TrainingData, time_slice: slice, tbptt_steps: int = 0
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

                # Loss on last half of chunk only: first half is burn-in so
                # the model can't cheat by draining initial storage.
                # Gradients still flow through the full chunk (via Q_chunk).
                burnin = chunk_len // 2
                q_obs_chunk = data.q_obs[obs_offset + burnin:obs_offset + chunk_len]
                Q_chunk_loss = Q_chunk[burnin:]
                loss_chunk, comps = self.loss_fn(
                    q_obs=q_obs_chunk,
                    q_sim=Q_chunk_loss,
                    station_mask=data.station_mask,
                    residual_gate_logits=(
                        self.model.residual_corrector.gate_logit
                        if self.model.use_residual and self.model.residual_corrector is not None
                        else None
                    ),
                )

            # Scale by chunk fraction so total gradient ≈ full-series gradient
            weight = chunk_len / n_train
            if not torch.isnan(loss_chunk):
                (loss_chunk * weight).backward()

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
            Q_sim, _ = self._simulate(data, data.train_slice, tbptt_steps=self.config.tbptt_steps)

            n_train = data.train_slice.stop - data.train_slice.start
            q_obs_train = data.q_obs[:n_train]

            loss, components = self.loss_fn(
                q_obs=q_obs_train,
                q_sim=Q_sim,
                station_mask=data.station_mask,
                residual_gate_logits=(
                    self.model.residual_corrector.gate_logit
                    if self.model.use_residual and self.model.residual_corrector is not None
                    else None
                ),
            )

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
        """Validation: simulate val period, compute evaluation metrics."""
        self.model.eval()
        data = self.val_data

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=self._amp_dtype, enabled=self._use_amp):
            Q_sim, _ = self._simulate(data, data.val_slice)

        # q_obs is pre-sliced to start at the beginning of the val period.
        n_val = data.val_slice.stop - data.val_slice.start
        q_obs_val = data.q_obs[:n_val]  # (T_val, n_stations)
        q_sim_at_stations = Q_sim[:, data.station_mask]

        # Flatten and mask NaN (in obs or sim — catches NaN model weights)
        q_o = q_obs_val.reshape(-1)
        q_s = q_sim_at_stations.reshape(-1)
        valid = ~torch.isnan(q_o) & ~torch.isnan(q_s)
        q_o, q_s = q_o[valid], q_s[valid]

        if q_o.numel() == 0:
            return {"nse": float("nan"), "kge": float("nan"), "pbias": float("nan"),
                    "rmse": float("nan"), "nrmse": float("nan"), "mae": float("nan"),
                    "r": float("nan"), "beta": float("nan"), "gamma": float("nan"),
                    "r_log": float("nan"), "beta_log": float("nan"), "gamma_log": float("nan"),
                    "kge_log": float("nan")}

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

        if hasattr(self.model.routing, "tta"):
            enabled = epoch >= cfg.enable_travel_time_attn_epoch
            self.model.routing.use_tta = enabled
            for p in self.model.routing.tta.parameters():
                p.requires_grad_(enabled)

    def _start_run(self) -> None:
        if self.run_logger is None:
            return
        self.run_logger.start_run(self.run_name)
        self.run_logger.log_params({
            k: v for k, v in vars(self.config).items()
            if not k.startswith("_")
        })
