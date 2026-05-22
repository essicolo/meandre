"""Tests for cold-start hardening of the autopilot.

Covers two behaviours added 2026-05-22 after the slso-kendall-gal-v2 run
showed the autopilot firing three smart restarts in the first 5 epochs
because of natural early-training oscillation:

1. Smart restart now requires epochs_without_improvement >=
   autopilot_restart_min_no_improve (default 3) — guards against a single
   transient bad epoch consuming the restart budget.

2. β/γ drift handler is skipped when the residual corrector is disabled
   (enable_residual_corrector_epoch == 9999 or model has no corrector) —
   incrementing w_residual otherwise has no effect on training and just
   pollutes the log.
"""

from types import SimpleNamespace

import torch
import torch.nn as nn

from meandre.training.trainer import Trainer, TrainingConfig


class _StubLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_nll = 0.0
        self.w_kge = 1.0
        self.w_residual = 0.0


class _StubModelNoCorrector(nn.Module):
    """HydroModel surface without a residual corrector."""

    def __init__(self):
        super().__init__()
        self.spatial_encoder = nn.Linear(2, 4)
        self.temporal_encoder = nn.Linear(4, 4)
        self.vertical_column = nn.Linear(4, 4)
        self.routing = nn.Linear(4, 4)
        # No residual_corrector attribute on purpose


class _StubModelWithCorrector(nn.Module):
    """HydroModel surface with an active residual corrector."""

    def __init__(self):
        super().__init__()
        self.spatial_encoder = nn.Linear(2, 4)
        self.temporal_encoder = nn.Linear(4, 4)
        self.vertical_column = nn.Linear(4, 4)
        self.routing = nn.Linear(4, 4)
        self.residual_corrector = nn.Linear(4, 1)


def _make_trainer(cfg: TrainingConfig, model: nn.Module, checkpoint_path: str | None = None) -> Trainer:
    """Build a Trainer bypassing __init__ — inject only what _run_autopilot needs."""
    trainer = Trainer.__new__(Trainer)
    trainer.config = cfg
    trainer.model = model
    trainer.loss_fn = _StubLoss()
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    trainer.checkpoint_path = checkpoint_path
    trainer._best_val_metric = 0.70           # last best kge_station
    trainer._best_metric_lower_is_better = False
    trainer._ap_w_residual_orig = 0.0
    trainer._ap_lr_plateau_count = 0
    trainer._ap_restart_count = 0
    trainer._ap_prev_kge_sta = None
    trainer._cached_spinup_state = None
    # KGA state (not exercised here but Trainer references it elsewhere)
    trainer._kga_phase = None
    return trainer


def _val_with_drift(kge_sta: float, beta: float = 1.30, gamma: float = 0.70) -> dict:
    """Validation metrics with explicit β/γ drift values."""
    return {
        "kge_station": kge_sta,
        "beta": beta,
        "gamma": gamma,
    }


def test_restart_requires_persistent_regression():
    """A single transient regression must NOT trigger smart restart."""
    cfg = TrainingConfig(
        autopilot=True,
        autopilot_grace_epochs=0,
        autopilot_restart_regression=0.05,
        autopilot_restart_min_no_improve=3,
        autopilot_beta_threshold=0.15,
        autopilot_gamma_threshold=0.20,
        enable_residual_corrector_epoch=9999,  # corrector off
    )
    t = _make_trainer(cfg, _StubModelNoCorrector(), checkpoint_path="dummy.pt")

    # Single bad epoch (kge=0.50 vs best=0.70 → 28% regression), but
    # epochs_without_improvement = 1 < min_no_improve = 3 → no restart.
    t._run_autopilot(epoch=5, val_metrics=_val_with_drift(0.50), epochs_without_improvement=1)
    assert t._ap_restart_count == 0


def test_restart_fires_on_persistent_regression():
    """Persistent regression (>= min_no_improve consecutive bad epochs)
    DOES trigger smart restart."""
    cfg = TrainingConfig(
        autopilot=True,
        autopilot_grace_epochs=0,
        autopilot_restart_regression=0.05,
        autopilot_restart_min_no_improve=3,
        autopilot_beta_threshold=0.15,
        autopilot_gamma_threshold=0.20,
        autopilot_restart_max=3,
        enable_residual_corrector_epoch=9999,
    )
    # Save a fake checkpoint so model.load() doesn't crash
    model = _StubModelNoCorrector()
    ckpt_path = "tests/test_training/_tmp_autopilot_ckpt.pt"
    model.save = lambda p: None  # noop save
    model.load = lambda p: None  # noop load
    t = _make_trainer(cfg, model, checkpoint_path=ckpt_path)

    t._run_autopilot(epoch=5, val_metrics=_val_with_drift(0.50), epochs_without_improvement=3)
    assert t._ap_restart_count == 1


def test_drift_handler_skipped_when_corrector_disabled():
    """β/γ drift must NOT increment w_residual when the corrector is off
    (enable_residual_corrector_epoch == 9999)."""
    cfg = TrainingConfig(
        autopilot=True,
        autopilot_grace_epochs=0,
        autopilot_beta_threshold=0.15,
        autopilot_gamma_threshold=0.20,
        autopilot_beta_penalty=0.005,
        autopilot_gamma_penalty=0.003,
        enable_residual_corrector_epoch=9999,  # OFF
    )
    t = _make_trainer(cfg, _StubModelNoCorrector())
    initial_w = t.loss_fn.w_residual

    # Heavy β + γ drift, should normally trigger increments
    t._run_autopilot(epoch=5, val_metrics=_val_with_drift(0.65, beta=1.40, gamma=0.60),
                     epochs_without_improvement=0)

    assert t.loss_fn.w_residual == initial_w, (
        f"w_residual moved from {initial_w} to {t.loss_fn.w_residual} "
        f"despite corrector being disabled"
    )


def test_drift_handler_fires_when_corrector_active():
    """β/γ drift DOES increment w_residual when the corrector is enabled
    and present on the model — control case for the previous test."""
    cfg = TrainingConfig(
        autopilot=True,
        autopilot_grace_epochs=0,
        autopilot_beta_threshold=0.15,
        autopilot_gamma_threshold=0.20,
        autopilot_beta_penalty=0.005,
        autopilot_gamma_penalty=0.003,
        enable_residual_corrector_epoch=0,  # ON from start
    )
    t = _make_trainer(cfg, _StubModelWithCorrector())
    initial_w = t.loss_fn.w_residual

    t._run_autopilot(epoch=5, val_metrics=_val_with_drift(0.65, beta=1.40, gamma=0.60),
                     epochs_without_improvement=0)

    assert t.loss_fn.w_residual > initial_w, (
        f"w_residual stayed at {initial_w} despite corrector being active"
    )


def test_grace_period_blocks_all_autopilot_actions():
    """Within the grace period, no autopilot action fires regardless of
    drift, regression, or no_improve."""
    cfg = TrainingConfig(
        autopilot=True,
        autopilot_grace_epochs=15,
        autopilot_beta_threshold=0.15,
        autopilot_gamma_threshold=0.20,
        autopilot_restart_min_no_improve=3,
        enable_residual_corrector_epoch=0,
    )
    t = _make_trainer(cfg, _StubModelWithCorrector(), checkpoint_path="dummy.pt")
    initial_w = t.loss_fn.w_residual

    # Epoch 5 < grace (15): nothing should fire
    t._run_autopilot(epoch=5, val_metrics=_val_with_drift(0.30, beta=2.0, gamma=0.30),
                     epochs_without_improvement=5)
    assert t._ap_restart_count == 0
    assert t.loss_fn.w_residual == initial_w
