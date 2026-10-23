"""Tests for cold-start hardening of the autopilot.

Covers a behaviour added 2026-05-22 after the slso-kendall-gal-v2 run
showed the autopilot firing three smart restarts in the first 5 epochs
because of natural early-training oscillation: smart restart now requires
epochs_without_improvement >= autopilot_restart_min_no_improve (default 3),
which guards against a single transient bad epoch consuming the restart budget.

Le gestionnaire de dérive qui augmentait w_residual est retiré avec le correcteur
résiduel (2026-10-23), et ses deux tests avec lui.
"""

from types import SimpleNamespace

import torch
import torch.nn as nn

from meandre.training.trainer import Trainer, TrainingConfig


class _StubLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_kge = 1.0


class _StubModel(nn.Module):
    """HydroModel surface."""

    def __init__(self):
        super().__init__()
        self.spatial_encoder = nn.Linear(2, 4)
        self.vertical_column = nn.Linear(4, 4)
        self.routing = nn.Linear(4, 4)


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
    trainer._ap_lr_plateau_count = 0
    trainer._ap_restart_count = 0
    trainer._ap_prev_kge_sta = None
    trainer._cached_spinup_state = None
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
    )
    t = _make_trainer(cfg, _StubModel(), checkpoint_path="dummy.pt")

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
    )
    # Save a fake checkpoint so model.load() doesn't crash
    model = _StubModel()
    ckpt_path = "tests/test_training/_tmp_autopilot_ckpt.pt"
    model.save = lambda p: None  # noop save
    model.load = lambda p: None  # noop load
    t = _make_trainer(cfg, model, checkpoint_path=ckpt_path)

    t._run_autopilot(epoch=5, val_metrics=_val_with_drift(0.50), epochs_without_improvement=3)
    assert t._ap_restart_count == 1


def test_grace_period_blocks_all_autopilot_actions():
    """Within the grace period, no autopilot action fires regardless of
    drift, regression, or no_improve."""
    cfg = TrainingConfig(
        autopilot=True,
        autopilot_grace_epochs=15,
        autopilot_beta_threshold=0.15,
        autopilot_gamma_threshold=0.20,
        autopilot_restart_min_no_improve=3,
    )
    t = _make_trainer(cfg, _StubModel(), checkpoint_path="dummy.pt")
    initial_lr = t.optimizer.param_groups[0]["lr"]

    # Epoch 5 < grace (15): nothing should fire
    t._run_autopilot(epoch=5, val_metrics=_val_with_drift(0.30, beta=2.0, gamma=0.30),
                     epochs_without_improvement=5)
    assert t._ap_restart_count == 0
    assert t.optimizer.param_groups[0]["lr"] == initial_lr
