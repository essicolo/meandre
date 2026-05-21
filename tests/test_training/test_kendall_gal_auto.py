"""Tests for the Kendall-Gal phase 1→2 auto-transition in the trainer.

The transition logic is exercised by directly calling _maybe_transition_kga
and _apply_phase2_config with a stub Trainer that has the minimum required
attributes — avoids spinning up a full HydroModel.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn as nn

from meandre.training.trainer import Trainer, TrainingConfig


class _StubLoss(nn.Module):
    """Minimal stand-in for HydroLoss exposing the weight attributes that
    _apply_phase2_config might try to setattr."""

    def __init__(self):
        super().__init__()
        self.w_nll = 0.0
        self.w_kge = 1.0
        self.w_mse = 0.5
        self.w_log_mse = 0.3
        self.w_pbias = 0.1
        self.w_nrmse = 0.0
        self.w_nse = 0.0
        self.w_log_nse = 0.0
        self.w_physics = 0.0
        self.w_residual = 0.0
        self.w_nll_et = 0.0
        self.w_nll_swe = 0.0


class _StubModel(nn.Module):
    """Mimics the HydroModel surface area used by _apply_phase2_config:
    spatial_encoder / temporal_encoder / vertical_column / routing each
    expose a Parameter so freezing is observable."""

    def __init__(self):
        super().__init__()
        self.spatial_encoder = nn.Linear(2, 4)
        self.temporal_encoder = nn.Sequential()
        # Two named params: one will be frozen, one (drop.*) stays trainable
        self.temporal_encoder.add_module("gru", nn.Linear(4, 4))
        self.temporal_encoder.add_module("drop", nn.Linear(4, 4))
        self.vertical_column = nn.Linear(4, 4)
        self.routing = nn.Linear(4, 4)


def _make_stub_trainer(cfg: TrainingConfig) -> Trainer:
    """Build a Trainer skipping __init__ (which needs full data + spinup)
    and inject just the attributes touched by the transition logic."""
    trainer = Trainer.__new__(Trainer)
    trainer.config = cfg
    trainer.model = _StubModel()
    trainer.loss_fn = _StubLoss()
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=1e-3)
    trainer._kga_phase = 1
    trainer._kga_best_kge = -float("inf")
    trainer._kga_plateau_counter = 0
    trainer._kga_phase1_start_epoch = 0
    trainer._best_metric_lower_is_better = False
    trainer._best_val_metric = -float("inf")
    return trainer


def test_transition_fires_on_kge_threshold():
    """kge_station >= threshold → immediate transition once min_epochs hit."""
    cfg = TrainingConfig(
        kendall_gal_auto=True,
        kga_phase1_kge_threshold=0.80,
        kga_phase1_plateau_patience=999,  # disable plateau path
        kga_phase1_min_epochs=3,
        kga_phase2_loss_weights={"w_nll": 0.1, "w_kge": 1.0, "w_mse": 0.0},
        kga_phase2_best_metric="nll",
    )
    t = _make_stub_trainer(cfg)

    # Before min_epochs: no transition even if kge >= threshold
    assert t._maybe_transition_kga(0, {"kge_station": 0.90}) is False
    assert t._kga_phase == 1

    # After min_epochs with kge above threshold: transition fires
    assert t._maybe_transition_kga(3, {"kge_station": 0.90}) is True
    assert t._kga_phase == 2
    assert t.loss_fn.w_nll == 0.1
    assert t.loss_fn.w_mse == 0.0
    assert t.config.best_metric == "nll"
    assert t._best_val_metric == float("inf")  # reset for lower-is-better


def test_transition_fires_on_plateau():
    """No KGE gain for N epochs → transition even below threshold."""
    cfg = TrainingConfig(
        kendall_gal_auto=True,
        kga_phase1_kge_threshold=0.99,  # too high to trigger by threshold
        kga_phase1_plateau_patience=3,
        kga_phase1_min_epochs=0,
    )
    t = _make_stub_trainer(cfg)

    # First call sets best, no plateau yet
    assert t._maybe_transition_kga(1, {"kge_station": 0.50}) is False
    # Three calls without improvement → plateau exhausted, transition
    assert t._maybe_transition_kga(2, {"kge_station": 0.49}) is False
    assert t._maybe_transition_kga(3, {"kge_station": 0.50}) is False
    assert t._maybe_transition_kga(4, {"kge_station": 0.48}) is True
    assert t._kga_phase == 2


def test_plateau_counter_resets_on_improvement():
    """Strict KGE gain resets the plateau counter."""
    cfg = TrainingConfig(
        kendall_gal_auto=True,
        kga_phase1_kge_threshold=0.99,
        kga_phase1_plateau_patience=3,
        kga_phase1_min_epochs=0,
    )
    t = _make_stub_trainer(cfg)
    t._maybe_transition_kga(1, {"kge_station": 0.50})
    t._maybe_transition_kga(2, {"kge_station": 0.49})
    t._maybe_transition_kga(3, {"kge_station": 0.55})  # improvement
    assert t._kga_plateau_counter == 0
    assert t._kga_best_kge == 0.55


def test_freeze_applied_correctly():
    """Phase 2 freeze flags actually flip requires_grad on the right modules."""
    cfg = TrainingConfig(
        kendall_gal_auto=True,
        kga_phase1_kge_threshold=0.0,
        kga_phase1_plateau_patience=999,
        kga_phase1_min_epochs=0,
        kga_phase2_freeze_spatial=True,
        kga_phase2_freeze_temporal=True,
        kga_phase2_freeze_backbone=True,
    )
    t = _make_stub_trainer(cfg)
    assert t._maybe_transition_kga(1, {"kge_station": 0.99}) is True

    # Spatial: all frozen
    assert not any(p.requires_grad for p in t.model.spatial_encoder.parameters())
    # Temporal: gru frozen, drop trainable (epistemic uncertainty)
    gru_params = list(t.model.temporal_encoder.gru.parameters())
    drop_params = list(t.model.temporal_encoder.drop.parameters())
    assert not any(p.requires_grad for p in gru_params)
    assert all(p.requires_grad for p in drop_params)
    # Vertical + routing: frozen
    assert not any(p.requires_grad for p in t.model.vertical_column.parameters())
    assert not any(p.requires_grad for p in t.model.routing.parameters())


def test_no_transition_when_disabled():
    """kendall_gal_auto=False → _kga_phase is None, no transition tracking."""
    cfg = TrainingConfig(kendall_gal_auto=False)
    t = _make_stub_trainer(cfg)
    t._kga_phase = None
    # Even at high kge, no transition occurs
    assert t._maybe_transition_kga(100, {"kge_station": 0.99}) is False


def test_phase2_lr_override():
    """kga_phase2_lr sets optimizer LR at transition."""
    cfg = TrainingConfig(
        kendall_gal_auto=True,
        kga_phase1_kge_threshold=0.0,
        kga_phase1_min_epochs=0,
        kga_phase2_lr=5e-5,
    )
    t = _make_stub_trainer(cfg)
    assert t._maybe_transition_kga(1, {"kge_station": 0.99}) is True
    assert t.optimizer.param_groups[0]["lr"] == 5e-5
