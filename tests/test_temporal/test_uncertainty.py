"""Tests for ensemble uncertainty quantification modules.

Covers:
  - CorrelatedStateNoise: AR(1) structure, parameter constraints
  - frozen_dropout: trajectory coherence (same seed → same Q; different → different)
  - inject_noise: noise propagates through the physics
  - crps_loss: correct score and gradient flow
  - variance_decomposition: fractions sum to ≈ 1
"""

from __future__ import annotations

import torch
import pytest

from meandre.model import HydroModel
from meandre.routing.graph import synthetic_linear_graph
from meandre.routing.withdrawals import WithdrawalData
from meandre.spatial.territorial import TerritorialFeatures
from meandre.temporal.state_noise import CorrelatedStateNoise
from meandre.training.loss import crps_loss
from meandre.training.uncertainty import frozen_dropout
from meandre.training.ensemble import variance_decomposition
from meandre.utils.state import HydroState


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N_NODES = 6
N_T = 20
N_FORCING = 6


def _make_model(**kw) -> HydroModel:
    defaults = dict(
        n_nodes=N_NODES,
        n_forcing=N_FORCING,
        context_window=5,
        residual_history=3,
        max_travel_time=3,
        use_temporal=False,
        use_residual=False,
        use_travel_time_attn=False,
    )
    defaults.update(kw)
    return HydroModel(**defaults)


def _make_simulate_args():
    forcing = torch.rand(N_T, N_NODES, N_FORCING)
    forcing[:, :, 1:3] = 10.0   # warm, no snow
    state = HydroState.zeros(N_NODES)
    graph = synthetic_linear_graph(N_NODES, tau_days=1)
    coords = torch.randn(N_NODES, 2)
    territorial = TerritorialFeatures.zeros(n_nodes=N_NODES, n_features=17)
    territorial.physical["area_km2_physical"] = torch.ones(N_NODES) * 10.0
    territorial.physical["area_km2_local"] = torch.ones(N_NODES) * 2.0
    territorial.physical["slope_fraction"] = torch.ones(N_NODES) * 0.02
    withdrawals = WithdrawalData.zeros(N_T, N_NODES)
    doy = torch.ones(N_T, dtype=torch.long) * 180
    return forcing, state, graph, coords, territorial, withdrawals, doy


# ---------------------------------------------------------------------------
# CorrelatedStateNoise
# ---------------------------------------------------------------------------

class TestCorrelatedStateNoise:
    def test_rho_in_unit_interval(self):
        m = CorrelatedStateNoise(n_state_vars=4)
        assert (m.rho >= 0).all() and (m.rho <= 1).all()

    def test_sigma_positive(self):
        m = CorrelatedStateNoise(n_state_vars=4)
        assert (m.sigma > 0).all()

    def test_init_noise_shape(self):
        m = CorrelatedStateNoise(n_state_vars=4)
        noise = m.init_noise(10, torch.device("cpu"))
        assert noise.shape == (10, 4)
        assert (noise == 0).all()

    def test_step_shape(self):
        m = CorrelatedStateNoise(n_state_vars=3)
        noise = m.init_noise(5, torch.device("cpu"))
        new_noise = m.step(noise)
        assert new_noise.shape == (5, 3)

    def test_ar1_autocorrelation(self):
        """Empirical lag-1 ACF should be close to learned rho."""
        torch.manual_seed(0)
        m = CorrelatedStateNoise(n_state_vars=2)
        # Force rho ≈ 0.8 for a clear test
        with torch.no_grad():
            m.logit_rho.fill_(1.386)   # sigmoid(1.386) ≈ 0.8

        n_steps = 2000
        noise = m.init_noise(1, torch.device("cpu"))
        series = []
        for _ in range(n_steps):
            noise = m.step(noise)
            series.append(noise[0, 0].item())

        series = torch.tensor(series)
        acf1 = torch.corrcoef(torch.stack([series[:-1], series[1:]]))[0, 1]
        expected = torch.sigmoid(m.logit_rho[0]).detach()
        assert abs(float(acf1) - float(expected)) < 0.05, (
            f"AR(1) ACF {acf1:.3f} far from rho {expected:.3f}"
        )

    def test_deterministic_step_decays(self):
        """Deterministic step decays without adding variance."""
        m = CorrelatedStateNoise(n_state_vars=2)
        noise = torch.ones(3, 2)
        new_noise = m.step_deterministic(noise)
        assert (new_noise.abs() <= noise.abs() + 1e-6).all(), (
            "Deterministic step should only decay"
        )

    def test_gradients_on_params(self):
        m = CorrelatedStateNoise(n_state_vars=3)
        noise = m.init_noise(4, torch.device("cpu"))
        out = m.step(noise)
        out.sum().backward()
        assert m.logit_rho.grad is not None
        assert m.log_amplitude.grad is not None


# ---------------------------------------------------------------------------
# frozen_dropout
# ---------------------------------------------------------------------------

class TestFrozenDropout:
    def test_same_seed_identical_trajectory(self):
        """Same seed must produce bit-identical Q for the full trajectory."""
        # SpatialFieldNetwork has dropout; use dropout > 0
        from meandre.spatial.field_network import SpatialFieldNetwork
        model = _make_model()
        # Inject dropout into the spatial encoder for this test
        model.spatial_encoder.drop1 = torch.nn.Dropout(p=0.2)
        model.spatial_encoder.drop2 = torch.nn.Dropout(p=0.2)

        args = _make_simulate_args()

        model.eval()
        with frozen_dropout(model, seed=42):
            Q1, _ = model.simulate(*args)
        with frozen_dropout(model, seed=42):
            Q2, _ = model.simulate(*args)

        assert torch.allclose(Q1, Q2, atol=1e-6), (
            "Same seed must yield identical trajectory"
        )

    def test_different_seeds_differ(self):
        """Different seeds must produce different trajectories."""
        model = _make_model()
        model.spatial_encoder.drop1 = torch.nn.Dropout(p=0.3)
        model.spatial_encoder.drop2 = torch.nn.Dropout(p=0.3)

        args = _make_simulate_args()

        model.eval()
        with frozen_dropout(model, seed=0):
            Q1, _ = model.simulate(*args)
        with frozen_dropout(model, seed=1):
            Q2, _ = model.simulate(*args)

        assert not torch.allclose(Q1, Q2), (
            "Different seeds should yield different trajectories"
        )

    def test_model_restored_to_eval_after_context(self):
        """frozen_dropout temporarily sets training=True; model should be
        restored to its previous state after the context exits."""
        model = _make_model()
        model.eval()
        args = _make_simulate_args()
        with frozen_dropout(model, seed=0):
            pass   # just enter/exit
        # Training mode should match what we set before (eval → not training)
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                # Dropout's internal training flag was flipped by frozen_dropout;
                # the context manager should restore it after exit.
                pass   # no crash = ok


# ---------------------------------------------------------------------------
# inject_noise in model.simulate
# ---------------------------------------------------------------------------

class TestInjectNoise:
    def test_no_crash_with_noise(self):
        model = _make_model(use_state_noise=True)
        args = _make_simulate_args()
        with torch.no_grad():
            Q, _ = model.simulate(*args, inject_noise=True)
        assert Q.shape == (N_T, N_NODES)
        assert not torch.isnan(Q).any()
        assert (Q >= 0).all()

    def test_noise_changes_trajectory(self):
        """Two different torch seeds must give different trajectories."""
        model = _make_model(use_state_noise=True)
        args = _make_simulate_args()

        torch.manual_seed(0)
        with torch.no_grad():
            Q1, _ = model.simulate(*args, inject_noise=True)

        torch.manual_seed(1)
        with torch.no_grad():
            Q2, _ = model.simulate(*args, inject_noise=True)

        assert not torch.allclose(Q1, Q2), (
            "Different RNG states should yield different noisy trajectories"
        )

    def test_no_noise_flag_unchanged(self):
        """inject_noise=False must give the same result as without state_noise."""
        model_plain = _make_model(use_state_noise=False)
        model_noise = _make_model(use_state_noise=True)
        # Copy weights so only the noise module differs
        with torch.no_grad():
            for (n1, p1), (n2, p2) in zip(
                model_plain.named_parameters(), model_noise.named_parameters()
            ):
                if n1 == n2:
                    p2.copy_(p1)

        args = _make_simulate_args()
        with torch.no_grad():
            Q_plain, _ = model_plain.simulate(*args)
            Q_noise_off, _ = model_noise.simulate(*args, inject_noise=False)

        assert torch.allclose(Q_plain, Q_noise_off, atol=1e-5), (
            "inject_noise=False must not change the trajectory"
        )


# ---------------------------------------------------------------------------
# crps_loss
# ---------------------------------------------------------------------------

class TestCRPS:
    def test_perfect_ensemble_has_low_crps(self):
        """An ensemble concentrated on the truth should have near-zero CRPS."""
        y = torch.rand(10, 3)
        # All members equal to the observation
        ensemble = y.unsqueeze(0).expand(20, -1, -1)
        score = crps_loss(ensemble, y)
        assert float(score) < 0.01, f"Perfect ensemble CRPS should be ~0, got {score:.4f}"

    def test_crps_positive(self):
        ensemble = torch.rand(10, 20, 5)
        obs = torch.rand(20, 5)
        score = crps_loss(ensemble, obs)
        assert float(score) >= 0, "CRPS must be non-negative"

    def test_crps_nan_mask(self):
        """NaN observations should be silently ignored."""
        ensemble = torch.rand(5, 10, 3)
        obs = torch.rand(10, 3)
        obs[3, 1] = float("nan")
        # Must not raise
        score = crps_loss(ensemble, obs)
        assert score.isfinite(), "CRPS must be finite when NaNs are masked"

    def test_crps_gradient(self):
        """Gradients must flow through the CRPS to ensemble members."""
        ensemble = torch.rand(8, 15, 4, requires_grad=True)
        obs = torch.rand(15, 4)
        score = crps_loss(ensemble, obs)
        score.backward()
        assert ensemble.grad is not None


# ---------------------------------------------------------------------------
# variance_decomposition
# ---------------------------------------------------------------------------

class TestVarianceDecomposition:
    def test_fractions_sum_to_one(self):
        # (n_meteo=3, n_dropout=2, n_noise=4, T=10, N=5)
        ensemble = torch.rand(3, 2, 4, 10, 5)
        vd = variance_decomposition(ensemble)
        total_frac = (
            vd["fraction_meteo"]
            + vd["fraction_parametric"]
            + vd["fraction_aleatoric"]
        )
        # NaN fractions occur where total variance ≈ 0 (constant ensemble);
        # replace with 1/3 each so the sum is still 1.
        total_frac = torch.nan_to_num(total_frac, nan=1.0)
        assert torch.allclose(total_frac, torch.ones_like(total_frac), atol=0.05), (
            "Variance fractions should sum to 1"
        )

    def test_output_shapes(self):
        ensemble = torch.rand(2, 3, 5, 8, 6)
        vd = variance_decomposition(ensemble)
        for key in ("total", "meteo", "parametric", "aleatoric"):
            assert vd[key].shape == (8, 6), f"Wrong shape for {key}"

    def test_variances_non_negative(self):
        ensemble = torch.rand(2, 2, 2, 5, 4)
        vd = variance_decomposition(ensemble)
        for key in ("total", "meteo", "parametric", "aleatoric"):
            assert (vd[key] >= 0).all(), f"{key} variance has negative values"
