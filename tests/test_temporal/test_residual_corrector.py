"""Tests for StateResidualCorrector."""

import torch
import pytest
from meandre.temporal.residual_corrector import StateResidualCorrector


def test_output_shape():
    corr = StateResidualCorrector(n_state_vars=7)
    history = torch.randn(10, 14, 7)
    physics = torch.randn(10, 7)
    out = corr(history, physics)
    assert out.shape == (10, 7)


def test_zero_sum_soil_layers():
    """Soil layer delta must sum to zero per node."""
    corr = StateResidualCorrector(n_state_vars=7, n_soil_layers=3)
    history = torch.randn(8, 14, 7)
    physics = torch.randn(8, 7)
    corrected = corr(history, physics)
    soil_delta = corrected[:, :3] - physics[:, :3]
    assert torch.allclose(soil_delta.sum(dim=1), torch.zeros(8), atol=1e-5)


def test_near_zero_gate_init():
    """Gate should start near 0 so model begins as pure physics."""
    corr = StateResidualCorrector(n_state_vars=5)
    gate = torch.sigmoid(corr.gate_logit)
    assert (gate < 0.1).all(), f"Gate not near zero at init: {gate}"


def test_gradients_flow():
    corr = StateResidualCorrector(n_state_vars=6)
    history = torch.randn(4, 14, 6)
    physics = torch.randn(4, 6, requires_grad=True)
    out = corr(history, physics)
    out.sum().backward()
    assert physics.grad is not None
