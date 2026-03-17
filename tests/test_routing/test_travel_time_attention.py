"""Tests for TravelTimeAttention."""

import torch
import pytest
from meandre.routing.travel_time_attention import TravelTimeAttention


def test_no_upstream_returns_zero():
    tta = TravelTimeAttention()
    node_state = torch.tensor([1.5])
    out = tta(node_state, [], [])
    assert out.item() == pytest.approx(0.0)


def test_single_upstream_output_shape():
    tta = TravelTimeAttention(d_flow=1, d_model=16, n_heads=2, max_tau_days=10)
    node_state = torch.tensor([2.0])
    hist = torch.rand(5, 1)   # 5 timesteps of history, d_flow=1
    out = tta(node_state, [hist], [3])
    assert out.shape == torch.Size([])  # scalar


def test_multiple_upstream():
    tta = TravelTimeAttention(d_flow=1, d_model=16, max_tau_days=15)
    node_state = torch.tensor([1.0])
    histories = [torch.rand(tau, 1) for tau in [2, 5, 3]]
    taus = [2, 5, 3]
    out = tta(node_state, histories, taus)
    assert out.numel() == 1


def test_gradients():
    tta = TravelTimeAttention(d_flow=1, d_model=16, max_tau_days=10)
    node_state = torch.tensor([1.0], requires_grad=True)
    hist = torch.rand(4, 1, requires_grad=True)
    out = tta(node_state, [hist], [4])
    out.backward()
    assert node_state.grad is not None
    assert hist.grad is not None
