"""Tests for Muskingum-Cunge kinematic wave routing."""

import torch
import pytest
from meandre.routing.kinematic import MuskingumCunge


def test_zero_inflow_decays():
    """With zero inflow, stored water should drain over time."""
    musk = MuskingumCunge()
    n = 3
    Q_in = torch.zeros(n)
    Q_out_prev = torch.ones(n) * 10.0
    q_lat = torch.zeros(n)
    K = torch.ones(n) * 3600.0 * 6  # 6 hours
    x = torch.full((n,), 0.2)

    Q_out = musk(Q_in, Q_out_prev, q_lat, K, x)
    assert (Q_out >= 0).all()
    assert (Q_out < Q_out_prev).all(), "Outflow should decay with zero inflow"


def test_gradients():
    musk = MuskingumCunge()
    n = 5
    Q_in = torch.rand(n, requires_grad=True)
    Q_out_prev = torch.rand(n)
    q_lat = torch.zeros(n)
    K = torch.ones(n) * 86400.0
    x = torch.full((n,), 0.2)

    Q_out = musk(Q_in, Q_out_prev, q_lat, K, x)
    Q_out.sum().backward()
    assert Q_in.grad is not None
