"""Tests for the spatial field network (NeRF-style MLP)."""

import torch
import pytest
from meandre.spatial.field_network import SpatialFieldNetwork, SpatialParams


def test_output_types():
    net = SpatialFieldNetwork(n_territorial=17, hidden=64)
    coords = torch.randn(10, 2)
    features = torch.randn(10, 17)
    params = net(coords, features)
    assert isinstance(params, SpatialParams)


def test_K_sat_positive():
    net = SpatialFieldNetwork(n_territorial=17, hidden=64)
    coords = torch.randn(20, 2)
    features = torch.randn(20, 17)
    params = net(coords, features)
    assert (params.K_sat_1 > 0).all()
    assert (params.K_sat_2 > 0).all()
    assert (params.K_sat_3 > 0).all()


def test_root_fractions_sum_to_one():
    net = SpatialFieldNetwork(n_territorial=17, hidden=64)
    coords = torch.randn(15, 2)
    features = torch.randn(15, 17)
    params = net(coords, features)
    f_sum = params.f_root_1 + params.f_root_2 + params.f_root_3
    assert torch.allclose(f_sum, torch.ones(15), atol=1e-5)


def test_parameter_ranges():
    net = SpatialFieldNetwork(n_territorial=17, hidden=64)
    coords = torch.randn(30, 2)
    features = torch.randn(30, 17)
    params = net(coords, features)

    assert (params.C_f >= 0).all() and (params.C_f <= 10).all()
    assert (params.T_melt >= -2).all() and (params.T_melt <= 2).all()
    assert (params.manning_n >= 0.01).all() and (params.manning_n <= 0.2).all()


def test_gradients_flow():
    net = SpatialFieldNetwork(n_territorial=5, hidden=32)
    coords = torch.randn(8, 2, requires_grad=True)
    features = torch.randn(8, 5)
    params = net(coords, features)
    params.C_f.sum().backward()
    assert coords.grad is not None
