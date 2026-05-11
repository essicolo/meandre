"""Gradient flow tests — every learnable parameter must receive a gradient.

Run: pytest tests/test_differentiability.py -v
"""

import pytest
import torch

from meandre.temporal.context_encoder import TemporalContextEncoder
from meandre.temporal.residual_corrector import StateResidualCorrector


def test_temporal_context_gradients():
    """Verify that gradients propagate back through the attention window."""
    encoder = TemporalContextEncoder(n_forcing=6, window=30, n_context_out=8)
    window = torch.randn(1, 30, 10, 6, requires_grad=True)
    doy = torch.randint(1, 366, (1, 30))

    ctx = encoder(window, doy)
    ctx.sum().backward()

    assert window.grad is not None, "No gradient on forcing window"
    assert not torch.all(window.grad == 0), "All-zero gradient on forcing window"


def test_residual_preserves_mass():
    """Zero-sum projection on soil layers must hold after correction."""
    corrector = StateResidualCorrector(n_state_vars=7, n_soil_layers=3)
    n_nodes = 10
    history = torch.randn(n_nodes, 14, 7)
    physics = torch.randn(n_nodes, 7)

    corrected = corrector(history, physics)
    soil_delta = corrected[:, 0:3] - physics[:, 0:3]

    assert torch.allclose(
        soil_delta.sum(dim=1), torch.zeros(n_nodes), atol=1e-5
    ), "Soil layer correction is not zero-sum (mass conservation violated)"


def test_residual_corrector_gradients():
    """Gradients must flow from loss through the residual corrector."""
    corrector = StateResidualCorrector(n_state_vars=7)
    history = torch.randn(5, 14, 7)
    physics = torch.randn(5, 7, requires_grad=True)

    corrected = corrector(history, physics)
    corrected.sum().backward()

    assert physics.grad is not None
    for name, param in corrector.named_parameters():
        assert param.grad is not None, f"No gradient for corrector param {name}"


def test_snow_module_gradients():
    """Gradients must flow through the snow module thresholds."""
    from meandre.vertical.snow import SnowModule

    snow = SnowModule()
    n = 20
    P = torch.rand(n, requires_grad=True)   # leaf tensor — no in-place ops
    T_air = torch.randn(n)
    SWE = torch.rand(n) * 50
    C_f = torch.ones(n) * 3.0
    T_melt = torch.zeros(n)
    T_snow = torch.zeros(n)

    P_eff, SWE_new, _ = snow(P, T_air, SWE, C_f, T_melt, T_snow)
    (P_eff.sum() + SWE_new.sum()).backward()

    assert P.grad is not None
    assert not torch.all(P.grad == 0)


def test_soil_module_gradients():
    """Gradients must flow through the soil water balance."""
    from meandre.vertical.soil import SoilModule

    soil = SoilModule()
    n = 8
    P_eff = torch.rand(n, requires_grad=True)  # leaf tensor
    ET1 = torch.rand(n) * 1
    ET2 = torch.rand(n) * 0.5
    ET3 = torch.rand(n) * 0.2

    theta1 = torch.rand(n) * 0.3 + 0.1
    theta2 = torch.rand(n) * 0.25 + 0.1
    theta3 = torch.rand(n) * 0.2 + 0.1
    K_sat = torch.ones(n) * 0.5
    por = torch.ones(n) * 0.45
    fc = torch.ones(n) * 0.35
    wp = torch.ones(n) * 0.15

    f_vert = torch.full((n,), 0.5, requires_grad=True)
    t1, t2, t3, R, interflow, baseflow = soil(
        P_eff, ET1, ET2, ET3,
        theta1, theta2, theta3,
        K_sat, K_sat, K_sat,
        por, por, por,
        fc, fc, fc,
        wp, wp, wp,
        f_vert, f_vert, f_vert,
    )
    (t1.sum() + t2.sum() + t3.sum() + R.sum() + interflow.sum() + baseflow.sum()).backward()
    assert P_eff.grad is not None


def test_spatial_field_network_gradients():
    """Gradients must flow through the NeRF spatial encoder."""
    from meandre.spatial.field_network import SpatialFieldNetwork

    net = SpatialFieldNetwork(n_territorial=17)
    coords = torch.randn(15, 2, requires_grad=True)
    features = torch.randn(15, 17)

    params = net(coords, features)
    params.C_f.sum().backward()

    assert coords.grad is not None
    for name, p in net.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"
