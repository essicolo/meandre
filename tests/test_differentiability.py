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


# test_snow_module_gradients / test_soil_module_gradients RETIRÉS 2026-06-27 :
# testaient les modules natifs (snow.py/soil.py) supprimés. La différentiabilité
# end-to-end de la colonne hydrotel (snow+gel+sol+UH) est couverte par
# tests/smoke_hydrotel_model.py (backward + gradient NeRF/colonne).


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
