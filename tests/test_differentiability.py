"""Gradient flow tests — every learnable parameter must receive a gradient.

Run: pytest tests/test_differentiability.py -v
"""

import pytest
import torch


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
