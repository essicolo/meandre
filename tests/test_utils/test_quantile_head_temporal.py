"""Terme temporel de la tête de quantiles (R551) : nul à l'initialisation, relit un ancien point."""
import torch

from meandre.utils import settings
from meandre.utils.quantile_head import QuantileHead


def _heads():
    torch.manual_seed(0)
    plain = QuantileHead(n_spatial_params=8)
    settings.override("quantile.temporal", True)
    try:
        torch.manual_seed(0)
        temporal = QuantileHead(n_spatial_params=8)
    finally:
        settings.override("quantile.temporal", False)
    return plain, temporal


def test_temporal_term_starts_at_zero():
    plain, temporal = _heads()
    temporal.load_state_dict(plain.state_dict(), strict=False)
    params = torch.randn(5, 8)
    q = torch.rand(30, 5) * 10 + 0.1
    assert torch.allclose(plain(params, q), temporal(params, q))


def test_rising_flow_widens_the_upper_quantile():
    _, temporal = _heads()
    with torch.no_grad():
        temporal.temporal.bias[: temporal.K] = 2.0
    params = torch.zeros(1, 8)
    flat = torch.full((10, 1), 5.0)
    rising = torch.linspace(1.0, 20.0, 10).unsqueeze(1)
    upper_flat = temporal(params, flat)[-1, 0, -1] / flat[-1, 0]
    upper_rising = temporal(params, rising)[-1, 0, -1] / rising[-1, 0]
    assert upper_rising > upper_flat
