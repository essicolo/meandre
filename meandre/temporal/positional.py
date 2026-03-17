"""Positional encodings for the temporal context encoder.

Two encodings are combined:
- SinusoidalDOYEncoding  seasonal cycle (day-of-year 1-366)
- Lag embeddings         learned per-position offsets ("3 days ago" vs "30 days ago")
"""

import math

import torch
import torch.nn as nn
from torch import Tensor


class SinusoidalDOYEncoding(nn.Module):
    """Encode day-of-year as (sin, cos) pair projected into d_model dimensions.

    The sin/cos encoding is periodic with period 366, capturing seasonality.
    A linear projection learns which aspects of the seasonal cycle matter.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.linear = nn.Linear(2, d_model)

    def forward(self, doy: Tensor) -> Tensor:
        """
        Args:
            doy: (...) integer day 1-366
        Returns:
            encoding: (..., d_model)
        """
        angle = 2.0 * math.pi * doy.float() / 366.0
        sincos = torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)
        return self.linear(sincos)


class FourierDOYEncoding(nn.Module):
    """Multi-harmonic Fourier encoding of day-of-year.

    Includes k harmonics: sin(2*pi*n*doy/366), cos(2*pi*n*doy/366) for n=1..k.
    Richer than a single harmonic; useful for capturing intra-seasonal patterns.
    """

    def __init__(self, d_model: int, n_harmonics: int = 4) -> None:
        super().__init__()
        self.n_harmonics = n_harmonics
        self.linear = nn.Linear(2 * n_harmonics, d_model)

    def forward(self, doy: Tensor) -> Tensor:
        """
        Args:
            doy: (...) integer day 1-366
        Returns:
            encoding: (..., d_model)
        """
        base = 2.0 * math.pi * doy.float() / 366.0  # (...)
        harmonics = []
        for n in range(1, self.n_harmonics + 1):
            harmonics.extend([torch.sin(n * base), torch.cos(n * base)])
        x = torch.stack(harmonics, dim=-1)  # (..., 2*n_harmonics)
        return self.linear(x)
