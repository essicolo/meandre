"""Fourier positional encoding (NeRF-style) for spatial coordinates.

Lifts (lon, lat) from 2D to a high-dimensional space using random Fourier
features or deterministic log-scale sinusoids, following NeRF (Mildenhall 2020).

    gamma(x) = [sin(2^0 * pi * x), cos(2^0 * pi * x),
                sin(2^1 * pi * x), cos(2^1 * pi * x), ...,
                sin(2^(L-1) * pi * x), cos(2^(L-1) * pi * x)]
"""

import math

import torch
import torch.nn as nn
from torch import Tensor


class FourierPositionalEncoding(nn.Module):
    """Log-scale Fourier features for spatial coordinates.

    Encodes each input dimension independently and concatenates the results.
    Input dimension d -> output dimension d * 2 * n_freqs.

    Parameters
    ----------
    n_freqs : int
        Number of frequency bands L (output dim = in_dim * 2 * L).
    include_input : bool
        If True, prepend the raw input to the encoding.
    """

    def __init__(self, n_freqs: int = 6, include_input: bool = True) -> None:
        super().__init__()
        self.n_freqs = n_freqs
        self.include_input = include_input
        # Frequencies: 2^0, 2^1, ..., 2^(L-1)
        freqs = 2.0 ** torch.arange(n_freqs).float()  # (L,)
        self.register_buffer("freqs", freqs)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (..., in_dim)
        Returns:
            encoded: (..., in_dim * 2 * n_freqs [+ in_dim if include_input])
        """
        parts = [x] if self.include_input else []
        for freq in self.freqs:
            parts.append(torch.sin(math.pi * freq * x))
            parts.append(torch.cos(math.pi * freq * x))
        return torch.cat(parts, dim=-1)

    def out_dim(self, in_dim: int) -> int:
        base = in_dim * 2 * self.n_freqs
        return base + in_dim if self.include_input else base


class RandomFourierFeatures(nn.Module):
    """Gaussian random Fourier features (Rahimi & Recht 2007).

    An alternative to the deterministic log-scale encoding when the
    input has more than 2 dimensions (e.g. when territorial features
    are also encoded positionally).
    """

    def __init__(self, in_dim: int, out_dim: int, sigma: float = 1.0) -> None:
        super().__init__()
        assert out_dim % 2 == 0, "out_dim must be even"
        B = torch.randn(in_dim, out_dim // 2) * sigma
        self.register_buffer("B", B)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (..., in_dim)
        Returns:
            features: (..., out_dim)
        """
        proj = x @ self.B  # (..., out_dim//2)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
