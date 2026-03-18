"""Concrete Dropout (Gal, Hron & Kendall, 2017).

Learns the dropout rate per layer via a differentiable relaxation of
discrete Bernoulli masks.  The regularisation term balances:

    - Weight regularisation  (λ_w · ||W||²)  →  shrinks weights
    - Entropy of dropout rate (-λ_d · H(p))  →  pushes p away from 0/1

This yields calibrated epistemic uncertainty when combined with MC sampling.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class ConcreteDropout(nn.Module):
    """Learnable dropout layer with variational regularisation.

    Parameters
    ----------
    n_data : int
        Number of data points (nodes × timesteps for spatial, nodes for NeRF).
        Controls regularisation strength: more data → weaker prior.
    init_p : float
        Initial dropout probability.
    temperature : float
        Concrete relaxation temperature.  Lower → sharper (more binary) masks.
        Default 0.1 follows Gal et al. (2017).
    length_scale : float
        Prior length-scale (l).  Higher → stronger weight regularisation.
    """

    def __init__(
        self,
        n_data: int,
        init_p: float = 0.1,
        temperature: float = 0.1,
        length_scale: float = 1e-2,
    ) -> None:
        super().__init__()
        init_p = max(1e-4, min(init_p, 1.0 - 1e-4))
        self.logit_p = nn.Parameter(
            torch.tensor(math.log(init_p / (1.0 - init_p)))
        )
        self.temperature = temperature

        # Regularisation coefficients (Gal et al. 2017, Eq. 3)
        tau = 1.0  # model precision (absorbed into loss weighting)
        self.weight_regularizer = length_scale ** 2 / (tau * n_data)
        self.dropout_regularizer = 2.0 / (tau * n_data)

    @property
    def p(self) -> Tensor:
        """Current dropout probability (differentiable)."""
        return torch.sigmoid(self.logit_p)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            # Concrete (binary concrete) relaxation
            u = torch.rand_like(x).clamp(1e-6, 1.0 - 1e-6)
            z = torch.sigmoid(
                (torch.log(u) - torch.log(1.0 - u) + self.logit_p)
                / self.temperature
            )
            # Inverted dropout scaling so E[output] ≈ x
            return x * z / (1.0 - self.p + 1e-8)
        return x

    def regularization(self, weight: Tensor) -> Tensor:
        """KL-divergence regularisation for this layer.

        Parameters
        ----------
        weight : Tensor
            Weight matrix of the associated linear layer (e.g. fc.weight).

        Returns
        -------
        Scalar tensor to ADD to the loss.
        """
        p = self.p
        # Entropy: H(p) = -p·log(p) - (1-p)·log(1-p)
        entropy = -(
            p * torch.log(p + 1e-8)
            + (1.0 - p) * torch.log(1.0 - p + 1e-8)
        )
        # Minimise weight norm, maximise entropy (push p away from 0 and 1)
        return self.weight_regularizer * (weight ** 2).sum() - self.dropout_regularizer * entropy
