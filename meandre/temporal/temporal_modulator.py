"""Temporal modulator — per-param seasonal + instantaneous modulation.

Pour chaque paramètre modulé, apprend 4 scalaires:
  - phase_logit  → sigmoid × 365.25 = jour de l'année du pic
  - amplitude    = force du cycle saisonnier (0 = aucun cycle)
  - P_coef       = réactivité à la pluie du jour (P_mm)
  - T_coef       = réactivité à la température (T_air)

Tous initialisés à 0 → modulator = 1.0 → comportement identité au démarrage.
Le modèle apprend à ajouter la modulation s'il y a un signal pour ça.

Pas de variation spatiale du modulateur lui-même (toutes les régions partagent
la même phénologie), mais le param spatial qu'il multiplie varie via le NeRF.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class TemporalModulator(nn.Module):
    """Differentiable seasonal + instantaneous param modulation."""

    def __init__(self, n_modulated: int) -> None:
        super().__init__()
        self.n_modulated = n_modulated
        # All zero init → modulator = 1.0 (identity)
        self.phase_logit = nn.Parameter(torch.zeros(n_modulated))
        self.amplitude   = nn.Parameter(torch.zeros(n_modulated))
        self.P_coef      = nn.Parameter(torch.zeros(n_modulated))
        self.T_coef      = nn.Parameter(torch.zeros(n_modulated))

    def forward(self, doy: Tensor, P_mm: Tensor, T_air: Tensor) -> Tensor:
        """
        Args:
            doy:   scalar tensor — current day of year (1-366)
            P_mm:  (n_nodes,) precipitation today (mm/day)
            T_air: (n_nodes,) air temperature (°C)
        Returns:
            modulators: (n_nodes, n_modulated) multiplicative factors in [0.1, 3.0]
        """
        phase = 365.25 * torch.sigmoid(self.phase_logit)        # (M,)
        # doy can be int or scalar tensor; cast to float
        d = doy.to(torch.float32) if isinstance(doy, Tensor) else torch.tensor(float(doy))
        season = self.amplitude * torch.cos(
            2.0 * math.pi * (d - phase) / 365.25
        )                                                        # (M,)
        season_b = season.unsqueeze(0)                           # (1, M)

        # Instantaneous response (linear in P, T)
        instant = (
            self.P_coef.unsqueeze(0) * (P_mm.unsqueeze(-1) - 10.0)
            + self.T_coef.unsqueeze(0) * (T_air.unsqueeze(-1) - 10.0)
        )                                                        # (n_nodes, M)

        return torch.clamp(1.0 + season_b + instant, min=0.1, max=3.0)
