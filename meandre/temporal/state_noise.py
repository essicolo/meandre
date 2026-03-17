"""AR(1) correlated state noise for ensemble uncertainty quantification.

Injects temporally-coherent noise UPSTREAM of the temporal dynamics so it
propagates through the physics (Darcy, ET, routing) — exactly like real
uncertainty is filtered by watershed inertia.

Rule: stochasticity must enter INSIDE the temporal scan, not at the output.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class CorrelatedStateNoise(nn.Module):
    """AR(1) noise process injected into hydrological state variables.

    noise_t = rho * noise_{t-1} + sigma * epsilon_t

    rho  (persistence) and sigma (innovation amplitude) are LEARNED per state
    variable.  The model can discover that soil moisture needs high persistence
    (~0.9) while canopy storage needs lower (~0.3).

    Parameters
    ----------
    n_state_vars : int
        Number of state variables to perturb.  Defaults to 4, covering the
        three soil layers and SWE — the variables with strongest memory and
        highest uncertainty.  t_soil, canopy, and wetland are left unperturbed
        by default because their time constants are short and they are strongly
        constrained by forcing.
    """

    def __init__(self, n_state_vars: int = 4) -> None:
        super().__init__()
        # logit(rho) initialised high (logit(2) → sigmoid ≈ 0.88) for soil memory
        self.logit_rho = nn.Parameter(torch.full((n_state_vars,), 2.0))
        # log(sigma) initialised small (log(-4) → exp ≈ 0.018) to avoid
        # overwhelming the physics during the first training phase
        self.log_amplitude = nn.Parameter(torch.full((n_state_vars,), -4.0))

    @property
    def rho(self) -> Tensor:
        """Persistence coefficient in (0, 1) per state variable."""
        return torch.sigmoid(self.logit_rho)

    @property
    def sigma(self) -> Tensor:
        """Innovation amplitude (> 0) per state variable."""
        return torch.exp(self.log_amplitude)

    def init_noise(self, n_nodes: int, device: torch.device) -> Tensor:
        """Initialise noise state to zero at the start of a trajectory.

        Returns
        -------
        noise : (n_nodes, n_state_vars)
        """
        return torch.zeros(n_nodes, len(self.logit_rho), device=device)

    def step(self, prev_noise: Tensor) -> Tensor:
        """One stochastic AR(1) step — call once per simulation timestep.

        Args:
            prev_noise : (n_nodes, n_state_vars) noise from the previous step.
        Returns:
            new_noise  : (n_nodes, n_state_vars)
        """
        eps = torch.randn_like(prev_noise)
        return self.rho * prev_noise + self.sigma * eps

    def step_deterministic(self, prev_noise: Tensor) -> Tensor:
        """Decay without a new innovation — useful during gradient-based training
        when stochastic sampling would break gradient flow."""
        return self.rho * prev_noise
