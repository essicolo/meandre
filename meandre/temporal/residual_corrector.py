"""State residual corrector — learns the systematic error of the physics modules.

The physics pipeline (snow -> frost -> soil -> ET -> wetland) handles mass
conservation and process equations. This module learns the residual: the portion
that simplified equations miss (preferential flow, macropore effects, vegetation
phenology, etc.).

    state_{t+1} = physics_update(state_t, forcing_t)
                  + gate * residual_net(state_history[t-H:t])

Critical constraints
--------------------
1. Gate is initialised near 0 so training starts as pure physics.
2. Corrections to soil layers are zero-sum (water redistributed, not created).
"""

import torch
import torch.nn as nn
from torch import Tensor


class StateResidualCorrector(nn.Module):
    """GRU-based corrector for physics state updates.

    Takes the last H physics states, runs them through a 2-layer GRU, and
    produces a per-variable delta that is added to the current physics state.

    Parameters
    ----------
    n_state_vars : int
        Total number of state variables per node (default 7, matches HydroState).
    hidden : int
        GRU hidden size (keep small to avoid overwhelming physics).
    history : int
        Number of past timesteps H consumed by the GRU.
    n_soil_layers : int
        Number of leading state variables that are soil moisture layers.
        Zero-sum projection is applied only to these.
    """

    def __init__(
        self,
        n_state_vars: int = 7,
        hidden: int = 32,
        history: int = 14,
        n_soil_layers: int = 3,
    ) -> None:
        super().__init__()
        self.history = history
        self.n_soil_layers = n_soil_layers

        self.gru = nn.GRU(n_state_vars, hidden, num_layers=2, batch_first=True)
        self.proj = nn.Linear(hidden, n_state_vars)

        # Gate per state variable, initialised to sigmoid(-3) ≈ 0.05
        # so the model begins as near-pure physics.
        self.gate_logit = nn.Parameter(torch.full((n_state_vars,), -3.0))

    def forward(self, state_history: Tensor, physics_state: Tensor) -> Tensor:
        """
        Args:
            state_history: (n_nodes, H, n_state_vars)
                Last H physics-computed states (from HydroState.to_tensor stacked).
            physics_state: (n_nodes, n_state_vars)
                Current timestep physics update.
        Returns:
            corrected_state: (n_nodes, n_state_vars)
                Physics state plus gated residual correction.
        """
        # Guard against NaN/Inf that can appear from extreme physics states and
        # corrupt the GRU hidden state, triggering CUBLAS_STATUS_EXECUTION_FAILED.
        # Clamp to physically plausible ranges before replacing NaN.
        state_history = torch.clamp(state_history, min=-50.0, max=500.0)
        state_history = torch.nan_to_num(state_history, nan=0.0)
        _, hidden = self.gru(state_history)
        delta = self.proj(hidden[-1])  # (n_nodes, n_state_vars)

        # Zero-sum projection on soil layers: water is redistributed, not created
        soil_delta = delta[:, : self.n_soil_layers]
        soil_delta = soil_delta - soil_delta.mean(dim=1, keepdim=True)
        delta = torch.cat([soil_delta, delta[:, self.n_soil_layers :]], dim=1)

        gate = torch.sigmoid(self.gate_logit)  # (n_state_vars,)
        return physics_state + gate * delta
