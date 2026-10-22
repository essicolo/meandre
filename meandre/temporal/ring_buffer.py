"""Outflow ring buffer for travel-time attention in the routing module.

Each node in the river graph maintains a short history of its outflow Q_out.
Downstream nodes query this buffer to retrieve the outflow that departed
tau days ago, implementing the travel-time lag used in TravelTimeAttention.
"""

from __future__ import annotations

import torch
from torch import Tensor


class OutflowRingBuffer:
    """Fixed-depth circular buffer of per-node outflow tensors.

    Parameters
    ----------
    n_nodes : int
        Number of graph nodes (subbasins/reaches).
    depth : int
        Maximum history depth in timesteps (typically max_travel_time_days).
    device : torch.device, optional
    """

    def __init__(
        self,
        n_nodes: int,
        depth: int,
        device: torch.device | None = None,
    ) -> None:
        self.n_nodes = n_nodes
        self.depth = depth
        self.device = device
        # Pre-allocate buffer: (depth, n_nodes)
        self._buf = torch.zeros(depth, n_nodes, device=device)
        self._ptr = 0       # write pointer (oldest slot)
        self._filled = 0    # how many valid timesteps are stored

    def push(self, Q_out: Tensor) -> None:
        """Append latest outflow (n_nodes,), overwriting the oldest entry."""
        self._buf[self._ptr] = Q_out.detach()
        self._ptr = (self._ptr + 1) % self.depth
        self._filled = min(self._filled + 1, self.depth)

    def get_all_history(self, tau: int) -> Tensor:
        """Return the last *tau* timesteps for ALL nodes.

        Returns a (min(tau, filled), n_nodes) tensor, chronological order.
        """
        if self._filled == 0:
            return torch.zeros(1, self.n_nodes, device=self.device)
        tau = min(tau, self._filled)
        end = self._ptr
        indices = [(end - tau + i) % self.depth for i in range(tau)]
        return self._buf[indices]  # (tau, n_nodes)

    def __len__(self) -> int:
        return self._filled
