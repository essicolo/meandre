"""Travel-time attention for routing — learned temporal aggregation of upstream flows.

Instead of naively summing upstream outflows, each node attends over the recent
outflow history of its upstream neighbours. Attention logits are biased by a
learned embedding of the travel time tau, so the model distinguishes a fast
tributary (tau = 1 day) from a slow main-stem contribution (tau = 10 days).

This is the core learned component of the routing module.

Two execution modes
-------------------
* ``forward_edges`` (fast, GPU-friendly): operates on pre-fetched edge histories
  packed into a padded (n_edges, max_tau, 1) tensor.  Used by the vectorized
  routing path in ``RoutingLayer``.

* ``forward`` (per-node, legacy): accepts Python lists of per-upstream histories.
  Retained for tests and ad-hoc inspection; not used during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TravelTimeAttention(nn.Module):
    """Attention over upstream contributors with travel-time positional bias.

    For node i at time t:
      - Query:  local inflow state of node i
      - Keys:   recent outflow history Q_out[j, t-tau:t] for each upstream j
      - Values: same as keys
      - Bias:   learned embedding of tau_{j->i} added to attention logits

    Parameters
    ----------
    d_flow : int
        Dimension of flow signal (1 for scalar Q).
    d_model : int
        Internal embedding dimension.
    n_heads : int
        Number of attention heads.
    max_tau_days : int
        Maximum travel time (days) in the network; determines embedding table size.
    """

    def __init__(
        self,
        d_flow: int = 1,
        d_model: int = 8,
        n_heads: int = 1,
        max_tau_days: int = 30,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.tau_embed = nn.Embedding(max_tau_days + 1, d_model)
        self.q_proj = nn.Linear(d_flow, d_model)
        self.k_proj = nn.Linear(d_flow, d_model)
        self.v_proj = nn.Linear(d_flow, d_model)
        self.out_proj = nn.Linear(d_model, 1)

    def forward_edges(
        self,
        edge_queries: Tensor,
        edge_hist: Tensor,
        edge_taus: Tensor,
        hist_mask: Tensor,
    ) -> Tensor:
        """Batched TTA for all edges simultaneously — GPU-friendly.

        Args:
            edge_queries: (n_edges, 1) lateral inflow of each destination node
            edge_hist:    (n_edges, max_tau, 1) padded upstream outflow histories
                          (oldest first; entries beyond tau are zeroed out)
            edge_taus:    (n_edges,) int64 travel time per edge
            hist_mask:    (n_edges, max_tau) bool — True = valid timestep

        Returns:
            (n_edges,) per-edge weighted upstream contribution
        """
        max_idx = self.tau_embed.num_embeddings - 1

        # Keys and values from history: (n_edges, max_tau, d_model)
        K = self.k_proj(edge_hist)
        V = self.v_proj(edge_hist)

        # Travel-time bias: one embedding per edge, broadcast over max_tau
        tau_clamped = edge_taus.clamp(max=max_idx)          # (n_edges,)
        B = self.tau_embed(tau_clamped)                      # (n_edges, d_model)
        B = B.unsqueeze(1).expand(-1, edge_hist.shape[1], -1)  # (n_edges, max_tau, d_model)

        # Query from destination node's lateral inflow: (n_edges, 1, d_model)
        Q = self.q_proj(edge_queries).unsqueeze(1)

        # Attention logits: (n_edges, 1, max_tau)
        logits = (Q @ (K + B).transpose(1, 2)) / (self.d_model ** 0.5)

        # Mask padding positions to -inf before softmax
        pad_mask = ~hist_mask                                # (n_edges, max_tau)
        logits = logits.masked_fill(pad_mask.unsqueeze(1), float("-inf"))

        weights = F.softmax(logits, dim=-1)                  # (n_edges, 1, max_tau)
        attended = weights @ V                               # (n_edges, 1, d_model)

        return self.out_proj(attended.squeeze(1)).squeeze(-1)  # (n_edges,)

    def forward(
        self,
        node_state: Tensor,
        upstream_histories: list[Tensor],
        travel_times: list[int],
    ) -> Tensor:
        """Aggregate upstream outflow histories into a single inflow estimate.

        Args:
            node_state: (d_flow,) or (1,) current node's local lateral inflow
            upstream_histories: list of (tau_i, d_flow) tensors — one per
                upstream node, containing its last tau_i outflow values
                in chronological order (oldest first)
            travel_times: list of int travel times (days) per upstream edge,
                must have the same length as upstream_histories
        Returns:
            aggregated_inflow: scalar tensor, weighted upstream contribution
        """
        if not upstream_histories:
            return torch.zeros(1, device=node_state.device)

        query = self.q_proj(node_state.view(1, -1))  # (1, d_model)

        max_idx = self.tau_embed.num_embeddings - 1
        keys, values, biases = [], [], []
        for hist, tau in zip(upstream_histories, travel_times):
            k = self.k_proj(hist)   # (tau, d_model)
            v = self.v_proj(hist)   # (tau, d_model)
            b = self.tau_embed(
                torch.tensor(min(tau, max_idx), device=k.device)
            )  # (d_model,)
            keys.append(k)
            values.append(v)
            biases.append(b.expand(k.shape[0], -1))

        K = torch.cat(keys, dim=0)    # (total_steps, d_model)
        V = torch.cat(values, dim=0)
        B = torch.cat(biases, dim=0)  # (total_steps, d_model)

        # Attention with travel-time positional bias
        logits = (query @ (K + B).T) / (self.d_model**0.5)  # (1, total_steps)
        weights = F.softmax(logits, dim=-1)
        attended = weights @ V  # (1, d_model)

        return self.out_proj(attended).squeeze()
