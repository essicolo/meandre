"""Temporal context encoder — causal GRU over forcing history.

Gives each timestep awareness of the meteorological history so downstream
modules can implicitly reason about antecedent moisture, thaw-refreeze
cycles, drought build-up, etc.

Architecture
------------
1. Linear projection:  n_forcing  ->  d_model
2. Add day-of-year encoding  (seasonal cycle)
3. GRU hidden state (causal by construction — processes left-to-right)
4. LayerNorm + linear output projection  ->  n_context_out

Design choice vs. attention
----------------------------
A sliding-window MultiheadAttention over (N_nodes, W=90, d_model) called once
per timestep costs O(T × N × W²) and was the main CPU bottleneck (~115s/epoch
for SLSO).  A single GRU pass over (N_nodes, T) costs O(T × N × d²) — the
same operation counted once, not T times — reducing this to ~1s/epoch.

The GRU is causal: output at step t only depends on steps ≤ t.

Usage
-----
* ``encoder.step(forcing_t, doy_t, h)``   — called per timestep inside the
  model loop; maintains hidden state h across steps.
* ``encoder(forcing_window, doy_window)``  — window-based API kept for
  backward compatibility with existing tests.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from meandre.temporal.positional import SinusoidalDOYEncoding


class TemporalContextEncoder(nn.Module):
    """Causal GRU temporal context encoder.

    Parameters
    ----------
    n_forcing : int
        Number of raw forcing variables (default 6).
    d_model : int
        GRU hidden size and intermediate embedding dimension.
    n_heads : int
        Kept for API compatibility; ignored (GRU has no heads).
    window : int
        Kept for API compatibility; controls BPTT truncation period in
        ``model.simulate``.
    n_context_out : int
        Size of the output context vector per node.
    """

    def __init__(
        self,
        n_forcing: int,
        d_model: int = 64,
        n_heads: int = 4,
        window: int = 60,
        n_context_out: int = 16,
        concrete_dropout: bool = False,
        concrete_init_p: float = 0.05,
        n_data: int = 2889,
    ) -> None:
        super().__init__()
        self.window = window
        self.d_model = d_model
        self.input_proj = nn.Linear(n_forcing, d_model)
        self.doy_encoding = SinusoidalDOYEncoding(d_model)
        self.rnn = nn.GRU(d_model, d_model, num_layers=1, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        if concrete_dropout:
            from meandre.spatial.concrete_dropout import ConcreteDropout
            self.drop = ConcreteDropout(n_data=n_data, init_p=concrete_init_p)
        else:
            self.drop = nn.Identity()
        self.output_proj = nn.Linear(d_model, n_context_out)

    # ------------------------------------------------------------------
    # Full-sequence API  (used by model.simulate — ONE call for all T steps)
    # ------------------------------------------------------------------

    def encode_sequence(
        self,
        forcing: Tensor,
        day_of_year: Tensor,
        chunk_size: int = 90,
        h0: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Precompute context vectors for ALL timesteps in one (chunked) GRU pass.

        Processes T timesteps in chunks of ``chunk_size`` days.  The GRU hidden
        state is carried across chunk boundaries (but detached for truncated
        BPTT), so memory stays bounded at O(chunk_size × N × d).

        Args:
            forcing:    (T, N, n_forcing)
            day_of_year:(T,) integer day 1-366
            chunk_size: days per GRU chunk; controls BPTT depth and peak memory.
                        365 ≈ 256 MB/chunk for SLSO (2889 nodes, d=64).
            h0:         optional initial hidden state (e.g. from spinup).
        Returns:
            context:    (T, N, n_context_out)
            h_final:    (1, N, d_model) final hidden state (detached)
        """
        T, N, _ = forcing.shape

        # Empty sequence (e.g. zero-length spinup): return zeros immediately
        if T == 0:
            n_out = self.output_proj.out_features
            empty = torch.zeros(0, N, n_out, device=forcing.device)
            return empty, h0

        doy_enc = self.doy_encoding(day_of_year)                 # (T, d_model)

        # Process in chunks to bound peak GPU memory at O(chunk × N × d).
        # Gradient checkpointing recomputes activations during backward,
        # trading ~30% extra compute for ~60% less memory per chunk.
        out_chunks: list[Tensor] = []
        h = h0
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            from torch.utils.checkpoint import checkpoint as _ckpt
            chunk_out, h = _ckpt(
                self._encode_chunk,
                forcing[start:end],
                doy_enc[start:end],
                h,
                use_reentrant=False,
            )
            h = h.detach()                                        # truncated BPTT
            out_chunks.append(chunk_out)                          # (C, N, n_ctx)

        return torch.cat(out_chunks, dim=0), h                   # (T,N,n_ctx), h

    def _encode_chunk(
        self, forcing_chunk: Tensor, doy_chunk: Tensor, h: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        """Encode one chunk — called under gradient checkpointing."""
        x = self.input_proj(forcing_chunk)                       # (C, N, d_model)
        x = x + doy_chunk.unsqueeze(1)                           # (C, N, d_model)
        x = x.permute(1, 0, 2)                                   # (N, C, d_model)
        out, h_new = self.rnn(x, h)                               # (N, C, d)
        out = self.output_proj(self.drop(self.norm(out.permute(1, 0, 2))))  # (C, N, n_ctx)
        return out, h_new

    def concrete_kl(self):
        """Sum of Concrete Dropout KL terms (0 if using Identity/standard)."""
        from meandre.spatial.concrete_dropout import ConcreteDropout
        if isinstance(self.drop, ConcreteDropout):
            return self.drop.regularization(self.output_proj.weight)
        return torch.tensor(0.0, device=next(self.parameters()).device)

    # ------------------------------------------------------------------
    # Window-based API  (kept for backward compatibility with tests)
    # ------------------------------------------------------------------

    def forward(self, forcing_window: Tensor, day_of_year: Tensor) -> Tensor:
        """Process a W-step window for B batch elements.

        Args:
            forcing_window: (B, W, N, n_forcing)
            day_of_year:    (B, W) integer day 1-366
        Returns:
            context:        (B, N, n_context_out)
        """
        B, W, N, F = forcing_window.shape

        # Merge batch and node dims: (B*N, W, F)
        x = forcing_window.permute(0, 2, 1, 3).reshape(B * N, W, F)
        x = self.input_proj(x)                                   # (B*N, W, d_model)

        # DOY encoding: (B, W, d_model) → (B*N, W, d_model)
        doy_enc = self.doy_encoding(day_of_year)                 # (B, W, d_model)
        doy_enc = (
            doy_enc.unsqueeze(2)
            .expand(-1, -1, N, -1)
            .reshape(B * N, W, -1)
        )
        x = x + doy_enc

        out, _ = self.rnn(x)                                     # (B*N, W, d_model)
        last = out[:, -1, :]                                     # (B*N, d_model)
        context = self.output_proj(self.drop(self.norm(last)))   # (B*N, n_ctx)
        return context.reshape(B, N, -1)                         # (B, N, n_ctx)
