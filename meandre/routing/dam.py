"""Regulated dam / reservoir release data.

Dams are lake nodes whose outflow is prescribed by observed historical releases
rather than estimated by the storage-discharge power law.  Any lake node can be
made regulated by providing a release time-series for it here.

Design
------
* ``DamData.releases`` is an (n_timesteps, n_nodes) float32 tensor.
* ``torch.nan`` marks unregulated nodes (natural lakes or river reaches).
* Regulated nodes receive the exact release as Q_out; storage is still tracked
  via the mass balance so the simulation remains differentiable w.r.t. inputs.

Constructors
------------
* ``DamData.unregulated(n_timesteps, n_nodes)`` — no dam regulation anywhere.
* ``DamData.from_node_series(node_releases, n_timesteps, n_nodes)`` — from a
  dict mapping node index → (n_timesteps,) release tensor (m3/s).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class DamData:
    """Per-timestep regulated outflow for dam / reservoir nodes.

    Attributes
    ----------
    releases:
        (n_timesteps, n_nodes) float32, m3/s.
        ``torch.nan`` at unregulated nodes (natural lakes or non-lake reaches).
    """

    releases: Tensor  # (n_timesteps, n_nodes) — nan = unregulated

    # ------------------------------------------------------------------
    # Per-step access used by RoutingLayer
    # ------------------------------------------------------------------

    def release_at(self, t: int, node: int) -> Tensor | None:
        """Return the forced release (m3/s) for *node* at timestep *t*, or
        ``None`` if the node is unregulated (no observed release available)."""
        v = self.releases[t, node]
        if torch.isnan(v):
            return None
        return v.unsqueeze(0)

    # ------------------------------------------------------------------
    # Device transfer
    # ------------------------------------------------------------------

    def to(self, device: torch.device) -> "DamData":
        return DamData(releases=self.releases.to(device))

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def unregulated(
        cls,
        n_timesteps: int,
        n_nodes: int,
        device: torch.device | None = None,
    ) -> "DamData":
        """All nodes unregulated — equivalent to no dam data."""
        return cls(
            releases=torch.full(
                (n_timesteps, n_nodes), float("nan"), device=device
            )
        )

    @classmethod
    def from_node_series(
        cls,
        node_releases: dict[int, Tensor],
        n_timesteps: int,
        n_nodes: int,
        device: torch.device | None = None,
    ) -> "DamData":
        """Build from a mapping of node index → release time-series.

        Parameters
        ----------
        node_releases:
            ``{node_index: tensor_of_shape_(n_timesteps,)}`` in m3/s.
            Nodes absent from the dict are treated as unregulated (nan).
        n_timesteps, n_nodes:
            Shape of the full network / simulation window.
        device:
            Target device.

        Returns
        -------
        DamData
        """
        releases = torch.full((n_timesteps, n_nodes), float("nan"))
        for node_idx, series in node_releases.items():
            if series.shape[0] != n_timesteps:
                raise ValueError(
                    f"Release series for node {node_idx} has length "
                    f"{series.shape[0]}, expected {n_timesteps}."
                )
            releases[:, node_idx] = series.float()
        return cls(releases=releases.to(device) if device else releases)
