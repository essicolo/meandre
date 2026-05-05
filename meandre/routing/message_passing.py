"""Custom PyTorch Geometric MessagePassing layer for river routing.

Implements topologically-sorted upstream->downstream message passing.
Each node aggregates upstream outflows via TravelTimeAttention before
applying Muskingum-Cunge routing.

Two execution paths
-------------------
* **Vectorized** (default, fast): used when TravelTimeAttention is not yet
  active or the outflow buffer is empty.  Processes nodes in topological
  levels (Kahn's algorithm) so upstream outflow propagates through the
  entire network within a single timestep.  Per-level aggregation uses
  vectorized ``scatter_add``.

* **Sequential** (TTA active): processes nodes in topological order so the
  attention module can read per-upstream outflow histories from the ring
  buffer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from meandre.routing.dam import DamData
from meandre.routing.graph import RiverGraph
from meandre.routing.kinematic import MuskingumCunge
from meandre.routing.lake import LakeModule
from meandre.routing.travel_time_attention import TravelTimeAttention
from meandre.routing.withdrawals import WithdrawalData
from meandre.temporal.ring_buffer import OutflowRingBuffer


class RoutingLayer(nn.Module):
    """One full routing step: propagate Q through the river graph."""

    def __init__(
        self,
        use_travel_time_attention: bool = True,
        max_tau_days: int = 30,
    ) -> None:
        super().__init__()
        self.use_tta = use_travel_time_attention
        if use_travel_time_attention:
            self.tta = TravelTimeAttention(max_tau_days=max_tau_days)
            # Warmup factor for blending simple-sum ↔ TTA aggregation.
            # 0.0 = pure simple sum (Σ Q_actuels), 1.0 = pure TTA.
            # Set by Trainer._apply_curriculum during TTA warmup.
            self.register_buffer("tta_warmup_factor", torch.tensor(0.0))
        # n_substeps=2 (12h) — compromis vitesse/précision.
        # Limite : K < 6h sera instable; les bornes K_musk_hours doivent rester ≥ 4h.
        self.muskingum = MuskingumCunge(n_substeps=2)
        self.lake = LakeModule()

    def forward(
        self,
        lateral_inflow: Tensor,
        graph: RiverGraph,
        Q_out_prev: Tensor,
        outflow_buffer: OutflowRingBuffer,
        withdrawals: WithdrawalData,
        t: int,
        K_musk: Tensor,
        x_musk: Tensor,
        dx: Tensor | None = None,
        lake_storage: Tensor | None = None,
        area_km2: Tensor | None = None,
        dam_data: DamData | None = None,
        area_km2_local: Tensor | None = None,
        Q_in_prev: Tensor | None = None,
        H_lateral: Tensor | None = None,
        T_air: Tensor | None = None,
        R_n: Tensor | None = None,
        K_atm: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        """
        Args:
            lateral_inflow: (n_nodes,) lateral inflow in mm/day
            graph:          RiverGraph
            Q_out_prev:     (n_nodes,) outflow at previous timestep (m3/s)
            outflow_buffer: OutflowRingBuffer for TravelTimeAttention
            withdrawals:    WithdrawalData
            t:              current timestep index
            K_musk, x_musk: Muskingum params per reach
            dx:             ignored (kept for backward compat)
            lake_storage:   (n_nodes,) m3 — only lake nodes used
            area_km2:       (n_nodes,) km2 — cumulative area for lake routing
            dam_data:       optional regulated releases
            area_km2_local: (n_nodes,) km2 — local sub-watershed area for
                            lateral inflow conversion (mm/day → m³/s).
                            Falls back to area_km2 if None.
            Q_in_prev:      (n_nodes,) upstream inflow at previous timestep (m3/s).
                            Used by Muskingum for proper I(t-1) attenuation.
            H_lateral:      (n_nodes,) lateral heat load (m³·°C/s). If provided,
                            temperature is propagated through the network.
            T_air:          (n_nodes,) air temperature (°C) for atmospheric exchange.
            R_n:            (n_nodes,) net radiation (MJ/m²/day).
            K_atm:          (n_nodes,) atmospheric exchange coefficient (1/day).
        Returns:
            Q_out:            (n_nodes,) routed outflow (m3/s)
            lake_storage_new: updated lake storage or None
            T_water:          (n_nodes,) stream temperature (°C) or None
        """
        # Convert lateral inflow mm/day -> m3/s using LOCAL area, not cumulative.
        conv_area = area_km2_local if area_km2_local is not None else area_km2
        if conv_area is not None:
            q_lat_m3s = lateral_inflow * 1e-3 * conv_area * 1e6 / 86400.0
        else:
            q_lat_m3s = lateral_inflow

        net_W = withdrawals.net_withdrawal(t)  # (n_nodes,) m3/s
        lake_storage_new = lake_storage.clone() if lake_storage is not None else None

        # Temperature kwargs bundled for internal methods
        temp_kwargs = None
        if H_lateral is not None:
            temp_kwargs = {
                "H_lateral": H_lateral,
                "T_air": T_air,
                "R_n": R_n,
                "K_atm": K_atm,
            }

        # TTA vectorized path: all edges computed in a single batched call.
        # The sequential path (Python loop) is removed; it was unnecessary because
        # TTA reads only from the ring buffer (past timesteps), so there is no
        # ordering dependency within a single routing step.
        if self.use_tta and len(outflow_buffer) > 0:
            Q_out, lake_storage_new, _ = self._route_vectorized_tta(
                q_lat_m3s, graph, outflow_buffer, net_W,
                K_musk, x_musk, lake_storage_new, area_km2, dam_data, t,
            )
            T_water = (
                self._temperature_sweep(Q_out, q_lat_m3s, graph, temp_kwargs)
                if temp_kwargs is not None else None
            )
            return Q_out, lake_storage_new, T_water

        return self._route_vectorized(
            q_lat_m3s, graph, Q_out_prev, net_W,
            K_musk, x_musk, lake_storage_new, area_km2, dam_data, t,
            Q_in_prev=Q_in_prev,
            temp_kwargs=temp_kwargs,
        )

    # ------------------------------------------------------------------
    # Vectorized TTA path  (GPU-friendly: all edges in one batched call)
    # ------------------------------------------------------------------

    def _route_vectorized_tta(
        self,
        q_lat_m3s: Tensor,
        graph: RiverGraph,
        outflow_buffer: OutflowRingBuffer,
        net_W: Tensor,
        K_musk: Tensor,
        x_musk: Tensor,
        lake_storage_new: Tensor | None,
        area_km2: Tensor | None,
        dam_data: DamData | None,
        t: int,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        """TTA routing: batched attention over all edges, no Python loop.

        All edges are processed simultaneously using ``TravelTimeAttention.
        forward_edges``.  Per-edge contributions are scatter-added to
        destination nodes, then Muskingum / lake routing proceeds as in the
        vectorized path.
        """
        device = q_lat_m3s.device
        n_nodes = graph.n_nodes

        if graph.n_edges == 0:
            Q_agg = torch.zeros(n_nodes, device=device)
        else:
            src = graph.edge_index[0]   # (n_edges,)
            dst = graph.edge_index[1]   # (n_edges,)
            taus = graph.travel_time_days  # (n_edges,) int64

            # Fetch histories for all source nodes at once
            max_tau_val = max(int(taus.max().item()), 1)
            all_hist = outflow_buffer.get_all_history(max_tau_val)  # (actual_tau, n_nodes)

            # Pad to max_tau_val with leading zeros if buffer not yet full
            if all_hist.shape[0] < max_tau_val:
                pad = torch.zeros(max_tau_val - all_hist.shape[0], n_nodes, device=device)
                all_hist = torch.cat([pad, all_hist], dim=0)  # (max_tau_val, n_nodes)

            # Per-edge padded history: (n_edges, max_tau_val, 1)
            edge_hist = all_hist[:, src].permute(1, 0).unsqueeze(-1)

            # Validity mask: position t is valid when t >= max_tau_val - tau_e
            t_idx = torch.arange(max_tau_val, device=device).unsqueeze(0)  # (1, max_tau_val)
            tau_thresh = (max_tau_val - taus).unsqueeze(1)                  # (n_edges, 1)
            hist_mask = t_idx >= tau_thresh                                 # (n_edges, max_tau_val)

            edge_queries = q_lat_m3s[dst].unsqueeze(-1)  # (n_edges, 1)

            edge_contrib = self.tta.forward_edges(
                edge_queries, edge_hist, taus, hist_mask
            )  # (n_edges,)

            # TTA aggregation
            Q_tta = torch.zeros(n_nodes, device=device)
            Q_tta.scatter_add_(0, dst, edge_contrib.float())

            # Simple sum aggregation (same as _route_vectorized)
            Q_out_proxy = outflow_buffer._buf[
                (outflow_buffer._ptr - 1) % outflow_buffer.depth
            ]
            Q_simple = torch.zeros(n_nodes, device=device)
            Q_simple.scatter_add_(0, dst, Q_out_proxy[src].float())

            # Blend: factor 0 = pure simple sum, 1 = pure TTA
            alpha = self.tta_warmup_factor
            Q_agg = (1.0 - alpha) * Q_simple + alpha * Q_tta

        # Upstream inflow + withdrawals (allow deficit — clamp Q_out later)
        Q_in_upstream = Q_agg + net_W
        # Total inflow for lake mass balance (lakes don't use Muskingum)
        Q_in_total = Q_agg + q_lat_m3s + net_W

        # River nodes: Muskingum
        Q_out = torch.zeros(n_nodes, device=device, dtype=Q_agg.dtype)
        # Use Q_agg as implicit Q_out_prev proxy — reuse last pushed buffer value
        Q_out_proxy = outflow_buffer._buf[(outflow_buffer._ptr - 1) % outflow_buffer.depth]
        river_mask = ~graph.is_lake
        if river_mask.any():
            Q_out[river_mask] = self.muskingum(
                Q_in_upstream[river_mask],
                Q_out_proxy[river_mask],
                q_lat_m3s[river_mask],
                K_musk[river_mask],
                x_musk[river_mask],
            )

        # Lake nodes
        if lake_storage_new is not None and graph.is_lake.any():
            lake_mask = graph.is_lake
            n_lakes = int(lake_mask.sum())
            zeros_l = torch.zeros(n_lakes, device=device)
            area_l = area_km2[lake_mask] if area_km2 is not None else torch.ones(n_lakes, device=device)

            Q_lake, S_lake = self.lake(
                Q_in_total[lake_mask],
                lake_storage_new[lake_mask],
                area_l,
                E_lake=zeros_l,
                P_lake=zeros_l,
                S_dead=zeros_l,
            )

            if dam_data is not None:
                forced = dam_data.releases[t][lake_mask]
                regulated = ~torch.isnan(forced)
                if regulated.any():
                    Q_lake = torch.where(regulated, forced, Q_lake)
                    S_reg_new = torch.clamp(
                        lake_storage_new[lake_mask] + (Q_in_total[lake_mask] - forced) * 86400.0,
                        min=0.0,
                    )
                    S_lake = torch.where(regulated, S_reg_new, S_lake)

            Q_out[lake_mask] = Q_lake
            lake_storage_new[lake_mask] = S_lake.detach()

        # Clamp after routing: withdrawals can reduce Q to 0 but not below
        Q_out = torch.clamp(Q_out, min=0.0)

        return Q_out, lake_storage_new, None

    # ------------------------------------------------------------------
    # Vectorized path  (O(n_edges) tensor ops, no Python loop over nodes)
    # ------------------------------------------------------------------

    def _route_vectorized(
        self,
        q_lat_m3s: Tensor,
        graph: RiverGraph,
        Q_out_prev: Tensor,
        net_W: Tensor,
        K_musk: Tensor,
        x_musk: Tensor,
        lake_storage_new: Tensor | None,
        area_km2: Tensor | None,
        dam_data: DamData | None,
        t: int,
        Q_in_prev: Tensor | None = None,
        temp_kwargs: dict | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        """Route water using level-by-level topological sweep.

        Processes nodes in topological order so that within a single daily
        timestep, upstream outflow propagates through the entire network.
        Without this, the explicit scheme limits propagation to 1 hop/day,
        causing 71+ day delays for deep stations in the SLSO network.

        When temp_kwargs is provided, heat loads (H = Q * T) are propagated
        alongside discharge through the same topological sweep.
        """
        device = q_lat_m3s.device
        n_nodes = graph.n_nodes
        Q_out = torch.zeros(n_nodes, device=device)

        # Temperature tracking
        do_temp = temp_kwargs is not None
        if do_temp:
            H_out = torch.zeros(n_nodes, device=device)  # heat load per node
            H_lateral = temp_kwargs["H_lateral"]
            T_air = temp_kwargs["T_air"]
            R_n = temp_kwargs["R_n"]
            K_atm = temp_kwargs["K_atm"]
            _eps = 1e-6

        # Pre-compute topological levels with per-level edge info (cached)
        if not hasattr(graph, '_topo_level_data') or graph._topo_level_data is None:
            graph._topo_level_data = self._build_topo_level_data(graph)

        level_data = graph._topo_level_data

        for level_nodes, edge_src, edge_dst_local in level_data:
            n_level = len(level_nodes)

            # Vectorized upstream aggregation via scatter_add
            Q_agg = torch.zeros(n_level, device=device)
            if edge_src is not None:
                Q_agg.scatter_add_(0, edge_dst_local, Q_out[edge_src])

            # Heat load aggregation (same scatter pattern)
            if do_temp:
                H_agg = torch.zeros(n_level, device=device)
                if edge_src is not None:
                    H_agg.scatter_add_(0, edge_dst_local, H_out[edge_src])

            Q_in_upstream = Q_agg + net_W[level_nodes]  # allow deficit
            q_lat_level = q_lat_m3s[level_nodes]

            # Separate river and lake nodes within this level
            is_lake_level = graph.is_lake[level_nodes]
            river_in_level = ~is_lake_level
            lake_in_level = is_lake_level

            # River nodes: Muskingum
            if river_in_level.any():
                river_idx = level_nodes[river_in_level]
                Q_out[river_idx] = self.muskingum(
                    Q_in_upstream[river_in_level],
                    Q_out_prev[river_idx],
                    q_lat_level[river_in_level],
                    K_musk[river_idx],
                    x_musk[river_idx],
                )

            # Lake nodes: storage-discharge
            if lake_in_level.any() and lake_storage_new is not None:
                lake_idx = level_nodes[lake_in_level]
                n_l = int(lake_in_level.sum())
                zeros_l = torch.zeros(n_l, device=device)
                area_l = area_km2[lake_idx] if area_km2 is not None else torch.ones(n_l, device=device)
                Q_in_total = (
                    Q_agg[lake_in_level] + q_lat_level[lake_in_level] + net_W[lake_idx]
                )

                Q_lake, S_lake = self.lake(
                    Q_in_total,
                    lake_storage_new[lake_idx],
                    area_l,
                    E_lake=zeros_l,
                    P_lake=zeros_l,
                    S_dead=zeros_l,
                )

                if dam_data is not None:
                    forced = dam_data.releases[t][lake_idx]
                    regulated = ~torch.isnan(forced)
                    if regulated.any():
                        Q_lake = torch.where(regulated, forced, Q_lake)
                        S_reg_new = torch.clamp(
                            lake_storage_new[lake_idx] + (Q_in_total - forced) * 86400.0,
                            min=0.0,
                        )
                        S_lake = torch.where(regulated, S_reg_new, S_lake)

                Q_out[lake_idx] = Q_lake
                lake_storage_new[lake_idx] = S_lake.detach()

            # Clamp after routing: withdrawals can reduce Q to 0 but not below
            Q_out[level_nodes] = torch.clamp(Q_out[level_nodes], min=0.0)

            # Temperature: mix upstream + lateral heat, then atmospheric exchange
            if do_temp:
                H_total = H_agg + H_lateral[level_nodes]
                Q_total = Q_agg + q_lat_level
                # Mixed temperature from advection
                T_mix = H_total / (Q_total + _eps)
                # Atmospheric exchange: relax toward equilibrium
                T_eq = T_air[level_nodes] + R_n[level_nodes] * 0.3
                T_node = T_mix + K_atm[level_nodes] * (T_eq - T_mix)
                T_node = torch.clamp(T_node, min=0.0, max=40.0)
                # Outgoing heat load = Q_out * T_node
                H_out[level_nodes] = Q_out[level_nodes] * T_node

        T_water = None
        if do_temp:
            # Recover temperature from heat load / discharge
            T_water = H_out / (Q_out + _eps)
            T_water = torch.clamp(T_water, min=0.0, max=40.0)

        return Q_out, lake_storage_new, T_water

    def _temperature_sweep(
        self,
        Q_out: Tensor,
        q_lat_m3s: Tensor,
        graph: RiverGraph,
        temp_kwargs: dict,
    ) -> Tensor:
        """Topological heat-load sweep to compute stream temperature.

        Separated from the routing sweep so it can be called after any routing
        path (Muskingum or TTA).  Uses the already-computed Q_out to partition
        heat loads downstream.
        """
        device = Q_out.device
        n_nodes = graph.n_nodes
        H_out = torch.zeros(n_nodes, device=device)

        H_lateral = temp_kwargs["H_lateral"]
        T_air     = temp_kwargs["T_air"]
        R_n       = temp_kwargs["R_n"]
        K_atm     = temp_kwargs["K_atm"]
        _eps = 1e-6

        if not hasattr(graph, '_topo_level_data') or graph._topo_level_data is None:
            graph._topo_level_data = self._build_topo_level_data(graph)

        for level_nodes, edge_src, edge_dst_local in graph._topo_level_data:
            n_level = len(level_nodes)

            H_agg = torch.zeros(n_level, device=device)
            Q_agg = torch.zeros(n_level, device=device)
            if edge_src is not None:
                H_agg.scatter_add_(0, edge_dst_local, H_out[edge_src])
                Q_agg.scatter_add_(0, edge_dst_local, Q_out[edge_src])

            q_lat_level = q_lat_m3s[level_nodes]
            H_total = H_agg + H_lateral[level_nodes]
            Q_total = Q_agg + q_lat_level
            T_mix = H_total / (Q_total + _eps)

            T_eq  = T_air[level_nodes] + R_n[level_nodes] * 0.3
            T_node = T_mix + K_atm[level_nodes] * (T_eq - T_mix)
            T_node = torch.clamp(T_node, min=0.0, max=40.0)

            H_out[level_nodes] = Q_out[level_nodes] * T_node

        T_water = H_out / (Q_out + _eps)
        return torch.clamp(T_water, min=0.0, max=40.0)

    @staticmethod
    def _build_topo_level_data(
        graph: RiverGraph,
    ) -> list[tuple[Tensor, Tensor | None, Tensor | None]]:
        """Build topological levels with per-level edge info for vectorized scatter.

        Returns list of (level_nodes, edge_src, edge_dst_local) tuples.
        - level_nodes: (n_level,) global node indices
        - edge_src: (n_edges_into_level,) global source node indices
        - edge_dst_local: (n_edges_into_level,) LOCAL index within level_nodes
        Both edge tensors are None if no edges feed into this level.
        """
        n = graph.n_nodes
        device = graph.edge_index.device if graph.n_edges > 0 else torch.device('cpu')

        # Build adjacency and incoming edges on CPU
        in_degree = [0] * n
        children: list[list[int]] = [[] for _ in range(n)]
        dst_to_src: dict[int, list[int]] = {}
        if graph.n_edges > 0:
            src_list = graph.edge_index[0].tolist()
            dst_list = graph.edge_index[1].tolist()
            for s, d in zip(src_list, dst_list):
                in_degree[d] += 1
                children[s].append(d)
                dst_to_src.setdefault(d, []).append(s)

        # Kahn's algorithm
        result: list[tuple[Tensor, Tensor | None, Tensor | None]] = []
        queue = [i for i in range(n) if in_degree[i] == 0]

        while queue:
            level_nodes = torch.tensor(queue, dtype=torch.long, device=device)

            # Build per-level edge lists for scatter_add
            edge_srcs: list[int] = []
            edge_dst_locals: list[int] = []
            for local_idx, node in enumerate(queue):
                srcs = dst_to_src.get(node)
                if srcs:
                    for s in srcs:
                        edge_srcs.append(s)
                        edge_dst_locals.append(local_idx)

            if edge_srcs:
                e_src = torch.tensor(edge_srcs, dtype=torch.long, device=device)
                e_dst = torch.tensor(edge_dst_locals, dtype=torch.long, device=device)
            else:
                e_src = None
                e_dst = None

            result.append((level_nodes, e_src, e_dst))

            next_queue: list[int] = []
            for node in queue:
                for child in children[node]:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        next_queue.append(child)
            queue = next_queue

        return result

