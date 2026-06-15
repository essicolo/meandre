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

from typing import NamedTuple

import torch
import torch.nn as nn
from torch import Tensor


class TopoLevelData(NamedTuple):
    """Pre-built flat tensors for level-by-level routing.

    Each per-level slice is obtained by indexing ``river_idx[
    river_offsets[L]:river_offsets[L+1]]`` (same for lake, edge). Slicing is
    O(1) (creates a view), avoiding the per-iteration overhead of unpacking
    Python list-of-tuples and computing river/lake masks on every level.
    """
    n_levels: int
    n_nodes: int
    river_idx: Tensor          # (sum_L n_river_L,) global indices, in level order
    river_offsets: Tensor      # (n_levels+1,) prefix sums into river_idx
    lake_idx: Tensor           # (sum_L n_lake_L,) global indices, in level order
    lake_offsets: Tensor       # (n_levels+1,) prefix sums into lake_idx
    edge_src: Tensor           # (n_edges,) src nodes, grouped by destination level
    edge_dst_global: Tensor    # (n_edges,) GLOBAL dst node indices — same buffer reuse across levels
    edge_offsets: Tensor       # (n_levels+1,) prefix sums into edge_src/edge_dst_global

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
        routing_mode: str = "level",
        routing_substeps: int = 2,
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
        self.muskingum = MuskingumCunge(n_substeps=routing_substeps)
        self.lake = LakeModule()
        # Routage : "level" (balayage par niveau), "operator" (solve par
        # étages de lacs, sémantique identique) ou "operator-lagged" (lacs
        # sur stockage de la veille, un seul solve). Cf operator_routing.py.
        if routing_mode not in ("level", "operator", "operator-lagged"):
            raise ValueError(f"routing_mode inconnu : {routing_mode!r}")
        self.routing_mode = routing_mode
        self._op_state = None    # opérateur du forward courant (rebâti à t=0)
        # Célérité dépendante du débit (Muskingum-Cunge non-linéaire, type onde
        # cinématique d'Hydrotel). Diagnostic 2026-06-15 : la GÉNÉRATION est
        # parfaite (peak_ratio 0.997 en routage instantané), tout le déficit de
        # pic (→0.74) vient de la diffusion du Muskingum LINÉAIRE à célérité
        # constante. Ici K_eff = K · (Qref/(Q+Qref))^β baisse à haut débit, donc
        # le PIC voyage plus vite et s'atténue moins, l'étiage garde son K lent.
        # Qref = qref_specific · aire (échelle de débit propre au tronçon).
        self.dq_celerity = False
        self.dq_beta = 0.4
        self.dq_qref_specific = 0.01   # m³/s par km²
        self.dq_kmin_frac = 0.15       # K_eff ≥ 15 % de K_base (stabilité)
        # Advection PURE (plug-flow / onde cinématique sans diffusion) : c01=1,
        # c2=0 → Q_out = Q_amont + apport, AUCUNE atténuation. Le Muskingum est
        # un schéma DIFFUSIF dont l'atténuation est verrouillée à bas x par la
        # stabilité (2Kx ≤ dt) ; il rabote les pics par construction. Hydrotel/
        # Raven préservent les pics avec une onde cinématique = advection pure.
        self.pure_advection = False
        # Atténuation DYNAMIQUE selon le régime (résout le mur de Pareto) : le
        # Muskingum lisse (atténue) à l'étiage pour le kge, mais NE doit PAS
        # atténuer en crue pour préserver le pic. Un coefficient statique ne peut
        # pas les deux. Ici c2_eff = c2·(1−α(Q)) avec α→1 à haut débit : pleine
        # atténuation à l'étiage, zéro atténuation en crue. STABLE (c2_eff ∈
        # [0, c2_musk]) contrairement aux tweaks de K/x. Piloté par le débit
        # retardé (un GRU pourra piloter α plus finement ensuite).
        self.dynamic_atten = False
        self.da_beta = 2.0           # raideur de la rampe α(Q)
        self.da_qref_specific = 0.05  # m³/s par km² (échelle de crue)
        # Params de lac par nœud (k_lake, beta) sortis du NeRF, posés par
        # HydroModel avant la boucle (constants dans le temps). None = scalaires
        # globaux du LakeModule (rétrocompat).
        self._lake_k = None
        self._lake_beta = None

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

        # Precompute Muskingum coefficients once per routing step. They depend
        # only on K, x, dt — invariant across the topological level loop. Saves
        # ~5 ops per node per level (denom/c0/c1/c2/clamp/scale) which on slso
        # (109 levels × 2 substeps) is the dominant per-timestep cost.
        # Onde cinématique d'Hydrotel : célérité c = (5/3)·v ∝ Q^0.4, donc le
        # temps de parcours K = dx/c ∝ Q^-0.4 BAISSE à haut débit → le pic
        # voyage plus vite et s'atténue moins. K_eff = K_base·(Qref/(Q+Qref))^β.
        # STABILITÉ : K_eff borné ≥ dt_sub (baisser K sous le sous-pas déstabilise
        # le Muskingum) — d'où la nécessité de sous-pas FINS (routing_substeps).
        if self.pure_advection:
            # Translation pure, zéro atténuation (onde cinématique limite).
            c01 = torch.ones_like(K_musk)
            c2 = torch.zeros_like(K_musk)
        elif self.dynamic_atten and area_km2 is not None and Q_out_prev is not None:
            # Atténuation dépendante du régime : c2_eff = c2·(1−α), α→1 en crue.
            c01, c2 = self.muskingum.precompute_coefficients(K_musk, x_musk)
            Qref = self.da_qref_specific * area_km2 + 1e-3
            q = Q_out_prev.clamp(min=0.0)
            alpha = (q / (q + Qref)) ** self.da_beta     # ∈[0,1), monte avec Q
            c2 = c2 * (1.0 - alpha)                       # moins d'atténuation en crue
            c01 = 1.0 - c2                                # conservation (c01+c2=1)
        elif self.dq_celerity and area_km2 is not None and Q_out_prev is not None:
            Qref = self.dq_qref_specific * area_km2 + 1e-3
            factor = (Qref / (Q_out_prev.clamp(min=0.0) + Qref)) ** self.dq_beta
            dt_sub = self.muskingum.dt / self.muskingum.n_substeps  # secondes
            K_eff = torch.clamp(K_musk * factor, min=dt_sub)
            c01, c2 = self.muskingum.precompute_coefficients(K_eff, x_musk)
        else:
            c01, c2 = self.muskingum.precompute_coefficients(K_musk, x_musk)

        # Routage par opérateur (solve triangulaire précalculé) : actif quand
        # demandé, hors thermie (non portée) et hors TTA actif. L'opérateur
        # est rebâti à t=0 (début de chaque simulate/chunk) et réutilisé sur
        # tous les pas — c01/c2 sont invariants dans le temps.
        if self.routing_mode != "level" and (
                temp_kwargs is not None or (self.use_tta and len(outflow_buffer) > 0)):
            if not getattr(self, "_op_fallback_warned", False):
                self._op_fallback_warned = True
                why = "thermie active" if temp_kwargs is not None else "TTA actif"
                print(f"[routing] routing_mode={self.routing_mode!r} ignoré ({why}) "
                      f"— fallback balayage par niveau. Couper use_temperature "
                      f"pour activer l'opérateur.", flush=True)
        if (self.routing_mode != "level" and temp_kwargs is None
                and not (self.use_tta and len(outflow_buffer) > 0)):
            from meandre.routing.operator_routing import (
                build_operator_topo, build_operator_state, route_operator,
            )
            lagged = self.routing_mode == "operator-lagged"
            topo_cache = getattr(graph, "_operator_topo", None)
            if topo_cache is None or getattr(graph, "_operator_topo_lagged", None) != lagged:
                graph._operator_topo = build_operator_topo(graph, lagged)
                graph._operator_topo_lagged = lagged
            # fp32 forcé : le solve/GEMV sous autocast bf16 mélange les dtypes
            # et dégrade les récurrences longues.
            with torch.autocast(device_type=q_lat_m3s.device.type, enabled=False):
                # Coefficients constants → opérateur bâti une fois (t=0). Avec
                # la célérité dépendante du débit, c01/c2 changent chaque pas →
                # on rebâtit l'opérateur à chaque pas (coût accepté ; à optimiser).
                if t == 0 or self._op_state is None or self.dq_celerity or self.dynamic_atten:
                    self._op_state = build_operator_state(
                        graph._operator_topo, c01, c2, self.muskingum.n_substeps,
                    )
                Q_out = route_operator(
                    self, graph._operator_topo, self._op_state, q_lat_m3s,
                    Q_out_prev, net_W, lake_storage_new, area_km2, dam_data, t,
                    lagged,
                )
            return Q_out, lake_storage_new, None

        # TTA vectorized path: all edges computed in a single batched call.
        # The sequential path (Python loop) is removed; it was unnecessary because
        # TTA reads only from the ring buffer (past timesteps), so there is no
        # ordering dependency within a single routing step.
        if self.use_tta and len(outflow_buffer) > 0:
            Q_out, lake_storage_new, _ = self._route_vectorized_tta(
                q_lat_m3s, graph, outflow_buffer, net_W,
                c01, c2, lake_storage_new, area_km2, dam_data, t,
            )
            T_water = (
                self._temperature_sweep(Q_out, q_lat_m3s, graph, temp_kwargs)
                if temp_kwargs is not None else None
            )
            return Q_out, lake_storage_new, T_water

        return self._route_vectorized(
            q_lat_m3s, graph, Q_out_prev, net_W,
            c01, c2, lake_storage_new, area_km2, dam_data, t,
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
        c01: Tensor,
        c2: Tensor,
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

        # River nodes: Muskingum (cached coefficients path)
        Q_out = torch.zeros(n_nodes, device=device, dtype=Q_agg.dtype)
        # Use Q_agg as implicit Q_out_prev proxy — reuse last pushed buffer value
        Q_out_proxy = outflow_buffer._buf[(outflow_buffer._ptr - 1) % outflow_buffer.depth]
        river_mask = ~graph.is_lake
        if river_mask.any():
            Q_out[river_mask] = self.muskingum.forward_cached(
                Q_in_upstream[river_mask],
                Q_out_proxy[river_mask],
                q_lat_m3s[river_mask],
                c01[river_mask],
                c2[river_mask],
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
                k_lake=self._lake_k[lake_mask] if self._lake_k is not None else None,
                beta=self._lake_beta[lake_mask] if self._lake_beta is not None else None,
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
        c01: Tensor,
        c2: Tensor,
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

        # Pre-compute topological levels (cached on the graph)
        if not hasattr(graph, '_topo_level_data') or graph._topo_level_data is None:
            graph._topo_level_data = self._build_topo_level_data(graph)
        topo: TopoLevelData = graph._topo_level_data

        # Single global buffers reused across levels (no per-level zeros() alloc).
        Q_out = torch.zeros(n_nodes, device=device)
        Q_agg = torch.zeros(n_nodes, device=device)

        do_temp = temp_kwargs is not None
        if do_temp:
            H_out = torch.zeros(n_nodes, device=device)
            H_agg = torch.zeros(n_nodes, device=device)
            H_lateral = temp_kwargs["H_lateral"]
            T_air = temp_kwargs["T_air"]
            R_n = temp_kwargs["R_n"]
            K_atm = temp_kwargs["K_atm"]
            _eps = 1e-6

        # Cache CPU-side offsets (Python ints) for the loop — avoids GPU→CPU sync
        # on every slice operation.
        riv_off = topo.river_offsets.tolist()
        lak_off = topo.lake_offsets.tolist()
        edg_off = topo.edge_offsets.tolist()

        for L in range(topo.n_levels):
            e_lo, e_hi = edg_off[L], edg_off[L + 1]
            if e_hi > e_lo:
                e_src = topo.edge_src[e_lo:e_hi]
                e_dst = topo.edge_dst_global[e_lo:e_hi]
                Q_agg.scatter_add_(0, e_dst, Q_out[e_src])
                if do_temp:
                    H_agg.scatter_add_(0, e_dst, H_out[e_src])

            # River nodes at this level (cached Muskingum, inlined)
            r_lo, r_hi = riv_off[L], riv_off[L + 1]
            if r_hi > r_lo:
                ri = topo.river_idx[r_lo:r_hi]
                Q_in = Q_agg[ri] + net_W[ri]
                q_lat_sub = q_lat_m3s[ri] / self.muskingum.n_substeps
                Q = Q_out_prev[ri]
                for _ in range(self.muskingum.n_substeps):
                    Q = torch.clamp(c01[ri] * Q_in + c2[ri] * Q + q_lat_sub, min=0.0)
                Q_out[ri] = Q

            # Lake nodes at this level
            l_lo, l_hi = lak_off[L], lak_off[L + 1]
            if l_hi > l_lo and lake_storage_new is not None:
                li = topo.lake_idx[l_lo:l_hi]
                n_l = l_hi - l_lo
                zeros_l = torch.zeros(n_l, device=device)
                area_l = area_km2[li] if area_km2 is not None else torch.ones(n_l, device=device)
                Q_in_total = Q_agg[li] + q_lat_m3s[li] + net_W[li]

                Q_lake, S_lake = self.lake(
                    Q_in_total,
                    lake_storage_new[li],
                    area_l,
                    E_lake=zeros_l,
                    P_lake=zeros_l,
                    S_dead=zeros_l,
                    k_lake=self._lake_k[li] if self._lake_k is not None else None,
                    beta=self._lake_beta[li] if self._lake_beta is not None else None,
                )

                if dam_data is not None:
                    forced = dam_data.releases[t][li]
                    regulated = ~torch.isnan(forced)
                    if regulated.any():
                        Q_lake = torch.where(regulated, forced, Q_lake)
                        S_reg_new = torch.clamp(
                            lake_storage_new[li] + (Q_in_total - forced) * 86400.0,
                            min=0.0,
                        )
                        S_lake = torch.where(regulated, S_reg_new, S_lake)

                Q_out[li] = torch.clamp(Q_lake, min=0.0)
                lake_storage_new[li] = S_lake.detach()

            # Temperature: per-level downstream propagation
            if do_temp:
                # Combine river+lake indices for this level for the temp update
                # (cheap concat — usually one is empty)
                all_idx_L = torch.cat([topo.river_idx[r_lo:r_hi], topo.lake_idx[l_lo:l_hi]])
                if all_idx_L.numel() > 0:
                    H_total = H_agg[all_idx_L] + H_lateral[all_idx_L]
                    Q_total = Q_agg[all_idx_L] + q_lat_m3s[all_idx_L]
                    T_mix = H_total / (Q_total + _eps)
                    T_eq = T_air[all_idx_L] + R_n[all_idx_L] * 0.3
                    T_node = T_mix + K_atm[all_idx_L] * (T_eq - T_mix)
                    T_node = torch.clamp(T_node, min=0.0, max=40.0)
                    H_out[all_idx_L] = Q_out[all_idx_L] * T_node

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
        topo: TopoLevelData = graph._topo_level_data

        H_agg = torch.zeros(n_nodes, device=device)
        Q_agg = torch.zeros(n_nodes, device=device)

        riv_off = topo.river_offsets.tolist()
        lak_off = topo.lake_offsets.tolist()
        edg_off = topo.edge_offsets.tolist()

        for L in range(topo.n_levels):
            e_lo, e_hi = edg_off[L], edg_off[L + 1]
            if e_hi > e_lo:
                e_src = topo.edge_src[e_lo:e_hi]
                e_dst = topo.edge_dst_global[e_lo:e_hi]
                H_agg.scatter_add_(0, e_dst, H_out[e_src])
                Q_agg.scatter_add_(0, e_dst, Q_out[e_src])

            r_lo, r_hi = riv_off[L], riv_off[L + 1]
            l_lo, l_hi = lak_off[L], lak_off[L + 1]
            all_idx_L = torch.cat([topo.river_idx[r_lo:r_hi], topo.lake_idx[l_lo:l_hi]])
            if all_idx_L.numel() > 0:
                H_total = H_agg[all_idx_L] + H_lateral[all_idx_L]
                Q_total = Q_agg[all_idx_L] + q_lat_m3s[all_idx_L]
                T_mix = H_total / (Q_total + _eps)
                T_eq = T_air[all_idx_L] + R_n[all_idx_L] * 0.3
                T_node = T_mix + K_atm[all_idx_L] * (T_eq - T_mix)
                T_node = torch.clamp(T_node, min=0.0, max=40.0)
                H_out[all_idx_L] = Q_out[all_idx_L] * T_node

        T_water = H_out / (Q_out + _eps)
        return torch.clamp(T_water, min=0.0, max=40.0)

    @staticmethod
    def _build_topo_level_data(
        graph: RiverGraph,
    ) -> "TopoLevelData":
        """Build topological levels for level-by-level routing (Kahn's algorithm).

        Returns a ``TopoLevelData`` namedtuple of flat tensors with offsets per
        level. River vs lake separation is precomputed once. Edge destinations
        are stored as GLOBAL node indices (not per-level local indices) so the
        accumulation can share a single global Q_agg buffer across all levels —
        eliminates ``torch.zeros(n_level)`` allocations on every iteration.
        """
        import numpy as np

        n = graph.n_nodes
        device = graph.edge_index.device if graph.n_edges > 0 else torch.device('cpu')
        is_lake_cpu = graph.is_lake.cpu().numpy().astype(bool)

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

        # Kahn's algorithm to produce per-level node lists
        per_level_nodes: list[list[int]] = []
        queue = [i for i in range(n) if in_degree[i] == 0]
        while queue:
            per_level_nodes.append(queue)
            next_queue: list[int] = []
            for node in queue:
                for child in children[node]:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        next_queue.append(child)
            queue = next_queue

        n_levels = len(per_level_nodes)

        # Flat node lists (global indices), separated river/lake per level.
        river_per_level: list[list[int]] = []
        lake_per_level: list[list[int]] = []
        for nodes_L in per_level_nodes:
            riv = [i for i in nodes_L if not is_lake_cpu[i]]
            lak = [i for i in nodes_L if is_lake_cpu[i]]
            river_per_level.append(riv)
            lake_per_level.append(lak)

        # Concatenate per-level node lists into flat tensors with offsets.
        river_flat: list[int] = []
        river_offsets = [0]
        for riv in river_per_level:
            river_flat.extend(riv)
            river_offsets.append(len(river_flat))

        lake_flat: list[int] = []
        lake_offsets = [0]
        for lak in lake_per_level:
            lake_flat.extend(lak)
            lake_offsets.append(len(lake_flat))

        # Flat edges: src and GLOBAL dst, one block per level.
        edge_src_flat: list[int] = []
        edge_dst_flat: list[int] = []
        edge_offsets = [0]
        for nodes_L in per_level_nodes:
            for node in nodes_L:
                for s in dst_to_src.get(node, ()):
                    edge_src_flat.append(s)
                    edge_dst_flat.append(node)  # GLOBAL dst index
            edge_offsets.append(len(edge_src_flat))

        return TopoLevelData(
            n_levels=n_levels,
            n_nodes=n,
            river_idx=torch.tensor(river_flat, dtype=torch.long, device=device),
            river_offsets=torch.tensor(river_offsets, dtype=torch.long, device=device),
            lake_idx=torch.tensor(lake_flat, dtype=torch.long, device=device),
            lake_offsets=torch.tensor(lake_offsets, dtype=torch.long, device=device),
            edge_src=torch.tensor(edge_src_flat, dtype=torch.long, device=device),
            edge_dst_global=torch.tensor(edge_dst_flat, dtype=torch.long, device=device),
            edge_offsets=torch.tensor(edge_offsets, dtype=torch.long, device=device),
        )

